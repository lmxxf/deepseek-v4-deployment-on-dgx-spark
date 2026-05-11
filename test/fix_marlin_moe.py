"""
Monkey-patch Marlin MoE kernel to use pure PyTorch on SM120.

Instead of calling ops.moe_wna16_marlin_gemm (which produces wrong results
on SM120), we intercept the fused_marlin_moe function and replace it with
a pure PyTorch implementation that dequantizes FP4 weights and does BF16 matmul.

Run inside the vllm container:
  python3 /workspace/fix_marlin_moe.py
"""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py"

with open(TARGET, "r") as f:
    content = f.read()

# We need to add a PyTorch fallback right at the beginning of _fused_marlin_moe
# The key insight: we can detect SM120 at runtime and use a different code path

PATCH_CODE = '''
# === SM120 PyTorch fallback for Marlin MoE ===
import os as _os
_SM120_FALLBACK = None

def _is_sm120():
    global _SM120_FALLBACK
    if _SM120_FALLBACK is None:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            _SM120_FALLBACK = (cap[0] == 12)
        else:
            _SM120_FALLBACK = False
    return _SM120_FALLBACK

def _dequant_mxfp4_weight(w_packed, w_scale, global_scale=None):
    """Dequantize MXFP4 packed weights to BF16.
    w_packed: [num_experts, N, K//2] uint8 (two FP4 values packed per byte)
    w_scale: [num_experts, N, K//32] uint8 (E8M0 block scales)
    Returns: [num_experts, N, K] bfloat16
    """
    import torch
    FP4_TABLE = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.float32, device=w_packed.device)

    shape = w_packed.shape  # [E, N, K//2]
    E, N, K_half = shape
    K = K_half * 2

    # Unpack two FP4 values from each uint8
    w_uint8 = w_packed.to(torch.uint8)
    low = w_uint8 & 0x0F
    high = (w_uint8 >> 4) & 0x0F

    # Lookup FP4 values
    low_vals = FP4_TABLE[low.to(torch.long)].to(torch.bfloat16)
    high_vals = FP4_TABLE[high.to(torch.long)].to(torch.bfloat16)

    # Interleave: [low0, high0, low1, high1, ...]
    result = torch.stack([low_vals, high_vals], dim=-1).reshape(E, N, K)

    # Apply block scales (E8M0 format, block_size=32)
    if w_scale is not None:
        # E8M0 to float: 2^(val - 127)
        scale_f = w_scale.to(torch.float32)
        scale_f = torch.pow(2.0, scale_f - 127.0)
        # Reshape scale to match blocks: [E, N, K//32, 1] -> broadcast to [E, N, K]
        n_blocks = K // 32
        result = result.reshape(E, N, n_blocks, 32)
        scale_f = scale_f.reshape(E, N, n_blocks, 1)
        result = (result * scale_f).reshape(E, N, K).to(torch.bfloat16)

    # Apply global scale if present
    if global_scale is not None:
        result = result * global_scale.to(torch.bfloat16)

    return result

def _pytorch_moe_gemm(
    hidden_states,  # [M, K] bf16
    w,              # [E, N, K] packed uint8
    w_scale,        # [E, N, K//32] scale
    global_scale,   # scalar or None
    topk_weights,   # [M, topk]
    topk_ids,       # [M, topk]
    activation_func,
    activation,
    w13_num_shards,
    N,
    K,
    apply_router_weight_on_input,
):
    """Pure PyTorch MoE GEMM - correct but slow."""
    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import apply_moe_activation

    M = hidden_states.shape[0]
    num_topk = topk_ids.shape[1]

    # Dequantize all expert weights
    w_deq = _dequant_mxfp4_weight(w, w_scale, global_scale)  # [E, N, K]

    # For each token, compute weighted sum over selected experts
    output = torch.zeros(M, K if w13_num_shards == 1 else N,
                         device=hidden_states.device, dtype=torch.bfloat16)

    for i in range(num_topk):
        expert_ids = topk_ids[:, i]  # [M]
        weights = topk_weights[:, i:i+1]  # [M, 1]

        # Gather expert weights for this topk slot
        unique_experts = expert_ids.unique()
        for eid in unique_experts:
            mask = (expert_ids == eid)
            if not mask.any():
                continue
            tokens = hidden_states[mask]  # [m, K]
            expert_w = w_deq[eid]  # [N, K]
            # GEMM: [m, K] @ [K, N] -> [m, N]
            out = torch.mm(tokens.to(torch.float32), expert_w.t().to(torch.float32))
            if apply_router_weight_on_input:
                out = out * weights[mask].to(torch.float32)
            output[mask] += out.to(torch.bfloat16)

    if not apply_router_weight_on_input:
        output = output  # weights applied later

    return output

# === End SM120 fallback ===
'''

# Insert the patch code right before the _fused_marlin_moe function
insert_point = "def _fused_marlin_moe("
if "_SM120_FALLBACK" in content:
    print("Already patched!")
else:
    idx = content.index(insert_point)
    content = content[:idx] + PATCH_CODE + "\n" + content[idx:]
    with open(TARGET, "w") as f:
        f.write(content)
    print("Patched: added SM120 fallback functions")
    print("NOTE: The actual interception of ops.moe_wna16_marlin_gemm")
    print("still needs to be wired up. This patch only adds the helper functions.")
