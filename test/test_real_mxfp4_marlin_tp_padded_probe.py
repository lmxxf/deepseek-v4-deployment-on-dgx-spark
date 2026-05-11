import sys

import torch

from test_real_mxfp4_marlin_tp_probe import (
    _dequant,
    _metadata,
    _read_u8,
    CKPT,
)

sys.path.append("/work/spark-vllm-docker/vllm-sm120/vllm-sm120")

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types


class _Layer:
    params_dtype = torch.bfloat16


def _pad_rows(x, rows):
    out = torch.zeros((rows, x.shape[1]), dtype=x.dtype)
    out[: x.shape[0]].copy_(x)
    return out


def _pad_cols(x, cols):
    out = torch.zeros((x.shape[0], cols), dtype=x.dtype)
    out[:, : x.shape[1]].copy_(x)
    return out


def _run_rank_padded(a, topk_ids, topk_weights, tensors, tp_rank, padded_n=1536):
    w1s, w2s, w3s, s1s, s2s, s3s = tensors
    n_full = w1s[0].shape[0]
    n_part = n_full // 2
    lo = tp_rank * n_part
    hi = lo + n_part

    w13_parts = []
    s13_parts = []
    w2_parts = []
    s2_parts = []
    for w1, w2, w3, s1, s2, s3 in zip(w1s, w2s, w3s, s1s, s2s, s3s):
        w1_part = _pad_rows(w1[lo:hi], padded_n)
        w3_part = _pad_rows(w3[lo:hi], padded_n)
        s1_part = _pad_rows(s1[lo:hi], padded_n)
        s3_part = _pad_rows(s3[lo:hi], padded_n)
        w13_parts.append(torch.cat([w1_part, w3_part], dim=0))
        s13_parts.append(torch.cat([s1_part, s3_part], dim=0))

        w2_parts.append(_pad_cols(w2[:, lo // 2 : hi // 2], padded_n // 2))
        s2_parts.append(_pad_cols(s2[:, lo // 32 : hi // 32], padded_n // 32))

    w13 = torch.stack(w13_parts).cuda()
    w2 = torch.stack(w2_parts).cuda()
    s13 = torch.stack(s13_parts).cuda()
    s2 = torch.stack(s2_parts).cuda()
    w13_m, w2_m, s13_m, s2_m, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        _Layer(), w13, w2, s13, s2, None, None
    )
    with set_current_vllm_config(VllmConfig()):
        return fused_marlin_moe(
            a,
            w13_m,
            w2_m,
            None,
            None,
            s13_m,
            s2_m,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=len(w1s),
            quant_type_id=scalar_types.float4_e2m1f.id,
            input_dtype=torch.bfloat16,
            is_k_full=True,
            clamp_limit=10.0,
        )


def main():
    header_len, meta = _metadata(CKPT)
    experts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    topk = min(experts, int(sys.argv[3]) if len(sys.argv) > 3 else 6)

    w1s, w2s, w3s = [], [], []
    s1s, s2s, s3s = [], [], []
    for expert in range(experts):
        base = f"layers.0.ffn.experts.{expert}"
        w1s.append(_read_u8(CKPT, header_len, meta, f"{base}.w1.weight"))
        w2s.append(_read_u8(CKPT, header_len, meta, f"{base}.w2.weight"))
        w3s.append(_read_u8(CKPT, header_len, meta, f"{base}.w3.weight"))
        s1s.append(_read_u8(CKPT, header_len, meta, f"{base}.w1.scale"))
        s2s.append(_read_u8(CKPT, header_len, meta, f"{base}.w2.scale"))
        s3s.append(_read_u8(CKPT, header_len, meta, f"{base}.w3.scale"))

    tensors = (w1s, w2s, w3s, s1s, s2s, s3s)
    torch.manual_seed(123)
    a = (torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) / 10)
    ids = torch.arange(topk, dtype=torch.int64).repeat(m, 1)
    topk_ids = ids.cuda()
    weights = torch.arange(1, topk + 1, dtype=torch.float32)
    topk_weights = (weights / weights.sum()).repeat(m, 1).cuda()

    ref = torch.zeros((m, 4096), device="cuda", dtype=torch.bfloat16)
    for slot in range(topk):
        expert = int(ids[0, slot])
        ref_w1 = _dequant(w1s[expert], s1s[expert]).cuda().to(torch.bfloat16)
        ref_w2 = _dequant(w2s[expert], s2s[expert]).cuda().to(torch.bfloat16)
        ref_w3 = _dequant(w3s[expert], s3s[expert]).cuda().to(torch.bfloat16)
        gate = a @ ref_w1.T
        up = a @ ref_w3.T
        gate = torch.clamp(gate, max=10.0)
        up = torch.clamp(up, min=-10.0, max=10.0)
        ref += (torch.nn.functional.silu(gate) * up) @ ref_w2.T * topk_weights[
            :, slot : slot + 1
        ].to(torch.bfloat16)

    out0 = _run_rank_padded(a, topk_ids, topk_weights, tensors, 0)
    out1 = _run_rank_padded(a, topk_ids, topk_weights, tensors, 1)
    out = out0 + out1
    diff = (out.float() - ref.float()).abs()
    print("experts", experts, "m", m, "topk", topk, "padded_n", 1536)
    print("out0", out[0, :8].float().tolist())
    print("ref0", ref[0, :8].float().tolist())
    print("max_abs", diff.max().item())
    print("mean_abs", diff.mean().item())
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=0)


if __name__ == "__main__":
    main()
