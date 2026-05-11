import json
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


class _DummyMark:
    def __getattr__(self, _name):
        def deco(*_args, **_kwargs):
            if _args and callable(_args[0]) and len(_args) == 1 and not _kwargs:
                return _args[0]
            return lambda f: f

        return deco


sys.modules.setdefault("pytest", types.SimpleNamespace(mark=_DummyMark()))

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types


ROOT = Path("/root/.cache/huggingface/deepseek-v4-flash")
INDEX = ROOT / "model.safetensors.index.json"


def _metadata(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return header_len, json.loads(f.read(header_len))


_meta_cache = {}


def _read_u8(name):
    weight_map = json.load(open(INDEX))["weight_map"]
    path = ROOT / weight_map[name]
    if path not in _meta_cache:
        _meta_cache[path] = _metadata(path)
    header_len, meta = _meta_cache[path]
    item = meta[name]
    begin, end = item["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + begin)
        data = f.read(end - begin)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(item["shape"])


def _fp4_values(packed_u8):
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    nibbles = torch.stack((lo, hi), dim=-1).reshape(*packed_u8.shape[:-1], -1)
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    return table[nibbles.long()]


def _e8m0_values(scale_u8):
    return ((scale_u8.to(torch.int32) << 23).view(torch.float32))


def _dequant(weight_u8, scale_u8):
    return _fp4_values(weight_u8) * _e8m0_values(scale_u8).repeat_interleave(32, dim=1)


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    local_experts = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    m = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    w1s, w2s, w3s, s1s, s2s, s3s = [], [], [], [], [], []
    for expert in range(start, start + local_experts):
        base = f"layers.{layer}.ffn.experts.{expert}"
        w1s.append(_read_u8(f"{base}.w1.weight"))
        w2s.append(_read_u8(f"{base}.w2.weight"))
        w3s.append(_read_u8(f"{base}.w3.weight"))
        s1s.append(_read_u8(f"{base}.w1.scale"))
        s2s.append(_read_u8(f"{base}.w2.scale"))
        s3s.append(_read_u8(f"{base}.w3.scale"))

    w13 = torch.stack([torch.cat([w1, w3], dim=0) for w1, w3 in zip(w1s, w3s)]).cuda()
    w2 = torch.stack(w2s).cuda()
    s13 = torch.stack([torch.cat([s1, s3], dim=0) for s1, s3 in zip(s1s, s3s)]).cuda()
    s2 = torch.stack(s2s).cuda()
    layer_ns = SimpleNamespace(params_dtype=torch.bfloat16)
    w13_m, w2_m, s13_m, s2_m, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        layer_ns, w13, w2, s13, s2, None, None
    )

    torch.manual_seed(123)
    a = (torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) / 10).contiguous()
    mode = sys.argv[6] if len(sys.argv) > 6 else "local"
    if mode == "mixed":
        vals = []
        for i in range(topk):
            vals.append((start + i) if i % 2 else i)
        ids_cpu = torch.tensor(vals, dtype=torch.int32).repeat(m, 1)
    else:
        ids_cpu = torch.arange(start, start + topk, dtype=torch.int32).repeat(m, 1)
    topk_ids = ids_cpu.cuda()
    weights = torch.arange(1, topk + 1, device="cuda", dtype=torch.float32)
    topk_weights = (weights / weights.sum()).repeat(m, 1).contiguous()

    expert_map = torch.full((256,), -1, device="cuda", dtype=torch.int32)
    expert_map[start:start + local_experts] = torch.arange(
        local_experts, device="cuda", dtype=torch.int32
    )

    ref = torch.zeros((m, 4096), device="cuda", dtype=torch.bfloat16)
    for slot in range(topk):
        expert = int(ids_cpu[0, slot])
        if expert < start or expert >= start + local_experts:
            continue
        local = expert - start
        ref_w1 = _dequant(w1s[local], s1s[local]).cuda().to(torch.bfloat16)
        ref_w2 = _dequant(w2s[local], s2s[local]).cuda().to(torch.bfloat16)
        ref_w3 = _dequant(w3s[local], s3s[local]).cuda().to(torch.bfloat16)
        expert_out = (
            torch.nn.functional.silu(a @ ref_w1.T) * (a @ ref_w3.T)
        ) @ ref_w2.T
        ref += expert_out * topk_weights[:, slot:slot + 1].to(torch.bfloat16)

    with set_current_vllm_config(VllmConfig()):
        out = fused_marlin_moe(
            a, w13_m, w2_m, None, None, s13_m, s2_m,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=256,
            expert_map=expert_map,
            quant_type_id=scalar_types.float4_e2m1f.id,
            input_dtype=torch.bfloat16,
            is_k_full=True,
            clamp_limit=10.0,
        )

    diff = (out.float() - ref.float()).abs()
    print("layer", layer, "start", start, "local", local_experts, "topk", topk, "mode", mode)
    print("out0", out[0, :8].float().tolist())
    print("ref0", ref[0, :8].float().tolist())
    print("max_abs", diff.max().item())
    print("mean_abs", diff.mean().item())
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=0)


if __name__ == "__main__":
    main()
