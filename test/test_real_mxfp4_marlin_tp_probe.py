import json
import struct
import sys
import types
from types import SimpleNamespace

import torch

sys.path.append("/work/spark-vllm-docker/vllm-sm120/vllm-sm120")


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


CKPT = "/work/deepseek-v4-flash/model-00002-of-00046.safetensors"


def _metadata(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return header_len, json.loads(f.read(header_len))


def _read_u8(path, header_len, meta, name):
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
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
    )
    return table[nibbles.long()]


def _e8m0_values(scale_u8):
    return ((scale_u8.to(torch.int32) << 23).view(torch.float32))


def _dequant(weight_u8, scale_u8):
    return _fp4_values(weight_u8) * _e8m0_values(scale_u8).repeat_interleave(32, dim=1)


def _run_rank(a, topk_ids, topk_weights, w1s, w2s, w3s, s1s, s2s, s3s, tp_rank):
    n_full = w1s[0].shape[0]
    n_part = n_full // 2
    lo = tp_rank * n_part
    hi = lo + n_part

    w13 = torch.stack([
        torch.cat([w1[lo:hi], w3[lo:hi]], dim=0) for w1, w3 in zip(w1s, w3s)
    ]).cuda()
    w2 = torch.stack([w2[:, lo // 2 : hi // 2] for w2 in w2s]).cuda()
    s13 = torch.stack([
        torch.cat([s1[lo:hi], s3[lo:hi]], dim=0) for s1, s3 in zip(s1s, s3s)
    ]).cuda()
    s2 = torch.stack([s2[:, lo // 32 : hi // 32] for s2 in s2s]).cuda()

    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    w13_m, w2_m, s13_m, s2_m, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        layer, w13, w2, s13, s2, None, None
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

    torch.manual_seed(123)
    a = (torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) / 10)
    ids = torch.arange(topk, dtype=torch.int64).repeat(m, 1)
    topk_ids = ids.cuda()
    weights = torch.arange(1, topk + 1, dtype=torch.float32)
    weights = weights / weights.sum()
    topk_weights = weights.repeat(m, 1).cuda()

    ref = torch.zeros((m, 4096), device="cuda", dtype=torch.bfloat16)
    for slot in range(topk):
        expert = int(ids[0, slot])
        ref_w1 = _dequant(w1s[expert], s1s[expert]).cuda().to(torch.bfloat16)
        ref_w2 = _dequant(w2s[expert], s2s[expert]).cuda().to(torch.bfloat16)
        ref_w3 = _dequant(w3s[expert], s3s[expert]).cuda().to(torch.bfloat16)
        expert_out = (
            torch.nn.functional.silu(a @ ref_w1.T) * (a @ ref_w3.T)
        ) @ ref_w2.T
        ref += expert_out * topk_weights[:, slot : slot + 1].to(torch.bfloat16)

    out0 = _run_rank(a, topk_ids, topk_weights, w1s, w2s, w3s, s1s, s2s, s3s, 0)
    out1 = _run_rank(a, topk_ids, topk_weights, w1s, w2s, w3s, s1s, s2s, s3s, 1)
    out = out0 + out1
    diff = (out.float() - ref.float()).abs()
    print("experts", experts, "m", m, "topk", topk)
    print("rank0", out0[0, :8].float().tolist())
    print("rank1", out1[0, :8].float().tolist())
    print("out0", out[0, :8].float().tolist())
    print("ref0", ref[0, :8].float().tolist())
    print("max_abs", diff.max().item())
    print("mean_abs", diff.mean().item())
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=0)


if __name__ == "__main__":
    main()
