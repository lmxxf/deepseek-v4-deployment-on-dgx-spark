"""Marlin sm_121 error STRUCTURE probe.

Old probe (test_real_mxfp4_marlin_probe.py) only checked mean/max with
randn/10 inputs and atol=5e-2. This one asks WHERE the error lives:
- fp32 end-to-end reference (gold), bf16 reference (rounding baseline)
- activation magnitude sweep + outlier injection (real LLM activations
  have +-20..100 outliers; randn/10 never triggers clamp/overflow paths)
- m sweep (different Marlin schedules / slice counts)
- error metrics: max, p99.9, frac>thresholds, column clustering

Run inside docker:
  docker run --rm --gpus all -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
    lmxxf/vllm-deepseek-v4-dgx-spark:latest python3 /work/test/test_marlin_err_structure.py
"""
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
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    return table[nibbles.long()]


def _e8m0_values(scale_u8):
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def _dequant_f32(weight_u8, scale_u8):
    return _fp4_values(weight_u8) * _e8m0_values(scale_u8).repeat_interleave(32, dim=1)


def make_input(m, mode, seed=123):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn((m, 4096), generator=g, dtype=torch.float32)
    if mode == "small":
        a = a / 10
    elif mode == "unit":
        pass
    elif mode == "x4":
        a = a * 4
    elif mode == "outlier":
        # realistic: mostly ~N(0,1) with a few huge channels
        idx = torch.randperm(4096, generator=g)[:32]
        a[:, idx] *= 30.0
    return a.cuda().to(torch.bfloat16)


def report(tag, out, ref32, ref_bf16):
    d_marlin = (out.float() - ref32).abs()
    d_bf16 = (ref_bf16.float() - ref32).abs()
    scale = ref32.abs().mean().item() + 1e-9

    def stats(d):
        flat = d.flatten()
        return dict(
            max=flat.max().item(),
            p999=torch.quantile(flat.float(), 0.999).item(),
            mean=flat.mean().item(),
            frac_1e2=(flat > 1e-2).float().mean().item(),
            frac_1e1=(flat > 1e-1).float().mean().item(),
        )

    sm, sb = stats(d_marlin), stats(d_bf16)
    # column clustering: how many output columns hold the worst errors
    col_max = d_marlin.max(dim=0).values
    bad_cols = (col_max > max(10 * sb["max"], 1e-2)).sum().item()
    print(f"[{tag}] ref_mean_abs={scale:.4f}")
    print(f"  marlin vs f32: max={sm['max']:.5f} p999={sm['p999']:.5f} mean={sm['mean']:.6f} "
          f"frac>1e-2={sm['frac_1e2']:.4f} frac>1e-1={sm['frac_1e1']:.5f}")
    print(f"  bf16ref vs f32: max={sb['max']:.5f} p999={sb['p999']:.5f} mean={sb['mean']:.6f}")
    print(f"  excess_ratio(max)={sm['max'] / (sb['max'] + 1e-12):.2f}  bad_cols={bad_cols}/4096")
    sys.stdout.flush()


def main():
    header_len, meta = _metadata(CKPT)
    experts = 8
    topk = 4

    w1s, w2s, w3s, s1s, s2s, s3s = [], [], [], [], [], []
    for expert in range(experts):
        base = f"layers.0.ffn.experts.{expert}"
        w1s.append(_read_u8(CKPT, header_len, meta, f"{base}.w1.weight"))
        w2s.append(_read_u8(CKPT, header_len, meta, f"{base}.w2.weight"))
        w3s.append(_read_u8(CKPT, header_len, meta, f"{base}.w3.weight"))
        s1s.append(_read_u8(CKPT, header_len, meta, f"{base}.w1.scale"))
        s2s.append(_read_u8(CKPT, header_len, meta, f"{base}.w2.scale"))
        s3s.append(_read_u8(CKPT, header_len, meta, f"{base}.w3.scale"))

    w13 = torch.stack([torch.cat([a, b], dim=0) for a, b in zip(w1s, w3s)]).cuda()
    w2 = torch.stack(w2s).cuda()
    s13 = torch.stack([torch.cat([a, b], dim=0) for a, b in zip(s1s, s3s)]).cuda()
    s2 = torch.stack(s2s).cuda()

    # fp32 dequantized weights (gold)
    deq1 = [_dequant_f32(w1s[e], s1s[e]).cuda() for e in range(experts)]
    deq2 = [_dequant_f32(w2s[e], s2s[e]).cuda() for e in range(experts)]
    deq3 = [_dequant_f32(w3s[e], s3s[e]).cuda() for e in range(experts)]

    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    w13_m, w2_m, s13_m, s2_m, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        layer, w13, w2, s13, s2, None, None
    )

    ids = torch.arange(topk, dtype=torch.int64)
    weights = torch.arange(1, topk + 1, dtype=torch.float32)
    weights = weights / weights.sum()

    for m in (1, 2, 16, 256):
        topk_ids = ids.repeat(m, 1).cuda()
        topk_weights = weights.repeat(m, 1).cuda()
        for mode in ("small", "unit", "x4", "outlier"):
            a = make_input(m, mode)
            a32 = a.float()

            # gold fp32 reference (plain silu*up, matches old probe semantics)
            ref32 = torch.zeros((m, 4096), device="cuda", dtype=torch.float32)
            refb = torch.zeros((m, 4096), device="cuda", dtype=torch.bfloat16)
            for slot in range(topk):
                e = int(ids[slot])
                g32 = a32 @ deq1[e].T
                u32 = a32 @ deq3[e].T
                ref32 += ((torch.nn.functional.silu(g32) * u32) @ deq2[e].T) * float(weights[slot])
                gb = a @ deq1[e].to(torch.bfloat16).T
                ub = a @ deq3[e].to(torch.bfloat16).T
                refb += ((torch.nn.functional.silu(gb) * ub) @ deq2[e].to(torch.bfloat16).T) \
                    * topk_weights[0, slot].to(torch.bfloat16)

            with set_current_vllm_config(VllmConfig()):
                out = fused_marlin_moe(
                    a, w13_m, w2_m, None, None, s13_m, s2_m,
                    topk_weights=topk_weights, topk_ids=topk_ids,
                    global_num_experts=experts,
                    quant_type_id=scalar_types.float4_e2m1f.id,
                    input_dtype=torch.bfloat16, is_k_full=True,
                )
            report(f"m={m} mode={mode}", out, ref32, refb)


if __name__ == "__main__":
    main()
