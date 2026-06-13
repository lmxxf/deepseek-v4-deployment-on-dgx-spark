"""Replay real layer42 MoE dumps against EP-aware references.

Run inside docker:
  docker run --rm --gpus all \
    -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
    --entrypoint python3 vllm-deepseek-v4-act-dump:latest \
    /work/test/replay_layer42_ep_ref.py
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
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

ROOT = Path("/work")
CKPT_DIR = ROOT / "deepseek-v4-flash"
INDEX = CKPT_DIR / "model.safetensors.index.json"
MIXED_DUMP = ROOT / "debug_acts_mixed/model.layers.42.ffn.experts.apply.0.pt"
MARLIN_DUMP = ROOT / "debug_acts_all_marlin/model.layers.42.ffn.experts.apply.0.pt"


def _metadata(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return header_len, json.loads(f.read(header_len))


def _read_u8(path: Path, header_len: int, meta: dict, name: str):
    item = meta[name]
    begin, end = item["data_offsets"]
    with path.open("rb") as f:
        f.seek(8 + header_len + begin)
        data = f.read(end - begin)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(item["shape"])


def _fp4_values(packed_u8: torch.Tensor):
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    nibbles = torch.stack((lo, hi), dim=-1).reshape(*packed_u8.shape[:-1], -1)
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    return table[nibbles.long()]


def _e8m0_values(scale_u8: torch.Tensor):
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def _dequant_f32(weight_u8: torch.Tensor, scale_u8: torch.Tensor):
    return _fp4_values(weight_u8) * _e8m0_values(scale_u8).repeat_interleave(32, dim=1)


class SafeTensorReader:
    def __init__(self, index_path: Path):
        with index_path.open("r") as f:
            self.weight_map = json.load(f)["weight_map"]
        self._metas: dict[str, tuple[int, dict]] = {}

    def read(self, name: str):
        filename = self.weight_map[name]
        if filename not in self._metas:
            self._metas[filename] = _metadata(CKPT_DIR / filename)
        header_len, meta = self._metas[filename]
        return _read_u8(CKPT_DIR / filename, header_len, meta, name)


def _load_expert(reader: SafeTensorReader, layer: int, expert: int):
    base = f"layers.{layer}.ffn.experts.{expert}"
    return {
        "w1": reader.read(f"{base}.w1.weight"),
        "w2": reader.read(f"{base}.w2.weight"),
        "w3": reader.read(f"{base}.w3.weight"),
        "s1": reader.read(f"{base}.w1.scale"),
        "s2": reader.read(f"{base}.w2.scale"),
        "s3": reader.read(f"{base}.w3.scale"),
    }


def _report(tag: str, candidate: torch.Tensor, target: torch.Tensor):
    diff = (candidate.float() - target.float()).abs().flatten()
    cos = torch.nn.functional.cosine_similarity(
        candidate.float().flatten(), target.float().flatten(), dim=0
    ).item()
    print(
        f"{tag}: max={diff.max().item():.8f} "
        f"p999={torch.quantile(diff, 0.999).item():.8f} "
        f"p99={torch.quantile(diff, 0.99).item():.8f} "
        f"mean={diff.mean().item():.8f} cos={cos:.9f}"
    )


def _reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    reader: SafeTensorReader,
    *,
    rank: int | None,
    tp_rank: int | None = None,
):
    used = sorted(set(int(v) for v in topk_ids.flatten().tolist()))
    if rank is not None:
        lo_expert = rank * 128
        hi_expert = lo_expert + 128
        used = [e for e in used if lo_expert <= e < hi_expert]

    out = torch.zeros((x.shape[0], 4096), device="cuda", dtype=torch.float32)
    for expert in used:
        rows, slots = torch.where(topk_ids == expert)
        if rows.numel() == 0:
            continue
        weights = topk_weights[rows, slots].float().unsqueeze(1)
        tensors = _load_expert(reader, 42, expert)
        if tp_rank is None:
            row_slice = slice(None)
            col_slice = slice(None)
            scale_col_slice = slice(None)
        else:
            n_full = tensors["w1"].shape[0]
            n_part = n_full // 2
            lo = tp_rank * n_part
            hi = lo + n_part
            row_slice = slice(lo, hi)
            col_slice = slice(lo // 2, hi // 2)
            scale_col_slice = slice(lo // 32, hi // 32)
        w1 = _dequant_f32(tensors["w1"][row_slice], tensors["s1"][row_slice]).cuda()
        w2 = _dequant_f32(
            tensors["w2"][:, col_slice], tensors["s2"][:, scale_col_slice]
        ).cuda()
        w3 = _dequant_f32(tensors["w3"][row_slice], tensors["s3"][row_slice]).cuda()
        a = x[rows].float()
        gate = torch.clamp(a @ w1.T, max=10.0)
        up = torch.clamp(a @ w3.T, min=-10.0, max=10.0)
        contrib = (torch.nn.functional.silu(gate) * up) @ w2.T
        out[rows] += contrib * weights
    return out.to(torch.bfloat16)


def _marlin_replay(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    reader: SafeTensorReader,
    *,
    experts_count: int,
    use_linear_ep_map: bool,
    tp_rank: int | None,
):
    experts = [_load_expert(reader, 42, expert) for expert in range(experts_count)]
    if tp_rank is None:
        row_slice = slice(None)
        col_slice = slice(None)
        scale_col_slice = slice(None)
    else:
        n_full = experts[0]["w1"].shape[0]
        n_part = n_full // 2
        lo = tp_rank * n_part
        hi = lo + n_part
        row_slice = slice(lo, hi)
        col_slice = slice(lo // 2, hi // 2)
        scale_col_slice = slice(lo // 32, hi // 32)
    w13 = torch.stack(
        [torch.cat([t["w1"][row_slice], t["w3"][row_slice]], dim=0) for t in experts]
    ).cuda()
    w2 = torch.stack([t["w2"][:, col_slice] for t in experts]).cuda()
    s13 = torch.stack(
        [torch.cat([t["s1"][row_slice], t["s3"][row_slice]], dim=0) for t in experts]
    ).cuda()
    s2 = torch.stack([t["s2"][:, scale_col_slice] for t in experts]).cuda()
    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    w13_m, w2_m, s13_m, s2_m, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        layer, w13, w2, s13, s2, None, None
    )
    expert_map = None
    if use_linear_ep_map:
        expert_map = torch.full((256,), -1, dtype=torch.int32, device="cuda")
        expert_map[:experts_count] = torch.arange(
            experts_count, dtype=torch.int32, device="cuda"
        )
    with set_current_vllm_config(VllmConfig()):
        return fused_marlin_moe(
            x,
            w13_m,
            w2_m,
            None,
            None,
            s13_m,
            s2_m,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=256 if use_linear_ep_map else experts_count,
            expert_map=expert_map,
            quant_type_id=scalar_types.float4_e2m1f.id,
            input_dtype=torch.bfloat16,
            is_k_full=True,
            clamp_limit=10.0,
        )


def main():
    mixed = torch.load(MIXED_DUMP, map_location="cpu")
    marlin = torch.load(MARLIN_DUMP, map_location="cpu")
    x = mixed["x"].cuda().to(torch.bfloat16).contiguous()
    topk_ids = mixed["kwargs"]["topk_ids"].cuda().long().contiguous()
    topk_weights = mixed["kwargs"]["topk_weights"].cuda().float().contiguous()
    mixed_out = mixed["out"].cuda().to(torch.bfloat16)
    marlin_out = marlin["out"].cuda().to(torch.bfloat16)

    print("mixed fused_experts_type", mixed.get("fused_experts_type"))
    print("marlin fused_experts_type", marlin.get("fused_experts_type"))
    print("x equal", torch.equal(mixed["x"], marlin["x"]))
    print(
        "topk_ids equal",
        torch.equal(mixed["kwargs"]["topk_ids"], marlin["kwargs"]["topk_ids"]),
    )
    print(
        "topk_weights equal",
        torch.equal(
            mixed["kwargs"]["topk_weights"], marlin["kwargs"]["topk_weights"]
        ),
    )

    reader = SafeTensorReader(INDEX)
    ref_rank0 = _reference(x, topk_ids, topk_weights, reader, rank=0)
    ref_all = _reference(x, topk_ids, topk_weights, reader, rank=None)
    ref_tp0 = _reference(x, topk_ids, topk_weights, reader, rank=None, tp_rank=0)
    marlin_replay_all = _marlin_replay(
        x,
        topk_ids,
        topk_weights,
        reader,
        experts_count=256,
        use_linear_ep_map=False,
        tp_rank=None,
    )
    marlin_replay_tp0 = _marlin_replay(
        x,
        topk_ids,
        topk_weights,
        reader,
        experts_count=256,
        use_linear_ep_map=False,
        tp_rank=0,
    )
    marlin_replay_ep0 = _marlin_replay(
        x,
        topk_ids,
        topk_weights,
        reader,
        experts_count=128,
        use_linear_ep_map=True,
        tp_rank=None,
    )

    _report("mixed(CDG rank0 dump) vs ref_rank0", mixed_out, ref_rank0)
    _report("marlin rank0 dump vs ref_rank0", marlin_out, ref_rank0)
    _report("mixed(CDG rank0 dump) vs ref_tp0", mixed_out, ref_tp0)
    _report("marlin rank0 dump vs ref_tp0", marlin_out, ref_tp0)
    _report("marlin rank0 dump vs marlin_replay_all256", marlin_out, marlin_replay_all)
    _report("mixed(CDG rank0 dump) vs marlin_replay_all256", mixed_out, marlin_replay_all)
    _report("marlin rank0 dump vs marlin_replay_tp0", marlin_out, marlin_replay_tp0)
    _report("mixed(CDG rank0 dump) vs marlin_replay_tp0", mixed_out, marlin_replay_tp0)
    _report("marlin rank0 dump vs marlin_replay_ep0", marlin_out, marlin_replay_ep0)
    _report("marlin rank0 dump vs ref_all_BAD", marlin_out, ref_all)
    _report("mixed(CDG rank0 dump) vs marlin rank0 dump", mixed_out, marlin_out)
    _report("ref_all_BAD vs ref_rank0", ref_all, ref_rank0)

    high_remote = (topk_ids >= 128).float().mean().item()
    print(f"remote_topk_fraction_for_rank0={high_remote:.4f}")


if __name__ == "__main__":
    main()
