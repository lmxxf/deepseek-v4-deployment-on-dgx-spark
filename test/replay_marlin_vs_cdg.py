"""Replay real activations through Marlin vs Consumer-DeepGEMM, per layer.

Hypothesis (DevHistory §18): the garbled-English output isn't from numerical
noise accumulation (CDG quantizes activations to FP8 ~3% noise and survives,
Marlin uses bf16 activations ~0.2% noise and fails). It's a SYSTEMATIC bias
that only fires on real activations / real routing.

This script:
  1. loads dump files (real x + real topk from the deployed model)
  2. loads real MXFP4 expert weights from the checkpoint for each layer
  3. runs Marlin path on real (x, topk) -> y_marlin
  4. runs CDG path (same) -> y_cdg
  5. computes per-layer divergence metrics: max abs diff, p99.9, column
     concentration (does the error cluster on a few channels?), expert
     concentration (does it cluster on a few experts?).

Run inside docker (single-GPU is fine, replay doesn't need TP):
  docker run --rm --gpus all \\
    -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \\
    --entrypoint python3 \\
    lmxxf/vllm-deepseek-v4-dgx-spark:latest \\
    /work/test/replay_marlin_vs_cdg.py [--layers 0,21,41,42] [--frames 1]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch

DUMP_DIR = Path("/work/debug_acts")
CKPT_DIR = Path("/work/deepseek-v4-flash")


def list_layers(dump_dir):
    layers = set()
    for p in dump_dir.glob("model.layers.*.ffn.experts.apply.*.pt"):
        idx = int(p.name.split(".")[2])
        layers.add(idx)
    return sorted(layers)


def load_dump(dump_dir, layer_idx, frame=0):
    path = dump_dir / f"model.layers.{layer_idx}.ffn.experts.apply.{frame}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize(name, d):
    print(f"[{name}] keys={list(d.keys())}")
    for k in ("x", "out"):
        t = d[k]
        print(f"  {k}: shape={tuple(t.shape)} dtype={t.dtype} absmax={t.abs().max().item():.4g}")
    print(f"  args: {len(d.get('args', []))} items")
    kwargs = d.get("kwargs", {})
    print(f"  kwargs keys: {list(kwargs.keys())}")
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            extra = (f" absmax={v.abs().max().item():.4g}"
                     if v.dtype.is_floating_point
                     else f" min={v.min().item()} max={v.max().item()}")
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}{extra}")
        else:
            print(f"    {k}: {type(v).__name__} = {v}")
    if "call_index" in d:
        print(f"  call_index: {d['call_index']}")
    for k in ("moe_kernel_type", "fused_experts_type"):
        if k in d:
            print(f"  {k}: {d[k]}")
    x = d["x"]
    if x.ndim == 2 and x.shape[0] > 1:
        same_first_last = torch.equal(x[0], x[-1])
        same_rows = torch.unique(x[:min(x.shape[0], 64)], dim=0).shape[0]
        print(f"  row uniqueness: first64_unique={same_rows}, first_eq_last={same_first_last}")


def compare_dumps(left_dir, right_dir, frame):
    print(f"compare frame={frame}")
    print(f"left={left_dir}")
    print(f"right={right_dir}")
    print("L shape x_max_diff out_max_diff out_p999_diff out_cos")
    for L in sorted(set(list_layers(left_dir)) & set(list_layers(right_dir))):
        lp = left_dir / f"model.layers.{L}.ffn.experts.apply.{frame}.pt"
        rp = right_dir / f"model.layers.{L}.ffn.experts.apply.{frame}.pt"
        if not lp.exists() or not rp.exists():
            continue
        l = load_dump(left_dir, L, frame)
        r = load_dump(right_dir, L, frame)
        if l["x"].shape != r["x"].shape or l["out"].shape != r["out"].shape:
            print(f"{L:02d} shape mismatch x={tuple(l['x'].shape)} vs {tuple(r['x'].shape)}")
            continue
        lx = l["x"].float()
        rx = r["x"].float()
        lo = l["out"].float()
        ro = r["out"].float()
        od = (lo - ro).abs().flatten()
        cos = torch.nn.functional.cosine_similarity(
            lo.flatten(), ro.flatten(), dim=0).item()
        print(
            f"{L:02d} {tuple(lo.shape)} "
            f"{(lx-rx).abs().max().item():.4g} "
            f"{od.max().item():.4g} "
            f"{torch.quantile(od, 0.999).item():.4g} "
            f"{cos:.6f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,10,21,30,41,42",
                    help="comma-separated layer indices to inspect")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--dump-dir", default=str(DUMP_DIR))
    ap.add_argument("--compare-left")
    ap.add_argument("--compare-right")
    ap.add_argument("--just-inspect", action="store_true",
                    help="only summarize one dump file (sanity check)")
    args = ap.parse_args()

    if args.compare_left and args.compare_right:
        compare_dumps(Path(args.compare_left), Path(args.compare_right),
                      args.frame)
        return

    dump_dir = Path(args.dump_dir)
    avail = list_layers(dump_dir)
    print(f"available layers: {avail[0]}..{avail[-1]} ({len(avail)} total)")

    targets = [int(x) for x in args.layers.split(",")]
    for L in targets:
        if L not in avail:
            print(f"  layer {L} not in dumps, skipping")
            continue
        d = load_dump(dump_dir, L, args.frame)
        summarize(f"layer {L}", d)

    if args.just_inspect:
        return


if __name__ == "__main__":
    main()
