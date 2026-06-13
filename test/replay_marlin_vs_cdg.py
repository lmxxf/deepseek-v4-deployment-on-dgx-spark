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


def list_layers():
    layers = set()
    for p in DUMP_DIR.glob("model.layers.*.ffn.experts.apply.*.pt"):
        idx = int(p.name.split(".")[2])
        layers.add(idx)
    return sorted(layers)


def load_dump(layer_idx, frame=0):
    path = DUMP_DIR / f"model.layers.{layer_idx}.ffn.experts.apply.{frame}.pt"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,10,21,30,41,42",
                    help="comma-separated layer indices to inspect")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--just-inspect", action="store_true",
                    help="only summarize one dump file (sanity check)")
    args = ap.parse_args()

    avail = list_layers()
    print(f"available layers: {avail[0]}..{avail[-1]} ({len(avail)} total)")

    targets = [int(x) for x in args.layers.split(",")]
    for L in targets:
        if L not in avail:
            print(f"  layer {L} not in dumps, skipping")
            continue
        d = load_dump(L, args.frame)
        summarize(f"layer {L}", d)

    if args.just_inspect:
        return


if __name__ == "__main__":
    main()
