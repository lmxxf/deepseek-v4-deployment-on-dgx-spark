#!/usr/bin/env bash
set -euo pipefail

# Build an activation-dump debug image from the deployed hybrid image.
# Adds an env-gated hook to Mxfp4MoEMethod.apply that saves each MoE layer's
# input activations / routing / output to VLLM_MXFP4_DUMP_DIR (rank0 only).
# Used to capture REAL activations for local Marlin-vs-reference replay.
#
# Usage:
#   ./scripts/build-act-dump-image.sh                 # build vllm-deepseek-v4-act-dump:latest
#   ./scripts/build-act-dump-image.sh --copy-to 169.254.30.81

BASE_IMAGE="lmxxf/vllm-deepseek-v4-dgx-spark:latest"
TAG="vllm-deepseek-v4-act-dump:latest"
COPY_HOSTS=()
SSH_USER="${USER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE_IMAGE="$2"; shift 2 ;;
    -t|--tag) TAG="$2"; shift 2 ;;
    --copy-to) IFS=',' read -ra COPY_HOSTS <<< "$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

tmpdir="$(mktemp -d --tmpdir "act-dump-build.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

cat > "$tmpdir/Dockerfile" <<EOF
ARG BASE_IMAGE=${BASE_IMAGE}
FROM \${BASE_IMAGE}

RUN python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py")
text = path.read_text()

MARK = "# === act dump hook ==="
if MARK in text:
    print("act dump hook already present")
else:
    hook = '''

''' + MARK + '''
import os as _os

_dump_counts: dict = {}

def _act_dump_wrap(cls, method_name):
    orig = getattr(cls, method_name)

    def wrapped(self, layer, x, *args, **kwargs):
        out = orig(self, layer, x, *args, **kwargs)
        d = _os.environ.get("VLLM_MXFP4_DUMP_DIR")
        if d:
            try:
                import torch as _t
                rank = (
                    _t.distributed.get_rank()
                    if _t.distributed.is_initialized()
                    else 0
                )
                if rank == 0:
                    name = getattr(layer, "layer_name", "unknown")
                    key = (method_name, name)
                    n = _dump_counts.get(key, 0)
                    if n < int(_os.environ.get("VLLM_MXFP4_DUMP_MAX", "8")):
                        _dump_counts[key] = n + 1
                        _os.makedirs(d, exist_ok=True)
                        payload = {
                            "layer": name,
                            "method": method_name,
                            "x": x.detach().to("cpu", copy=True),
                            "args": [
                                a.detach().to("cpu", copy=True)
                                if isinstance(a, _t.Tensor)
                                else a
                                for a in args
                            ],
                            "out": out.detach().to("cpu", copy=True)
                            if isinstance(out, _t.Tensor)
                            else None,
                        }
                        _t.save(payload, f"{d}/{name}.{method_name}.{n}.pt")
            except Exception as e:  # noqa: BLE001
                print("act dump hook error:", e)
        return out

    setattr(cls, method_name, wrapped)

_act_dump_wrap(Mxfp4MoEMethod, "apply")
_act_dump_wrap(Mxfp4MoEMethod, "apply_monolithic")
'''
    path.write_text(text + hook)
    print("act dump hook installed")
PY
EOF

docker build -f "$tmpdir/Dockerfile" -t "$TAG" "$tmpdir"
echo "Built: $TAG"

if [[ "${#COPY_HOSTS[@]}" -gt 0 ]]; then
  tmp_image="$(mktemp --tmpdir "act-dump-image.XXXXXX.tar")"
  docker save "$TAG" -o "$tmp_image"
  for host in "${COPY_HOSTS[@]}"; do
    host="${host//[[:space:]]/}"
    [[ -z "$host" ]] && continue
    echo "Loading image into ${SSH_USER}@${host}..."
    ssh "${SSH_USER}@${host}" docker load < "$tmp_image"
  done
  rm -f "$tmp_image"
fi
echo "Done."
