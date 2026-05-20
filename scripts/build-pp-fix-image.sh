#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="lmxxf/vllm-deepseek-v4-dgx-spark:latest"
TAG="vllm-deepseek-v4-pp-fix:latest"
DO_PULL=0
DO_PUSH=0
COPY_HOSTS=()
SSH_USER="${USER}"

usage() {
  cat <<'EOF'
Build a small PP-fix image from the already uploaded DeepSeek V4 Spark image.

The patch updates vLLM's MXFP4 layer-name parser inside the image so
VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 also matches PP layer names like:
  model.layers.42.ffn
  model.layers.42.ffn.experts

Usage:
  ./scripts/build-pp-fix-image.sh [options]

Options:
  --base IMAGE        Base image to patch.
                      Default: lmxxf/vllm-deepseek-v4-dgx-spark:latest
  -t, --tag IMAGE     Output image tag.
                      Default: vllm-deepseek-v4-pp-fix:latest
  --pull              docker pull the base image before building.
  --push              docker push the output image after building.
  --copy-to HOSTS     Save/load the output image to remote hosts over ssh.
                      HOSTS may be comma-separated, e.g. host1,host2
  --ssh-user USER     SSH user for --copy-to. Default: current user.
  -h, --help          Show this help.

Examples:
  ./scripts/build-pp-fix-image.sh --pull
  ./scripts/build-pp-fix-image.sh --base lmxxf/vllm-deepseek-v4-dgx-spark:latest -t lmxxf/vllm-deepseek-v4-pp-fix:latest --push
  ./scripts/build-pp-fix-image.sh --copy-to 169.254.30.81
EOF
}

add_copy_hosts() {
  local token part
  for token in "$@"; do
    IFS=',' read -ra PARTS <<< "$token"
    for part in "${PARTS[@]}"; do
      part="${part//[[:space:]]/}"
      if [[ -n "$part" ]]; then
        COPY_HOSTS+=("$part")
      fi
    done
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_IMAGE="${2:?missing value for --base}"
      shift 2
      ;;
    -t|--tag)
      TAG="${2:?missing value for --tag}"
      shift 2
      ;;
    --pull)
      DO_PULL=1
      shift
      ;;
    --push)
      DO_PUSH=1
      shift
      ;;
    --copy-to)
      add_copy_hosts "${2:?missing value for --copy-to}"
      shift 2
      ;;
    --ssh-user)
      SSH_USER="${2:?missing value for --ssh-user}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "Base image:   $BASE_IMAGE"
echo "Output image: $TAG"

if [[ "$DO_PULL" -eq 1 ]]; then
  docker pull "$BASE_IMAGE"
fi

tmpdir="$(mktemp -d --tmpdir "pp-fix-build.XXXXXX")"
tmp_image=""
cleanup() {
  rm -rf "$tmpdir"
  if [[ -n "$tmp_image" ]]; then
    rm -f "$tmp_image"
  fi
}
trap cleanup EXIT

cat > "$tmpdir/Dockerfile" <<EOF
ARG BASE_IMAGE=${BASE_IMAGE}
FROM \${BASE_IMAGE}

RUN python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py")
text = path.read_text()

old = r'r"\.layers\.(\d+)\.ffn\.experts$"'
new = r'r"(?:^|\.)layers\.(\d+)\.ffn(?:\.experts)?$"'

if new in text:
    print("PP layer-name parser is already patched")
elif old in text:
    path.write_text(text.replace(old, new))
    print("Patched MXFP4 layer-name parser for PP")
else:
    raise SystemExit("Could not find the expected MXFP4 layer-name regex in " + str(path))
PY
EOF

docker build -f "$tmpdir/Dockerfile" -t "$TAG" "$tmpdir"

echo
echo "Built: $TAG"

if [[ "$DO_PUSH" -eq 1 ]]; then
  docker push "$TAG"
fi

if [[ "${#COPY_HOSTS[@]}" -gt 0 ]]; then
  tmp_image="$(mktemp --tmpdir "pp-fix-image.XXXXXX.tar")"
  echo "Saving image for remote copy: $tmp_image"
  docker save "$TAG" -o "$tmp_image"

  for host in "${COPY_HOSTS[@]}"; do
    echo "Loading image into ${SSH_USER}@${host}..."
    ssh "${SSH_USER}@${host}" docker load < "$tmp_image"
  done
fi

echo "Done."
