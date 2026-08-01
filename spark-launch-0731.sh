#!/usr/bin/env bash
# 我们双机的适配版（2026-08-01，v3）
# TAG=production-hybrid-1.6（默认，支持 DSpark）| production-ready（旧，投机必须 SPEC=off）
# SPEC=dspark（默认，仅新镜像）| mtp | off
#
# 用法（先 worker 后 head）：
#   slave：  ssh 169.254.30.81 "bash ~/work/dgx-spark-slave/spark-launch.sh 1"
#   host ：  ~/work/deepseek-v4-flash-dgx-spark/launch/spark-launch.sh 0
# 回退旧镜像：两条前缀加  TAG=production-ready SPEC=off
# 看日志： docker logs -f vllm_unholy    验活： curl http://localhost:8000/v1/models

set -uo pipefail
NODE_RANK="${1:?usage: spark-launch.sh <0|1>   (0=head, 1=worker)}"
HEADLESS_FLAG=""
[ "$NODE_RANK" = "1" ] && HEADLESS_FLAG="--headless"

TAG="${TAG:-production-hybrid-1.6}"
SPEC="${SPEC:-dspark}"

SPEC_FLAG=""
case "$SPEC" in
  dspark) SPEC_FLAG="--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":2}'" ;;
  mtp)    SPEC_FLAG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'" ;;
esac

docker rm -f vllm_unholy 2>/dev/null || true

COMMON_DOCKER_ARGS=(--gpus all -d --privileged --network host --ipc host --shm-size 10g
  --ulimit memlock=-1
  --device /dev/infiniband:/dev/infiniband
  -v "$HOME/work:/work"
  --name vllm_unholy
  -e HF_HUB_OFFLINE=1
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_GID_INDEX=3
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e TP_SOCKET_IFNAME=enp1s0f1np1
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN)

if [ "$TAG" = "production-ready" ]; then
  # ---- 旧镜像路径（06-01 构建，验证过：无投机 24.5 tok/s）----
  docker run "${COMMON_DOCKER_ARGS[@]}" \
    -e VLLM_CACHE_ROOT=/work/models/vllm-cache \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
    --entrypoint bash \
    aidendle94/sparkrun-vllm-ds4-gb10:$TAG \
    -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve /work/models/DeepSeek-V4-Flash-0731 --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --kv-cache-dtype fp8 --block-size 256 --max-model-len 500000 --max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8 --enable-prefix-caching $SPEC_FLAG --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 --enable-flashinfer-autotune --nnodes 2 --node-rank $NODE_RANK --master-addr 169.254.248.35 --master-port 29501 $HEADLESS_FLAG"
else
  # ---- hybrid-1.6 路径：镜像自带 serve-ds4-flash.sh 是单机版（NCCL_IB_DISABLE=1/TP=4/PCIe allreduce），
  # 这里手动起 vllm：保留它的 B12X kernel 环境，替换成我们的双机 RoCE 配置 ----
  docker run "${COMMON_DOCKER_ARGS[@]}" \
    -e CUTE_DSL_ARCH=sm_121a \
    \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --entrypoint bash \
    aidendle94/sparkrun-vllm-ds4-gb10:$TAG \
    -lc "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /work/models/DeepSeek-V4-Flash-0731 --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --kv-cache-dtype fp8 --block-size 256 --moe-backend b12x --linear-backend b12x --max-model-len 500000 --max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8 --enable-chunked-prefill --enable-prefix-caching $SPEC_FLAG --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 --enable-flashinfer-autotune --nnodes 2 --node-rank $NODE_RANK --master-addr 169.254.248.35 --master-port 29501 $HEADLESS_FLAG"
fi

echo "launched vllm_unholy tag=$TAG spec=$SPEC node-rank=$NODE_RANK rc=$?"
sleep 2
docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_unholy || echo "WARN: container not in ps (check: docker logs vllm_unholy)"
