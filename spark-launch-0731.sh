#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 双 DGX Spark (GB10) vLLM TP=2 启动脚本
# 用法：两台机器各放一份本脚本，先 worker 后 head：
#   worker（从机）:  ./spark-launch-0731.sh 1
#   head  （主机）:  ./spark-launch-0731.sh 0
# 详细流程见同仓库《如何使用tonyd2wild的ds4推理框架.md》

set -uo pipefail
NODE_RANK="${1:?usage: spark-launch-0731.sh <0|1>   (0=head, 1=worker)}"

# ======== 按你自己的机器改这五个值（查法见文档第 4 步）========
MASTER_ADDR="${MASTER_ADDR:-169.254.248.35}"                       # head 机在 RoCE 直连口上的 IP
NCCL_HCA="${NCCL_HCA:-rocep1s0f1}"                                 # 接了线的那个 RoCE 设备名（ibstatus 查）
NET_IF="${NET_IF:-enp1s0f1np1}"                                    # 上面那个设备对应的网口名
GID_INDEX="${GID_INDEX:-3}"                                        # RoCE v2/IPv4 的 GID index（show_gids 查）
MODEL_DIR="${MODEL_DIR:-$HOME/work/models/DeepSeek-V4-Flash-0731}" # 权重目录（两台都要有全量）
# =============================================================

# 可选开关：
#   TAG=production-hybrid-1.6（默认，支持 DSpark 投机）| production-ready（旧稳定版，投机必须 SPEC=off）
#   SPEC=dspark（默认）| mtp | off
TAG="${TAG:-production-hybrid-1.6}"
SPEC="${SPEC:-dspark}"

HEADLESS_FLAG=""
[ "$NODE_RANK" = "1" ] && HEADLESS_FLAG="--headless"

SPEC_FLAG=""
case "$SPEC" in
  dspark) SPEC_FLAG="--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":2}'" ;;
  mtp)    SPEC_FLAG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'" ;;
esac

docker rm -f vllm_unholy 2>/dev/null || true

COMMON_DOCKER_ARGS=(--gpus all -d --privileged --network host --ipc host --shm-size 10g
  --ulimit memlock=-1
  --device /dev/infiniband:/dev/infiniband
  -v "$MODEL_DIR:/model"
  --name vllm_unholy
  -e HF_HUB_OFFLINE=1
  -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$NCCL_HCA" -e "NCCL_IB_GID_INDEX=$GID_INDEX"
  -e "NCCL_SOCKET_IFNAME=$NET_IF" -e "GLOO_SOCKET_IFNAME=$NET_IF" -e "TP_SOCKET_IFNAME=$NET_IF"
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN)

COMMON_VLLM_ARGS="--served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --kv-cache-dtype fp8 --block-size 256 --max-model-len 500000 --max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8 --enable-prefix-caching --enable-chunked-prefill $SPEC_FLAG --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 --enable-flashinfer-autotune --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER_ADDR --master-port 29501 $HEADLESS_FLAG"

if [ "$TAG" = "production-ready" ]; then
  # ---- 旧镜像路径（验证过：无投机 24.5 tok/s）----
  docker run "${COMMON_DOCKER_ARGS[@]}" \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
    --entrypoint bash \
    aidendle94/sparkrun-vllm-ds4-gb10:$TAG \
    -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve /model --pipeline-parallel-size 1 $COMMON_VLLM_ARGS"
else
  # ---- hybrid-1.6 路径（DSpark，51 tok/s）：镜像自带 serve 脚本是单机版，这里手动起 vllm ----
  # 三处与镜像自带脚本的关键差异（都是双机/sm_121 踩坑换来的，详见文档第 7 步）：
  #   CUTE_DSL_ARCH 必须 sm_121a（sm_120a 的 arch-specific 二进制在 GB10 上不兼容）
  #   不开 VLLM_USE_AOT_COMPILE / B12X_MLA_SM120_UNIFIED / --attention-backend B12X_MLA_SPARSE（接口错位会炸）
  #   NCCL 走 RoCE（镜像自带脚本 NCCL_IB_DISABLE=1 是单机配置）
  docker run "${COMMON_DOCKER_ARGS[@]}" \
    -e CUTE_DSL_ARCH=sm_121a \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --entrypoint bash \
    aidendle94/sparkrun-vllm-ds4-gb10:$TAG \
    -lc "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model --moe-backend b12x --linear-backend b12x $COMMON_VLLM_ARGS"
fi

echo "launched vllm_unholy tag=$TAG spec=$SPEC node-rank=$NODE_RANK rc=$?"
sleep 2
docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_unholy || echo "WARN: container not in ps (check: docker logs vllm_unholy)"
