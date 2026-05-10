#!/bin/bash
# 一键：stop → build → 安装CDG → 同步slave → 启动服务
set -e

cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

echo "=== Step 1: Stop existing service ==="
./launch-cluster.sh -t vllm-node-jasl-fix stop 2>/dev/null || true
ssh lmxxf@169.254.30.81 "docker rm -f vllm_node 2>/dev/null" || true

echo "=== Step 2: Build vLLM ==="
./build-and-copy.sh -t vllm-node-jasl-fix -c 169.254.30.81 --vllm-ref ds4-sm120

echo "=== Step 3: Install Consumer-DeepGEMM ==="
docker run --gpus all \
  -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
  --name jasl-cdg \
  vllm-node-jasl-fix:latest \
  bash -lc 'cd /work/Consumer-DeepGEMM && pip install . && python3 -c "import deep_gemm; print(\"CDG installed ✅\")"'

echo "=== Step 4: Commit + sync slave ==="
docker commit jasl-cdg vllm-node-jasl-fix:latest
docker rm jasl-cdg
docker save vllm-node-jasl-fix:latest | ssh lmxxf@169.254.30.81 docker load

echo "=== Step 5: Start V4 Flash ==="
ssh lmxxf@169.254.30.81 "docker rm -f vllm_node 2>/dev/null" || true

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
VLLM_SPARK_EXTRA_DOCKER_ARGS="-e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1" \
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-node-jasl-fix exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --enforce-eager
