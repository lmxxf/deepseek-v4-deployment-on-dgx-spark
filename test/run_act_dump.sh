#!/bin/bash
# 抓真实激活：用 act-dump 镜像跑混合后端（正确输出的配置），
# 每层 MoE 的输入/路由/输出存到 debug_acts/，供本地 Marlin-vs-ref 回放。
# 用法：
#   第 1 步（拷镜像到 slave，约几分钟）: ./run_act_dump.sh sync
#   第 2 步（启动服务，等 "Application startup complete"）: ./run_act_dump.sh serve
#   第 3 步（另开终端发请求，抓完即可 Ctrl+C 停服务）: ./run_act_dump.sh probe
set -e
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

case "${1:-}" in
sync)
  docker save vllm-deepseek-v4-act-dump:latest | ssh lmxxf@169.254.30.81 docker load
  ;;
serve)
  ssh lmxxf@169.254.30.81 "docker rm -f vllm_node 2>/dev/null" || true
  HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
  VLLM_SPARK_EXTRA_DOCKER_ARGS="\
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
    -e VLLM_MXFP4_DUMP_DIR=/root/.cache/huggingface/debug_acts \
    -e VLLM_MXFP4_DUMP_MAX=8 \
  " \
  ./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-deepseek-v4-act-dump exec \
    vllm serve /root/.cache/huggingface/deepseek-v4-flash \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.80 \
    --kv-cache-dtype fp8 \
    --max-model-len 4096 \
    --enforce-eager \
    --moe-backend marlin
  ;;
probe)
  # 英文 prompt（乱码触发场景）
  curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
    "model": "/root/.cache/huggingface/deepseek-v4-flash",
    "prompt": "Write a quicksort function in Python and explain how it works.",
    "max_tokens": 64, "temperature": 0
  }' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
  echo "--- dump files: ---"
  ls /home/lmxxf/work/deepseek-v4-flash-deployment/debug_acts/ | head
  ls /home/lmxxf/work/deepseek-v4-flash-deployment/debug_acts/ | wc -l
  ;;
*)
  echo "usage: $0 {sync|serve|probe}"
  ;;
esac
