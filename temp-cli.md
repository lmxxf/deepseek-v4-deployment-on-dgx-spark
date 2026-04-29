```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
./launch-cluster.sh -t vllm-node-sm120 exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --enforce-eager
```
