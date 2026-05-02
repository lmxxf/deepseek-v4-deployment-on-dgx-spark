## 启动 Consumer-DeepGEMM 修复版服务

当前双机镜像：

```text
vllm-node-sm121-cdg:latest ab948bd4d9d9
```

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
VLLM_SPARK_EXTRA_DOCKER_ARGS="-e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1" \
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-node-sm121-cdg exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --enforce-eager
```

## 验证镜像内 Consumer-DeepGEMM

```bash
docker run --rm --gpus all vllm-node-sm121-cdg:latest bash -lc 'python3 - <<PY
import consumer_deep_gemm as dg
import vllm.third_party.deep_gemm as vdg
print("consumer", dg.native_build_info())
print("vllm", vdg.native_build_info())
PY'
```

## 长输出测试

```bash
curl -w '\n---\nTotal time: %{time_total}s\n' \
  http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/huggingface/deepseek-v4-flash",
    "messages": [{"role":"user","content":"请用500字介绍万里长城"}],
    "max_tokens": 600,
    "temperature": 0.7,
    "stream": false
  }'
```
