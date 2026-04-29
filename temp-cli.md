## 在现有容器里用 jasl fork 替换 vllm

### 步骤1：启动集群容器（不跑模型）

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
./launch-cluster.sh -t vllm-node-tf5 start
```

### 步骤2：在 host 容器里替换 vllm（jasl fork 带 sm120 支持）

```bash
docker exec vllm_node bash -c "pip install git+https://github.com/jasl/vllm.git@ds4-sm120 --no-build-isolation 2>&1 | tail -5"
```

### 步骤3：在 slave 容器里也替换

```bash
ssh lmxxf@169.254.30.81 'docker exec vllm_node bash -c "pip install git+https://github.com/jasl/vllm.git@ds4-sm120 --no-build-isolation 2>&1 | tail -5"'
```

### 步骤4：在 host 容器里手动启动 vllm

```bash
docker exec -it vllm_node bash -c "\
  export VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel && \
  export FLASHINFER_DISABLE_VERSION_CHECK=1 && \
  export VLLM_TRITON_MLA_SPARSE=1 && \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --enforce-eager"
```
