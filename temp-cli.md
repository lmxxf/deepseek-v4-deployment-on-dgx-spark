## 1. 启动服务

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
VLLM_SPARK_EXTRA_DOCKER_ARGS="-e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1" \
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-node-sm120 exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --enforce-eager
```

## 2. 先检查服务是否起来了

```bash
curl http://localhost:8000/v1/models
```

## 3. 服务起来后，跑速度测试

```bash
curl -w '\n---\nTotal time: %{time_total}s\n' \
  http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/huggingface/deepseek-v4-flash",
    "messages": [{"role":"user","content":"请用100字介绍一下北京"}],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": false
  }'
```

返回的 JSON 里 `usage` 字段有 `completion_tokens`，除以 `Total time` 就是粗略 tok/s。

### 第一轮结果：88 tokens / 22.7s = 3.9 tok/s（含 prefill）

### Clone DeepGEMM fork

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment
git clone --recursive git@github.com:lmxxf/DeepGEMM.git
```

## 测速流程

### 0. 先看服务还活着不
```bash
curl -s http://localhost:8000/v1/models | head -1
```

### 长输出测试（减少 prefill 占比，看纯 decode 速度）

```bash
curl -w '\n---\nTotal time: %{time_total}s\n' \
  http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/huggingface/deepseek-v4-flash",
    "messages": [{"role":"user","content":"请详细介绍中国的四大发明，每个发明写200字以上"}],
    "max_tokens": 1000,
    "temperature": 0.7,
    "stream": false
  }'
```

## 排障：Ray 集群没组起来

### 先停掉
```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker
./launch-cluster.sh -t vllm-node-sm120 stop
```

### 检查 slave 能不能 SSH 通
```bash
ssh lmxxf@169.254.30.81 "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv"
```

### 检查 slave 上有没有 worker 容器在跑
```bash
ssh lmxxf@169.254.30.81 "docker ps -a | grep vllm"
```

### 检查 host 上 Ray 状态
```bash
docker exec vllm-node-sm120 ray status 2>/dev/null || echo "容器不在"
```

## 排障结果

问题1：slave 上跑着 3 天前的旧容器，占着 GPU
问题2：launch-cluster.sh 没发现 slave，进了 solo mode

### 修复步骤

#### 1. 先杀 slave 上的旧容器
```bash
ssh lmxxf@169.254.30.81 "docker rm -f ray-worker-spark-e8bb"
```

#### 2. 检查为什么 launch-cluster.sh 发现不了 slave
```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker
grep -n "discover\|detect\|solo\|Only local" launch-cluster.sh | head -20
```

#### 3. 手动 ping slave CX7 确认网络通
```bash
ping -c 2 169.254.30.81
```
