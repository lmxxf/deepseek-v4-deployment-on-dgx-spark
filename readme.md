# DeepSeek V4 Flash 双机部署指南

两台 DGX Spark 通过 ConnectX-7 + QSFP56 线缆组 256GB 共享显存，跑 DeepSeek V4 Flash (280B)。

## 硬件

- DGX Spark ×2，各 128GB GPU 显存
- QSFP56 DAC 线缆（200Gbps RDMA）
- 模型权重：`deepseek-ai/DeepSeek-V4-Flash`，~160GB FP4
- **host (spark-3a10)**：192.168.31.198 / CX7: 169.254.248.35
- **slave (spark-e8bb)**：192.168.31.172 / CX7: 169.254.30.81

## 第一步：网络配置（两台都执行）✅

```bash
sudo wget -O /etc/netplan/40-cx7.yaml \
  https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/cx7-netplan.yaml

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

## 第二步：自动发现 + SSH 免密（两台都执行）✅

```bash
wget https://github.com/NVIDIA/dgx-spark-playbooks/raw/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks

chmod +x discover-sparks
./discover-sparks
```

## 第三步：验证连通 ✅

```bash
# 在 host 上
ping -c 3 192.168.31.172

# 在 slave 上
ping -c 3 192.168.31.198
```

## 第四步：下载权重 ✅

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash
```

## 第五步：部署 vLLM 双机推理

使用 Docker 容器方案，NCCL/Ray/vLLM 全部打包在 NGC 容器里，不需要手动编译。

参考：https://github.com/mark-ramsey-ri/vllm-dgx-spark

### 5.1 克隆部署工具（在 host 上）

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment
git clone https://github.com/mark-ramsey-ri/vllm-dgx-spark.git
cd vllm-dgx-spark
```

### 5.2 查 CX7 InfiniBand IP（在 slave 上）

```bash
ip addr show enp1s0f1np1 | grep "inet "
```

记下 169.254.x.x 地址。

### 5.3 配置（在 host 上）

```bash
cp config.env config.local.env
```

编辑 `config.local.env`：

```
WORKER_HOST="192.168.31.172"
WORKER_IB_IP="169.254.30.81"
WORKER_USER="lmxxf"
TENSOR_PARALLEL="2"
MODEL="deepseek-ai/DeepSeek-V4-Flash"
```

### 5.4 启动集群

```bash
./start_cluster.sh
```

自动拉取 NGC vLLM 容器、启动 Ray 集群、加载模型。

### 5.5 验证

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

### 5.6 停止

```bash
./stop_cluster.sh
```
