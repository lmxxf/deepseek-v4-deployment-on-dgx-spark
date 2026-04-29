# DeepSeek V4 Flash 双机部署指南

[English](README.md) | 中文

两台 DGX Spark (128GB×2) 通过 ConnectX-7 200Gbps 运行 DeepSeek V4 Flash (280B, 158GB FP4)。

## 硬件

- DGX Spark ×2，各 128GB 统一内存，GPU: NVIDIA GB10 (sm_121)
- QSFP56 DAC 线缆，ConnectX-7 200Gbps RDMA 直连
- **host (spark-3a10)**：192.168.31.198 / CX7: 169.254.248.35
- **slave (spark-e8bb)**：192.168.31.172 / CX7: 169.254.30.81

## 预构建镜像

不想自己编译的话，直接拉取预构建镜像（ARM64/aarch64 only）：

```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

然后跳到第 4 步启动推理服务，把 `-t vllm-node-sm120` 改成 `-t lmxxf/vllm-deepseek-v4-dgx-spark`。

## 从零开始

### 1. 双机组网

两台都执行：

```bash
sudo wget -O /etc/netplan/40-cx7.yaml \
  https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/cx7-netplan.yaml
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply

wget https://github.com/NVIDIA/dgx-spark-playbooks/raw/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
./discover-sparks
```

### 2. 下载模型权重（两台都需要）

host 上下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash
```

rsync 到 slave（走 CX7，581MB/s）：

```bash
rsync -avP /home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/ \
  lmxxf@<slave_CX7_IP>:/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/
```

### 3. 构建 Docker 镜像

核心方案：eugr/spark-vllm-docker 基础设施 + jasl/vllm ds4-sm120 fork（sm_120 Triton fallback）。

```bash
# 克隆构建工具
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker

# 克隆依赖源码（Docker 内无法访问 GitHub）
git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git nccl-src
git clone --depth 1 -b ds4-sm120 https://github.com/jasl/vllm.git vllm-sm120

# 删除 vllm requirements 里的 hash 锁定
find vllm-sm120/ -name "*.txt" -path "*/require*" -exec sed -i '/--hash/d; s/ \\$//' {} +
```

修改 Dockerfile（关键改动）：
- NCCL：`COPY nccl-src/` 替代 `git clone`
- vLLM：用 jasl 源码编译替代预编译 whl，`TORCH_CUDA_ARCH_LIST="12.0"` + `CMAKE_CUDA_ARCHITECTURES="120-real"`
- 额外依赖：`cmake`、`setuptools_scm`、`setuptools>=75,<81`、`pybind11`

```bash
# 构建并自动复制到 slave（约 1 小时）
./build-and-copy.sh -t vllm-node-sm120 -c
```

### 4. 启动推理服务

```bash
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

### 5. 测试

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":100}'
```

## 显存分配（双机 256GB）

| 项目 | 单机 | 双机合计 |
|------|------|----------|
| 统一内存 | 128 GB | 256 GB |
| 可用 GPU 显存 (~121 GB) × 0.85 | ~103 GB | ~206 GB |
| 模型权重（TP=2，每台存一半） | 73.85 GB | ~148 GB |
| **剩余给 KV Cache** | **~29 GB** | **~58 GB** |

DeepSeek V4 Flash 采用 CSA + HCA 混合注意力架构，KV cache 极小——仅为传统 GQA 模型的 **~2%**。1M 上下文 fp8 KV cache 只需约 5GB。所以 58GB 的 KV cache 空间**完全支持 1M token 上下文**。

推荐 `--max-model-len` 设置：

| 设置 | 场景 |
|------|------|
| `8192` | 快速测试 |
| `65536` | 日常多轮对话 |
| `1000000` | 完整 1M 上下文（支持，KV cache 仅需约 5GB） |

## 架构说明

```
用户 → host:8000 (vLLM API)
              ↓
        Ray 调度器 (host)
        ↙          ↘
  GPU 0 (host)   GPU 1 (slave)
  73.85 GiB      73.85 GiB
        ↘          ↙
     NCCL AllReduce (CX7 200Gbps)
              ↓
        输出 token
```

- **Ray**：调度员，管进程启停和资源分配
- **NCCL**：通信层，GPU 间张量传输
- **Tensor Parallel**：每层矩阵乘法水平切分，两台同时算，每层 AllReduce 同步
- MoE 模型只激活 13B/280B 参数，跨机通信量相对较小

## 关键技术点

| 问题 | 解法 |
|------|------|
| DeepSeek V4 太新，NGC 26.03 不支持 | jasl/vllm ds4-sm120 fork，Triton fallback |
| DeepGEMM 不支持 sm_121 | jasl fork 用 TileLang 重写 hyperconnection kernel |
| torch 社区版只到 sm_120 | 编译时 `TORCH_CUDA_ARCH_LIST="12.0"`，sm_120 前向兼容 sm_121 |
| Docker 内无法访问 GitHub | 宿主机预先 clone，COPY 进容器 |
| vLLM requirements hash 锁定 | `sed` 删除所有 `--hash` 行 |

## 踩坑详情

见 [DevHistory.md](DevHistory.md)——48 小时，18 个坑的完整记录。
