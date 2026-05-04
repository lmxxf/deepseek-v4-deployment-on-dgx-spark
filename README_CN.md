# DeepSeek V4 Flash 双机部署指南

[English](README.md) | 中文

两台 DGX Spark (128GB×2) 通过 ConnectX-7 200Gbps 运行 DeepSeek V4 Flash (280B, 158GB FP4)。

## 当前状态（2026-05-04）

| 场景 | 状态 | 速度 |
|------|------|------|
| 中文问答/写作/代码 | 完美 | ~14 tok/s |
| 英文问答/代码 | 偶尔开头出垃圾 token | ~14 tok/s |

基于 jasl/vllm fork（`ds4-sm120` 分支），用 Triton 重写了 DeepSeek V4 的关键 kernel，绕过 Marlin 在 sm_120+ 上的静默计算错误。英文残留问题是社区已知的，等 jasl 继续覆盖残余路径或 vLLM 官方修复（issue #40928）。

## 硬件

- DGX Spark ×2，各 128GB 统一内存，GPU: NVIDIA GB10 (sm_121)
- QSFP56 DAC 线缆，ConnectX-7 200Gbps RDMA 直连
- **host (spark-3a10)**：192.168.31.198 / CX7: 169.254.248.35
- **slave (spark-e8bb)**：192.168.31.172 / CX7: 169.254.30.81

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

核心方案：eugr/spark-vllm-docker 基础设施 + jasl/vllm ds4-sm120 fork（Triton 重写 DeepSeek V4 关键 kernel）。

```bash
# 克隆构建工具
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

修改 Dockerfile 第 190 行的 clone URL：

```diff
- git clone --recursive https://github.com/vllm-project/vllm.git
+ git clone --recursive https://github.com/jasl/vllm.git
```

构建并同步到 slave：

```bash
./build-and-copy.sh -t vllm-node-jasl -c <slave_CX7_IP> --vllm-ref ds4-sm120
```

约 30 分钟编译 + 1 分钟同步。

### 4. 启动推理服务

```bash
HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
VLLM_SPARK_EXTRA_DOCKER_ARGS="-e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1" \
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-node-jasl exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --enforce-eager
```

关键参数：
- `--enforce-eager`：禁用 CUDA graph（sm_120+ 兼容性问题）
- `--kv-cache-dtype fp8`：DeepSeek V4 要求 FP8 KV cache
- `--max-model-len 131072`：128K 上下文，可按需调整

### 5. 停止

```bash
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-node-jasl stop
```

重启前必须先 stop——否则 slave 上的容器还在跑。不要直接 ctrl-c。

### 6. 测试

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
| Marlin FP4 MoE 在 sm_120+ 静默算错 | jasl/vllm fork 用 Triton 重写关键 kernel，绕过 Marlin |
| DeepGEMM 不支持 sm_121 | jasl fork 内置 Triton fallback |
| vLLM 0.20.x 未修复 sm_120+ MXFP4 MoE | 使用 jasl fork 而非官方 main |
| PR #40082 (FlashInfer b12x) 对 V4 无效 | b12x 只注册在 NVFP4 oracle，V4 用 MXFP4 |
| 英文输出偶尔有垃圾 token | 已知问题，等 jasl 继续覆盖残余 Marlin 路径 |

## 方案对比

| 方案 | 中文 | 英文 | 速度 | 状态 |
|------|------|------|------|------|
| vLLM 官方 main | 静默算错 | 静默算错 | N/A | 不可用 |
| Consumer-DeepGEMM（CUTLASS 替换 Marlin） | 正确 | 正确 | 0.8 tok/s | 正确但太慢 |
| **jasl/vllm fork（推荐）** | **正确** | **偶尔垃圾** | **14 tok/s** | **中文日常可用** |
| eugr exp-b12x (PR #40082) | N/A | N/A | N/A | 不适用 DeepSeek V4 |

## 相关链接

- jasl fork: https://github.com/jasl/vllm （分支 ds4-sm120，PR #40991）
- eugr 构建工具: https://github.com/eugr/spark-vllm-docker
- Consumer-DeepGEMM: https://github.com/lmxxf/Consumer-DeepGEMM （CUTLASS 方案，正确但慢，学习用）
- vLLM issue: https://github.com/vllm-project/vllm/issues/40928 https://github.com/vllm-project/vllm/issues/41063

## 踩坑详情

见 [DevHistory.md](DevHistory.md)——完整记录。
