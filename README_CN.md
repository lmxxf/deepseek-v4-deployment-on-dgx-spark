# DeepSeek V4 Flash 双机部署指南

[English](README.md) | 中文

两台 DGX Spark (128GB×2) 通过 ConnectX-7 200Gbps 运行 DeepSeek V4 Flash (280B, 158GB FP4)。

## 当前状态（2026-05-11）

| 场景 | 状态 | 速度 |
|------|------|------|
| 中文问答/写作/代码 | **完美** | **~12 tok/s** |
| 英文问答/代码 | **完美** | **~12 tok/s** |

基于**混合后端**方案：42 层走 Marlin（快），最后一层 MoE 回退到 DeepGEMM（[Consumer-DeepGEMM](https://github.com/lmxxf/Consumer-DeepGEMM) 的 Triton `tl.dot_scaled` kernel，正确）。Marlin 的数值误差在单层内可接受，但经 43 层累积后在 logits 处放大导致英文输出崩坏——只回退最后一层就能修复。

底层基于 [jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120) fork，用 Triton 重写了 attention/MLA kernel 适配 SM120+。

## 硬件

- DGX Spark ×2，各 128GB 统一内存，GPU: NVIDIA GB10 (sm_121)
- QSFP56 DAC 线缆，ConnectX-7 200Gbps RDMA 直连

## 预构建镜像

不想自己编译的话，直接拉取（ARM64/aarch64 only）：

```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

基于 jasl/vllm `ds4-sm120` + 混合后端补丁（2026-05-11），跳到第 4 步启动服务，把 `-t vllm-node-jasl-marlin-fix` 改成 `-t lmxxf/vllm-deepseek-v4-dgx-spark`。

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
  --local-dir ./deepseek-v4-flash
```

rsync 到 slave（走 CX7，581MB/s）：

```bash
rsync -avP ./deepseek-v4-flash/ user@<slave_CX7_IP>:./deepseek-v4-flash/
```

### 3. 构建 Docker 镜像

核心方案：eugr/spark-vllm-docker 基础设施 + jasl/vllm ds4-sm120 fork + 混合后端补丁。

```bash
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
HF_HOME=/path/to/weights/parent \
VLLM_SPARK_EXTRA_DOCKER_ARGS="\
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
" \
./launch-cluster.sh -n <head_IP>,<worker_IP> -t vllm-node-jasl-marlin-fix exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --enforce-eager \
  --moe-backend marlin
```

关键参数：
- `--moe-backend marlin`：MoE 使用 Marlin 后端（快）
- `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42`：最后一层 MoE 回退到 DeepGEMM（正确）。这是核心修复——Marlin 的数值误差在 layer 42 累积到不可接受
- `--enforce-eager`：禁用 CUDA graph（sm_120+ 兼容性）
- `--kv-cache-dtype fp8`：DeepSeek V4 要求
- `--max-model-len 4096`：先用小上下文；131072 在 Marlin 权重加载时可能卡住
- `--gpu-memory-utilization 0.80`：防止 worker 节点 OOM

启动后确认日志里有：
```
Using 'MARLIN' Mxfp4 MoE backend.
MXFP4 layer 42 forced to DEEPGEMM_MXFP4 by VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

### 5. 停止

```bash
./launch-cluster.sh -n <head_IP>,<worker_IP> -t vllm-node-jasl-marlin-fix stop
```

重启前必须先 stop——否则 slave 上的容器还在跑。不要直接 ctrl-c。

### 6. 测试

```bash
# 中文
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"请用200字介绍万里长城"}],"max_tokens":300}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); u=r['usage']; print(f\"tokens: {u['completion_tokens']}\"); print(r['choices'][0]['message']['content'][:300])"

# 英文
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"What is quicksort? Explain with code."}],"max_tokens":200}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); u=r['usage']; print(f\"tokens: {u['completion_tokens']}\"); print(r['choices'][0]['message']['content'][:300])"

# 数学
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"17 * 23 = ?"}],"max_tokens":50}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); u=r['usage']; print(f\"tokens: {u['completion_tokens']}\"); print(r['choices'][0]['message']['content'])"
```

预期结果：

| 测试 | 预期 | 速度 |
|------|------|------|
| 中文（万里长城） | 正确，finish_reason=stop | ~12 tok/s |
| 英文（quicksort） | **正确，干净输出** | ~12 tok/s |
| 数学（17×23） | 正确 | ~12 tok/s |

中英文全对 = 部署成功。

## 显存分配（双机 256GB）

| 项目 | 单机 | 双机合计 |
|------|------|----------|
| 统一内存 | 128 GB | 256 GB |
| 可用 GPU 显存 (~121 GB) × 0.80 | ~97 GB | ~194 GB |
| 模型权重（TP=2，每台存一半） | 74.13 GB | ~148 GB |
| **剩余给 KV Cache** | **~23 GB** | **~46 GB** |

DeepSeek V4 Flash 采用 CSA + HCA 混合注意力架构，KV cache 极小——仅为传统 GQA 模型的 ~2%。4K 上下文足够日常测试。

## 混合后端原理

Marlin 的 FP4 MoE kernel 在 SM120+ 上有数据布局 bug（`ldmatrix` 指令行为和 SM80/90 不同）。每一层引入的误差很小，前 42 层都可容忍。但到第 42 层（最后一个 MoE 层），累积误差污染了 logits，输出变成垃圾。

修复方法：0-41 层走 Marlin（快，单层误差可接受），第 42 层回退到 DeepGEMM（Consumer-DeepGEMM 的 Triton `tl.dot_scaled` kernel，正确但稍慢）。这个发现来自系统性的按层二分测试。

`VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS` 支持逗号分隔的层号或半开区间（如 `42`、`40:43`、`0,42`）。

## 方案对比

| 方案 | 中文 | 英文 | 速度 | 状态 |
|------|------|------|------|------|
| vLLM 官方 main | 静默算错 | 静默算错 | N/A | 不可用 |
| jasl/vllm fork（纯 Marlin） | 正确 | 垃圾 | 14 tok/s | 仅中文 |
| Consumer-DeepGEMM（全层） | 正确 | 正确 | 4.1 tok/s | 正确但慢 |
| **混合后端（推荐）** | **正确** | **正确** | **12 tok/s** | **当前最优** |

## 相关链接

- jasl fork: https://github.com/jasl/vllm （分支 ds4-sm120，PR #40991）
- eugr 构建工具: https://github.com/eugr/spark-vllm-docker
- Consumer-DeepGEMM: https://github.com/lmxxf/Consumer-DeepGEMM （SM120+ 的 Triton FP4 MoE kernel）
- vLLM issue: https://github.com/vllm-project/vllm/issues/40928

## 踩坑详情

见 [DevHistory.md](DevHistory.md)——完整记录。
