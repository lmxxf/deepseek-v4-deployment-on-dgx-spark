# DeepSeek V4 Flash on Dual DGX Spark

[中文文档](README_CN.md) | English

Run **DeepSeek V4 Flash (280B, FP4)** on two **NVIDIA DGX Spark** nodes (128GB x2) connected via ConnectX-7 200Gbps RDMA.

## Current Status (2026-05-04)

| Scenario | Status | Speed |
|----------|--------|-------|
| Chinese Q&A / writing / code | Perfect | ~14 tok/s |
| English Q&A / code | Occasional garbage tokens at start | ~14 tok/s |

Uses [jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120) fork with Triton-rewritten kernels, bypassing Marlin's silent computation errors on SM120+. English issue is known; tracking jasl's ongoing work and vLLM issue #40928.

## Hardware Requirements

- 2x NVIDIA DGX Spark (128GB unified memory each, GPU: GB10 sm_121)
- QSFP56 DAC cable (ConnectX-7 200Gbps point-to-point)
- Model weights: ~160GB (FP4 quantized)

## Setup Guide

### 1. Network Configuration

Run on **both** nodes:

```bash
sudo wget -O /etc/netplan/40-cx7.yaml \
  https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/cx7-netplan.yaml
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply

wget https://github.com/NVIDIA/dgx-spark-playbooks/raw/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
./discover-sparks
```

### 2. Download Model Weights

On **head node**:

```bash
export HF_ENDPOINT=https://hf-mirror.com  # China mirror, optional
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ./deepseek-v4-flash
```

Sync to **worker node** via CX7 (581MB/s):

```bash
rsync -avP ./deepseek-v4-flash/ user@<worker_CX7_IP>:./deepseek-v4-flash/
```

### 3. Build Docker Image

The solution combines two projects:
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) - DGX Spark Docker infrastructure (NCCL, PyTorch, FlashInfer, Ray)
- [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) - Triton-rewritten kernels for SM120+ (sparse MLA, FP8 einsum, paged MQA)

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Modify Dockerfile line 190 — change the vLLM clone URL:

```diff
- git clone --recursive https://github.com/vllm-project/vllm.git
+ git clone --recursive https://github.com/jasl/vllm.git
```

Build and sync to worker:

```bash
./build-and-copy.sh -t vllm-node-jasl -c <worker_CX7_IP> --vllm-ref ds4-sm120
```

~30 minutes compile + ~1 minute sync.

### 4. Start Inference

```bash
HF_HOME=/path/to/weights/parent \
VLLM_SPARK_EXTRA_DOCKER_ARGS="-e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1" \
./launch-cluster.sh -n <head_IP>,<worker_IP> -t vllm-node-jasl exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --enforce-eager
```

Key flags:
- `--enforce-eager`: Disable CUDA graph (SM120+ compatibility)
- `--kv-cache-dtype fp8`: Required by DeepSeek V4
- `--max-model-len 131072`: 128K context, adjust as needed

### 5. Stop

```bash
./launch-cluster.sh -n <head_IP>,<worker_IP> -t vllm-node-jasl stop
```

Must run `stop` before restarting — otherwise the worker node's container keeps running.

### 6. Test

```bash
# Short output (Chinese, should be correct)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"2+2等于几"}],"max_tokens":50}' | python3 -m json.tool

# Long output (Chinese, should be correct, ~14 tok/s)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"请用500字介绍万里长城"}],"max_tokens":600}' | python3 -m json.tool

# Code generation (Chinese prompt, should be correct)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"写一个Python快速排序函数"}],"max_tokens":500}' | python3 -m json.tool

# English test (known issue: may produce garbage tokens)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"What is quicksort? Explain with code."}],"max_tokens":500}' | python3 -m json.tool
```

Expected results:

| Test | Expected | Speed |
|------|----------|-------|
| Chinese short (2+2) | Correct | First request ~17s (warmup), then <1s |
| Chinese long (Great Wall) | Correct, finish_reason=stop | ~14 tok/s |
| Chinese code (quicksort) | Correct, runnable code | ~14 tok/s |
| English code (quicksort) | May produce garbage tokens | ~14 tok/s |

Chinese correct + English broken = deployment successful (known jasl fork limitation).

## Memory Budget (Dual Spark, 256GB total)

| Item | Per Node | Total |
|------|----------|-------|
| Unified Memory | 128 GB | 256 GB |
| Usable GPU Memory (~121 GB) × 0.85 | ~103 GB | ~206 GB |
| Model Weights (TP=2, split across nodes) | 73.85 GB | ~148 GB |
| **Available for KV Cache** | **~29 GB** | **~58 GB** |

DeepSeek V4 Flash uses CSA + HCA hybrid attention with extremely compact KV cache — only **~2% of traditional GQA models**. At 1M context with fp8 KV cache, it needs only ~5 GB. So 58 GB of KV cache can comfortably support **1M token context** on dual Spark.

Recommended `--max-model-len` settings:

| Setting | Use Case |
|---------|----------|
| `8192` | Quick testing, minimal memory |
| `65536` | Daily use, multi-turn conversations |
| `1000000` | Full 1M context (supported, ~5 GB KV cache) |

## Architecture

```
Client -> head:8000 (vLLM API)
                |
          Ray Scheduler (head)
          /              \
  GPU 0 (head)     GPU 1 (worker)
  73.85 GiB        73.85 GiB
          \              /
       NCCL AllReduce (CX7 200Gbps)
                |
          Output tokens
```

| Component | Role |
|-----------|------|
| **Ray** | Distributed executor, process orchestration |
| **NCCL** | GPU-to-GPU tensor communication over RoCE |
| **vLLM** | Inference engine (jasl fork with SM120 Triton fallback) |
| **FlashInfer** | Optimized attention kernels |
| **TileLang** | Hyperconnection kernel replacement for SM120 |

## Key Technical Challenges

| Problem | Solution |
|---------|----------|
| Marlin FP4 MoE silent errors on SM120+ | jasl/vllm fork rewrites critical kernels in Triton, bypasses Marlin |
| DeepGEMM doesn't support SM120+ | jasl fork includes Triton fallbacks |
| vLLM 0.20.x doesn't fix SM120+ MXFP4 MoE | Use jasl fork instead of official main |
| PR #40082 (FlashInfer b12x) doesn't help V4 | b12x only registered in NVFP4 oracle; V4 uses MXFP4 |
| English output has occasional garbage tokens | Known issue; awaiting jasl's coverage of remaining Marlin paths |

## Solution Comparison

| Approach | Chinese | English | Speed | Status |
|----------|---------|---------|-------|--------|
| vLLM official main | Silent errors | Silent errors | N/A | Broken |
| Consumer-DeepGEMM (CUTLASS replace Marlin) | Correct | Correct | 0.8 tok/s | Correct but too slow |
| **jasl/vllm fork (recommended)** | **Correct** | **Occasional garbage** | **14 tok/s** | **Usable for Chinese** |
| eugr exp-b12x (PR #40082) | N/A | N/A | N/A | Not applicable to V4 |

## Troubleshooting

See [DevHistory.md](DevHistory.md) for the full journey.

## Related Links

- jasl fork: https://github.com/jasl/vllm (branch ds4-sm120, PR #40991)
- eugr build tool: https://github.com/eugr/spark-vllm-docker
- Consumer-DeepGEMM: https://github.com/lmxxf/Consumer-DeepGEMM (CUTLASS approach, correct but slow, educational)
- vLLM issues: https://github.com/vllm-project/vllm/issues/40928 https://github.com/vllm-project/vllm/issues/41063

## License

MIT
