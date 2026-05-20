# DeepSeek V4 Flash on Dual DGX Spark

[中文文档](README_CN.md) | English

Run **DeepSeek V4 Flash (280B, FP4)** on two **NVIDIA DGX Spark** nodes (128GB x2) connected via ConnectX-7 200Gbps RDMA.

## Current Status (2026-05-21)

| Scenario | Status | Speed |
|----------|--------|-------|
| TP=2, Chinese Q&A / writing / code | **Perfect** | **~12 tok/s** |
| TP=2, English Q&A / code | **Perfect** | **~12 tok/s** |
| PP=2, Chinese / English / code | **Works** | **~10-11 tok/s** |

Uses a **mixed backend** approach: 42 layers run Marlin (fast), the last MoE layer falls back to DeepGEMM via [Consumer-DeepGEMM](https://github.com/lmxxf/Consumer-DeepGEMM) (correct). This avoids Marlin's accumulated numerical error that corrupts English output at the final layer, while keeping 85% of Marlin's speed.

Built on [jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120) fork with Triton-rewritten attention/MLA kernels for SM120+.

## Pre-built Docker Image

```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

Based on jasl/vllm `ds4-sm120` + mixed backend patches (2026-05-11). Skip to [Step 4](#4-start-inference) and use `-t lmxxf/vllm-deepseek-v4-dgx-spark`.

For the optional PP=2 mode, build a tiny derived image from the uploaded image. This only replaces `mxfp4.py` so the layer-42 DeepGEMM fallback also matches PP layer names:

```bash
./scripts/build-pp-fix-image.sh --pull
```

This creates `vllm-deepseek-v4-pp-fix:latest`. To push it:

```bash
./scripts/build-pp-fix-image.sh \
  --base lmxxf/vllm-deepseek-v4-dgx-spark:latest \
  -t lmxxf/vllm-deepseek-v4-pp-fix:latest \
  --push
```

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

### 3. Pull Docker Image

The mixed backend patches are baked into the pre-built image. No source compilation needed:

```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

Pull on both nodes. Or pull on head and transfer via CX7: `docker save | ssh worker docker load`.

> **Want to build from source?** First build the base image using [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) + [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) (~30 min), then apply patches via `Dockerfile.marlin-fix` in this repo. See [DevHistory.md](DevHistory.md) for details.

### 4. Start Inference

#### Recommended: TP=2 mixed backend

```bash
cd /path/to/spark-vllm-docker

HF_HOME=/path/to/weights/parent \
VLLM_SPARK_EXTRA_DOCKER_ARGS="\
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
" \
./launch-cluster.sh -n <head_IP>,<worker_IP> -t lmxxf/vllm-deepseek-v4-dgx-spark exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --enforce-eager \
  --moe-backend marlin
```

Key flags:
- `--moe-backend marlin`: Use Marlin for MoE FP4 GEMM (fast)
- `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42`: Fall back the last MoE layer to DeepGEMM (correct). This is the key fix — Marlin's accumulated numerical error corrupts output at layer 42
- `--enforce-eager`: Disable CUDA graph (SM120+ compatibility)
- `--kv-cache-dtype fp8`: Required by DeepSeek V4
- `--max-model-len 4096`: Start small; 131072 may hang during Marlin weight loading under memory pressure
- `--gpu-memory-utilization 0.80`: Prevents OOM on worker node

You should see these in the startup logs:
```
Using 'MARLIN' Mxfp4 MoE backend.
MXFP4 layer 42 forced to DEEPGEMM_MXFP4 by VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

#### Optional: PP=2 pipeline parallel

PP=2 is verified to work, but for single-request generation it is slower than TP=2 (~10-11 tok/s vs ~12 tok/s). Use it as a pipeline-parallel baseline or for future concurrency testing, not as the default fastest mode.

It requires the PP fix image:

```bash
cd /path/to/deepseek-v4-flash-deployment
./scripts/build-pp-fix-image.sh --pull
```

Start with TP disabled and PP enabled:

```bash
cd /path/to/spark-vllm-docker

HF_HOME=/path/to/weights/parent \
VLLM_SPARK_EXTRA_DOCKER_ARGS="\
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
" \
./launch-cluster.sh -n <head_IP>,<worker_IP> -t vllm-deepseek-v4-pp-fix exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --enforce-eager \
  --moe-backend marlin
```

PP-specific requirements:
- `vllm-deepseek-v4-pp-fix`: Derived image that fixes layer-name parsing in `mxfp4.py`; without it, layer 42 fallback may not match and output can become corrupted
- `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`: Required for stable PP execution; the default Ray compiled-graph PP path can hang on the second request
- `--tensor-parallel-size 1 --pipeline-parallel-size 2`: Split layers across the two nodes instead of splitting tensors inside every layer

Expected PP=2 steady-state speed:

| Test | Output | Speed |
|------|--------|-------|
| `17 * 23 = ?` | `391` | sub-second after warmup |
| Chinese writing | Correct | ~10-11 tok/s |
| English code / quicksort | Correct | ~10-11 tok/s |

### 5. Stop

```bash
# TP=2 image
./launch-cluster.sh -n <head_IP>,<worker_IP> -t lmxxf/vllm-deepseek-v4-dgx-spark stop

# PP=2 image
./launch-cluster.sh -n <head_IP>,<worker_IP> -t vllm-deepseek-v4-pp-fix stop
```

Must run `stop` before restarting — otherwise the worker node's container keeps running.

### 6. Test

```bash
# Short output (Chinese, should be correct)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"2+2等于几"}],"max_tokens":50}' | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin),ensure_ascii=False,indent=2))"

# Long output (Chinese, should be correct, ~14 tok/s)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"请用500字介绍万里长城"}],"max_tokens":600}' | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin),ensure_ascii=False,indent=2))"

# Code generation (Chinese prompt, should be correct)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"写一个Python快速排序函数"}],"max_tokens":500}' | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin),ensure_ascii=False,indent=2))"

# English test (should be correct with mixed backend)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"What is quicksort? Explain with code."}],"max_tokens":200}' | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin),ensure_ascii=False,indent=2))"
```

Expected results:

| Test | Expected | Speed |
|------|----------|-------|
| Chinese short (2+2) | Correct | First request ~4s (warmup), then <1s |
| Chinese long (Great Wall) | Correct, finish_reason=stop | ~12 tok/s |
| Chinese code (quicksort) | Correct, runnable code | ~12 tok/s |
| English code (quicksort) | **Correct, clean output** | ~12 tok/s |

All tests should produce correct output — both Chinese and English.

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

For PP=2, the model is split by layer instead of tensor shard: rank0 owns the first 22 layers and rank1 owns the remaining 21 layers. This reduces per-layer all-reduce traffic, but single-request decode has pipeline bubbles, so TP=2 is still faster for interactive use.

## Key Technical Challenges

| Problem | Solution |
|---------|----------|
| Marlin FP4 MoE silent errors on SM120+ | Mixed backend: 42 layers Marlin + last layer DeepGEMM fallback |
| DeepGEMM doesn't support SM120+ | jasl fork includes Triton fallbacks; Consumer-DeepGEMM for last layer |
| vLLM 0.20.x doesn't fix SM120+ MXFP4 MoE | Use jasl fork instead of official main |
| PR #40082 (FlashInfer b12x) doesn't help V4 | b12x only registered in NVFP4 oracle; V4 uses MXFP4 |
| English output garbage (full Marlin) | Layer-by-layer bisection found error accumulates across all 43 layers; falling back only the last layer fixes it |
| TP padding source offset bug | Fixed in patched `layer.py` — checkpoint source offset now uses unpadded shard size |
| PP=2 second request hangs | Use `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` instead of the default Ray compiled-graph PP path |
| PP=2 output corruption | Use `vllm-deepseek-v4-pp-fix`; it lets the layer-42 fallback match PP layer names |

## Solution Comparison

| Approach | Chinese | English | Speed | Status |
|----------|---------|---------|-------|--------|
| vLLM official main | Silent errors | Silent errors | N/A | Broken on SM120+ |
| jasl/vllm fork (pure Marlin) | Correct | Garbage | 14 tok/s | Chinese only |
| Consumer-DeepGEMM (all layers) | Correct | Correct | 4.1 tok/s | Correct but slow |
| **Mixed backend (recommended)** | **Correct** | **Correct** | **12 tok/s** | **Best available** |
| Mixed backend + PP=2 | Correct | Correct | 10-11 tok/s | Works; useful PP baseline |

## Troubleshooting

See [DevHistory.md](DevHistory.md) for the full journey.

## How the Mixed Backend Works

Marlin's FP4 MoE kernel has a data layout bug on SM120+ (`ldmatrix` behaves differently than SM80/90). Each layer introduces a small numerical error. For the first 42 layers, the error is tolerable. But by layer 42 (the last MoE layer), the accumulated error corrupts the logits and produces garbage output.

The fix is simple: run layers 0-41 with Marlin (fast, small per-layer error acceptable) and fall back layer 42 to DeepGEMM via Consumer-DeepGEMM's Triton `tl.dot_scaled` kernel (correct, slightly slower). This was discovered through systematic layer-by-layer bisection testing.

`VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS` accepts comma-separated layer numbers or ranges (e.g., `42`, `40:43`, `0,42`).

## Related Links

- jasl fork: https://github.com/jasl/vllm (branch ds4-sm120, PR #40991)
- eugr build tool: https://github.com/eugr/spark-vllm-docker
- Consumer-DeepGEMM: https://github.com/lmxxf/Consumer-DeepGEMM (Triton `tl.dot_scaled` FP4 MoE kernel for SM120+)
- vLLM issues: https://github.com/vllm-project/vllm/issues/40928 https://github.com/vllm-project/vllm/issues/41063

## License

MIT
