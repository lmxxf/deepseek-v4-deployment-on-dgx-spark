# DeepSeek V4 Flash on Dual DGX Spark

[中文文档](README_CN.md) | English

Run **DeepSeek V4 Flash (280B, FP4)** on two **NVIDIA DGX Spark** nodes (128GB x2) connected via ConnectX-7 200Gbps RDMA.

## Pre-built Docker Image

```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

Skip to [Step 4](#4-start-inference) if using the pre-built image.

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

### 3. Build Docker Image (skip if using pre-built)

The solution combines three projects:
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) - DGX Spark Docker infrastructure
- [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) - SM120 Triton fallback kernels
- [zyang-dev/nccl](https://github.com/zyang-dev/nccl/tree/dgxspark-3node-ring) - NCCL mesh support

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker

# Pre-clone dependencies (Docker build can't access GitHub)
git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git nccl-src
git clone --depth 1 -b ds4-sm120 https://github.com/jasl/vllm.git vllm-sm120

# Remove hash locks from vllm requirements
find vllm-sm120/ -name "*.txt" -path "*/require*" -exec sed -i '/--hash/d; s/ \\$//' {} +
```

Key Dockerfile modifications:
- Replace `git clone` with `COPY nccl-src/` and `COPY vllm-sm120/`
- Build vLLM from jasl source instead of pre-built wheels
- Set `TORCH_CUDA_ARCH_LIST="12.0"` + `CMAKE_CUDA_ARCHITECTURES="120-real"` (sm_120 forward-compatible with sm_121)
- Add build deps: `cmake`, `setuptools_scm`, `setuptools>=75,<81`, `pybind11`

```bash
# Build and auto-copy to worker (~1 hour on Grace CPU)
./build-and-copy.sh -t vllm-node-sm120 -c
```

### 4. Start Inference

```bash
HF_HOME=/path/to/weights/parent \
./launch-cluster.sh -t vllm-node-sm120 exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --enforce-eager
```

Or with pre-built image:

```bash
HF_HOME=/path/to/weights/parent \
./launch-cluster.sh -t lmxxf/vllm-deepseek-v4-dgx-spark exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --enforce-eager
```

### 5. Test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

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
| DeepSeek V4 too new for NGC 26.03 | jasl/vllm ds4-sm120 fork with Triton fallback |
| DeepGEMM doesn't support sm_121 | TileLang kernels replace hyperconnection |
| PyTorch only ships sm_120 kernels | `TORCH_CUDA_ARCH_LIST="12.0"`, sm_120 forward-compatible with sm_121 |
| Docker build can't reach GitHub | Pre-clone on host, COPY into container |
| vLLM requirements hash mismatch | `sed` to remove all `--hash` lines |
| NGC torch can't be pip-upgraded | Use community torch from PyTorch whl index |

## Troubleshooting

See [DevHistory.md](DevHistory.md) for the full 48-hour, 18-pitfall journey.

## Known Limitations

- `--enforce-eager` required (torch.compile not yet compatible with SM120 kernels)
- SM121 runs SM120 kernels via forward compatibility (minor performance overhead possible)
- DeepSeek V4 ecosystem is still maturing (released 2026-04-24)

## Credits

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
- [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120)
- [zyang-dev/nccl](https://github.com/zyang-dev/nccl/tree/dgxspark-3node-ring)
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

## License

MIT
