# DeepSeek V4 Flash on DGX Spark (ARM64/sm_121)

Pre-built Docker image for running **DeepSeek V4 Flash (280B, FP4)** on **NVIDIA DGX Spark** dual-node clusters.

## Key Features

- Based on [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) infrastructure
- Includes [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) fork with Triton fallback kernels for SM120/121
- Supports dual DGX Spark (128GB x2) with Ray + NCCL over ConnectX-7 200Gbps
- DeepGEMM hyperconnection replaced with TileLang kernels
- CUDA arch: sm_120 (forward compatible with sm_121/GB10)

## Quick Start

```bash
# Pull image
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest

# Download model weights
export HF_ENDPOINT=https://hf-mirror.com  # China mirror, optional
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ./deepseek-v4-flash

# Start with eugr's launch-cluster.sh
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker

HF_HOME=/path/to/weights/parent \
./launch-cluster.sh -t lmxxf/vllm-deepseek-v4-dgx-spark exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --enforce-eager

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

## Hardware Requirements

- 2x NVIDIA DGX Spark (128GB each)
- QSFP56 DAC cable (ConnectX-7 200Gbps)
- Model weights: ~160GB (FP4)

## Build from Source

See [GitHub repo](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) for full build instructions and troubleshooting guide.

## Architecture

| Component | Role |
|-----------|------|
| vLLM 0.1.dev (jasl fork) | Inference engine with SM120 Triton fallback |
| Ray | Distributed executor across 2 nodes |
| NCCL | GPU-to-GPU communication over RoCE |
| FlashInfer | Optimized attention kernels |
| TileLang | Hyperconnection kernel replacement |

## Known Limitations

- `--enforce-eager` required (torch.compile not yet compatible)
- SM121 forward compatibility via SM120 kernels (minor perf overhead)
- DeepSeek V4 is 5 days old (2026-04-24), ecosystem still maturing

## Credits

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) - DGX Spark Docker infrastructure
- [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) - SM120 Triton fallback kernels
- [zyang-dev/nccl](https://github.com/zyang-dev/nccl/tree/dgxspark-3node-ring) - NCCL mesh support
