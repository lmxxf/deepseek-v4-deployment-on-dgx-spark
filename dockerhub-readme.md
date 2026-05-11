# DeepSeek V4 Flash on DGX Spark (ARM64/sm_121)

Pre-built Docker image for running **DeepSeek V4 Flash (280B, FP4)** on **NVIDIA DGX Spark** dual-node clusters. **Chinese and English output both correct, ~12 tok/s.**

## What's Inside

- [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) fork with Triton fallback kernels for SM120/121
- [Consumer-DeepGEMM](https://github.com/lmxxf/Consumer-DeepGEMM) `tl.dot_scaled` FP4 MoE kernel for the last layer fallback
- Mixed backend patches: TP padding offset fix, swiglu_limit fix, workspace alias fix
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) infrastructure (Ray, NCCL, FlashInfer, TileLang)

## Quick Start

```bash
# Pull image (both nodes)
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest

# Download model weights
export HF_ENDPOINT=https://hf-mirror.com  # China mirror, optional
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ./deepseek-v4-flash

# Start with eugr's launch-cluster.sh
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker

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

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"What is quicksort?"}],"max_tokens":200}'
```

Key: `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42` falls back the last MoE layer to Consumer-DeepGEMM (correct), while layers 0-41 use Marlin (fast). This avoids Marlin's accumulated numerical error on SM120+.

## Hardware Requirements

- 2x NVIDIA DGX Spark (128GB each)
- QSFP56 DAC cable (ConnectX-7 200Gbps)
- Model weights: ~160GB (FP4)

## Performance

| Scenario | Speed | Status |
|----------|-------|--------|
| Chinese Q&A / code | ~12 tok/s | Correct |
| English Q&A / code | ~12 tok/s | Correct |

## Known Limitations

- `--enforce-eager` required (torch.compile not yet compatible with SM121)
- `--max-model-len 4096` recommended for initial testing; 131072 may hang during weight loading
- `--gpu-memory-utilization 0.80` to prevent worker OOM

## Full Documentation

See [GitHub repo](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) for build instructions, troubleshooting, and the full development history.

## Credits

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) - DGX Spark Docker infrastructure
- [jasl/vllm ds4-sm120](https://github.com/jasl/vllm/tree/ds4-sm120) - SM120 Triton fallback kernels
- [Consumer-DeepGEMM](https://github.com/lmxxf/Consumer-DeepGEMM) - Triton FP4 MoE kernel for SM120+
