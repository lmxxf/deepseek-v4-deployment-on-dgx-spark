#!/bin/bash
# Benchmark: dot_scaled FP4 MoE kernel vs float32 fused kernel
#
# 在 DGX Spark 上复现 182/183 期的 kernel 性能对比。
# 需要 vllm-node-sm120:latest 或任何含 CUDA 13.0+ 和 Triton 3.6 的容器。
#
# 用法：
#   docker run --rm --gpus all \
#     -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
#     vllm-node-sm120:latest \
#     bash /work/scripts/benchmark_dot_scaled.sh

set -e
cd /work

# 确保 Consumer-DeepGEMM 存在
if [ ! -d Consumer-DeepGEMM ]; then
    echo "Cloning Consumer-DeepGEMM..."
    git clone https://github.com/lmxxf/Consumer-DeepGEMM.git
fi

echo "=== Step 1: INT8 tensor core 验证 ==="
python3 test_fp8_dot.py

echo ""
echo "=== Step 2: FP8 tensor core 验证 ==="
# 已包含在 test_fp8_dot.py 中

echo ""
echo "=== Step 3: tl.dot_scaled FP4 验证 ==="
python3 test_dot_scaled.py

echo ""
echo "=== Step 4: dot_scaled Kernel Benchmark ==="
PYTHONPATH=/work/Consumer-DeepGEMM python3 Consumer-DeepGEMM/tests/test_dot_scaled_fused.py

echo ""
echo "=== Step 5: Dispatch Profiling ==="
PYTHONPATH=/work/Consumer-DeepGEMM python3 test_dispatch_profile.py

echo ""
echo "=== Done ==="
echo "对比 179 期 float32 fused kernel (5.16ms FC1) vs dot_scaled (0.69ms FC1) = 7.5x"
echo "端到端：0.79 → 4.1 tok/s (5.2x 提速)"
