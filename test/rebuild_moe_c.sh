#!/bin/bash
# Rebuild _moe_C.so inside jasl container with the ldmatrix fix.
# Only recompiles the MoE C extension, not all of vLLM.

set -e

VLLM_SRC=/work/spark-vllm-docker/vllm-sm120
SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages
MARLIN_ARCH_LIST=${MARLIN_ARCH_LIST:-8.0,12.0}
MARLIN_NVCC_ARCH=${MARLIN_NVCC_ARCH:-sm_120}

echo "=== Rebuilding _moe_C.so with ldmatrix sm_120 fix ==="
echo "Kernel generator arch list: $MARLIN_ARCH_LIST"
echo "NVCC arch: $MARLIN_NVCC_ARCH"

# Verify the fix is in the source
if ! grep -q "shfl_sync" $VLLM_SRC/csrc/moe/marlin_moe_wna16/marlin_template.h; then
    echo "ERROR: ldmatrix fix not found in marlin_template.h"
    exit 1
fi
echo "ldmatrix fix present in source ✅"

cd $VLLM_SRC

python3 csrc/moe/marlin_moe_wna16/generate_kernels.py "$MARLIN_ARCH_LIST"

# Build just _moe_C using the same approach vLLM uses
python3 -c "
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup
import glob
import os

# Find all moe source files
moe_sources = glob.glob('csrc/moe/**/*.cu', recursive=True)
moe_sources += glob.glob('csrc/moe/**/*.cpp', recursive=True)

# Also need the marlin quantization utils (dequant.h is in csrc/quantization/marlin/)
include_dirs = [
    os.path.abspath('csrc'),
    os.path.abspath('.deps/vllm-flash-attn-src/csrc/cutlass/tools/util/include'),
    '/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include',
    '/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include',
    os.path.join(torch.utils.cmake_prefix_path, '..', '..', 'include'),
]

print(f'Sources: {moe_sources}')
print(f'Include dirs: {include_dirs}')

ext = CUDAExtension(
    name='vllm._moe_C',
    sources=moe_sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': ['-O3', '-std=c++17', f'-arch={os.environ.get(\"MARLIN_NVCC_ARCH\", \"sm_120\")}',
                 '-U__CUDA_NO_HALF_OPERATORS__',
                 '-U__CUDA_NO_HALF_CONVERSIONS__',
                 '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                 '-U__CUDA_NO_HALF2_OPERATORS__',
                 '-static-global-template-stub=false',
                 '--expt-relaxed-constexpr', '--expt-extended-lambda',
                 '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1'],
    },
)

setup(
    name='vllm_moe_C',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
" build_ext --inplace 2>&1

# Find the built .so
BUILT_SO=$(find . -name '_moe_C*.so' -newer csrc/moe/marlin_moe_wna16/marlin_template.h | head -1)
if [ -z "$BUILT_SO" ]; then
    echo "ERROR: _moe_C.so not found after build"
    exit 1
fi

echo "Built: $BUILT_SO"
echo "Replacing $SITE_PACKAGES/vllm/_moe_C.abi3.so"
cp "$BUILT_SO" "$SITE_PACKAGES/vllm/_moe_C.abi3.so"
echo "=== Done ✅ ==="
