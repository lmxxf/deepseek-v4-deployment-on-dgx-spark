#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=${VLLM_SM120_SRC:-"$SCRIPT_DIR/spark-vllm-docker/vllm-sm120/vllm-sm120"}
cd "$SRC"

python3 csrc/moe/marlin_moe_wna16/generate_kernels.py 8.0,12.0,12.1

python3 - <<'PY'
import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    "csrc/moe/torch_bindings.cpp",
    "csrc/moe/moe_align_sum_kernels.cu",
    "csrc/moe/topk_softmax_kernels.cu",
    "csrc/moe/moe_wna16.cu",
    "csrc/moe/grouped_topk_kernels.cu",
    "csrc/moe/router_gemm.cu",
    "csrc/moe/topk_softplus_sqrt_kernels.cu",
    "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu",
    "csrc/moe/moe_permute_unpermute_op.cu",
]
sources += sorted(glob.glob("csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu"))
sources += sorted(glob.glob("csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu"))
sources += ["csrc/moe/marlin_moe_wna16/ops.cu"]

include_dirs = [
    os.path.abspath("csrc"),
    os.path.abspath("/work/spark-vllm-docker/vllm-sm120/.deps/vllm-flash-attn-src/csrc/cutlass/tools/util/include"),
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include",
    os.path.join(torch.utils.cmake_prefix_path, "..", "..", "include"),
]

print("source_count", len(sources))
ext = CUDAExtension(
    name="vllm._moe_C",
    sources=sources,
    include_dirs=include_dirs,
    py_limited_api=True,
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-arch=sm_120",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-static-global-template-stub=false",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
    },
)

setup(
    name="vllm_exact_moe_C",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext", "--inplace"],
)
PY

python3 - <<'PY'
import glob
import os
import subprocess

import torch

src_root = os.getcwd()
build_temp = "build/temp.linux-aarch64-cpython-312"
build_lib = "build/lib.linux-aarch64-cpython-312/vllm/_moe_C.abi3.so"

include_dirs = [
    os.path.abspath("csrc"),
    os.path.abspath("/work/spark-vllm-docker/vllm-sm120/.deps/vllm-flash-attn-src/csrc/cutlass/tools/util/include"),
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include",
    os.path.join(torch.utils.cmake_prefix_path, "..", "..", "include"),
    "/usr/local/lib/python3.12/dist-packages/torch/include",
    "/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include",
    "/usr/local/cuda/include",
    "/usr/include/python3.12",
]

common = [
    "/usr/local/cuda/bin/nvcc",
    "-c",
    "-O3",
    "-std=c++17",
    "-arch=sm_80",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-static-global-template-stub=false",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-DPy_LIMITED_API=0x030A0000",
    "-DTORCH_EXTENSION_NAME=_moe_C",
    "--compiler-options",
    "-fPIC",
]
for inc in include_dirs:
    common.append("-I" + inc)

sources = [
    "csrc/moe/torch_bindings.cpp",
    "csrc/moe/moe_align_sum_kernels.cu",
    "csrc/moe/topk_softmax_kernels.cu",
    "csrc/moe/moe_wna16.cu",
    "csrc/moe/grouped_topk_kernels.cu",
    "csrc/moe/router_gemm.cu",
    "csrc/moe/topk_softplus_sqrt_kernels.cu",
    "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu",
    "csrc/moe/moe_permute_unpermute_op.cu",
]
sources += sorted(glob.glob("csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu"))
sources += sorted(glob.glob("csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu"))
sources += ["csrc/moe/marlin_moe_wna16/ops.cu"]

sm80_sources = sorted(glob.glob("csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu"))
sm80_sources += ["csrc/moe/marlin_moe_wna16/ops.cu"]
print("recompile_sm80_count", len(sm80_sources))
for src in sm80_sources:
    obj = os.path.join(build_temp, src[:-3] + ".o")
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    cmd = common + [src, "-o", obj]
    print("SM80", src)
    subprocess.check_call(cmd)

objects = []
for src in sources:
    if src.endswith(".cpp"):
        obj = os.path.join(build_temp, src[:-4] + ".o")
    else:
        obj = os.path.join(build_temp, src[:-3] + ".o")
    objects.append(obj)
print("relink_object_count", len(objects))
link = [
    "aarch64-linux-gnu-g++",
    "-fno-strict-overflow",
    "-Wsign-compare",
    "-DNDEBUG",
    "-g",
    "-O2",
    "-Wall",
    "-shared",
    "-Wl,-O1",
    "-Wl,-Bsymbolic-functions",
    *objects,
    "-L/usr/local/lib/python3.12/dist-packages/torch/lib",
    "-L/usr/local/cuda/lib64",
    "-L/usr/lib/aarch64-linux-gnu",
    "-lc10",
    "-ltorch",
    "-ltorch_cpu",
    "-lcudart",
    "-lc10_cuda",
    "-ltorch_cuda",
    "-o",
    build_lib,
]
subprocess.check_call(link)
subprocess.check_call(["cp", build_lib, "vllm/_moe_C.abi3.so"])
PY
