"""Compare loop kernel: pure GPU kernel time vs wall time.

Uses torch.cuda.Event to measure only kernel execution, excluding Python overhead.
"""
import time
import torch
import triton
import sys
sys.path.insert(0, "/work/Consumer-DeepGEMM")

from consumer_deep_gemm.triton_moe import (
    triton_fused_fp4_matmul_nt,
    _dot_scaled_matmul_kernel,
    _ensure_e8m0_scales,
)


def main():
    torch.manual_seed(42)

    M, K, N = 384, 7168, 4096
    K_PACKED = K // 2
    K_SCALE = K // 32
    G = 256

    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 256, (G, N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_scale = torch.randint(119, 123, (G, N, K_SCALE), device='cuda', dtype=torch.uint8)

    active_experts = [3, 17, 42, 100, 155, 200]
    n_segs = len(active_experts)

    # Pre-slice everything
    expert_a = [a_bf16[i*64:(i+1)*64].contiguous() for i in range(n_segs)]
    expert_b = [b_packed[eid] for eid in active_experts]
    expert_s = [b_scale[eid] for eid in active_experts]
    expert_d = [torch.empty(64, N, dtype=torch.bfloat16, device='cuda') for _ in range(n_segs)]

    N_WARMUP = 5
    N_ITER = 100

    # === Method 1: Loop with triton_fused_fp4_matmul_nt (wrapper) ===
    for _ in range(N_WARMUP):
        for i in range(n_segs):
            triton_fused_fp4_matmul_nt(expert_a[i], expert_b[i], expert_s[i], expert_d[i])
    torch.cuda.synchronize()

    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start1.record()
    for _ in range(N_ITER):
        for i in range(n_segs):
            triton_fused_fp4_matmul_nt(expert_a[i], expert_b[i], expert_s[i], expert_d[i])
    end1.record()
    torch.cuda.synchronize()
    loop_gpu_ms = start1.elapsed_time(end1) / N_ITER

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        for i in range(n_segs):
            triton_fused_fp4_matmul_nt(expert_a[i], expert_b[i], expert_s[i], expert_d[i])
    torch.cuda.synchronize()
    loop_wall_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # === Method 2: Loop calling _dot_scaled_matmul_kernel directly (skip wrapper) ===
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 64
    K_packed = K // 2
    K_scale = K // 32

    for _ in range(N_WARMUP):
        for i in range(n_segs):
            M_i = expert_a[i].shape[0]
            grid = (triton.cdiv(M_i, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _dot_scaled_matmul_kernel[grid](
                expert_a[i], expert_a[i].stride(0), expert_a[i].stride(1),
                expert_b[i], expert_b[i].stride(0), expert_b[i].stride(1),
                expert_s[i], expert_s[i].stride(0), expert_s[i].stride(1),
                expert_d[i], expert_d[i].stride(0), expert_d[i].stride(1),
                M_i, N, K, K_packed, K_scale,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
    torch.cuda.synchronize()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(N_ITER):
        for i in range(n_segs):
            M_i = expert_a[i].shape[0]
            grid = (triton.cdiv(M_i, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _dot_scaled_matmul_kernel[grid](
                expert_a[i], expert_a[i].stride(0), expert_a[i].stride(1),
                expert_b[i], expert_b[i].stride(0), expert_b[i].stride(1),
                expert_s[i], expert_s[i].stride(0), expert_s[i].stride(1),
                expert_d[i], expert_d[i].stride(0), expert_d[i].stride(1),
                M_i, N, K, K_packed, K_scale,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
    end2.record()
    torch.cuda.synchronize()
    direct_gpu_ms = start2.elapsed_time(end2) / N_ITER

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        for i in range(n_segs):
            M_i = expert_a[i].shape[0]
            grid = (triton.cdiv(M_i, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _dot_scaled_matmul_kernel[grid](
                expert_a[i], expert_a[i].stride(0), expert_a[i].stride(1),
                expert_b[i], expert_b[i].stride(0), expert_b[i].stride(1),
                expert_s[i], expert_s[i].stride(0), expert_s[i].stride(1),
                expert_d[i], expert_d[i].stride(0), expert_d[i].stride(1),
                M_i, N, K, K_packed, K_scale,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
    torch.cuda.synchronize()
    direct_wall_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # === Method 3: Single kernel (1 expert) for baseline ===
    for _ in range(N_WARMUP):
        triton_fused_fp4_matmul_nt(expert_a[0], expert_b[0], expert_s[0], expert_d[0])
    torch.cuda.synchronize()

    start3 = torch.cuda.Event(enable_timing=True)
    end3 = torch.cuda.Event(enable_timing=True)
    start3.record()
    for _ in range(N_ITER):
        triton_fused_fp4_matmul_nt(expert_a[0], expert_b[0], expert_s[0], expert_d[0])
    end3.record()
    torch.cuda.synchronize()
    single_gpu_ms = start3.elapsed_time(end3) / N_ITER

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        triton_fused_fp4_matmul_nt(expert_a[0], expert_b[0], expert_s[0], expert_d[0])
    torch.cuda.synchronize()
    single_wall_ms = (time.perf_counter() - t0) / N_ITER * 1000

    print("=== 6x Kernel Launch Analysis ===")
    print(f"  {'Method':<30s} {'GPU (ms)':>10s} {'Wall (ms)':>10s} {'Py OH (ms)':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'1x kernel (baseline)':.<30s} {single_gpu_ms:10.3f} {single_wall_ms:10.3f} {single_wall_ms - single_gpu_ms:10.3f}")
    print(f"  {'6x loop (wrapper)':.<30s} {loop_gpu_ms:10.3f} {loop_wall_ms:10.3f} {loop_wall_ms - loop_gpu_ms:10.3f}")
    print(f"  {'6x loop (direct kernel)':.<30s} {direct_gpu_ms:10.3f} {direct_wall_ms:10.3f} {direct_wall_ms - direct_gpu_ms:10.3f}")
    print()
    print(f"  6x GPU / 1x GPU = {loop_gpu_ms / single_gpu_ms:.2f}x  (ideal=6.0, >6 means serialized)")
    print(f"  6x Wall / 1x Wall = {loop_wall_ms / single_wall_ms:.2f}x")
    print(f"  Direct vs Wrapper overhead: {loop_wall_ms - direct_wall_ms:.3f} ms")
    print()
    print(f"  Per token (×120):")
    print(f"    Loop wrapper:  GPU {loop_gpu_ms * 120:.0f}ms  Wall {loop_wall_ms * 120:.0f}ms  = {1000 / (loop_wall_ms * 120):.1f} tok/s")
    print(f"    Loop direct:   GPU {direct_gpu_ms * 120:.0f}ms  Wall {direct_wall_ms * 120:.0f}ms  = {1000 / (direct_wall_ms * 120):.1f} tok/s")
    print(f"    Ideal (1x×120):GPU {single_gpu_ms * 120:.0f}ms  Wall {single_wall_ms * 120:.0f}ms  = {1000 / (single_wall_ms * 120):.1f} tok/s")


if __name__ == "__main__":
    main()
