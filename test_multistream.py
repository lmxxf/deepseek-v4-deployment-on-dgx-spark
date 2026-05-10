"""Test: 6 kernels on 6 separate CUDA streams for parallel execution."""
import time
import torch
import triton
import sys
sys.path.insert(0, "/work/Consumer-DeepGEMM")

from consumer_deep_gemm.triton_moe import (
    triton_fused_fp4_matmul_nt,
    _dot_scaled_matmul_kernel,
)


def main():
    torch.manual_seed(42)

    M, K, N = 384, 7168, 4096
    K_PACKED = K // 2
    K_SCALE = K // 32
    G = 256
    n_segs = 6

    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 256, (G, N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_scale = torch.randint(119, 123, (G, N, K_SCALE), device='cuda', dtype=torch.uint8)

    active_experts = [3, 17, 42, 100, 155, 200]
    expert_a = [a_bf16[i*64:(i+1)*64].contiguous() for i in range(n_segs)]
    expert_b = [b_packed[eid] for eid in active_experts]
    expert_s = [b_scale[eid] for eid in active_experts]
    expert_d = [torch.empty(64, N, dtype=torch.bfloat16, device='cuda') for _ in range(n_segs)]

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 64
    K_packed = K // 2
    K_scale = K // 32

    N_WARMUP = 5
    N_ITER = 100

    # Create 6 streams
    streams = [torch.cuda.Stream() for _ in range(n_segs)]

    # Warmup on all streams
    for _ in range(N_WARMUP):
        for i in range(n_segs):
            with torch.cuda.stream(streams[i]):
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

    # === Multi-stream: 6 kernels on 6 streams ===
    start_ms = torch.cuda.Event(enable_timing=True)
    end_ms = torch.cuda.Event(enable_timing=True)

    start_ms.record()
    for _ in range(N_ITER):
        for i in range(n_segs):
            with torch.cuda.stream(streams[i]):
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
        # Sync all streams
        for s in streams:
            torch.cuda.current_stream().wait_stream(s)
    end_ms.record()
    torch.cuda.synchronize()
    multi_gpu_ms = start_ms.elapsed_time(end_ms) / N_ITER

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        for i in range(n_segs):
            with torch.cuda.stream(streams[i]):
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
        for s in streams:
            torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    multi_wall_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # === Single stream (baseline from previous test) ===
    start_ss = torch.cuda.Event(enable_timing=True)
    end_ss = torch.cuda.Event(enable_timing=True)

    start_ss.record()
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
    end_ss.record()
    torch.cuda.synchronize()
    single_gpu_ms = start_ss.elapsed_time(end_ss) / N_ITER

    # === 1x kernel baseline ===
    start_1x = torch.cuda.Event(enable_timing=True)
    end_1x = torch.cuda.Event(enable_timing=True)
    start_1x.record()
    for _ in range(N_ITER):
        M_i = expert_a[0].shape[0]
        grid = (triton.cdiv(M_i, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _dot_scaled_matmul_kernel[grid](
            expert_a[0], expert_a[0].stride(0), expert_a[0].stride(1),
            expert_b[0], expert_b[0].stride(0), expert_b[0].stride(1),
            expert_s[0], expert_s[0].stride(0), expert_s[0].stride(1),
            expert_d[0], expert_d[0].stride(0), expert_d[0].stride(1),
            M_i, N, K, K_packed, K_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    end_1x.record()
    torch.cuda.synchronize()
    one_gpu_ms = start_1x.elapsed_time(end_1x) / N_ITER

    print("=== Multi-Stream vs Single-Stream ===")
    print(f"  {'Method':<30s} {'GPU (ms)':>10s} {'Wall (ms)':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'1x kernel':.<30s} {one_gpu_ms:10.3f}")
    print(f"  {'6x single stream':.<30s} {single_gpu_ms:10.3f}")
    print(f"  {'6x multi stream (6 streams)':.<30s} {multi_gpu_ms:10.3f} {multi_wall_ms:10.3f}")
    print()
    print(f"  Multi/Single GPU ratio: {multi_gpu_ms / single_gpu_ms:.2f}x  (ideal < 1.0 = parallel)")
    print(f"  Multi/1x GPU ratio:     {multi_gpu_ms / one_gpu_ms:.2f}x  (ideal = 1.0 = fully parallel)")
    print()
    print(f"  Per token (×120):")
    print(f"    Single stream: {single_gpu_ms * 120:.0f}ms = {1000 / (single_gpu_ms * 120):.1f} tok/s")
    print(f"    Multi stream:  {multi_gpu_ms * 120:.0f}ms = {1000 / (multi_gpu_ms * 120):.1f} tok/s")
    print(f"    Ideal (1x):    {one_gpu_ms * 120:.0f}ms = {1000 / (one_gpu_ms * 120):.1f} tok/s")


if __name__ == "__main__":
    main()
