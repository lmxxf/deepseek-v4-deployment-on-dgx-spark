"""Profile with FP8 activation dequant included — matches real vLLM path."""
import time
import torch
import sys
sys.path.insert(0, "/work/Consumer-DeepGEMM")

from consumer_deep_gemm.gemm import _dequant_fp8_block
from consumer_deep_gemm.triton_moe import (
    triton_fused_fp4_matmul_nt, _ensure_e8m0_scales,
    m_grouped_fp8_fp4_gemm_nt_contiguous_triton,
)


def main():
    torch.manual_seed(42)

    M, K, N = 384, 7168, 4096
    K_PACKED = K // 2
    K_SCALE = K // 32
    G = 256

    # FP8 activation (what vLLM actually passes)
    a_fp8 = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    a_scale = torch.full((M, K // 128), 127, device='cuda', dtype=torch.uint8)

    b_packed = torch.randint(0, 256, (G, N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_scale = torch.randint(119, 123, (G, N, K_SCALE), device='cuda', dtype=torch.uint8)
    d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    active_experts = [3, 17, 42, 100, 155, 200]
    m_indices = torch.zeros(M, device='cuda', dtype=torch.int32)
    for i, eid in enumerate(active_experts):
        m_indices[i*64:(i+1)*64] = eid

    # Warmup
    for _ in range(3):
        m_grouped_fp8_fp4_gemm_nt_contiguous_triton(
            (a_fp8, a_scale), (b_packed, b_scale), d, m_indices
        )
    torch.cuda.synchronize()

    N_ITER = 50
    timings = {}

    def timed(name, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_ITER * 1000
        timings[name] = ms

    # A: FP8 dequant alone
    timed("fp8_dequant", lambda: _dequant_fp8_block(a_fp8, a_scale))

    # B: full dispatch (what actually runs)
    timed("full_with_fp8", lambda: m_grouped_fp8_fp4_gemm_nt_contiguous_triton(
        (a_fp8, a_scale), (b_packed, b_scale), d, m_indices
    ))

    # C: full dispatch with pre-dequanted bf16 (skip FP8 dequant)
    a_bf16 = _dequant_fp8_block(a_fp8, a_scale)
    timed("full_bf16_input", lambda: m_grouped_fp8_fp4_gemm_nt_contiguous_triton(
        a_bf16, (b_packed, b_scale), d, m_indices
    ))

    print("=== Real Dispatch Profiling (1 call, ms) ===")
    for name, ms in timings.items():
        print(f"  {name:<25s} {ms:8.3f} ms")
    print()

    fp8_overhead = timings["full_with_fp8"] - timings["full_bf16_input"]
    print(f"  FP8 dequant overhead:   {fp8_overhead:.3f} ms ({fp8_overhead/timings['full_with_fp8']*100:.1f}%)")
    print(f"  Per token (×120):       {timings['full_with_fp8'] * 120:.0f} ms = {1000 / (timings['full_with_fp8'] * 120):.2f} tok/s")
    print(f"  Per token no-fp8 (×120): {timings['full_bf16_input'] * 120:.0f} ms = {1000 / (timings['full_bf16_input'] * 120):.2f} tok/s")

    # Marlin comparison point
    marlin_ms_per_token = 1000 / 14  # jasl = 14 tok/s
    print(f"\n  jasl/Marlin reference:  {marlin_ms_per_token:.0f} ms/token (14 tok/s)")
    print(f"  Our overhead vs Marlin: {timings['full_with_fp8'] * 120 - marlin_ms_per_token:.0f} ms/token")


if __name__ == "__main__":
    main()
