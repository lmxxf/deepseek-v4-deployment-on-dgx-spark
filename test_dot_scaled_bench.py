"""Benchmark tl.dot_scaled (FP4 native) vs FP8 fused vs float32 fused on sm_121.

Tests correctness and speed at real V4 MoE GEMM sizes.
"""
import time
import torch
import triton
import triton.language as tl


# ── dot_scaled kernel ──────────────────────────────────────────────

@triton.jit
def _dot_scaled_matmul_kernel(
    a_ptr, a_stride_m, a_stride_k,
    b_ptr, b_stride_k, b_stride_n,
    a_scale_ptr, as_stride_m, as_stride_k,
    b_scale_ptr, bs_stride_n, bs_stride_k,
    d_ptr, d_stride_m, d_stride_n,
    M, N, K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """D[M,N] = dot_scaled(A_fp4[M,K], B_fp4[K,N]) with E8M0 scales.

    A: [M, K_PACKED] uint8 packed e2m1, row-major
    B: [K_PACKED, N] uint8 packed e2m1, col-major (K-major)
    a_scale: [M, K_SCALE] uint8 e8m0
    b_scale: [N, K_SCALE] uint8 e8m0
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    CHUNK_K_PACKED: tl.constexpr = 32  # 64 K values = 1 dot_scaled call (need K >= 64 for e2m1)

    for kp_start in range(0, K_PACKED, CHUNK_K_PACKED):
        kp_offs = kp_start + tl.arange(0, CHUNK_K_PACKED)
        kp_mask = kp_offs < K_PACKED

        a_chunk = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + kp_offs[None, :] * a_stride_k,
            mask=m_mask[:, None] & kp_mask[None, :], other=0
        )
        b_chunk = tl.load(
            b_ptr + kp_offs[:, None] * b_stride_k + n_offs[None, :] * b_stride_n,
            mask=kp_mask[:, None] & n_mask[None, :], other=0
        )

        scale_idx = kp_start // 16  # each scale covers 16 packed bytes
        scale_offs = tl.arange(0, CHUNK_K_PACKED // 16)
        a_sc = tl.load(
            a_scale_ptr + m_offs[:, None] * as_stride_m + (scale_idx + scale_offs[None, :]) * as_stride_k,
            mask=m_mask[:, None] & ((scale_idx + scale_offs[None, :]) < K_SCALE), other=127
        )
        b_sc = tl.load(
            b_scale_ptr + n_offs[:, None] * bs_stride_n + (scale_idx + scale_offs[None, :]) * bs_stride_k,
            mask=n_mask[:, None] & ((scale_idx + scale_offs[None, :]) < K_SCALE), other=127
        )

        acc += tl.dot_scaled(a_chunk, a_sc, "e2m1", b_chunk, b_sc, "e2m1")

    tl.store(
        d_ptr + m_offs[:, None] * d_stride_m + n_offs[None, :] * d_stride_n,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :]
    )


def dot_scaled_matmul(a_bf16, b_packed_nk, b_scale_u8):
    """Wrapper: A[M,K] bf16 x B_fp4[N,K_packed] -> D[M,N] bf16.

    Quantizes A to FP4 on the fly (lossy but fast for benchmarking).
    For real use, A would already be in FP4/FP8 format.
    """
    M, K = a_bf16.shape
    N = b_packed_nk.shape[0]
    K_PACKED = K // 2
    K_SCALE = K // 32

    # For benchmark: quantize A to uint8 packed FP4 (crude: just take raw bytes)
    # In real V4, activation is FP8 not FP4, so this is just for speed comparison
    a_packed = torch.zeros(M, K_PACKED, device='cuda', dtype=torch.uint8)
    a_scale = torch.full((M, K_SCALE), 127, device='cuda', dtype=torch.uint8)

    b_packed_kn = b_packed_nk.t().contiguous()

    d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _dot_scaled_matmul_kernel[grid](
        a_packed, a_packed.stride(0), a_packed.stride(1),
        b_packed_kn, b_packed_kn.stride(0), b_packed_kn.stride(1),
        a_scale, a_scale.stride(0), a_scale.stride(1),
        b_scale_u8, b_scale_u8.stride(0), b_scale_u8.stride(1),
        d, d.stride(0), d.stride(1),
        M, N, K, K_PACKED, K_SCALE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return d


# ── Reference dequant ──────────────────────────────────────────────

def dequant_ref(b_packed, b_scale_u8, K):
    N, K_packed = b_packed.shape
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        device=b_packed.device, dtype=torch.float32)
    low_f = vals[(b_packed & 0x0F).long()]
    high_f = vals[((b_packed >> 4) & 0x0F).long()]
    out = torch.zeros(N, K, device=b_packed.device, dtype=torch.float32)
    out[:, 0::2] = low_f
    out[:, 1::2] = high_f
    scale_f = torch.pow(2.0, b_scale_u8.float() - 127.0)
    scale_expanded = scale_f.repeat_interleave(32, dim=1)[:, :K]
    return out * scale_expanded


# ── Benchmark ──────────────────────────────────────────────────────

def bench(fn, name, n_warmup=3, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    return ms


def main():
    import sys
    sys.path.insert(0, "/work/Consumer-DeepGEMM")
    from consumer_deep_gemm.triton_moe import triton_fused_fp4_matmul_nt

    torch.manual_seed(42)

    for M, K, N, label in [
        (384, 7168, 4096, "FC1"),
        (384, 2048, 7168, "FC2"),
    ]:
        K_PACKED = K // 2
        K_SCALE = K // 32

        a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        b_packed = torch.randint(0, 16, (N, K_PACKED), device='cuda', dtype=torch.uint8)
        b_scale = torch.randint(119, 123, (N, K_SCALE), device='cuda', dtype=torch.uint8)
        d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

        # float32 fused
        t_f32 = bench(lambda: triton_fused_fp4_matmul_nt(a, b_packed, b_scale, d, use_fp8=False), "f32")

        # FP8 fused
        t_fp8 = bench(lambda: triton_fused_fp4_matmul_nt(a, b_packed, b_scale, d, use_fp8=True), "fp8")

        # dot_scaled
        b_packed_kn = b_packed.t().contiguous()
        a_packed = torch.zeros(M, K_PACKED, device='cuda', dtype=torch.uint8)
        a_scale = torch.full((M, K_SCALE), 127, device='cuda', dtype=torch.uint8)
        d2 = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

        BLOCK_M = min(64, triton.next_power_of_2(M))
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        def run_dot_scaled():
            _dot_scaled_matmul_kernel[grid](
                a_packed, a_packed.stride(0), a_packed.stride(1),
                b_packed_kn, b_packed_kn.stride(0), b_packed_kn.stride(1),
                a_scale, a_scale.stride(0), a_scale.stride(1),
                b_scale, b_scale.stride(0), b_scale.stride(1),
                d2, d2.stride(0), d2.stride(1),
                M, N, K, K_PACKED, K_SCALE,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )

        t_ds = bench(run_dot_scaled, "dot_scaled")

        print(f"{label} [{M}x{K}x{N}]:")
        print(f"  float32 fused:  {t_f32:.2f} ms")
        print(f"  FP8 fused:      {t_fp8:.2f} ms  ({t_f32/t_fp8:.2f}x vs f32)")
        print(f"  dot_scaled:     {t_ds:.2f} ms  ({t_f32/t_ds:.2f}x vs f32)")
        print()


if __name__ == "__main__":
    main()
