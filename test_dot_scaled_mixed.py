"""Test tl.dot_scaled with mixed formats: e4m3 (FP8 activation) x e2m1 (FP4 weight)."""
import time
import torch
import triton
import triton.language as tl


@triton.jit
def _dot_scaled_fp8xfp4_kernel(
    # A: [M, K] float8_e4m3 activation (stored as uint8)
    a_ptr, a_stride_m, a_stride_k,
    a_scale_ptr, as_stride_m, as_stride_k,
    # B: [K_PACKED, N] uint8 packed e2m1 weight
    b_ptr, b_stride_k, b_stride_n,
    b_scale_ptr, bs_stride_n, bs_stride_k,
    # D: [M, N] bf16 output
    d_ptr, d_stride_m, d_stride_n,
    M, N, K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # FP8 e4m3: 1 byte per value, scale per 32 values
    # FP4 e2m1: 0.5 byte per value (packed), scale per 32 values = 16 packed bytes
    # Process 32 K values per chunk = 32 FP8 bytes + 16 FP4 packed bytes
    CHUNK_K: tl.constexpr = 32
    CHUNK_K_PACKED: tl.constexpr = 16

    for k_start in range(0, K, CHUNK_K):
        k_offs = k_start + tl.arange(0, CHUNK_K)
        kp_start = k_start // 2
        kp_offs = kp_start + tl.arange(0, CHUNK_K_PACKED)

        # Load A: [BLOCK_M, 32] uint8 (FP8 e4m3)
        a_chunk = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_offs[None, :] * a_stride_k,
            mask=m_mask[:, None] & (k_offs[None, :] < K), other=0
        )

        # Load B: [16, BLOCK_N] uint8 (packed FP4 e2m1), K-major
        b_chunk = tl.load(
            b_ptr + kp_offs[:, None] * b_stride_k + n_offs[None, :] * b_stride_n,
            mask=(kp_offs[:, None] < K_PACKED) & n_mask[None, :], other=0
        )

        # Scales
        scale_idx = k_start // 32
        a_sc = tl.load(
            a_scale_ptr + m_offs[:, None] * as_stride_m + scale_idx,
            mask=m_mask[:, None], other=127
        )
        # a_sc needs shape [BLOCK_M, 1] for dot_scaled
        b_sc = tl.load(
            b_scale_ptr + n_offs[:, None] * bs_stride_n + scale_idx,
            mask=n_mask[:, None], other=127
        )
        # b_sc needs shape [BLOCK_N, 1]

        acc += tl.dot_scaled(a_chunk, a_sc, "e4m3", b_chunk, b_sc, "e2m1")

    tl.store(
        d_ptr + m_offs[:, None] * d_stride_m + n_offs[None, :] * d_stride_n,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :]
    )


def main():
    M, K, N = 16, 64, 16
    K_PACKED = K // 2
    K_SCALE = K // 32

    # FP8 activation
    a_fp8 = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
    a_scale = torch.full((M, K_SCALE), 127, device='cuda', dtype=torch.uint8)

    # FP4 weight [N, K_PACKED] -> transpose to [K_PACKED, N]
    b_packed_nk = torch.randint(0, 256, (N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_packed_kn = b_packed_nk.t().contiguous()
    b_scale = torch.full((N, K_SCALE), 127, device='cuda', dtype=torch.uint8)

    d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    BLOCK_M = 16
    BLOCK_N = 16
    grid = (1, 1)

    try:
        _dot_scaled_fp8xfp4_kernel[grid](
            a_fp8, a_fp8.stride(0), a_fp8.stride(1),
            a_scale, a_scale.stride(0), a_scale.stride(1),
            b_packed_kn, b_packed_kn.stride(0), b_packed_kn.stride(1),
            b_scale, b_scale.stride(0), b_scale.stride(1),
            d, d.stride(0), d.stride(1),
            M, N, K, K_PACKED, K_SCALE,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        torch.cuda.synchronize()
        print(f"dot_scaled e4m3 x e2m1: compiled and ran ✅")
        print(f"d[:4,:4] =\n{d[:4,:4]}")

        # Speed test at real sizes
        for M, K, N, label in [(384, 7168, 4096, "FC1"), (384, 2048, 7168, "FC2")]:
            K_PACKED = K // 2
            K_SCALE = K // 32
            a_fp8 = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
            a_scale = torch.full((M, K_SCALE), 127, device='cuda', dtype=torch.uint8)
            b_nk = torch.randint(0, 256, (N, K_PACKED), device='cuda', dtype=torch.uint8)
            b_kn = b_nk.t().contiguous()
            b_scale = torch.randint(119, 123, (N, K_SCALE), device='cuda', dtype=torch.uint8)
            d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

            BLOCK_M = min(64, triton.next_power_of_2(M))
            BLOCK_N = 32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            # warmup
            for _ in range(3):
                _dot_scaled_fp8xfp4_kernel[grid](
                    a_fp8, a_fp8.stride(0), a_fp8.stride(1),
                    a_scale, a_scale.stride(0), a_scale.stride(1),
                    b_kn, b_kn.stride(0), b_kn.stride(1),
                    b_scale, b_scale.stride(0), b_scale.stride(1),
                    d, d.stride(0), d.stride(1),
                    M, N, K, K_PACKED, K_SCALE,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                )
            torch.cuda.synchronize()

            N_ITER = 20
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                _dot_scaled_fp8xfp4_kernel[grid](
                    a_fp8, a_fp8.stride(0), a_fp8.stride(1),
                    a_scale, a_scale.stride(0), a_scale.stride(1),
                    b_kn, b_kn.stride(0), b_kn.stride(1),
                    b_scale, b_scale.stride(0), b_scale.stride(1),
                    d, d.stride(0), d.stride(1),
                    M, N, K, K_PACKED, K_SCALE,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                )
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / N_ITER * 1000
            print(f"{label} [{M}x{K}x{N}]: dot_scaled e4m3×e2m1 = {ms:.2f} ms")

    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
