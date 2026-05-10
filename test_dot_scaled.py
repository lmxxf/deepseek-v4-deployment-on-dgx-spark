"""Test tl.dot_scaled with e2m1 (FP4) on sm_121."""
import torch
import triton
import triton.language as tl


@triton.jit
def _dot_scaled_test(
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    K_PACKED: tl.constexpr, K_SCALE: tl.constexpr,
):
    # A: [M, K_PACKED] uint8 (packed e2m1), scale [M, K_SCALE] uint8 e8m0
    # B: [K_PACKED, N] uint8 (packed e2m1), scale [N, K_SCALE] uint8 e8m0
    # note: rhs is [K, N] not [N, K] for dot_scaled
    m_offs = tl.arange(0, M)
    n_offs = tl.arange(0, N)
    k_packed_offs = tl.arange(0, K_PACKED)
    scale_offs = tl.arange(0, K_SCALE)

    a = tl.load(a_ptr + m_offs[:, None] * K_PACKED + k_packed_offs[None, :])
    a_scale = tl.load(a_scale_ptr + m_offs[:, None] * K_SCALE + scale_offs[None, :])

    b = tl.load(b_ptr + k_packed_offs[:, None] * N + n_offs[None, :])
    b_scale = tl.load(b_scale_ptr + n_offs[:, None] * K_SCALE + scale_offs[None, :])

    acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1")

    tl.store(c_ptr + m_offs[:, None] * N + n_offs[None, :], acc)


def dequant_e2m1(packed_uint8, scale_uint8, rows, K):
    """Reference dequant for verification."""
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        device=packed_uint8.device, dtype=torch.float32)
    low = vals[(packed_uint8 & 0x0F).long()]
    high = vals[((packed_uint8 >> 4) & 0x0F).long()]
    out = torch.zeros(rows, K, device=packed_uint8.device, dtype=torch.float32)
    out[:, 0::2] = low
    out[:, 1::2] = high
    scale_f = torch.pow(2.0, scale_uint8.float() - 127.0)
    scale_expanded = scale_f.repeat_interleave(32, dim=1)[:, :K]
    return out * scale_expanded


def main():
    print(f"Triton version: {triton.__version__}")

    M, N, K = 16, 16, 64
    K_PACKED = K // 2
    K_SCALE = K // 32

    a_packed = torch.randint(0, 256, (M, K_PACKED), device='cuda', dtype=torch.uint8)
    a_scale = torch.full((M, K_SCALE), 127, device='cuda', dtype=torch.uint8)

    # B for dot_scaled: [K_PACKED, N] (K-major)
    b_packed_nk = torch.randint(0, 256, (N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_packed_kn = b_packed_nk.t().contiguous()  # [K_PACKED, N]
    b_scale = torch.full((N, K_SCALE), 127, device='cuda', dtype=torch.uint8)

    c = torch.empty(M, N, device='cuda', dtype=torch.float32)

    try:
        _dot_scaled_test[(1,)](
            a_packed, a_scale,
            b_packed_kn, b_scale,
            c,
            M, N, K, K_PACKED, K_SCALE,
        )
        torch.cuda.synchronize()

        # Reference
        a_deq = dequant_e2m1(a_packed, a_scale, M, K)
        b_deq = dequant_e2m1(b_packed_nk, b_scale, N, K)
        ref = a_deq @ b_deq.t()

        diff = (c - ref).abs().max().item()
        rel = diff / (ref.abs().max().item() + 1e-12)
        print(f"dot_scaled e2m1 result: max_diff={diff:.4f} rel_err={rel:.4f}")
        print(f"{'PASS ✅' if rel < 0.05 else 'FAIL ❌'}")
    except Exception as e:
        print(f"dot_scaled failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
