"""Test FP8 tl.dot on sm_121."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_dot_test(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m_offs = tl.arange(0, M)
    n_offs = tl.arange(0, N)
    k_offs = tl.arange(0, K)

    a = tl.load(a_ptr + m_offs[:, None] * K + k_offs[None, :])
    b = tl.load(b_ptr + n_offs[:, None] * K + k_offs[None, :])

    acc = tl.dot(a, tl.trans(b))

    tl.store(c_ptr + m_offs[:, None] * N + n_offs[None, :], acc)


def main():
    M, N, K = 16, 16, 32
    a = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')

    _fp8_dot_test[(1,)](a, b, c, M, N, K)

    ref = a.float() @ b.float().t()
    diff = (c - ref).abs().max().item()
    print(f"FP8 tl.dot on sm_121: max_diff = {diff}")
    print("PASS" if diff < 0.5 else "FAIL")


if __name__ == "__main__":
    main()
