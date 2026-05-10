"""Profile CDG dispatch with FP8 round-trip removed.

Measures time of each step in the current optimized path.
"""
import time
import torch
import sys
sys.path.insert(0, "/work/Consumer-DeepGEMM")

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

    # BF16 activation (no FP8 quantize!)
    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)

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
            a_bf16, (b_packed, b_scale), d, m_indices
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

    # Full dispatch (BF16 input, no FP8)
    timed("full_bf16", lambda: m_grouped_fp8_fp4_gemm_nt_contiguous_triton(
        a_bf16, (b_packed, b_scale), d, m_indices
    ))

    # For comparison: with FP8 input (old path)
    from consumer_deep_gemm.gemm import _dequant_fp8_block
    a_fp8 = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    a_scale_fp8 = torch.full((M, K // 128), 127, device='cuda', dtype=torch.uint8)
    timed("full_fp8_input", lambda: m_grouped_fp8_fp4_gemm_nt_contiguous_triton(
        (a_fp8, a_scale_fp8), (b_packed, b_scale), d, m_indices
    ))

    # Break down BF16 path
    b_scale_u8 = _ensure_e8m0_scales(b_scale)

    # Step: argsort + segment detection
    def step_sort():
        si = torch.argsort(m_indices)
        sid = m_indices[si]
        vm = sid >= 0
        nv = vm.sum().item()
        return si, sid, vm, nv
    timed("sort+segment", lambda: step_sort())

    si, sid, vm, nv = step_sort()
    vsi = si[vm]
    veid = sid[vm]
    ch = torch.zeros(nv, dtype=torch.bool, device='cuda')
    ch[0] = True
    if nv > 1:
        ch[1:] = veid[1:] != veid[:-1]
    ss = ch.nonzero(as_tuple=False).flatten()
    ssc = ss.cpu()
    eias = veid[ss].cpu()
    n_segs = ssc.numel()

    # Step: index_select
    timed("index_select", lambda: a_bf16.index_select(0, vsi))

    a_sorted = a_bf16.index_select(0, vsi)

    # Step: 6x kernel launches
    d_sorted = torch.empty(nv, N, dtype=d.dtype, device=d.device)
    launches = []
    for i in range(n_segs):
        gid = eias[i].item()
        start = ssc[i].item()
        end = ssc[i+1].item() if i+1 < n_segs else nv
        launches.append((gid, start, end))

    def step_kernels():
        for gid, start, end in launches:
            gs = b_scale_u8[gid]
            triton_fused_fp4_matmul_nt(a_sorted[start:end], b_packed[gid], gs, d_sorted[start:end])
    timed("6x_kernels", step_kernels)

    # Step: single kernel
    gid0, s0, e0 = launches[0]
    def step_1kernel():
        triton_fused_fp4_matmul_nt(a_sorted[s0:e0], b_packed[gid0], b_scale_u8[gid0], d_sorted[s0:e0])
    timed("1x_kernel", step_1kernel)

    # Step: index_copy
    timed("index_copy", lambda: d.index_copy_(0, vsi, d_sorted))

    # Print
    full = timings["full_bf16"]
    print("=== CDG Dispatch Profiling (BF16 input, no FP8) ===")
    print(f"  {'Step':<20s} {'ms':>8s} {'%':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*6}")
    for name, ms in timings.items():
        if name.startswith("full"):
            continue
        pct = ms / full * 100
        print(f"  {name:<20s} {ms:8.3f} {pct:5.1f}%")
    print(f"  {'-'*20} {'-'*8} {'-'*6}")
    print(f"  {'full_bf16':<20s} {full:8.3f} 100.0%")
    print(f"  {'full_fp8_input':<20s} {timings['full_fp8_input']:8.3f}  (old path for comparison)")
    print()
    print(f"  Per token (×120): {full * 120:.0f} ms = {1000 / (full * 120):.2f} tok/s")
    print(f"  Old FP8 path:     {timings['full_fp8_input'] * 120:.0f} ms = {1000 / (timings['full_fp8_input'] * 120):.2f} tok/s")
    print(f"  Kernel only:      {timings['6x_kernels'] * 120:.0f} ms")
    print(f"  Overhead:         {(full - timings['6x_kernels']) * 120:.0f} ms")


if __name__ == "__main__":
    main()
