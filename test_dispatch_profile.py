"""Profile the dispatch overhead in m_grouped_fp8_fp4_gemm_nt_contiguous_triton.

Simulates real V4 MoE decode: M=384 (6 experts × 64 rows), K=7168, N=4096.
Measures each step's time to find the bottleneck.
"""
import time
import torch
import sys
sys.path.insert(0, "/work/Consumer-DeepGEMM")


def profile_dispatch():
    torch.manual_seed(42)

    # Real V4 decode params
    M = 384  # 6 experts × 64 rows
    K = 7168
    N = 4096
    K_PACKED = K // 2
    K_SCALE = K // 32
    G = 256  # total experts

    # Simulate inputs
    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 256, (G, N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_scale = torch.randint(119, 123, (G, N, K_SCALE), device='cuda', dtype=torch.uint8)
    d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    # Simulate m_indices: 6 active experts, 64 rows each
    active_experts = [3, 17, 42, 100, 155, 200]
    m_indices = torch.zeros(M, device='cuda', dtype=torch.int32)
    for i, eid in enumerate(active_experts):
        m_indices[i*64:(i+1)*64] = eid

    from consumer_deep_gemm.triton_moe import (
        triton_fused_fp4_matmul_nt, _ensure_e8m0_scales
    )

    # Warmup
    b_scale_u8 = _ensure_e8m0_scales(b_scale)
    for _ in range(3):
        # Full dispatch
        sort_indices = torch.argsort(m_indices)
        sorted_indices = m_indices[sort_indices]
        valid_mask = sorted_indices >= 0
        n_valid = valid_mask.sum().item()
        valid_sort_indices = sort_indices[valid_mask]
        valid_expert_ids = sorted_indices[valid_mask]
        changes = torch.zeros(n_valid, dtype=torch.bool, device='cuda')
        changes[0] = True
        if n_valid > 1:
            changes[1:] = valid_expert_ids[1:] != valid_expert_ids[:-1]
        seg_starts = changes.nonzero(as_tuple=False).flatten()
        seg_starts_cpu = seg_starts.cpu()
        expert_ids_at_starts = valid_expert_ids[seg_starts].cpu()
        a_sorted = a_bf16.index_select(0, valid_sort_indices)
        d_sorted = torch.empty(n_valid, N, dtype=d.dtype, device=d.device)
        n_segs = seg_starts_cpu.numel()
        for i in range(n_segs):
            gid = expert_ids_at_starts[i].item()
            start = seg_starts_cpu[i].item()
            end = seg_starts_cpu[i+1].item() if i+1 < n_segs else n_valid
            triton_fused_fp4_matmul_nt(a_sorted[start:end], b_packed[gid], b_scale_u8[gid], d_sorted[start:end])
        d.index_copy_(0, valid_sort_indices, d_sorted)
    torch.cuda.synchronize()

    # Now profile each step
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

    # Step 1: argsort
    timed("1_argsort", lambda: torch.argsort(m_indices))

    # Step 2: valid_mask + n_valid
    sort_indices = torch.argsort(m_indices)
    sorted_indices = m_indices[sort_indices]

    def step2():
        vm = sorted_indices >= 0
        return vm.sum().item()
    timed("2_valid_mask+item", step2)

    # Step 3: extract valid + diff + nonzero
    valid_mask = sorted_indices >= 0
    n_valid = valid_mask.sum().item()
    valid_sort_indices = sort_indices[valid_mask]
    valid_expert_ids = sorted_indices[valid_mask]

    def step3():
        ch = torch.zeros(n_valid, dtype=torch.bool, device='cuda')
        ch[0] = True
        if n_valid > 1:
            ch[1:] = valid_expert_ids[1:] != valid_expert_ids[:-1]
        return ch.nonzero(as_tuple=False).flatten()
    timed("3_diff+nonzero", step3)

    # Step 4: seg_starts.cpu() + expert_ids.cpu()
    seg_starts = step3()
    def step4():
        return seg_starts.cpu(), valid_expert_ids[seg_starts].cpu()
    timed("4_cpu_transfer", step4)

    # Step 5: index_select (gather all valid rows)
    timed("5_index_select", lambda: a_bf16.index_select(0, valid_sort_indices))

    # Step 6: kernel launches (6 experts)
    a_sorted = a_bf16.index_select(0, valid_sort_indices)
    seg_starts_cpu = seg_starts.cpu()
    expert_ids_at_starts = valid_expert_ids[seg_starts].cpu()
    n_segs = seg_starts_cpu.numel()
    d_sorted = torch.empty(n_valid, N, dtype=d.dtype, device=d.device)

    def step6():
        for i in range(n_segs):
            gid = expert_ids_at_starts[i].item()
            start = seg_starts_cpu[i].item()
            end = seg_starts_cpu[i+1].item() if i+1 < n_segs else n_valid
            triton_fused_fp4_matmul_nt(a_sorted[start:end], b_packed[gid], b_scale_u8[gid], d_sorted[start:end])
    timed("6_kernels_x6", step6)

    # Step 6b: single kernel (largest expert) for comparison
    def step6b():
        triton_fused_fp4_matmul_nt(a_sorted[:64], b_packed[3], b_scale_u8[3], d_sorted[:64])
    timed("6b_single_kernel", step6b)

    # Step 7: index_copy (scatter back)
    timed("7_index_copy", lambda: d.index_copy_(0, valid_sort_indices, d_sorted))

    # Step 8: full dispatch (end to end for one call)
    def full():
        si = torch.argsort(m_indices)
        sid = m_indices[si]
        vm = sid >= 0
        nv = vm.sum().item()
        vsi = si[vm]
        veid = sid[vm]
        ch = torch.zeros(nv, dtype=torch.bool, device='cuda')
        ch[0] = True
        if nv > 1:
            ch[1:] = veid[1:] != veid[:-1]
        ss = ch.nonzero(as_tuple=False).flatten()
        ssc = ss.cpu()
        eias = veid[ss].cpu()
        a_s = a_bf16.index_select(0, vsi)
        ns = ssc.numel()
        ds = torch.empty(nv, N, dtype=d.dtype, device=d.device)
        d.zero_()
        for i in range(ns):
            gid = eias[i].item()
            st = ssc[i].item()
            en = ssc[i+1].item() if i+1 < ns else nv
            triton_fused_fp4_matmul_nt(a_s[st:en], b_packed[gid], b_scale_u8[gid], ds[st:en])
        d.index_copy_(0, vsi, ds)
    timed("FULL_dispatch", full)

    # Print results
    print("=== Dispatch Profiling (1 call, ms) ===")
    print(f"  {'Step':<25s} {'ms':>8s} {'%':>6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*6}")
    full_ms = timings["FULL_dispatch"]
    for name, ms in timings.items():
        if name == "FULL_dispatch":
            continue
        pct = ms / full_ms * 100
        print(f"  {name:<25s} {ms:8.3f} {pct:5.1f}%")
    print(f"  {'-'*25} {'-'*8} {'-'*6}")
    print(f"  {'FULL_dispatch':<25s} {full_ms:8.3f} 100.0%")
    print()
    print(f"  Per token (×120 calls): {full_ms * 120:.0f} ms = {1000 / (full_ms * 120):.2f} tok/s")
    print(f"  Kernel only (×120):     {timings['6_kernels_x6'] * 120:.0f} ms")
    print(f"  Overhead (×120):        {(full_ms - timings['6_kernels_x6']) * 120:.0f} ms")


if __name__ == "__main__":
    profile_dispatch()
