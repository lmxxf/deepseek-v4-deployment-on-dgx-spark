/*
 * Verify ldmatrix + __shfl_sync fix on sm_121.
 *
 * After ldmatrix on sm_121 (linear), apply shuffle to get sm_80-compatible layout.
 * Fix: src_lane = (laneid % 8) * 4 + laneid / 8
 *
 * nvcc -arch=sm_121 -o test_ldmatrix_shfl test_ldmatrix_shfl_fix.cu && ./test_ldmatrix_shfl
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void test_shfl_fix(uint32_t* output) {
    __shared__ uint16_t smem[256];

    int tid = threadIdx.x;
    int laneid = tid % 32;

    // Fill: smem[i] = i
    if (tid < 256) smem[tid] = (uint16_t)tid;
    __syncthreads();

    if (tid < 32) {
        // ldmatrix.x4: provide row addresses (same as Marlin does)
        int row = laneid % 8;
        int mat = laneid / 8;
        uint32_t addr = __cvta_generic_to_shared(&smem[mat * 64 + row * 8]);

        uint32_t frag[4];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
            : "r"(addr)
        );

        // Apply shuffle fix
        int src_lane = (laneid % 8) * 4 + laneid / 8;
        frag[0] = __shfl_sync(0xFFFFFFFF, frag[0], src_lane);
        frag[1] = __shfl_sync(0xFFFFFFFF, frag[1], src_lane);
        frag[2] = __shfl_sync(0xFFFFFFFF, frag[2], src_lane);
        frag[3] = __shfl_sync(0xFFFFFFFF, frag[3], src_lane);

        // Store results
        for (int i = 0; i < 4; i++)
            output[tid * 4 + i] = frag[i];
    }
}

int main() {
    uint32_t* d_output;
    uint32_t h_output[128];

    cudaMalloc(&d_output, 128 * sizeof(uint32_t));
    test_shfl_fix<<<1, 256>>>(d_output);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_output, d_output, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("ldmatrix + __shfl_sync fix on sm_121:\n\n");

    // Expected: sm_80-compatible layout
    // Thread T, frag[k]: should have smem[k*64 + (T%%8)*8 + 2*(T/8)] and +1
    int errors = 0;
    for (int k = 0; k < 4; k++) {
        printf("frag[%d]:\n", k);
        printf("  Thread  actual     expected   match\n");
        for (int t = 0; t < 32; t++) {
            uint32_t f = h_output[t * 4 + k];
            uint16_t got_lo = f & 0xFFFF;
            uint16_t got_hi = f >> 16;
            int exp_lo = k * 64 + (t % 8) * 8 + 2 * (t / 8);
            int exp_hi = exp_lo + 1;
            int ok = (got_lo == exp_lo && got_hi == exp_hi);
            if (!ok) errors++;
            if (t < 8 || !ok) {
                printf("  T%-5d  [%3d,%3d]  [%3d,%3d]  %s\n",
                       t, got_lo, got_hi, exp_lo, exp_hi, ok ? "OK" : "FAIL <<<");
            }
        }
        if (k < 3) printf("  ...\n");
    }

    printf("\n%d / 128 values correct\n", 128 - errors);
    printf("%s\n", errors == 0 ? "ALL MATCH ✅" : "MISMATCHES FOUND ❌");

    cudaFree(d_output);
    return errors;
}
