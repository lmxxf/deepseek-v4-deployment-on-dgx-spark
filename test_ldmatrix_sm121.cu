/*
 * Test ldmatrix.sync.aligned behavior on sm_121.
 *
 * ldmatrix loads matrix fragments from shared memory into registers.
 * Each thread in a warp gets specific elements based on lane ID.
 * If sm_120's mapping differs from sm_80/sm_90, Marlin's data layout is wrong.
 *
 * Test: fill shared memory with known values (element = flat_index),
 * call ldmatrix, dump per-thread results, verify against PTX ISA spec.
 *
 * nvcc -arch=sm_121 -o test_ldmatrix test_ldmatrix_sm121.cu && ./test_ldmatrix
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// ldmatrix.sync.aligned.m8n8.x4.shared.b16
// Loads 4 x (8x8) matrices = 4 registers per thread from shared memory
// Input: 32 threads, each provides a shared memory address
// Output: 4 uint32 registers per thread

__global__ void test_ldmatrix_x4(uint16_t* smem_data, uint32_t* output, int* errors) {
    __shared__ uint16_t smem[4 * 8 * 8];  // 4 matrices of 8x8 half values

    int tid = threadIdx.x;  // 0-31 (one warp)

    // Fill shared memory with known pattern: smem[i] = i
    if (tid < 128) {
        smem[tid] = (uint16_t)tid;
    }
    if (tid + 128 < 256) {
        smem[tid + 128] = (uint16_t)(tid + 128);
    }
    __syncthreads();

    // ldmatrix: each thread provides address for its row
    // For m8n8.x4: threads 0-7 address matrix 0, 8-15 matrix 1, 16-23 matrix 2, 24-31 matrix 3
    // Each thread's address = start of its row in the 8x8 matrix
    int matrix_id = tid / 8;
    int row_in_matrix = tid % 8;
    uint32_t smem_addr = __cvta_generic_to_shared(&smem[matrix_id * 64 + row_in_matrix * 8]);

    uint32_t frag[4];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(smem_addr)
    );

    // Store results: output[tid * 4 + i] = frag[i]
    for (int i = 0; i < 4; i++) {
        output[tid * 4 + i] = frag[i];
    }

    // Verify against expected layout from PTX ISA
    // For m8n8.x4 with .b16:
    // Each frag[i] contains 2 x uint16 values
    // The mapping from PTX ISA (Figure 80, m16n8k16 A fragment for .f16):
    //
    // For ldmatrix.m8n8.x4, the 4 output registers correspond to
    // 4 consecutive 8x8 matrices loaded from shared memory.
    //
    // For thread T (laneid), register frag[k] contains:
    //   low16  = smem[matrix_k * 64 + (T % 8) * 8 + (T / 8) * 2]
    //   high16 = smem[matrix_k * 64 + (T % 8) * 8 + (T / 8) * 2 + 1]
    //
    // Wait -- actually ldmatrix transposes. Let me just verify empirically.

    // For now, just dump the first few threads' results
    __syncthreads();
    if (tid == 0) {
        *errors = 0;
    }
    __syncthreads();

    // Each thread verifies its own fragments
    // Expected: for simple row-major 8x8 layout loaded by ldmatrix,
    // thread T gets: frag[k].low16 = smem[k*64 + row*8 + col*2]
    // where row and col depend on the ldmatrix mapping.
    //
    // Rather than guess, let's just dump and inspect.
}

__global__ void test_ldmatrix_simple(uint32_t* output) {
    __shared__ uint16_t smem[256];

    int tid = threadIdx.x;

    // Fill: smem[i] = i (each uint16 = its index)
    if (tid < 256) {
        smem[tid] = (uint16_t)tid;
    }
    __syncthreads();

    if (tid < 32) {
        // ldmatrix.x4: each thread addresses a row
        int row = tid % 8;
        int mat = tid / 8;
        uint32_t addr = __cvta_generic_to_shared(&smem[mat * 64 + row * 8]);

        uint32_t frag[4];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
            : "r"(addr)
        );

        output[tid * 4 + 0] = frag[0];
        output[tid * 4 + 1] = frag[1];
        output[tid * 4 + 2] = frag[2];
        output[tid * 4 + 3] = frag[3];
    }
}

int main() {
    uint32_t* d_output;
    uint32_t h_output[128];  // 32 threads * 4 regs

    cudaMalloc(&d_output, 128 * sizeof(uint32_t));

    test_ldmatrix_simple<<<1, 256>>>(d_output);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_output, d_output, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("ldmatrix.sync.aligned.m8n8.x4.shared.b16 results on sm_121:\n\n");
    printf("smem layout: smem[i] = i, 4 matrices of 8x8 uint16\n");
    printf("matrix k: smem[k*64 .. k*64+63]\n\n");

    // Print per-thread results
    printf("Thread  frag[0]          frag[1]          frag[2]          frag[3]\n");
    printf("------  ---------------  ---------------  ---------------  ---------------\n");

    for (int t = 0; t < 32; t++) {
        uint32_t f0 = h_output[t * 4 + 0];
        uint32_t f1 = h_output[t * 4 + 1];
        uint32_t f2 = h_output[t * 4 + 2];
        uint32_t f3 = h_output[t * 4 + 3];

        uint16_t f0_lo = f0 & 0xFFFF, f0_hi = f0 >> 16;
        uint16_t f1_lo = f1 & 0xFFFF, f1_hi = f1 >> 16;
        uint16_t f2_lo = f2 & 0xFFFF, f2_hi = f2 >> 16;
        uint16_t f3_lo = f3 & 0xFFFF, f3_hi = f3 >> 16;

        printf("T%-5d  [%3d, %3d]       [%3d, %3d]       [%3d, %3d]       [%3d, %3d]\n",
               t, f0_lo, f0_hi, f1_lo, f1_hi, f2_lo, f2_hi, f3_lo, f3_hi);
    }

    // Also verify: for standard m8n8 layout, thread T row R=T%8 should get:
    // frag[k] = {smem[k*64 + R*8 + 2*(T/8)], smem[k*64 + R*8 + 2*(T/8) + 1]}
    // This is the "broadcast across groups of 8" pattern.
    printf("\nVerification (expected vs actual for frag[0]):\n");
    printf("Thread  expected_lo  expected_hi  actual_lo  actual_hi  match\n");
    for (int t = 0; t < 32; t++) {
        int row = t % 8;
        int group = t / 8;
        // Standard ldmatrix mapping for m8n8:
        // thread T loads from address of row (T%8),
        // gets columns based on (T/8): frag contains cols 2*(T/8) and 2*(T/8)+1
        // But wait — ldmatrix with x4 distributes 4 matrices across 4 groups of 8 threads
        // Actually let me just check: for matrix 0, all 32 threads get something
        uint16_t actual_lo = h_output[t * 4] & 0xFFFF;
        uint16_t actual_hi = h_output[t * 4] >> 16;

        // Expected for frag[0] (matrix 0, smem[0..63]):
        // ldmatrix distributes: thread T gets smem values from row (T%8)
        // columns depend on T/8. For x4, frag[0] corresponds to the first 8x8 block
        // addressed by threads 0-7 (since those threads point to matrix 0)
        // But threads 8-31 also get frag[0] — those come from the addresses
        // provided by threads 0-7 (broadcast within the instruction)
        //
        // Standard mapping: frag[0] for all threads comes from the addresses
        // given by threads 0-7. Thread T gets:
        //   frag[0].lo = smem[addr_of_thread(T%8) + 2*(T/8)]
        //   frag[0].hi = smem[addr_of_thread(T%8) + 2*(T/8) + 1]
        // where addr_of_thread(r) = row r * 8 = r * 8
        // So: expected_lo = row * 8 + 2 * group
        //     expected_hi = row * 8 + 2 * group + 1
        int exp_lo = row * 8 + 2 * group;
        int exp_hi = row * 8 + 2 * group + 1;

        printf("T%-5d  %3d          %3d          %3d        %3d        %s\n",
               t, exp_lo, exp_hi, actual_lo, actual_hi,
               (actual_lo == exp_lo && actual_hi == exp_hi) ? "OK" : "MISMATCH <<<");
    }

    cudaFree(d_output);
    return 0;
}
