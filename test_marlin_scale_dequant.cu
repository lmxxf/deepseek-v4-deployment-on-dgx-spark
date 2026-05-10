/*
 * Test Marlin's dequant_fp8_scales on sm_121.
 *
 * Hypothesis: E8M0 scale dequant has exponent widening bug on sm_120+,
 * causing scale values to underflow to ~2^-112 instead of correct 2^(e-127).
 *
 * Compile and run inside vllm container:
 *   nvcc -arch=sm_121 -o test_marlin_scale test_marlin_scale_dequant.cu && ./test_marlin_scale
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

// Copied from Marlin's dequant.h — the exact function we're testing
__device__ inline void dequant_fp8_scales_e8m0_bf16(int q, nv_bfloat162* frag_b) {
    // In this conversion, 2 ** -127 in FP8E8M0 would become 0 in BF16,
    // but we assume that such a extreme value would not occur in real models.
    int Out1 = (q & 0xFF00FF00) >> 1;
    q <<= 7;
    int Out2 = q & 0x7F807F80;

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
    frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

// Reference: correct E8M0 -> float conversion
__device__ float e8m0_to_float(uint8_t e) {
    return powf(2.0f, (float)e - 127.0f);
}

__global__ void test_kernel(uint8_t* e8m0_vals, float* marlin_results, float* ref_results, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;

    uint8_t e = e8m0_vals[idx];

    // Reference
    ref_results[idx] = e8m0_to_float(e);

    // Marlin's dequant: pack 4 E8M0 values into int32, call dequant
    // Pack: q = [e, 0, e, 0] so we test both Out1 and Out2 paths
    int q = ((int)e << 24) | ((int)e << 8);  // bytes: [e, 0, e, 0]

    nv_bfloat162 frag[2];
    dequant_fp8_scales_e8m0_bf16(q, frag);

    // Extract results — frag[0] and frag[1] each contain 2 bf16 values
    // frag[1] = Out1 (from high bytes), frag[0] = Out2 (from low bytes shifted)
    float r1_0 = __bfloat162float(__low2bfloat16(frag[1]));
    float r1_1 = __bfloat162float(__high2bfloat16(frag[1]));
    float r0_0 = __bfloat162float(__low2bfloat16(frag[0]));
    float r0_1 = __bfloat162float(__high2bfloat16(frag[0]));

    // Store the first non-zero result we find
    marlin_results[idx] = (r1_1 != 0.0f) ? r1_1 : (r0_1 != 0.0f) ? r0_1 : r1_0;
}

int main() {
    // Test E8M0 values that appear in real V4 weights: 119-122
    // Plus some edge cases
    uint8_t test_vals[] = {0, 1, 64, 119, 120, 121, 122, 126, 127, 128, 134, 200, 254, 255};
    int n = sizeof(test_vals) / sizeof(test_vals[0]);

    uint8_t *d_vals;
    float *d_marlin, *d_ref;
    float h_marlin[256], h_ref[256];

    cudaMalloc(&d_vals, n);
    cudaMalloc(&d_marlin, n * sizeof(float));
    cudaMalloc(&d_ref, n * sizeof(float));

    cudaMemcpy(d_vals, test_vals, n, cudaMemcpyHostToDevice);

    test_kernel<<<1, n>>>(d_vals, d_marlin, d_ref, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_marlin, d_marlin, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, d_ref, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("E8M0 -> BF16 scale dequant test on sm_121:\n\n");
    printf("  %-6s  %-15s  %-15s  %-8s\n", "E8M0", "Reference", "Marlin", "Status");
    printf("  %-6s  %-15s  %-15s  %-8s\n", "------", "---------------", "---------------", "--------");

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float ref = h_ref[i];
        float mar = h_marlin[i];
        float ratio = (ref != 0) ? mar / ref : (mar == 0 ? 1.0f : 999.0f);
        bool ok = fabsf(ratio - 1.0f) < 0.01f || (ref == 0 && mar == 0);
        if (!ok) failures++;
        printf("  %-6d  %-15.6e  %-15.6e  %s%s\n",
               test_vals[i], ref, mar, ok ? "OK" : "FAIL", ok ? "" : " <<<");
    }

    printf("\n%d / %d passed\n", n - failures, n);

    cudaFree(d_vals);
    cudaFree(d_marlin);
    cudaFree(d_ref);
    return failures > 0 ? 1 : 0;
}
