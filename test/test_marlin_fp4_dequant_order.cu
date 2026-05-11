/*
 * Dump Marlin FP4 E2M1 -> BF16 dequant output order for controlled packed
 * words. This is a small companion to the B-fragment repack analysis.
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -o test_marlin_fp4_dequant_order test_marlin_fp4_dequant_order.cu
 *
 * Run:
 *   ./test_marlin_fp4_dequant_order
 */

#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ void dequant_e2m1_bf16_skip(int q, uint32_t out[2]) {
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  int out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Marlin reverse indexing: frag_b[1] = out1, frag_b[0] = out2.
  out[1] = static_cast<uint32_t>(out1);
  out[0] = static_cast<uint32_t>(out2);
}

__global__ void probe(uint32_t* out) {
  int lane = threadIdx.x;
  if (lane != 0) return;

  uint32_t words[] = {
      0x76543210u,
      0xfedcba98u,
      0xeca86420u,
      0xfdb97531u,
  };
  for (int i = 0; i < 4; ++i) {
    uint32_t frag[2];
    dequant_e2m1_bf16_skip(static_cast<int>(words[i]), frag);
    out[i * 4 + 0] = words[i];
    out[i * 4 + 1] = frag[0];
    out[i * 4 + 2] = frag[1];

    uint32_t shifted[2];
    dequant_e2m1_bf16_skip(static_cast<int>(words[i] >> 8), shifted);
    out[i * 4 + 3] = shifted[0];
  }

  for (int pos = 0; pos < 8; ++pos) {
    uint32_t word = 0x7u << (pos * 4);
    uint32_t frag[2];
    dequant_e2m1_bf16_skip(static_cast<int>(word), frag);
    out[16 + pos * 2 + 0] = frag[0];
    out[16 + pos * 2 + 1] = frag[1];
  }
}

static void print_pair(uint32_t raw) {
  uint16_t lo = raw & 0xffff;
  uint16_t hi = raw >> 16;
  printf("raw=0x%08x halves=[0x%04x,0x%04x]", raw, lo, hi);
}

int main() {
  uint32_t* d_out = nullptr;
  uint32_t h[32] = {};
  cudaMalloc(&d_out, sizeof(h));
  probe<<<1, 32>>>(d_out);
  cudaDeviceSynchronize();
  cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; ++i) {
    printf("word=0x%08x frag0 ", h[i * 4 + 0]);
    print_pair(h[i * 4 + 1]);
    printf(" frag1 ");
    print_pair(h[i * 4 + 2]);
    printf(" shifted_frag0 ");
    print_pair(h[i * 4 + 3]);
    printf("\n");
  }
  printf("\none-hot nibble position map:\n");
  for (int pos = 0; pos < 8; ++pos) {
    printf("pos %d -> frag0 ", pos);
    print_pair(h[16 + pos * 2 + 0]);
    printf(" frag1 ");
    print_pair(h[16 + pos * 2 + 1]);
    printf("\n");
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaFree(d_out);
  return 0;
}
