/*
 * Probe which per-lane addresses ldmatrix.x4 actually uses on sm_121.
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -o test_ldmatrix_address_probe test_ldmatrix_address_probe.cu
 *
 * Run:
 *   ./test_ldmatrix_address_probe
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

enum Mode {
  ROW_ADDRS = 0,
  ALL_ZERO = 1,
  ALL_OFFSET_32 = 2,
  REVERSED_ROWS = 3,
  STRIDED_ROWS = 4,
};

__global__ void probe(uint32_t* output, int mode) {
  __shared__ uint16_t smem[512];
  int tid = threadIdx.x;
  int lane = tid & 31;

  for (int i = tid; i < 512; i += blockDim.x) {
    smem[i] = static_cast<uint16_t>(i);
  }
  __syncthreads();

  if (lane < 32) {
    int row = lane % 8;
    int mat = lane / 8;
    int offset = 0;
    if (mode == ROW_ADDRS) {
      offset = mat * 64 + row * 8;
    } else if (mode == ALL_ZERO) {
      offset = 0;
    } else if (mode == ALL_OFFSET_32) {
      offset = 32;
    } else if (mode == REVERSED_ROWS) {
      offset = mat * 64 + (7 - row) * 8;
    } else if (mode == STRIDED_ROWS) {
      offset = mat * 64 + ((row * 3) & 7) * 8;
    }

    uint32_t addr = __cvta_generic_to_shared(&smem[offset]);
    uint32_t frag[4];
    asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(addr));

    int base = mode * 32 * 4 + lane * 4;
    output[base + 0] = frag[0];
    output[base + 1] = frag[1];
    output[base + 2] = frag[2];
    output[base + 3] = frag[3];
  }
}

static void print_mode(const char* name, uint32_t* h, int mode) {
  printf("\n=== %s ===\n", name);
  printf("lane  f0        f1        f2        f3\n");
  for (int lane = 0; lane < 32; ++lane) {
    int base = mode * 32 * 4 + lane * 4;
    uint32_t f0 = h[base + 0], f1 = h[base + 1], f2 = h[base + 2], f3 = h[base + 3];
    printf("%2d    [%3u,%3u] [%3u,%3u] [%3u,%3u] [%3u,%3u]\n",
           lane,
           f0 & 0xffff, f0 >> 16,
           f1 & 0xffff, f1 >> 16,
           f2 & 0xffff, f2 >> 16,
           f3 & 0xffff, f3 >> 16);
  }
}

int main() {
  uint32_t* d_out = nullptr;
  constexpr int modes = 5;
  constexpr int words = modes * 32 * 4;
  uint32_t h_out[words];
  cudaMalloc(&d_out, words * sizeof(uint32_t));
  cudaMemset(d_out, 0, words * sizeof(uint32_t));

  for (int mode = 0; mode < modes; ++mode) {
    probe<<<1, 32>>>(d_out, mode);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error after mode %d: %s\n", mode, cudaGetErrorString(err));
      return 1;
    }
  }

  cudaMemcpy(h_out, d_out, words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  print_mode("ROW_ADDRS", h_out, ROW_ADDRS);
  print_mode("ALL_ZERO", h_out, ALL_ZERO);
  print_mode("ALL_OFFSET_32", h_out, ALL_OFFSET_32);
  print_mode("REVERSED_ROWS", h_out, REVERSED_ROWS);
  print_mode("STRIDED_ROWS", h_out, STRIDED_ROWS);

  cudaFree(d_out);
  return 0;
}
