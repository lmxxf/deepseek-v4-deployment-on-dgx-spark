/*
 * Search B-fragment packing permutations for SM121 mma.bf16.
 *
 * This compares ldmatrix.x2.trans B fragments against Marlin-style manual
 * fragments built from the eight values produced by one repack lane:
 *   {k0,n0}, {k1,n0}, {k8,n0}, {k9,n0}, {k0,n8}, ...
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -o test_b_pack_perm_search test_b_pack_perm_search.cu
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ uint32_t bf16_pair(float lo, float hi) {
  nv_bfloat16 x = __float2bfloat16(lo);
  nv_bfloat16 y = __float2bfloat16(hi);
  uint16_t xl = *reinterpret_cast<uint16_t*>(&x);
  uint16_t yh = *reinterpret_cast<uint16_t*>(&y);
  return static_cast<uint32_t>(xl) | (static_cast<uint32_t>(yh) << 16);
}

__device__ void mma_bf16(const uint32_t a[4], const uint32_t b[2], float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ void nth_perm4(int idx, int p[4]) {
  int elems[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int n = 8;
  for (int i = 0; i < 4; ++i) {
    int f = 1;
    for (int j = 1; j < n; ++j) f *= j;
    int q = idx / f;
    idx %= f;
    p[i] = elems[q];
    for (int j = q; j < n - 1; ++j) elems[j] = elems[j + 1];
    --n;
  }
}

__global__ void search_kernel(int* diffs, uint32_t* best_dump) {
  __shared__ uint16_t a_smem[256];
  __shared__ uint16_t b_smem[256];

  int lane = threadIdx.x & 31;
  int perm_id = blockIdx.x;

  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    a_smem[i] = 0;
    b_smem[i] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < 16; i += blockDim.x) {
    nv_bfloat16 one = __float2bfloat16(1.0f);
    a_smem[i * 16 + i] = *reinterpret_cast<uint16_t*>(&one);
  }
  for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
    int k = i / 16;
    int n = i % 16;
    nv_bfloat16 v = __float2bfloat16(static_cast<float>(100 * k + n + 1));
    b_smem[k * 16 + n] = *reinterpret_cast<uint16_t*>(&v);
  }
  __syncthreads();

  uint32_t a[4];
  int a_row = lane % 8;
  int a_mat = lane / 8;
  uint32_t a_addr = __cvta_generic_to_shared(&a_smem[a_mat * 64 + a_row * 8]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(a_addr));
  int src_lane = (lane % 8) * 4 + lane / 8;
  for (int i = 0; i < 4; ++i) a[i] = __shfl_sync(0xffffffff, a[i], src_lane);

  uint32_t b_ref[2];
  int b_row = lane % 8;
  int b_mat = lane / 8;
  uint32_t b_addr = __cvta_generic_to_shared(&b_smem[b_mat * 64 + b_row * 8]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(b_ref[0]), "=r"(b_ref[1])
      : "r"(b_addr));

  int tc_col = lane / 4;
  int tc_row = (lane % 4) * 2;
  int offs[4] = {0, 1, 8, 9};
  float vals[8];
  for (int i = 0; i < 4; ++i) {
    vals[i] = static_cast<float>(100 * (tc_row + offs[i]) + tc_col + 1);
    vals[4 + i] = static_cast<float>(100 * (tc_row + offs[i]) + tc_col + 8 + 1);
  }

  int p[4];
  nth_perm4(perm_id, p);
  uint32_t b_try[2] = {bf16_pair(vals[p[0]], vals[p[1]]),
                       bf16_pair(vals[p[2]], vals[p[3]])};

  float c_ref[4] = {0, 0, 0, 0};
  float c_try[4] = {0, 0, 0, 0};
  mma_bf16(a, b_ref, c_ref);
  mma_bf16(a, b_try, c_try);

  int d = 0;
  for (int i = 0; i < 4; ++i) {
    if (c_ref[i] != c_try[i]) ++d;
  }
  atomicAdd(&diffs[perm_id], d);

  if (perm_id == 0 && lane < 8) {
    best_dump[lane * 4 + 0] = b_ref[0];
    best_dump[lane * 4 + 1] = b_ref[1];
    best_dump[lane * 4 + 2] = b_try[0];
    best_dump[lane * 4 + 3] = b_try[1];
  }
}

static void print_perm4(int idx) {
  int elems[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int p[4];
  int n = 8;
  for (int i = 0; i < 4; ++i) {
    int f = 1;
    for (int j = 1; j < n; ++j) f *= j;
    int q = idx / f;
    idx %= f;
    p[i] = elems[q];
    for (int j = q; j < n - 1; ++j) elems[j] = elems[j + 1];
    --n;
  }
  printf("{%d,%d,%d,%d}", p[0], p[1], p[2], p[3]);
}

int main() {
  constexpr int perms = 8 * 7 * 6 * 5;
  int* d_diffs = nullptr;
  uint32_t* d_dump = nullptr;
  int h_diffs[perms];
  cudaMalloc(&d_diffs, sizeof(h_diffs));
  cudaMalloc(&d_dump, sizeof(uint32_t) * 32);
  cudaMemset(d_diffs, 0, sizeof(h_diffs));
  search_kernel<<<perms, 32>>>(d_diffs, d_dump);
  cudaDeviceSynchronize();
  cudaMemcpy(h_diffs, d_diffs, sizeof(h_diffs), cudaMemcpyDeviceToHost);

  int best = 999999;
  for (int i = 0; i < perms; ++i) best = std::min(best, h_diffs[i]);
  printf("best diff %d / 128\n", best);
  int shown = 0;
  for (int i = 0; i < perms && shown < 20; ++i) {
    if (h_diffs[i] == best) {
      printf("perm_id=%d p=", i);
      print_perm4(i);
      printf("\n");
      ++shown;
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaFree(d_diffs);
  cudaFree(d_dump);
  return 0;
}
