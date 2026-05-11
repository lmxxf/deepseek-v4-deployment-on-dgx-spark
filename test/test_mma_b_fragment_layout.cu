/*
 * Compare BF16 mma B fragments loaded by ldmatrix.trans with manually packed
 * register fragments. This isolates Marlin's B-side register layout from the
 * A-side ldmatrix issue.
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -o test_mma_b_fragment_layout test_mma_b_fragment_layout.cu
 *
 * Run:
 *   ./test_mma_b_fragment_layout
 */

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

__global__ void probe(float* out, uint32_t* b_dump) {
  __shared__ uint16_t a_smem[256];
  __shared__ uint16_t b_smem[256];

  int lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    a_smem[i] = 0;
    b_smem[i] = 0;
  }
  __syncthreads();

  // A = identity(16x16), row-major.
  for (int i = threadIdx.x; i < 16; i += blockDim.x) {
    nv_bfloat16 one = __float2bfloat16(1.0f);
    a_smem[i * 16 + i] = *reinterpret_cast<uint16_t*>(&one);
  }

  // B[k,n] = 100*k + n + 1. Store as row-major 16x8 first.
  for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) {
    int k = i / 8;
    int n = i % 8;
    nv_bfloat16 v = __float2bfloat16(static_cast<float>(100 * k + n + 1));
    b_smem[k * 8 + n] = *reinterpret_cast<uint16_t*>(&v);
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
  // Try the standard B operand loader: m8n8.x2.trans from row-major shared B.
  int b_row = lane % 8;
  int b_mat = lane / 8;
  uint32_t b_addr = __cvta_generic_to_shared(&b_smem[b_mat * 64 + b_row * 8]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(b_ref[0]), "=r"(b_ref[1])
      : "r"(b_addr));

  // Manual candidate matching Marlin's dequant result order for two bf162 regs.
  uint32_t b_manual_a[2];
  b_manual_a[0] = bf16_pair(static_cast<float>(lane * 4 + 1),
                            static_cast<float>(lane * 4 + 2));
  b_manual_a[1] = bf16_pair(static_cast<float>(lane * 4 + 3),
                            static_cast<float>(lane * 4 + 4));

  // Candidate with pair order swapped, mirroring dequant's reverse indexing.
  uint32_t b_manual_b[2] = {b_manual_a[1], b_manual_a[0]};

  float c_ref[4] = {0, 0, 0, 0};
  float c_a[4] = {0, 0, 0, 0};
  float c_b[4] = {0, 0, 0, 0};
  mma_bf16(a, b_ref, c_ref);
  mma_bf16(a, b_manual_a, c_a);
  mma_bf16(a, b_manual_b, c_b);

  for (int i = 0; i < 4; ++i) {
    out[(0 * 32 + lane) * 4 + i] = c_ref[i];
    out[(1 * 32 + lane) * 4 + i] = c_a[i];
    out[(2 * 32 + lane) * 4 + i] = c_b[i];
  }
  b_dump[lane * 4 + 0] = b_ref[0];
  b_dump[lane * 4 + 1] = b_ref[1];
  b_dump[lane * 4 + 2] = b_manual_a[0];
  b_dump[lane * 4 + 3] = b_manual_a[1];
}

static int compare(const float* h, int lhs, int rhs) {
  int diff = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int i = 0; i < 4; ++i) {
      float a = h[(lhs * 32 + lane) * 4 + i];
      float b = h[(rhs * 32 + lane) * 4 + i];
      if (a != b) ++diff;
    }
  }
  return diff;
}

static void print_b(uint32_t v) {
  uint16_t lo = v & 0xffff;
  uint16_t hi = v >> 16;
  nv_bfloat16 blo = *reinterpret_cast<nv_bfloat16*>(&lo);
  nv_bfloat16 bhi = *reinterpret_cast<nv_bfloat16*>(&hi);
  printf("[%.0f,%.0f]", __bfloat162float(blo), __bfloat162float(bhi));
}

int main() {
  float* d_out = nullptr;
  uint32_t* d_b = nullptr;
  float h_out[3 * 32 * 4] = {};
  uint32_t h_b[32 * 4] = {};
  cudaMalloc(&d_out, sizeof(h_out));
  cudaMalloc(&d_b, sizeof(h_b));
  probe<<<1, 32>>>(d_out, d_b);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, sizeof(h_b), cudaMemcpyDeviceToHost);

  printf("mma output diff: ref vs manual_a = %d / 128\n", compare(h_out, 0, 1));
  printf("mma output diff: ref vs manual_b = %d / 128\n", compare(h_out, 0, 2));
  printf("\nB fragment dump, first 8 lanes:\n");
  for (int lane = 0; lane < 8; ++lane) {
    printf("lane %2d ref0=", lane);
    print_b(h_b[lane * 4 + 0]);
    printf(" ref1=");
    print_b(h_b[lane * 4 + 1]);
    printf(" manual0=");
    print_b(h_b[lane * 4 + 2]);
    printf(" manual1=");
    print_b(h_b[lane * 4 + 3]);
    printf("\n");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaFree(d_out);
  cudaFree(d_b);
  return 0;
}
