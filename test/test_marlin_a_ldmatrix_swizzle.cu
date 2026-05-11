/*
 * Probe Marlin's A shared-memory swizzle with sm_121 ldmatrix.x4.
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -o test_marlin_a_ldmatrix_swizzle test_marlin_a_ldmatrix_swizzle.cu
 *
 * Run:
 *   ./test_marlin_a_ldmatrix_swizzle
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

struct Cfg {
  const char* name;
  int threads;
  int thread_m_blocks;
  int thread_n_blocks;
  int thread_k_blocks;
  int m_block_size_8;
};

static constexpr int kModes = 2;
static constexpr int kViews = 2;
static constexpr int kMaxWords = kModes * kViews * 32 * 4;
static constexpr int kMaxAddrs = 32;

__device__ __host__ int transform_a(int i, int a_gl_rd_delta_o) {
  int row = i / a_gl_rd_delta_o;
  return a_gl_rd_delta_o * row + ((i % a_gl_rd_delta_o) ^ (row % 8));
}

__global__ void probe(uint32_t* out, int* addrs, Cfg cfg, int mode) {
  __shared__ uint16_t smem[12288];
  int tid = threadIdx.x;
  int lane = tid & 31;

  int a_sh_stride = 16 * cfg.thread_k_blocks / 8;       // BF16 A
  int a_gl_rd_delta_o = 16 * cfg.thread_k_blocks / 8;   // BF16 A
  int a_sh_stage = a_sh_stride * (16 * cfg.thread_m_blocks);
  int b_sh_stride = ((cfg.thread_n_blocks * 16) * 16 / 8) / 4;  // FP4 B
  int b_sh_stage = b_sh_stride * cfg.thread_k_blocks;
  int b_sh_wr_iters = b_sh_stage / cfg.threads;
  int tb_n_warps = cfg.thread_n_blocks / 4;             // BF16 A

  for (int i = tid; i < 12288; i += blockDim.x) smem[i] = 0xffff;
  __syncthreads();

  for (int i = tid; i < a_sh_stage; i += blockDim.x) {
    int dst = mode == 0 ? i : transform_a(i, a_gl_rd_delta_o);
    for (int j = 0; j < 8; ++j) {
      smem[dst * 8 + j] = static_cast<uint16_t>(i * 8 + j);
    }
  }
  __syncthreads();

  int row_threads = 16 / (cfg.m_block_size_8 ? 2 : 1);
  int a_sh_rd = a_sh_stride * (lane % row_threads) + lane / row_threads;
  a_sh_rd += 2 * ((tid / 32) / tb_n_warps) * b_sh_wr_iters;
  int rd = transform_a(a_sh_rd, a_gl_rd_delta_o);

  if (lane < 32) addrs[lane] = rd * 8;

  uint32_t frag[4];
  uint32_t sh_addr = __cvta_generic_to_shared(&smem[rd * 8]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(sh_addr));

  uint32_t shfl[4];
  int src_lane = (lane % 8) * 4 + lane / 8;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    shfl[i] = __shfl_sync(0xffffffff, frag[i], src_lane);
  }

  uint32_t* raw_dst = out + (mode * kViews + 0) * 32 * 4 + lane * 4;
  uint32_t* shfl_dst = out + (mode * kViews + 1) * 32 * 4 + lane * 4;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    raw_dst[i] = frag[i];
    shfl_dst[i] = shfl[i];
  }
}

static uint16_t smem_value(int idx, const Cfg& cfg, int mode) {
  int a_gl_rd_delta_o = 16 * cfg.thread_k_blocks / 8;
  if (mode == 0) return static_cast<uint16_t>(idx);

  // Shared memory stores logical int4 slot i at transform_a(i). Invert by search; the
  // probe is tiny and this keeps the host oracle unambiguous.
  int a_sh_stride = 16 * cfg.thread_k_blocks / 8;
  int a_sh_stage = a_sh_stride * (16 * cfg.thread_m_blocks);
  int slot = idx / 8;
  int inner = idx % 8;
  for (int i = 0; i < a_sh_stage; ++i) {
    if (transform_a(i, a_gl_rd_delta_o) == slot) return static_cast<uint16_t>(i * 8 + inner);
  }
  return 0xffff;
}

static uint32_t pack_expected(int lane, int frag, const int* addrs,
                              const Cfg& cfg, int mode) {
  int src_addr_lane = frag * 8 + (lane % 8);
  int offset = 2 * (lane / 8);
  uint16_t lo = smem_value(addrs[src_addr_lane] + offset, cfg, mode);
  uint16_t hi = smem_value(addrs[src_addr_lane] + offset + 1, cfg, mode);
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

static void run_cfg(const Cfg& cfg) {
  uint32_t* d_out = nullptr;
  int* d_addrs = nullptr;
  uint32_t h_out[kMaxWords] = {};
  int h_addrs[kMaxAddrs] = {};

  cudaMalloc(&d_out, sizeof(h_out));
  cudaMalloc(&d_addrs, sizeof(h_addrs));
  cudaMemset(d_out, 0, sizeof(h_out));
  cudaMemset(d_addrs, 0, sizeof(h_addrs));

  probe<<<1, cfg.threads>>>(d_out, d_addrs, cfg, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_addrs, d_addrs, sizeof(h_addrs), cudaMemcpyDeviceToHost);

  printf("\n=== %s / linear fill ===\n", cfg.name);
  int mismatches = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int frag = 0; frag < 4; ++frag) {
      uint32_t got = h_out[(0 * kViews + 0) * 32 * 4 + lane * 4 + frag];
      uint32_t exp = pack_expected(lane, frag, h_addrs, cfg, 0);
      mismatches += got != exp;
    }
  }
  printf("raw  vs sm80 oracle mismatches: %d / 128\n", mismatches);
  mismatches = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int frag = 0; frag < 4; ++frag) {
      uint32_t got = h_out[(0 * kViews + 1) * 32 * 4 + lane * 4 + frag];
      uint32_t exp = pack_expected(lane, frag, h_addrs, cfg, 0);
      mismatches += got != exp;
    }
  }
  printf("shfl vs sm80 oracle mismatches: %d / 128\n", mismatches);
  printf("addr lanes 0..7:");
  for (int i = 0; i < 8; ++i) printf(" %d", h_addrs[i]);
  printf("\n");

  cudaMemset(d_out, 0, sizeof(h_out));
  cudaMemset(d_addrs, 0, sizeof(h_addrs));
  probe<<<1, cfg.threads>>>(d_out, d_addrs, cfg, 1);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_addrs, d_addrs, sizeof(h_addrs), cudaMemcpyDeviceToHost);

  printf("=== %s / Marlin transform_a fill ===\n", cfg.name);
  mismatches = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int frag = 0; frag < 4; ++frag) {
      uint32_t got = h_out[(1 * kViews + 0) * 32 * 4 + lane * 4 + frag];
      uint32_t exp = pack_expected(lane, frag, h_addrs, cfg, 1);
      mismatches += got != exp;
    }
  }
  printf("raw  vs sm80 oracle mismatches: %d / 128\n", mismatches);
  mismatches = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int frag = 0; frag < 4; ++frag) {
      uint32_t got = h_out[(1 * kViews + 1) * 32 * 4 + lane * 4 + frag];
      uint32_t exp = pack_expected(lane, frag, h_addrs, cfg, 1);
      mismatches += got != exp;
    }
  }
  printf("shfl vs sm80 oracle mismatches: %d / 128\n", mismatches);
  printf("addr lanes 0..7:");
  for (int i = 0; i < 8; ++i) printf(" %d", h_addrs[i]);
  printf("\n");

  cudaFree(d_out);
  cudaFree(d_addrs);
}

int main() {
  Cfg cfgs[] = {
      {"large 64x256, moe_block=64", 256, 4, 16, 4, 0},
      {"large 64x128, moe_block=64", 128, 4, 8, 4, 0},
      {"large 128x64, moe_block=64", 128, 4, 4, 8, 0},
      {"small 128x128, moe_block=64", 256, 4, 8, 8, 0},
  };
  for (const Cfg& cfg : cfgs) run_cfg(cfg);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
