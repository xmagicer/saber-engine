// Private gfx1201 decode experiment. Launch one 256-thread workgroup per 128
// output columns and per split. Geometry: 1 <= M <= 16, N % 16 == 0,
// K % 128 == 0; pb1 and pb2 use the stock 128-column partition alignment.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

namespace {

__device__ __forceinline__ int2_t expand_eight(unsigned word, unsigned low, unsigned high) {
  const unsigned even = word & 0x0f0f0f0fu;
  const unsigned odd = (word >> 4) & 0x0f0f0f0fu;
  const unsigned be = __builtin_amdgcn_perm(high, low, even & 0x07070707u)
                    | ((even & 0x08080808u) << 4);
  const unsigned bo = __builtin_amdgcn_perm(high, low, odd & 0x07070707u)
                    | ((odd & 0x08080808u) << 4);
  return int2_t{(int)__builtin_amdgcn_perm(bo, be, 0x05010400u),
                (int)__builtin_amdgcn_perm(bo, be, 0x07030602u)};
}

__device__ __forceinline__ void local_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int Splits, bool ShareA>
__device__ __forceinline__ void reg_decode(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) {
  constexpr int TileN = 128, SlabK = 128, AStride = SlabK + 8;
  __shared__ __align__(8) unsigned char shared_a[ShareA ? 16 * AStride : 1];
  __shared__ int last;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int row = lane & 15;
  const int half_k = (lane >> 4) * 8;
  const int n0 = blockIdx.x * TileN;
  const int n = n0 + (tid >> 5) * 16 + row;
  const int weight_row = min(n, N - 1);
  const int input_row = min(row, M - 1);
  const int partition = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
  A += (size_t)partition * astride;
  As += (size_t)partition * M;

  // A WPERM tile is [K/16][32] uint32 words. Lane l owns the
  // eight neighboring nibbles consumed by its one B fragment.
  const unsigned *lane_word = reinterpret_cast<const unsigned *>(W)
      + (size_t)(weight_row / 16) * (K / 16) * 32
      + (lane & 16) + (weight_row & 15);
  const int exponent = Wref[weight_row];
  const int split = blockIdx.z;
  const int slabs = K / SlabK;
  const int per_split = (slabs + Splits - 1) / Splits;
  const int end = min(slabs, (split + 1) * per_split);
  floatx8 acc = {};

  for (int slab = split * per_split; slab < end; ++slab) {
    const int k0 = slab * SlabK;
    unsigned packed[8];
    unsigned char scales[4];
    int2_t activations[8];
#pragma unroll
    for (int step = 0; step < 8; ++step)
      packed[step] = __builtin_nontemporal_load(lane_word + (size_t)(slab * 8 + step) * 32);
#pragma unroll
    for (int group = 0; group < 4; ++group)
      scales[group] = Ws[(size_t)(slab * 4 + group) * N + weight_row];

    if constexpr (ShareA) {
      // One 8-byte copy per thread stages the whole [16,128] A slab.
      const int arow = tid / 16, ak = (tid % 16) * 8;
      *reinterpret_cast<int2_t *>(shared_a + arow * AStride + ak)
          = *reinterpret_cast<const int2_t *>(A + (size_t)min(arow, M-1) * K + k0 + ak);
      local_barrier();
    }
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const unsigned char *src;
      if constexpr (ShareA) src = shared_a + row * AStride + step * 16 + half_k;
      else src = A + (size_t)input_row * K + k0 + step * 16 + half_k;
      activations[step] = *reinterpret_cast<const int2_t *>(src);
    }
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const int delta = min(15, max(0, exponent - (int)scales[group]));
      const unsigned low = kMag[delta][0], high = kMag[delta][1];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int step = group * 2 + i;
        const int2_t weight = expand_eight(packed[step], low, high);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(activations[step], weight, acc);
      }
    }
    if constexpr (ShareA) local_barrier();
  }

  // Stock TM1 accumulator writeout, split-buffer order and last-arriver fold.
  if constexpr (Splits == 1) {
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = half_k + e;
        if (m < M) C[(size_t)m * N + n] = (__bf16)(acc[e] * rf * As[m]);
      }
    }
    return;
  }
  if (n < N) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = half_k + e;
      if (m < M) P[((size_t)split * M + m) * N + n] = acc[e];
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    last = (atomicAdd(&cnt[blockIdx.x], 1) == Splits - 1);
  }
  __syncthreads();
  if (!last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;
  const int high_n = min(n0 + TileN, N);
  for (int nn = n0 + tid; nn < high_n; nn += 256) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int s = 0; s < Splits; ++s) sum += P[((size_t)s * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)(sum * rf * As[m]);
    }
  }
}
} // namespace

#define REG_ENTRY(NAME, KS, SHARE) \
  extern "C" __global__ __launch_bounds__(256) void NAME( \
      const unsigned char *__restrict__ A, const unsigned char *__restrict__ W, \
      const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref, \
      const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt, \
      __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) { \
    reg_decode<KS, SHARE>(A,W,Ws,Wref,As,P,cnt,C,M,N,K,pb1,pb2,astride); \
  }

REG_ENTRY(mxfp4_gemm_decode_reg_b1_areg_ks1, 1, false)
REG_ENTRY(mxfp4_gemm_decode_reg_b1_areg_ks2, 2, false)
REG_ENTRY(mxfp4_gemm_decode_reg_b1_areg_ks4, 4, false)
REG_ENTRY(mxfp4_gemm_decode_reg_b1_alds_ks1, 1, true)
REG_ENTRY(mxfp4_gemm_decode_reg_b1_alds_ks2, 2, true)
REG_ENTRY(mxfp4_gemm_decode_reg_b1_alds_ks4, 4, true)
