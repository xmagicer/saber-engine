// Saber source (CMP27E, 2026-10-26): the b8 (tm4) MXFP4 decode GEMMs with a REGISTER PREFETCH.
//
// The shipped decode body (`mxfp4_gemm_decode.hip`) stages each K slab global -> LDS and then runs
// its WMMA, one slab at a time: at 64 rows a block's slab is ~4 us of serial latency and the b8 GEMMs
// stream their weights at 240-400 GB/s against a 540 GB/s cap (LEDGER CMP27E's census). Here slab
// s+1's A rows, W fragment words and Ws / Wref scale bytes are loaded into registers BEFORE slab s's
// WMMA, and the loop's barriers are LDS-only (a `__syncthreads()` is a workgroup release fence =
// `s_wait_loadcnt 0` on gfx12, which would drain the prefetch). Every slab's LDS bytes and the WMMA
// order are the shipped body's, so the accumulators, the split-K partials, the last-arriver reduce
// and the direct store are the same bits (gated: `tools/capture/cmp27e/pair_probe.py --mode v5`,
// 0 differing outputs on 16 + 3 x 8 real layer copies, both cards).
//
// The PAIR twin also lays problem 1 (the 48-wide `in_proj_ba`, 4 short split blocks) FIRST in the
// grid: the shipped order dispatches them last, behind 128 long qkvz blocks, where their 10-slab
// latency chain is the kernel's tail (100 us isolated; 82 us only when the dispatcher happens to
// place them well -- CMP27D's per-boot, per-layer "rank asymmetry"). Same blocks, same bytes.
//
// Bound only under the `decode_pf` knob (`--engine-decode-pf`, an ARM, default off) by
// `saber_kernels::rules`, never by a registry row; the shipped `mxfp4_gemm_decode` keeps its bytes.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

#define DEC_DWN 8
#define DEC_MTILE 16
#define DEC_PAD 8
static __device__ __forceinline__ unsigned int wq_dec(uint2_t v, int q) { return q ? v[1] : v[0]; }
template <bool NT, typename T>
static __device__ __forceinline__ T ld_w(const T *p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

#define CMP27E_LDS_BARRIER()                                          \
  do {                                                                \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");   \
    __builtin_amdgcn_s_barrier();                                     \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");   \
  } while (0)

template <int DWN, int DBK, int DTM, bool NT>
__device__ __forceinline__ void cmp27e_decode_pf_impl(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride,
    const int DKS, const int bx_, const int bz_) {
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
  constexpr int NA = (DEC_MTILE * DTM * DBK + DNTHREADS * 16 - 1) / (DNTHREADS * 16);
  constexpr int NW = (NTILES_T * KSTEPS_T * 32) / (DNTHREADS * 2);
  static_assert((NTILES_T * KSTEPS_T * 32) % (DNTHREADS * 2) == 0, "W slots per thread");
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = bx_ * BND;
  const int ks = bz_;
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  const unsigned int *__restrict__ Wp = (const unsigned int *)W;
  const int ksteps_g = K / 16;
  uint4_t ra[NA];
  uint2_t rw[NW];
  unsigned int rmeta[NW];

  auto load = [&](int k0) {
#pragma unroll
    for (int j = 0; j < NA; ++j) {
      const int idx = j * DNTHREADS * 16 + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;
        ra[j] = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);
      }
    }
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const int sl = j * DNTHREADS * 2 + tid * 2;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
      const int r = ntl * 16 + (lane_ & 15);
      const int gn = n0 + r;
      const int gc = gn < N - 2 ? gn : N - 2;
      const int lanec = (lane_ & 16) | (gc & 15);
      rw[j] = ld_w<NT>((const uint2_t *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
      const int blk = (k0 + kst * 16) / 32;
      const unsigned int refs = *(const unsigned short *)(Wref + gc);
      const unsigned int scales = *(const unsigned short *)(Ws + (size_t)blk * N + gc);
      rmeta[j] = refs | (scales << 16);
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int j = 0; j < NA; ++j) {
      const int idx = j * DNTHREADS * 16 + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        *(uint4_t *)(&sA[r * DASTR + c]) = ra[j];
      }
    }
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const int sl = j * DNTHREADS * 2 + tid * 2;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
      const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        int d = (int)((rmeta[j] >> (q * 8)) & 255u) - (int)((rmeta[j] >> (16 + q * 8)) & 255u);
        d = d < 0 ? 0 : (d > 15 ? 15 : d);
        const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
        const unsigned int ev = wq_dec(rw[j], q) & 0x0F0F0F0Fu;
        const unsigned int od = (wq_dec(rw[j], q) >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
        *(uint2_t *)(&sW[(r + q) * DWSTR + kloc]) =
            uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u), __builtin_amdgcn_perm(bo, be, 0x07030602u)};
      }
    }
  };
  auto mma_slab = [&]() {
    CMP27E_LDS_BARRIER();
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[i][0] = *(const int *)pa; af[i][1] = *(const int *)(pa + 4);
      }
      const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
      wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
    }
    CMP27E_LDS_BARRIER();
  };

  if (s_lo < s_hi) load(s_lo * DBK);
  for (int s = s_lo; s < s_hi; ++s) {
    store();
    if (s + 1 < s_hi) load((s + 1) * DBK);
    mma_slab();
  }

  const int n = n0 + wave * 16 + col;
  if (DKS == 1) {
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n] = (__bf16)(acc[i][e] * rf * As[m]);
        }
    }
    return;
  }

  if (DKS == 1) {
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n] = (__bf16)(acc[i][e] * rf * As[m]);
        }
    }
    return;
  }
  if (n < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = acc[i][e];
      }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[bx_], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[bx_] = 0;
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DNTHREADS) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)(sum * rf * As[m]);
    }
  }
}

#define CMP27E_PAIR(NAME, BK, TM, BAFIRST)                                                              \
  extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void NAME(                                       \
      const unsigned char *__restrict__ A, const float *__restrict__ As,                                 \
      const unsigned char *__restrict__ W0, const unsigned char *__restrict__ Ws0,                       \
      const unsigned char *__restrict__ Wref0, float *__restrict__ P0, int *__restrict__ cnt0,           \
      __bf16 *__restrict__ C0, const unsigned char *__restrict__ W1, const unsigned char *__restrict__ Ws1, \
      const unsigned char *__restrict__ Wref1, float *__restrict__ P1, int *__restrict__ cnt1,           \
      __bf16 *__restrict__ C1, int M, int K, int N0, int ks0, int nb0, int N1, int ks1, int nb1) {       \
    const int b = blockIdx.x;                                                                            \
    const int g = BAFIRST ? nb1 * ks1 : nb0 * ks0;                                                       \
    const bool first0 = !BAFIRST;                                                                        \
    const bool is0 = (b < g) == first0;                                                                  \
    const int bb = b < g ? b : b - g;                                                                    \
    if (is0)                                                                                             \
      cmp27e_decode_pf_impl<DEC_DWN, BK, TM, true>(A, W0, Ws0, Wref0, As, P0, cnt0, C0, M, N0, K, 1 << 30, \
                                                  1 << 30, 0L, ks0, bb % nb0, bb / nb0);                  \
    else                                                                                                 \
      cmp27e_decode_pf_impl<DEC_DWN, BK, TM, true>(A, W1, Ws1, Wref1, As, P1, cnt1, C1, M, N1, K, 1 << 30, \
                                                  1 << 30, 0L, ks1, bb % nb1, bb / nb1);                  \
  }
CMP27E_PAIR(mxfp4_gemm_decode_pair_pf_bk128_tm4_wperm1_nt1, 128, 4, true)

#define CMP27E_DEC(NAME, BK, KS, TM)                                                                    \
  extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void NAME(                                       \
      const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,                          \
      const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,                      \
      const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C, \
      int M, int N, int K, int pb1, int pb2, long astride) {                                             \
    cmp27e_decode_pf_impl<DEC_DWN, BK, TM, true>(A, W, Ws, Wref, As, P, cnt, C, M, N, K, pb1, pb2, astride, \
                                                KS, blockIdx.x, blockIdx.z);                             \
  }
CMP27E_DEC(mxfp4_gemm_decode_pf_bk128_ks2_tm4_wperm1_nt1, 128, 2, 4)
CMP27E_DEC(mxfp4_gemm_decode_pf_bk64_ks1_tm4_wperm1_nt1, 64, 1, 4)

// CMP27T (2026-10-28): the same register-prefetch body at the conc-4 `bk128 ks2 tm2` cell (out_proj /
// o_proj / qkv / down at 25..32 rows). Bound only under the `decode_pf_tm2` knob (an ARM, default
// off). Card-1 probe p2 (IC-cold, M 32, real records, us): out_proj 25.69 -> 24.41, down 48.94 ->
// 47.19, o_proj 25.25 -> 24.26, qkv 41.91 -> 42.78; the ks1 gate_up (82.85 -> 86.24) and the tm2 pair
// (54.10 -> 55.87) LOST and are not compiled.
CMP27E_DEC(mxfp4_gemm_decode_pf_bk128_ks2_tm2_wperm1_nt1, 128, 2, 2)

// CMP27X (2026-10-29): the same register-prefetch body with FEWER waves a workgroup (`DWN` 4 = 64
// columns a block, 2 = 32), so the N 5120 cells (out_proj / o_proj / down: 40 blocks x ks2 at
// DWN 8) launch 2x / 4x the workgroups. Each wave still owns its 16 columns and walks the split's
// K slabs in the same order; the split-K partials and the last block's fold are the same sums in the
// same order -> the DWN 8 cell's bytes. Bound only under `--engine-step-arms dwn4` (an ARM, default
// off); `_dwn2` is a probe candidate.
#define CMP27X_DECW(NAME, BK, KS, TM, DW)                                                               \
  extern "C" __global__ __launch_bounds__(DW * 32) void NAME(                                            \
      const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,                          \
      const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,                      \
      const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C, \
      int M, int N, int K, int pb1, int pb2, long astride) {                                             \
    cmp27e_decode_pf_impl<DW, BK, TM, true>(A, W, Ws, Wref, As, P, cnt, C, M, N, K, pb1, pb2, astride,   \
                                           KS, blockIdx.x, blockIdx.z);                                  \
  }
CMP27X_DECW(mxfp4_gemm_decode_pf_bk128_ks2_tm2_wperm1_nt1_dwn4, 128, 2, 2, 4)
CMP27X_DECW(mxfp4_gemm_decode_pf_bk128_ks2_tm4_wperm1_nt1_dwn4, 128, 2, 4, 4)
CMP27X_DECW(mxfp4_gemm_decode_pf_bk128_ks2_tm2_wperm1_nt1_dwn2, 128, 2, 2, 2)

// Private b1 experiment: packed metadata prefetch and the ks1 direct store.

CMP27E_DEC(mxfp4_gemm_decode_pf_bk128_ks1_tm1_wperm1_nt1, 128, 1, 1)

CMP27E_DEC(mxfp4_gemm_decode_pf_bk128_ks2_tm1_wperm1_nt1, 128, 2, 1)

CMP27E_DEC(mxfp4_gemm_decode_pf_bk128_ks4_tm1_wperm1_nt1, 128, 4, 1)
