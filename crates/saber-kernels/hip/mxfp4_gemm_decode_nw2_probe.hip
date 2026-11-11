// Saber source (CLEANROOM-1, 2026-10-25: frozen from a retired generator's output; edit this file directly)
//   reference sha256 79a7eb6f261c2939386a4c231cf5727a0a1f2325d8a391419178778da1fd2ae2
//   device code sliced VERBATIM by line range; only the __global__ signatures became
//   __device__ __forceinline__ *_impl and the extern "C" wrappers at the end are new.
// Do not edit by hand: re-run the generator. No host code, no pybind, no getenv in this file --
// every tile parameter the prod launcher chose from an env var is an explicit symbol here.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

#define DEC_DWN 2   // waves per decode block; the N tile is DEC_DWN*16 = 128 wide

// ---------------------------------------------------------------- decode path (M <= 16)
// The tiles above are sized for prefill: BMF=256 via TM=4. At decode M is 5 (batch 1 x SPEC+1), so
// per wave the folded kernel issues K/DBK * DBK/16 * TM * TN = 4352 WMMA against 5 real rows -- 51x
// more MACs than useful, a ~70 us matrix-issue floor on a shape whose DRAM floor is 41 us. TM=1
// cuts issued MACs 16x and shrinks the A tile from 256xBK to 16xBK at the same time.
//
// Split-K is not optional here: with TM=1 the independent N-fragments number N/16, so N=5120 gives
// only 320 waves = 2.5 per SIMD no matter how the N tile is chosen. Measured, DKS=1 is 2.4x worse
// than DKS=4 on mlp.down. Partials are fp32 [DKS][M][N]; the folded scale composes for free because
// Wref[n] is a whole-row max over k, so both per-row factors are applied once in the reduce.
//
// DBK=128 here, which REVERSES the prefill answer (where it measured -34%). That loss was pure LDS
// occupancy: 22.5 -> 43.5 KB took resident blocks 2 -> 1. With a 16-row A tile, LDS is
// (16+BND)*(DBK+PAD) = 10.88 KB, so 5-6 blocks still fit. Measured 1.87x on gate_up. See
// ~/mxfp4_work/tier5/RESULTS.md.
#define DEC_MTILE 16
#define DEC_PAD 8
static __device__ __forceinline__ unsigned int wq_dec(uint2_t v, int q) { return q ? v[1] : v[0]; }
// Nontemporal (streaming) weight loads, `global_load ... th:TH_LOAD_NT` on gfx1201. The decode
// weight stream is read exactly once per step, so a streaming hint keeps it from displacing the
// split-K partials and A in L2/MALL. Applied to the W loads ONLY: Ws/Wref are re-read per slab and
// the partials must stay cached for the fused reduce. SABER_MXFP4_DECODE_NT=1 selects it.
template <bool NT, typename T>
static __device__ __forceinline__ T ld_w(const T *p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}


// DTM = number of 16-row M-fragments; the launcher picks ceil(M/16). The reference server captures decode at
// num_seqs*(SPEC+1): 5/10/20/25/35/40 under mtp at SPEC=4, and batch x 8 under a dflash drafter
// at SPEC=7, which reaches 64 at max_num_seqs=8. DTM 1..4 covers every reachable batch of both.
// Use the SMALLEST DTM that covers M: measured at M=20, DTM=2 beats DTM=3 by 12% (89.1 vs 99.9 us
// on gate_up) purely because the wider tile computes rows nobody asked for.
template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool DIRECT = true, bool NT = false>
__device__ __forceinline__ void saber_mxfp4_fp8_gemm_decode_impl(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) {
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  // Merged-linear partition select (paro: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
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

  // ---- per-slab pieces, shared by the three slab loops below ----
  auto stage_a = [&](int k0) {
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;                  // clamp, never predicate
        *(uint4_t *)(&sA[r * DASTR + c]) = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);
      }
    }
  };
  // Row-order (checkpoint layout) W staging from a packed 8-byte word per (iteration, thread).
  // getw(it, gc, c) supplies the word so the plain and line-group loops share the fold.
  constexpr int NITW = BND * DBK / 2 / (DNTHREADS * 8);
  auto stage_w_row = [&](int k0, auto &&getw) {
#pragma unroll
    for (int it = 0; it < NITW; ++it) {
      const int idx = it * DNTHREADS * 8 + tid * 8;
      const int r = idx / (DBK / 2), c = idx % (DBK / 2), gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;                  // clamp, never predicate
      const unsigned long long packed = getw(it, gc, c);
      const int blk = (k0 + c * 2) / 32;
      int dsh = (int)Wref[gc] - (int)Ws[(size_t)blk * N + gc];
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int wv = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = wv & 0x0F0F0F0Fu;
        const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * DWSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
  };
  auto stage_w_wperm = [&](int k0) {
    if constexpr (WPERM) {
      // Fragment order, as in the folded kernel: consecutive threads take consecutive uint32, so
      // a wave's weight read is 128 contiguous bytes instead of sixteen rows K/2 bytes apart.
      // TWO lane slots per thread (an 8 B request), the width the int4 twin measured optimal
      // (4 B was 11% slower there; 16 B identical). Adjacent slots are adjacent rows at the same
      // k, exactly as in the folded kernel above.
      constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < NTILES_T * KSTEPS_T * 32; off += DNTHREADS * 2) {
        const int sl = off + tid * 2;                   // first of the pair; always even
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // Clamp, never predicate; gn is always even here and N % 16 == 0 under WPERM, so
        // clamping to the last even row keeps the uint2 read 8-byte aligned and gc + 1 in
        // bounds; the fragment address is recomputed from the clamped row.
        const int gc = gn < N - 2 ? gn : N - 2;
        const int lanec = (lane_ & 16) | (gc & 15);
        const uint2_t wv = ld_w<NT>(
            (const uint2_t *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
        const int blk = (k0 + kst * 16) / 32;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq_dec(wv, q) & 0x0F0F0F0Fu;
          const unsigned int od = (wq_dec(wv, q) >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(r + q) * DWSTR + kloc]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    }
  };
  auto mma_slab = [&]() {
    __syncthreads();
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
    __syncthreads();
  };

  if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_row(k0, [&](int, int gc, int c) {
        return ld_w<NT>((const unsigned long long *)(W + ((size_t)gc * K + k0) / 2 + c));
      });
      mma_slab();
    }
  }

  const int n = n0 + wave * 16 + col;

  // At DKS==1 the accumulator already holds the whole K range, so the partial buffer, the
  // threadfence, the atomic and the reduction pass are all pure overhead -- and that overhead is
  // not small. Partial traffic is DKS*M*N floats: constant in the weight stream but LINEAR IN M,
  // 11.1 MB against gate_up's 44.6 MB of weights at M=40 and 17.8 MB at M=64. Picking DKS by shape
  // and by M is the launcher's job (split_k_for below); this is the path that makes DKS==1 free.
  // Bit-identical to the DKS==1 reduction path, which summed a single term -- the DIRECT=false
  // instantiation exists so that equivalence can be tested rather than asserted.
  if constexpr (DKS == 1 && DIRECT) {
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

  // Fused reduction. The DKS blocks covering one n-range race on an atomic counter and whoever
  // arrives last reduces in place, so there is no second kernel launch -- 13,376 of them per
  // decode run in the profile, at a 4.5 us mean host gap. It is also faster in pure kernel time
  // (-5.5% on gate_up to -29% on gdn.out) because the partials are still in cache when the
  // reduction reads them. The last block resets the counter, leaving it zeroed for the next
  // launch; launches on one stream are serialised, so no separate clearing pass is needed.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;

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


extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void mxfp4_gemm_decode_bk128_ks1_tm1_wperm1_nt1(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_decode_impl<DEC_DWN, 128, 1, 1, true,
                                      true, true>(
      A, W, Ws, Wref, As, P, cnt, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void mxfp4_gemm_decode_bk128_ks2_tm1_wperm1_nt1(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_decode_impl<DEC_DWN, 128, 2, 1, true,
                                      true, true>(
      A, W, Ws, Wref, As, P, cnt, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_decode_impl<DEC_DWN, 128, 4, 1, true,
                                      true, true>(
      A, W, Ws, Wref, As, P, cnt, C, M, N, K, pb1, pb2, astride);
}
