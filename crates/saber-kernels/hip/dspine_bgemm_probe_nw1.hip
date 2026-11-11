// The drafter's bf16 projections: `kernel_projection` and the selector's `hidden_projection`.
//
// These are the two Linear layers the drafter's checkpoint lists in `modules_to_not_convert`, so
// they stay bf16 while everything around them is W8A8, and they are the largest single item the
// census left uncovered: `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16...` x10 at 38.06 us
// (380.6 us -- the conv's `kernel_projection`, twice per layer) plus `MT32...` x1 at 89.96 us (the
// selector's `hidden_projection`), 470 us and 2.4% of a decode step. The reference is hipBLASLt,
// so as with the reference's GEMM there is no source to port: this is written to its SHAPE and gated
// bit-exact against its output.
//
// WHAT THE REFERENCE'S OWN GEOMETRY SAYS, and why this kernel copies it. The traced launch is
// 2560 threads at **32 per workgroup** -- one WAVE per workgroup -- and MT16x16 macro tiles, so
// hipBLASLt puts one wave on each 16x16 output tile and walks all of K in it. That is also the
// only parallelism that can be bit-exact: splitting K across waves would re-associate the sum.
// So the tile is the reference's, and the ONE thing this kernel does differently is depth --
// eight k-steps of both operands are in flight before the first WMMA, where the reference's
// 345 GB/s (13.1 MB in 38 us) and 29 GB/s (2.6 MB in 90 us) say it is waiting on memory.
//
//   C[m][n] = sum over k of A[m][k] * W[n][k],  A [M, K] bf16, W [N, K] bf16 row-major
//             (`nn.Linear.weight` is [out, in], so this is the natural layout and no transpose)
// accumulated in ONE fp32 accumulator, k ascending, and rounded once to bf16 at the store.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));

#define DB_M     16    // the WMMA M tile, and the largest M one launch serves
#define DB_MB    32    // the M-TILED band: two m-fragments in one wave, one weight pass for both
#define DB_N     16    // one wave's output columns: the reference's macro tile
#define DB_NW     1    // waves per workgroup, so a workgroup covers DB_NW * DB_N columns
#define DB_NT   (DB_NW * 32)
#define DB_UNROLL 8    // k-steps in flight: 8 x 2 operands x 16 B = 256 B per lane

__device__ __forceinline__ floatx8 db_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// The gfx12 w32 layout, from this tree's own WMMA code: an A or B fragment holds row `l & 15`,
// k elements `(l >> 4) * 8 .. +8`; accumulator element `e` of lane `l` is row `(l >> 4) * 8 + e`,
// column `l & 15`.
extern "C" __global__ __launch_bounds__(DB_NT) void dflash_bgemm_bf16_m16n16(
    const __bf16 *__restrict__ a,          // [M, K]
    const __bf16 *__restrict__ w,          // [N, K]
    __bf16 *__restrict__ c,                // [M, N]
    int m, int n, int k) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int n0 = (int)(blockIdx.x * DB_NW + wave) * DB_N;
  if (n0 >= n) return;
  const int row = lane & 15;
  // Rows past M are clamped to the last real row rather than predicated -- a masked load on
  // gfx1201 is a full memory round trip (LEDGER, the predicated-staging finding) -- and their
  // results are simply not stored.
  const int arow = row < m ? row : m - 1;
  const int nrow = n0 + row < n ? n0 + row : n - 1;
  const __bf16 *ap = a + (long)arow * k + (lane >> 4) * 8;
  const __bf16 *wp = w + (long)nrow * k + (lane >> 4) * 8;

  floatx8 acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;

  const int ksteps = k / 16;
  int ks = 0;
  bf16x8 af[DB_UNROLL], wf[DB_UNROLL];
  for (; ks + DB_UNROLL <= ksteps; ks += DB_UNROLL) {
    // Every load of the group is issued before the first WMMA waits on one, which is the whole
    // difference from the reference here.
#pragma unroll
    for (int u = 0; u < DB_UNROLL; ++u) {
      af[u] = *(const bf16x8 *)(ap + (ks + u) * 16);
      wf[u] = *(const bf16x8 *)(wp + (ks + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < DB_UNROLL; ++u) acc = db_wmma(af[u], wf[u], acc);
  }
  for (; ks < ksteps; ++ks)
    acc = db_wmma(*(const bf16x8 *)(ap + ks * 16), *(const bf16x8 *)(wp + ks * 16), acc);

  const int col = n0 + (lane & 15);
  if (col >= n) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int r = (lane >> 4) * 8 + e;
    if (r < m) c[(long)r * n + col] = (__bf16)acc[e];
  }
}

// ---- the K-REDUCTION VARIANTS, for the N the envelope stopped at -------------------------------
//
// The rules' `[bgemm] max_n` was lowered to 2560 by a gate rather than raised by a source: at
// M = 8, K = 5120 this kernel is identical to `F.linear` at N = 1280 and N = 2560 and differs in
// 19 of 40960 at N = 5120, and the paired time crosses 1.0 at the same point -- which is what
// hipBLASLt switching algorithm looks like from the outside. The context store's fused KV
// projection is N = 10240 at TP=1 (`L * 2 * kv_size`, and kv_size is 1024 on one rank), so it sat
// outside the envelope and the plan declared it absent.
//
// A wider launch was never the question. The question is WHICH ORDER the reference reduces K in,
// and there is no source to read: these are the arms that answer it, each a different association
// of the SAME 5120 products, all bit-identical to the route wherever the association does not
// matter. `acc2` splits the unrolled group into two accumulators and adds them at the end (the
// shape a DepthU-32 double-buffered macro tile has); `acc4` splits it four ways and folds the four
// pairwise. One of them agrees with the reference at the wide N or none does, and the answer is a
// measurement rather than a preference.
//
// They are NOT routes: the rules name `dflash_bgemm_bf16_m16n16` and nothing else.
#define DB_VARIANT(NAME, NACC)                                                                     \
  extern "C" __global__ __launch_bounds__(DB_NT) void NAME(                                        \
      const __bf16 *__restrict__ a, const __bf16 *__restrict__ w, __bf16 *__restrict__ c,          \
      int m, int n, int k) {                                                                       \
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;                                    \
    const int n0 = (int)(blockIdx.x * DB_NW + wave) * DB_N;                                        \
    if (n0 >= n) return;                                                                           \
    const int row = lane & 15;                                                                     \
    const int arow = row < m ? row : m - 1;                                                        \
    const int nrow = n0 + row < n ? n0 + row : n - 1;                                              \
    const __bf16 *ap = a + (long)arow * k + (lane >> 4) * 8;                                       \
    const __bf16 *wp = w + (long)nrow * k + (lane >> 4) * 8;                                       \
    floatx8 acc[NACC];                                                                             \
    _Pragma("unroll") for (int t = 0; t < NACC; ++t) {                                             \
      _Pragma("unroll") for (int e = 0; e < 8; ++e) acc[t][e] = 0.f;                               \
    }                                                                                              \
    const int ksteps = k / 16;                                                                     \
    int ks = 0;                                                                                    \
    for (; ks + DB_UNROLL <= ksteps; ks += DB_UNROLL) {                                            \
      bf16x8 af[DB_UNROLL], wf[DB_UNROLL];                                                         \
      _Pragma("unroll") for (int u = 0; u < DB_UNROLL; ++u) {                                      \
        af[u] = *(const bf16x8 *)(ap + (ks + u) * 16);                                             \
        wf[u] = *(const bf16x8 *)(wp + (ks + u) * 16);                                             \
      }                                                                                            \
      _Pragma("unroll") for (int u = 0; u < DB_UNROLL; ++u)                                        \
          acc[u % NACC] = db_wmma(af[u], wf[u], acc[u % NACC]);                                    \
    }                                                                                              \
    for (; ks < ksteps; ++ks)                                                                      \
      acc[ks % NACC] = db_wmma(*(const bf16x8 *)(ap + ks * 16),                                    \
                               *(const bf16x8 *)(wp + ks * 16), acc[ks % NACC]);                   \
    for (int s = NACC / 2; s > 0; s >>= 1) {                                                       \
      for (int t = 0; t < s; ++t) {                                                                \
        _Pragma("unroll") for (int e = 0; e < 8; ++e) acc[t][e] += acc[t + s][e];                  \
      }                                                                                            \
    }                                                                                              \
    const int col = n0 + (lane & 15);                                                              \
    if (col >= n) return;                                                                          \
    _Pragma("unroll") for (int e = 0; e < 8; ++e) {                                                \
      const int r = (lane >> 4) * 8 + e;                                                           \
      if (r < m) c[(long)r * n + col] = (__bf16)acc[0][e];                                         \
    }                                                                                              \
  }

DB_VARIANT(dflash_bgemm_bf16_m16n16_acc2, 2)
DB_VARIANT(dflash_bgemm_bf16_m16n16_acc4, 4)

// ---- THE M-TILED INSTANTIATION, for a BATCHED proposal -----------------------------------------
//
// One proposal per request is what the M tile above costs: the fused context-KV projection runs at
// `reqs * block` context rows and stops at 16, so four sequences at prod's block of 8 is 32 rows
// and the committed rules REFUSE it (`docs/LEDGER.md`, workstreams S3 and S4's open item 6).
//
// The tile does not change. `blockIdx.y` picks which sixteen rows a workgroup owns and everything
// inside is the kernel above, unrolled over the same k-steps in the same order into the same one
// fp32 accumulator -- so **every output row's arithmetic is the m16 kernel's, element for element**,
// and the residual class of this kernel against `F.linear` is the one the envelope already carries:
// the bf16 GEMM's ACCUMULATION ORDER, which is hipBLASLt's algorithm choice per shape and not a
// function of M (each output element is an independent dot product over K; M enters only through
// which rows a workgroup reads). That is the claim the gate tests from both ends: bit-identical to
// `dflash_bgemm_bf16_m16n16` on each 16-row slice, and against `F.linear` at the new M.
//
// It is a SECOND SYMBOL rather than a widened first one so that the shape the whole proposal has
// been gated at keeps the code object it was gated with: the rules route `bgemm_bf16` to the m16
// kernel at M <= 16 and `bgemm_bf16_mt` here above it, and the m16 kernel still refuses a taller M
// by name.
// PARAMETERISED ON THE BAND, and the reason is a MEASUREMENT rather than symmetry with the head.
// At M = 32 the row slice (MROWS = 16, two workgroups on the y axis) costs 334.38 us against
// hipBLASLt's 184.56 -- 1.81x -- and the M tile is not what costs it: two launches of the m16
// symbol are 374.34, so the tile is already the cheaper of the two. What costs it is the WEIGHT
// STREAM. This kernel's parallelism is the reference's: one wave owns one 16x16 output tile and
// walks all of K, so a wave reads 16 whole weight rows, and at N = 10240, K = 5120 the launch reads
// 10240 x 5120 x 2 = 104.9 MB of bf16 weight -- 165 us at the card's measured 635 GB/s. One weight
// pass is therefore the floor, hipBLASLt is AT it (184.56 = 1.12x), and a y-axis row slice reads
// the weight ONCE PER M TILE: 209.7 MB at M = 32, whose own floor is 330 us. The row slice is at
// 334.38. It was never off its roofline; it was on the wrong one.
//
// MROWS = 32 puts the second m tile in the SAME wave, against the same B fragment: `acc[mf]`
// accumulates `A[m0 + mf*16 + row] . W[n]` over the same k-steps in the same ascending order into
// the same single fp32 accumulator, one WMMA per fragment per k-step, and the weight fragment is
// loaded once and used twice. Every output element's arithmetic is the 16-row symbol's, element for
// element -- which is what keeps the declared residual class (the bf16 GEMM's accumulation order
// against hipBLASLt's algorithm choice, 3 elements per row at N = 10240, linear in M with the same
// constant) exactly where it was, and what the gate checks per 16-row slice.
//
// The A side doubles in CACHE traffic and not in DRAM traffic: A is [M, K] bf16 = 327 KB at M = 32,
// which every wave reads out of L2 and no wave reads from memory twice.
// ---- THE WEIGHT LOADER, the one part the fp8 head swaps (workstream N4) ------------------------
//
// A loader stages one lane's weight fragment for k-step `ks` as its RAW bytes (`load`, issued for
// the whole unrolled group before the first WMMA, as above) and turns the raw bytes into the bf16
// fragment the WMMA reads (`expand`, just before the WMMA that consumes it). The bf16 loader's
// expand is the identity, so the bf16 bands are the code they were; the fp8 loader is the only
// other one.
struct DbWBf16 {
  typedef bf16x8 raw_t;
  const __bf16 *p;
  __device__ __forceinline__ DbWBf16(const void *w, const float *, long nrow, int k, int lane)
      : p((const __bf16 *)w + nrow * k + (lane >> 4) * 8) {}
  __device__ __forceinline__ raw_t load(int ks) const { return *(const bf16x8 *)(p + ks * 16); }
  __device__ __forceinline__ bf16x8 expand(raw_t r) const { return r; }
};

// THE CHECKPOINT'S OWN e4m3 `lm_head`, dequantized IN REGISTER exactly as prod's loader does it on
// the host (`ref_nvfp4.RefFp8ChannelToBf16`): `bf16(f32(w8) * f32(scale))`, one multiply
// and one rounding per element. The multiply is EXACT -- an e4m3 value has 4 significant bits and
// the checkpoint's bf16 per-channel scale 8, so the product has at most 12 and fits f32's 24 --
// which leaves the bf16 round-to-nearest-even as the only rounding, the same one torch's
// `.to(torch.bfloat16)` does. So the fragment the WMMA reads is the bf16 head's record BYTE FOR
// BYTE, and the GEMM sees the same operands in the same k order as the bf16 band: ids and logits
// are claimed bit-identical to the bf16 head, not close to it.
//
// `w` is [N, K] e4m3 row-major (the checkpoint's tensor, dim-0 sharded, not reordered), `ws` is the
// per-output-row scale as f32 (prod's own parameter dtype; an exact widening of the checkpoint's
// bf16). A lane's fragment is 8 consecutive k bytes -- the same k elements the bf16 loader reads,
// at half the bytes.
__device__ __forceinline__ bf16x8 db_w8_expand(uint2 r, float s) {
  typedef float floatx2 __attribute__((ext_vector_type(2)));
  const floatx2 f0 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, false);
  const floatx2 f1 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, true);
  const floatx2 f2 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, false);
  const floatx2 f3 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, true);
  bf16x8 o;
  o[0] = (__bf16)(f0[0] * s);
  o[1] = (__bf16)(f0[1] * s);
  o[2] = (__bf16)(f1[0] * s);
  o[3] = (__bf16)(f1[1] * s);
  o[4] = (__bf16)(f2[0] * s);
  o[5] = (__bf16)(f2[1] * s);
  o[6] = (__bf16)(f3[0] * s);
  o[7] = (__bf16)(f3[1] * s);
  return o;
}

// The same expansion with the bf16 rounding written out as the integer round-to-nearest-even
// (`bits + 0x7fff + lsb`, keep the high half) and two results packed per `v_perm`, instead of the
// compiler's `(__bf16)` lowering, which also tests every product for NaN and selects a quiet NaN.
// For every FINITE product the two are the same function; a NaN product needs an e4m3 NaN code
// (0x7f / 0xff) in the weight, and the whole-head expand gate is what says the checkpoint has none.
// It is the VALU budget that asks: at M = 1 the WMMA is one instruction per k-step and the
// expansion is eight elements' worth of it.
__device__ __forceinline__ unsigned db_bf16_rne_pair(float lo, float hi) {
  unsigned a = __float_as_uint(lo), b = __float_as_uint(hi);
  a += 0x7fffu + ((a >> 16) & 1u);
  b += 0x7fffu + ((b >> 16) & 1u);
  return __builtin_amdgcn_perm(b, a, 0x07060302u);
}

__device__ __forceinline__ bf16x8 db_w8_expand_rne(uint2 r, float s) {
  typedef float floatx2 __attribute__((ext_vector_type(2)));
  const floatx2 f0 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, false);
  const floatx2 f1 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, true);
  const floatx2 f2 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, false);
  const floatx2 f3 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, true);
  union { uint4 u; bf16x8 h; } o;
  o.u.x = db_bf16_rne_pair(f0[0] * s, f0[1] * s);
  o.u.y = db_bf16_rne_pair(f1[0] * s, f1[1] * s);
  o.u.z = db_bf16_rne_pair(f2[0] * s, f2[1] * s);
  o.u.w = db_bf16_rne_pair(f3[0] * s, f3[1] * s);
  return o.h;
}

template <bool RNE>
struct DbWFp8Channel {
  typedef uint2 raw_t;
  const unsigned char *p;
  float s;
  __device__ __forceinline__ DbWFp8Channel(const void *w, const float *ws, long nrow, int k,
                                           int lane)
      : p((const unsigned char *)w + nrow * k + (lane >> 4) * 8), s(ws[nrow]) {}
  __device__ __forceinline__ raw_t load(int ks) const { return *(const uint2 *)(p + ks * 16); }
  __device__ __forceinline__ bf16x8 expand(raw_t r) const {
    return RNE ? db_w8_expand_rne(r, s) : db_w8_expand(r, s);
  }
};

template <int MROWS, class WL = DbWBf16, int UNROLL = DB_UNROLL>
__device__ __forceinline__ void db_mtile(const __bf16 *__restrict__ a, const void *__restrict__ w,
                                         const float *__restrict__ ws, __bf16 *__restrict__ c,
                                         int m, int n, int k) {
  constexpr int MF = MROWS / DB_M;   // WMMA m-fragments one wave holds: 1 at the row slice, 2
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int n0 = (int)(blockIdx.x * DB_NW + wave) * DB_N;
  if (n0 >= n) return;
  const int m0 = (int)blockIdx.y * MROWS;
  if (m0 >= m) return;
  const int row = lane & 15;
  // Clamped, never predicated, and clamped WITHIN this band: a masked load on gfx1201 is a full
  // memory round trip (LEDGER, the predicated-staging finding). The clamp target is the last real
  // row of the matrix, exactly as in the kernel above -- the rows it produces are simply not
  // stored.
  const int nrow = n0 + row < n ? n0 + row : n - 1;
  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
    const int ar = m0 + mf * DB_M + row;
    ap[mf] = a + (long)(ar < m ? ar : m - 1) * k + (lane >> 4) * 8;
  }
  const WL wl(w, ws, (long)nrow, k, lane);

  floatx8 acc[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[mf][e] = 0.f;
  }

  const int ksteps = k / 16;
  int ks = 0;
  bf16x8 af[MF][UNROLL];
  typename WL::raw_t wf[UNROLL];
  for (; ks + UNROLL <= ksteps; ks += UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[mf][u] = *(const bf16x8 *)(ap[mf] + (ks + u) * 16);
      wf[u] = wl.load(ks + u);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const bf16x8 wv = wl.expand(wf[u]);
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = db_wmma(af[mf][u], wv, acc[mf]);
    }
  }
  for (; ks < ksteps; ++ks) {
    const bf16x8 wv = wl.expand(wl.load(ks));
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
      acc[mf] = db_wmma(*(const bf16x8 *)(ap[mf] + ks * 16), wv, acc[mf]);
  }

  const int col = n0 + (lane & 15);
  if (col >= n) return;
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + mf * DB_M + (lane >> 4) * 8 + e;
      if (r < m) c[(long)r * n + col] = (__bf16)acc[mf][e];
    }
  }
}

#define DB_MTILE_ENTRY(name, MROWS)                                                                \
  extern "C" __global__ __launch_bounds__(DB_NT) void name(                                        \
      const __bf16 *__restrict__ a, const __bf16 *__restrict__ w, __bf16 *__restrict__ c, int m,    \
      int n, int k) {                                                                              \
    db_mtile<MROWS>(a, w, nullptr, c, m, n, k);                                                    \
  }

// The ROW SLICE: one workgroup per 16 rows, which is the tile the m16 symbol serves and therefore
// the fallback for an M that is a whole number of 16-row tiles but not of 32.
DB_MTILE_ENTRY(dflash_bgemm_bf16_mtn16, DB_M)
// ... and the 32-ROW BAND, which is what the rules pick above 16: one weight pass for two m tiles.
DB_MTILE_ENTRY(dflash_bgemm_bf16_m32n16, DB_MB)

// ---- THE fp8-WEIGHT HEAD, W8A16 (workstream N4) ------------------------------------------------
//
// The target's `lm_head` from the checkpoint's own e4m3 tensor and its per-channel scale (0.636 GB
// per rank at TP=2, half the bf16 head's 1.271), expanded in register to the bf16 head's bytes and
// then the bands above unchanged: the same tile, the same k-steps in the same ascending order into
// the same single fp32 accumulator, the same bf16 store. The m16 symbol is the ROW SLICE template at
// one band (at M <= 16 its grid is the m16 kernel's and its `m0` is 0), and the m32 symbol is the
// 32-row band, so every output element's arithmetic is the bf16 head's whenever the expanded
// fragment is -- which is what the gate measures, byte for byte and then element for element.
//
// Two things are template parameters, and both are TIME questions with one arithmetic answer.
// The k-steps in flight, because the fp8 fragment is HALF the bytes: at the bf16 bands' 8 a lane
// has 64 B of weight outstanding instead of 128, and the `_u16` instantiations restore the bytes
// at the cost of the registers the A side of the band needs. And the bf16 rounding's lowering
// (`_rne`: the integer round-to-nearest-even, identical for every finite product, see
// `db_bf16_rne_pair`). The rules name whichever instantiation the paired time picked; the others
// are gate arms, not routes, and every one of them is gated bit-identical to the bf16 band.
#define DB_W8_ENTRY(name, MROWS, RNE, UNROLL)                                                      \
  extern "C" __global__ __launch_bounds__(DB_NT) void name(                                        \
      const __bf16 *__restrict__ a, const unsigned char *__restrict__ w,                           \
      const float *__restrict__ ws, __bf16 *__restrict__ c, int m, int n, int k) {                 \
    db_mtile<MROWS, DbWFp8Channel<RNE>, UNROLL>(a, w, ws, c, m, n, k);                             \
  }

DB_W8_ENTRY(dflash_bgemm_w8_m16n16, DB_M, false, DB_UNROLL)
DB_W8_ENTRY(dflash_bgemm_w8_m32n16, DB_MB, false, DB_UNROLL)
DB_W8_ENTRY(dflash_bgemm_w8_m16n16_u16, DB_M, false, 16)
DB_W8_ENTRY(dflash_bgemm_w8_m32n16_u16, DB_MB, false, 16)
DB_W8_ENTRY(dflash_bgemm_w8_m16n16_rne, DB_M, true, DB_UNROLL)
DB_W8_ENTRY(dflash_bgemm_w8_m32n16_rne, DB_MB, true, DB_UNROLL)
DB_W8_ENTRY(dflash_bgemm_w8_m16n16_rne_u16, DB_M, true, 16)
DB_W8_ENTRY(dflash_bgemm_w8_m32n16_rne_u16, DB_MB, true, 16)

// THE WIDE HEAD BANDS (workstream H33). Above 32 rows the 32-row band runs twice and streams the
// 0.636 GB head twice: at the served b8s5 (40 rows) that is 2,292 us against the tile's 1,130 for
// one pass. `db_mtile<48>` / `<64>` hold three / four m-fragments against ONE expanded weight
// fragment and read the weight once: the same k-steps in the same ascending order into one fp32
// accumulator per fragment, the same bf16 store, so every output element is the 16-row symbol's
// and the gate is that, per 16-row slice (0 differing elements at M = 8..64 on both ranks).
//
// Measured (card 1, 210 W, 2026-09-22/23, two holds): the 48-row band is 1,426 / 1,487 us at
// M = 35 / 40 (0.62x / 0.65x the 32-row band's two passes) and is the rules' `bgemm_w8_head_wide`.
// It is NOT a weight-bound kernel: a one-pass band's time grows with the number of DISTINCT A rows
// (the 64-row band: 1,493 us at M = 32, 2,169 at M = 64, the same instruction stream -- a clamped
// row is one address across sixteen lanes), so the 64-row band at M = 64 reads the weight at
// 293 GB/s and saves only 7 % over two 32-row passes; it is compiled in and gated but not routed.
// Rejected in the same holds (LEDGER H33): the weight group prefetched one ahead (ping-pong
// registers, 1.02-1.36x slower), 4 k-steps in flight (1.05-1.36x), and N-wide waves (two or four
// 16-column weight fragments per wave against one A fragment: 2-5x slower).
DB_W8_ENTRY(dflash_bgemm_w8_m48n16_rne, 48, true, DB_UNROLL)
DB_W8_ENTRY(dflash_bgemm_w8_m64n16_rne, 64, true, DB_UNROLL)

// THE fp8 HEAD'S 16-BYTE WEIGHT LOAD (workstream H40's lab, moved into the plugin by H41). The
// served loader (`DbWFp8Channel`) gives each lane ONE 8-byte half of its row's 16-byte k-step: the
// low half-wave (lanes 0-15) bytes 0-7, the high half-wave bytes 8-15. Here each lane issues ONE
// 16-byte load covering a WHOLE k-step of its row -- the low half-wave loads k-step 2j, the high
// half-wave k-step 2j+1 -- and `v_permlanex16` swaps the 8-byte halves each lane does not own
// with the lane sixteen away (same row). So every WMMA sees the served fragment byte for byte, in
// the served ascending k order into the same single fp32 accumulator per m-fragment, the same
// integer RNE expansion and the same bf16 store: the output is the served symbol's element for
// element, which is what the gate measures (0 differing elements at every M, both ranks). The
// k-steps a pair of halves cannot cover (K/16 odd past the unrolled loop) take the served 8-byte
// load. `NT` is the cache policy of the 16-byte load only (same bytes).
//
// Measured (H40 lab, card 1, 210 W, 2026-09-23, rank 0's N = 124,160, us per launch, median of 30,
// served -> this): M = 1 1,088 -> 1,050 (u16), M = 8 1,130 -> 1,080, M = 16 1,153 -> 1,106,
// M = 32 1,313 -> 1,109 (u16, NT), M = 48 1,731 -> 1,200 (u8): 606 / 589 / 575 / 574 / 530 GB/s.
typedef unsigned db_u4v __attribute__((ext_vector_type(4)));
template <bool NT>
__device__ __forceinline__ uint4 db_ld16(const void *p) {
  db_u4v v;
  if constexpr (NT) v = __builtin_nontemporal_load((const db_u4v *)p);
  else v = *(const db_u4v *)p;
  return make_uint4(v.x, v.y, v.z, v.w);
}

template <int MROWS, bool NT, int UNROLL>
__device__ __forceinline__ void db_mtile_w8_w128(const __bf16 *__restrict__ a,
                                                 const unsigned char *__restrict__ w,
                                                 const float *__restrict__ ws,
                                                 __bf16 *__restrict__ c, int m, int n, int k) {
  static_assert(UNROLL % 2 == 0, "one 16-byte load covers two k-steps");
  constexpr int MF = MROWS / DB_M;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int n0 = (int)(blockIdx.x * DB_NW + wave) * DB_N;
  if (n0 >= n) return;
  const int m0 = (int)blockIdx.y * MROWS;
  if (m0 >= m) return;
  const int row = lane & 15, h = lane >> 4;
  const int nrow = n0 + row < n ? n0 + row : n - 1;
  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
    const int ar = m0 + mf * DB_M + row;
    ap[mf] = a + (long)(ar < m ? ar : m - 1) * k + h * 8;
  }
  const unsigned char *wrow = w + (long)nrow * k;
  const unsigned char *wp = wrow + h * 16;
  const float s = ws[nrow];
  floatx8 acc[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[mf][e] = 0.f;
  const int ksteps = k / 16;
  constexpr int UP = UNROLL / 2;
  int ks = 0;
  for (; ks + UNROLL <= ksteps; ks += UNROLL) {
    uint4 wr[UP];
    bf16x8 af[MF][UNROLL];
#pragma unroll
    for (int u = 0; u < UP; ++u) wr[u] = db_ld16<NT>(wp + (ks + 2 * u) * 16);
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[mf][u] = *(const bf16x8 *)(ap[mf] + (ks + u) * 16);
#pragma unroll
    for (int u = 0; u < UP; ++u) {
      const unsigned sx = h ? wr[u].x : wr[u].z, sy = h ? wr[u].y : wr[u].w;
      const unsigned rx =
          __builtin_amdgcn_permlanex16(sx, sx, 0x76543210u, 0xfedcba98u, false, false);
      const unsigned ry =
          __builtin_amdgcn_permlanex16(sy, sy, 0x76543210u, 0xfedcba98u, false, false);
      const uint2 e0 = h ? make_uint2(rx, ry) : make_uint2(wr[u].x, wr[u].y);
      const uint2 e1 = h ? make_uint2(wr[u].z, wr[u].w) : make_uint2(rx, ry);
      const bf16x8 w0 = db_w8_expand_rne(e0, s), w1 = db_w8_expand_rne(e1, s);
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = db_wmma(af[mf][2 * u], w0, acc[mf]);
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = db_wmma(af[mf][2 * u + 1], w1, acc[mf]);
    }
  }
  for (; ks < ksteps; ++ks) {
    const uint2 r = *(const uint2 *)(wrow + h * 8 + ks * 16);
    const bf16x8 wv = db_w8_expand_rne(r, s);
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
      acc[mf] = db_wmma(*(const bf16x8 *)(ap[mf] + ks * 16), wv, acc[mf]);
  }
  const int col = n0 + (lane & 15);
  if (col >= n) return;
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + mf * DB_M + (lane >> 4) * 8 + e;
      if (r < m) c[(long)r * n + col] = (__bf16)acc[mf][e];
    }
}

#define DB_W8_W128_ENTRY(name, MROWS, NT, UNROLL)                                                  \
  extern "C" __global__ __launch_bounds__(DB_NT) void name(                                        \
      const __bf16 *__restrict__ a, const unsigned char *__restrict__ w,                           \
      const float *__restrict__ ws, __bf16 *__restrict__ c, int m, int n, int k) {                 \
    db_mtile_w8_w128<MROWS, NT, UNROLL>(a, w, ws, c, m, n, k);                                     \
  }

// The three routed bands (the lab's exact winning arms): the tile at 16 k-steps in flight, the
// 32-row band at 16 with the non-temporal 16-byte load, the 48-row wide band at 8.
DB_W8_W128_ENTRY(dflash_bgemm_w8_m16n16_w128_u16, DB_M, false, 16)
DB_W8_W128_ENTRY(dflash_bgemm_w8_m32n16_w128_u16_nt, DB_MB, true, 16)
DB_W8_W128_ENTRY(dflash_bgemm_w8_m48n16_w128_u8, 48, false, DB_UNROLL)

// GATE-ONLY, not a route: the fp8 loader's own `expand` over a whole [N, K] matrix, written out
// as bf16 so the gate can byte-compare it against the bf16 head's pack record. One thread per 8
// elements (one lane fragment). `mode` 0 is `db_w8_expand`, `mode` 2 is `db_w8_expand_rne`, and
// `mode` 1 is the NEGATIVE CONTROL -- the same product rounded
// through f16 before bf16, the way a loader that staged in half precision would -- which must
// differ from the record where the case must not.
extern "C" __global__ __launch_bounds__(256) void dflash_w8_expand_gate(
    const unsigned char *__restrict__ w, const float *__restrict__ ws, __bf16 *__restrict__ out,
    int n, int k, int mode) {
  const long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long groups = (long)n * (k / 8);
  if (g >= groups) return;
  const long row = g / (k / 8);
  const uint2 r = ((const uint2 *)w)[g];
  bf16x8 o = mode == 2 ? db_w8_expand_rne(r, ws[row]) : db_w8_expand(r, ws[row]);
  if (mode == 1) {
    typedef float floatx2 __attribute__((ext_vector_type(2)));
    const float s = ws[row];
    const floatx2 f[4] = {__builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, false),
                          __builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, true),
                          __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, false),
                          __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, true)};
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = (__bf16)(float)(_Float16)(f[j >> 1][j & 1] * s);
  }
  ((bf16x8 *)out)[g] = o;
}
