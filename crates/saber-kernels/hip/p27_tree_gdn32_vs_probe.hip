// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the WY tree verify of the 27B (spec 2.9) and the
// advance that applies an accepted row's update later (spec 2.9.1).
//
// Written from the CR2-3a specification. A workgroup takes one (sequence, v head[, v slice]) with the decode
// recurrence's thread map (two threads a v row, 64 k each). From the committed state, every row r of a tree of
// up to TM rows is answered at once in the WY form over the row's ancestor set (bit c of anc[r]):
//   u_r  = b_r (v_r - e^{P_r} k_r.h) - sum_{c proper ancestor} A[r][c] u_c
//   on_r = e^{P_r} q_r.h + sum_{c ancestor or self} M[r][c] u_c
// with P_r the cumulated gate along the path, M[r][c] = e^{P_r - P_c} q_r.k_c, A[r][c] = b_r e^{P_r - P_c} k_r.k_c.
// The step's record (U, normalised k, P, ancestor words) lets the advance apply the accepted row later.
#include "sgdn_rec.h"


namespace sgdn {

// record layout (floats), TM rows
template <int TM> struct TreeRec {
  static constexpr int U = 0;
  static constexpr int K = TM * kDim;
  static constexpr int P = 2 * TM * kDim;
  static constexpr int A = 2 * TM * kDim + TM;  // P[32] then ancestor u32[32]
};

// h <- the state after accepting row acc of the record (no-op, false, when acc is outside [0, TM))
template <int TM>
__device__ __forceinline__ bool tree_advance(float* h, const float* __restrict__ rec, int acc, int row, int k0) {
  if (acc < 0 || acc >= TM) return false;
  const float pa = rec[TreeRec<TM>::P + acc];
  const u32 wa = (u32)((const int*)rec)[TreeRec<TM>::A + acc];
  const float ea = E(pa);
#pragma unroll
  for (int j = 0; j < kHalf; ++j) h[j] = h[j] * ea;
#pragma unroll 1
  for (int c = 0; c <= acc; ++c) {
    if (!((wa >> c) & 1u)) continue;
    const float coef = E(pa - rec[TreeRec<TM>::P + c]) * rec[TreeRec<TM>::U + c * kDim + row];
    const float* kc = rec + TreeRec<TM>::K + c * kDim + k0;
#pragma unroll
    for (int j = 0; j < kHalf; j += 4) {
      const float4 w = *(const float4*)(kc + j);
      h[j] = h[j] + coef * w.x;
      h[j + 1] = h[j + 1] + coef * w.y;
      h[j + 2] = h[j + 2] + coef * w.z;
      h[j + 3] = h[j + 3] + coef * w.w;
    }
  }
  return true;
}

// the standalone advance: load the committed state, apply, store at slot_out if it moved or the slots differ
template <int TM>
__device__ __forceinline__ void tree_advance_kernel(u16* __restrict__ state, long st_slot, long st_head,
                                                    const int* __restrict__ slot, const int* __restrict__ slot_out,
                                                    const float* __restrict__ ts, long ts_seq, long ts_head,
                                                    const int* __restrict__ acc, int n, int hv) {
  const int si = slot[n], so = slot_out[n];
  if (si <= 0 || so <= 0) return;
  const int tid = threadIdx.x, row = tid >> 1, k0 = (tid & 1) * kHalf;
  const long off = (long)hv * st_head + (long)row * kDim + k0;
  float h[kHalf];
  state_load<kStateF16, kHalf>(h, state + (long)si * st_slot + off);
  const bool moved = tree_advance<TM>(h, ts + (long)n * ts_seq + (long)hv * ts_head, acc[n], row, k0);
  if (moved || so != si) state_store<kStateF16, kHalf>(state + (long)so * st_slot + off, h);
}

struct TreeArgs {
  const u16* q; const u16* k; const u16* v; const u16* av; const u16* bv; long ab_stride; const float* a_log;
  const float* dt_bias; u16* state; long st_slot, st_head; const int* slot; u16* o; const int* cu;
  const u16* anc; const float* ts_in; float* ts_out; long ts_seq, ts_head; const int* acc; const u16* zg;
  const float* nw; float neps; int nact; int H, Hg; float scale, sp_thr;
};

template <int TM> struct TreeLdsBase {
  float q[TM][kDim], k[TM][kDim];
  float M[TM][TM], A[TM][TM];
  float g[TM], b[TM], eP[TM], P[TM];
  int anc[TM];
};
template <int TM, bool NORM> struct TreeLds : TreeLdsBase<TM> {
  float wsum[8];   // the row norm's per-wave sums
};
template <int TM> struct TreeLds<TM, false> : TreeLdsBase<TM> {};   // no norm, no per-wave sums

template <int TM, int VS, bool NORM, bool ADV>
__device__ __forceinline__ void tree_update(const TreeArgs& a) {
  constexpr int NR = kDim / VS;
  constexpr int NT = 2 * NR;
  constexpr int NW = NT / 32;
  __shared__ __attribute__((aligned(16))) TreeLds<TM, NORM> L;
  const int n = blockIdx.x, hv = blockIdx.y, z = (VS > 1) ? (int)blockIdx.z : 0;
  const int bos = a.cu[n];
  const int T = min(a.cu[n + 1] - bos, TM);
  if (T <= 0) return;
  const int si = a.slot[n];
  if (si <= 0) return;
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);   // wave-uniform: the per-wave loops run on the scalar unit
  const int row = (tid >> 1) + z * NR, half = tid & 1, k0 = half * kHalf;
  const int hk = hv / (a.H / a.Hg);

  // ---- 1. the committed state, the pending advance applied first (ADV)
  u16* sp = a.state + (long)si * a.st_slot + (long)hv * a.st_head + (long)row * kDim + k0;
  float h[kHalf];
  state_load<kStateF16, kHalf>(h, sp);
  if constexpr (ADV) {
    const float* rec = a.ts_in + (long)n * a.ts_seq + (long)hv * a.ts_head;
    if (tree_advance<TM>(h, rec, a.acc[n], row, k0)) {
      state_store<kStateF16, kHalf>(sp, h);
      round_through_f16<kHalf>(h);
    }
  }

  // ---- 2. raw q / k rows into LDS; the gates and the ancestor words
  {
    // every load first, then the LDS stores (one global round trip, whatever the slice count)
    // (in passes of at most three items a thread: the registers stay within the budget at every slice count)
    constexpr int ITEMS = (TM * kDim / 4 + NT - 1) / NT;
    constexpr int PASS = ITEMS < 3 ? ITEMS : 3;
#pragma unroll
    for (int j0 = 0; j0 < ITEMS; j0 += PASS) {
      u32x2 qw[PASS], kw[PASS];
#pragma unroll
      for (int j = 0; j < PASS; ++j) {
        const int i = tid + (j0 + j) * NT;
        if (j0 + j < ITEMS && i < T * kDim / 4) {
          const int r = i / (kDim / 4), c4 = (i % (kDim / 4)) * 4;
          const long src = ((long)(bos + r) * a.Hg + hk) * kDim + c4;
          qw[j] = *(const u32x2*)(a.q + src);
          kw[j] = *(const u32x2*)(a.k + src);
        }
      }
#pragma unroll
      for (int j = 0; j < PASS; ++j) {
        const int i = tid + (j0 + j) * NT;
        if (j0 + j < ITEMS && i < T * kDim / 4) {
          const int r = i / (kDim / 4), c4 = (i % (kDim / 4)) * 4;
          *(float4*)&L.q[r][c4] = make_float4(bf_lo(qw[j].x), bf_hi(qw[j].x), bf_lo(qw[j].y), bf_hi(qw[j].y));
          *(float4*)&L.k[r][c4] = make_float4(bf_lo(kw[j].x), bf_hi(kw[j].x), bf_lo(kw[j].y), bf_hi(kw[j].y));
        }
      }
    }
  }
  if (tid < T) {
    const long tok = bos + tid;
    const float alog = E(a.a_log[hv]);
    L.g[tid] = -alog * softplus(bf(a.av[tok * a.ab_stride + hv]) + a.dt_bias[hv], a.sp_thr);
    L.b[tid] = sigmoid_div(bf(a.bv[tok * a.ab_stride + hv]));
    const u32 word = (u32)a.anc[tok * 2] | ((u32)a.anc[tok * 2 + 1] << 16);
    L.anc[tid] = (int)((word | 1u | (1u << tid)) & (~0u >> (31 - tid)));
  }
  __syncthreads();

  // ---- 3. L2-normalise q (times scale) and k in place, one wave a vector (B vectors in flight a wave, each
  // with its own butterfly); the path sums of the gates
  constexpr int B = 2;
#pragma unroll 1
  for (int it = 0; it < (2 * TM + NW * B - 1) / (NW * B); ++it) {
    const int v0 = wave + it * NW * B;
    if (v0 >= 2 * T) break;
    float4 xv[B];
    float ss[B];
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const int vi = min(v0 + b * NW, 2 * T - 1);
      const float* x = (vi & 1) ? &L.k[vi >> 1][4 * lane] : &L.q[vi >> 1][4 * lane];
      xv[b] = *(const float4*)x;
      float t = xv[b].x * xv[b].x + xv[b].y * xv[b].y;
      t = t + xv[b].z * xv[b].z;
      ss[b] = t + xv[b].w * xv[b].w;
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
      for (int b = 0; b < B; ++b) ss[b] += __shfl_xor(ss[b], d, 32);
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const int vi = v0 + b * NW;
      if (vi < 2 * T) {
        const bool isk = vi & 1;
        float sc = R(ss[b] + 1e-6f);
        if (!isk) sc = sc * a.scale;
        float* x = isk ? &L.k[vi >> 1][4 * lane] : &L.q[vi >> 1][4 * lane];
        *(float4*)x = make_float4(xv[b].x * sc, xv[b].y * sc, xv[b].z * sc, xv[b].w * sc);
      }
    }
  }
  if (tid < T) {
    const u32 w = (u32)L.anc[tid];
    float p = 0.0f;
    for (int c = 0; c <= tid; ++c)
      if ((w >> c) & 1u) p = p + L.g[c];
    L.P[tid] = p;
    L.eP[tid] = E(p);
  }
  __syncthreads();

  // ---- 4. the masked Gram, one wave a (r, c) pair with c an ancestor of r (or r itself), B pairs in flight.
  // Lane t holds row t's ancestor word, path sum and beta; a pair reads them by lane index (no LDS round trip).
  constexpr int NP = TM * (TM + 1) / 2;
  {
    const int lt = min(lane, TM - 1);
    const int anc_l = L.anc[lt];
    const float p_l = L.P[lt], b_l = L.b[lt];
#pragma unroll 1
    for (int it = 0; it < (NP + NW * B - 1) / (NW * B); ++it) {
      const int p0 = wave + it * NW * B;
      float kk[B], qk[B];
      int rr[B], cc[B];
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const int pi = min(p0 + b * NW, NP - 1);
        int r = 0;
#pragma unroll
        for (int x = 1; x < TM; ++x) r += (pi >= x * (x + 1) / 2) ? 1 : 0;
        rr[b] = r;
        cc[b] = pi - r * (r + 1) / 2;
        const float4 kr = *(const float4*)&L.k[rr[b]][4 * lane];
        const float4 kc = *(const float4*)&L.k[cc[b]][4 * lane];
        const float4 qr = *(const float4*)&L.q[rr[b]][4 * lane];
        float t = kr.x * kc.x + kr.y * kc.y;
        t = t + kr.z * kc.z;
        kk[b] = t + kr.w * kc.w;
        float u = qr.x * kc.x + qr.y * kc.y;
        u = u + qr.z * kc.z;
        qk[b] = u + qr.w * kc.w;
      }
#pragma unroll
      for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
        for (int b = 0; b < B; ++b) {
          kk[b] += __shfl_xor(kk[b], d, 32);
          qk[b] += __shfl_xor(qk[b], d, 32);
        }
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const int r = rr[b], c = cc[b];
        const u32 wr = (u32)__builtin_amdgcn_readlane(anc_l, r);
        const bool live = (p0 + b * NW < NP) && r < T && ((wr >> c) & 1u);
        const float pr = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, p_l), r));
        const float pc = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, p_l), c));
        const float br = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, b_l), r));
        const float d = E(pr - pc);
        if (live && lane == 0) {
          L.M[r][c] = d * qk[b];
          if (c < r) L.A[r][c] = (br * d) * kk[b];
        }
      }
    }
  }
  __syncthreads();

  // ---- 5. k_t . h and q_t . h for every row (4-term groups added to the running sum), pair-summed
  float kh[TM], qh[TM];
#pragma unroll
  for (int t = 0; t < TM; ++t) {
    __builtin_amdgcn_sched_barrier(0);   // one row's operands live at a time (register pressure)
    if (t < T) {
      float sk = 0.0f, sq = 0.0f;
#pragma unroll
      for (int j = 0; j < kHalf; j += 4) {
        const float4 kv = *(const float4*)&L.k[t][k0 + j];
        const float4 qv = *(const float4*)&L.q[t][k0 + j];
        float gk = h[j] * kv.x + h[j + 1] * kv.y;
        gk = gk + h[j + 2] * kv.z;
        gk = gk + h[j + 3] * kv.w;
        float gq = h[j] * qv.x + h[j + 1] * qv.y;
        gq = gq + h[j + 2] * qv.z;
        gq = gq + h[j + 3] * qv.w;
        sk = sk + gk;
        sq = sq + gq;
        if (j == kHalf / 2 - 4) __builtin_amdgcn_sched_barrier(0);   // half a row's operands in flight at once
      }
      kh[t] = pair_add(sk);
      qh[t] = pair_add(sq);
    }
  }

  // every row's v (and norm gate) value, issued together once the state is no longer
  // needed (its 64 registers are free): the forward substitution must not wait on them one by one
  u32 vraw[TM], zraw[TM];
#pragma unroll
  for (int r = 0; r < TM; ++r) {
    if (r < T) {
      vraw[r] = a.v[((long)(bos + r) * a.H + hv) * kDim + row];
      if constexpr (NORM) zraw[r] = a.zg[((long)(bos + r) * a.H + hv) * kDim + row];
    }
  }

  // ---- 6. forward substitution down the tree; 7. the outputs
  float u[TM];
#pragma unroll
  for (int r = 0; r < TM; ++r) {
    if (r < T) {
      const long tok = bos + r;
      const float vr = bf(vraw[r]);
      float ur = L.b[r] * (vr - L.eP[r] * kh[r]);
      const u32 w = (u32)L.anc[r];
#pragma unroll
      for (int c = 0; c < r; ++c)
        if ((w >> c) & 1u) ur = ur - L.A[r][c] * u[c];
      u[r] = ur;
      float on = L.eP[r] * qh[r];
#pragma unroll
      for (int c = 0; c <= r; ++c)
        if ((w >> c) & 1u) on = on + L.M[r][c] * u[c];
      if constexpr (NORM)
        on = row_norm<NW>(on, half, tid, L.wsum, a.nw[row], bf(zraw[r]), a.neps, a.nact);
      if (half == 0) a.o[(tok * a.H + hv) * kDim + row] = away16(on);
    }
  }

  // ---- 8. the record: U (half 0 of each row), normalised k (slice 0), P and the ancestor words (rows < TM)
  float* rec = a.ts_out + (long)n * a.ts_seq + (long)hv * a.ts_head;
  if (half == 0) {
#pragma unroll
    for (int r = 0; r < TM; ++r)
      if (r < T) rec[TreeRec<TM>::U + r * kDim + row] = u[r];
  }
  if (z == 0) {
    for (int i = tid; i < T * kDim / 4; i += NT) {
      const int r = i / (kDim / 4), c4 = (i % (kDim / 4)) * 4;
      *(float4*)&rec[TreeRec<TM>::K + r * kDim + c4] = *(const float4*)&L.k[r][c4];
    }
    if (tid < TM) {   // P and the ancestor words of rows T..TM-1 are zero; the slots past TM are left as they are
      rec[TreeRec<TM>::P + tid] = tid < T ? L.P[tid] : 0.0f;
      ((int*)rec)[TreeRec<TM>::A + tid] = tid < T ? L.anc[tid] : 0;
    }
  }
}

}  // namespace sgdn

#define SGDN_TREE_PARAMS                                                                                         \
  sgdn::cbf q, sgdn::cbf k,                                    \
      sgdn::cbf v, sgdn::cbf av,                               \
      sgdn::cbf bv, long ab_stride, sgdn::cf32 a_log,                    \
      sgdn::cf32 dt_bias, sgdn::wbf state, long st_slot, long st_head,         \
      sgdn::ci32 slot, sgdn::wbf o, sgdn::ci32 cu,                  \
      sgdn::cbf anc, sgdn::cf32 ts_in, sgdn::wf32 ts_out,       \
      long ts_seq, long ts_head, sgdn::ci32 acc, sgdn::cbf zg,             \
      sgdn::cf32 nw, float neps, int nact, int h, int hg, float scale, float sp_thr
#define SGDN_TREE_ARGS                                                                                           \
  sgdn::TreeArgs{q,      k,      v,   av,  bv,     ab_stride, a_log,   dt_bias, state, st_slot,                  \
                 st_head, slot,  o,   cu,  anc,    ts_in,     ts_out,  ts_seq,  ts_head, acc,                    \
                 zg,     nw,     neps, nact, h,    hg,        scale,   sp_thr}

#define SGDN_TREE(TM, NAME, VS, NORM, ADV)                                                                       \
  extern "C" __global__ __launch_bounds__(256 / VS) void NAME(SGDN_TREE_PARAMS) {                               \
    sgdn::tree_update<TM, VS, NORM != 0, ADV != 0>(SGDN_TREE_ARGS);                                              \
  }

#define SGDN_TREE_ADVANCE(TM, NAME)                                                                              \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                                        \
      sgdn::wbf state, long st_slot, long st_head, sgdn::ci32 slot,              \
      sgdn::ci32 slot_out, sgdn::cf32 ts, long ts_seq, long ts_head,                 \
      sgdn::ci32 acc) {                                                                             \
    sgdn::tree_advance_kernel<TM>(state, st_slot, st_head, slot, slot_out, ts, ts_seq, ts_head, acc, blockIdx.x, \
                                  blockIdx.y);                                                                   \
  }


SGDN_TREE(32, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs2_adv1_dense32, 2, 0, 1)
SGDN_TREE(32, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs4_adv1_dense32, 4, 0, 1)
