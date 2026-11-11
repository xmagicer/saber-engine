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
  static constexpr int A = 2 * TM * kDim + 16;
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

  static_assert(TM==16 && !NORM, "split W16 product");
  constexpr int Q = 0, K = TM*kDim, QQ = 2*TM*kDim, KK = QQ + TM*TM;
  constexpr int PREP = KK + TM*TM;
  const float* pre = a.nw + ((long)n*a.Hg + hk)*PREP;
  for (int i=tid; i<T*kDim/4; i+=NT) {
    ((float4*)L.q)[i] = ((const float4*)(pre+Q))[i];
    ((float4*)L.k)[i] = ((const float4*)(pre+K))[i];
  }
  if (tid<T) {
    const long tok=bos+tid;
    L.g[tid]=-E(a.a_log[hv])*softplus(bf(a.av[tok*a.ab_stride+hv])+a.dt_bias[hv],a.sp_thr);
    L.b[tid]=sigmoid_div(bf(a.bv[tok*a.ab_stride+hv]));
    L.anc[tid]=(int)(((u32)a.anc[tok]|1u|(1u<<tid)) & ((2u<<tid)-1u));
  }
  __syncthreads();
  if (tid<T) {
    float sum=0.0f;
    const u32 bits=(u32)L.anc[tid];
    for(int c=0;c<=tid;++c) if ((bits>>c)&1u) sum=sum+L.g[c];
    L.P[tid]=sum;
    L.eP[tid]=E(sum);
  }
  __syncthreads();
  for(int cell=tid;cell<T*TM;cell+=NT) {
    const int r=cell/TM,c=cell%TM;
    if (((u32)L.anc[r]>>c)&1u) {
      const float decay=E(L.P[r]-L.P[c]);
      L.M[r][c]=decay*pre[QQ+cell];
      if(c<r) L.A[r][c]=(L.b[r]*decay)*pre[KK+cell];
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


// PREP layout belongs to one (sequence, key head); each query-row tile writes its own Gram rows.
template <int TILES>
__device__ __forceinline__ void split_prep16(const TreeArgs& a) {
  constexpr int TM=16, NRW=TM/TILES, NW=8;
  constexpr int K=TM*kDim, QQ=2*TM*kDim, KK=QQ+TM*TM, PREP=KK+TM*TM;
  __shared__ __attribute__((aligned(16))) float kshared[TM][kDim];
  __shared__ __attribute__((aligned(16))) float qshared[NRW][kDim];
  const int n=blockIdx.x,hk=blockIdx.y,tile=blockIdx.z,first=tile*NRW;
  const int bos=a.cu[n],T=min(a.cu[n+1]-bos,TM);
  if(first>=T || a.slot[n]<=0) return;
  const int wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5),lane=threadIdx.x&31;
  float* pre=a.ts_out+((long)n*a.Hg+hk)*PREP;
  for(int job=wave;job<TM+NRW;job+=NW) {
    const bool key=job<TM;
    const int r=key?job:first+(job-TM);
    if(r>=T) continue;
    const u16* raw=(key?a.k:a.q)+((long)(bos+r)*a.Hg+hk)*kDim+4*lane;
    const u32x2 packed=*(const u32x2*)raw;
    const float4 v=make_float4(bf_lo(packed.x),bf_hi(packed.x),bf_lo(packed.y),bf_hi(packed.y));
    float ss=v.x*v.x+v.y*v.y; ss=ss+v.z*v.z; ss=ss+v.w*v.w;
#pragma unroll
    for(int d=16;d>=1;d>>=1) ss+=__shfl_xor(ss,d,32);
    float scale=R(ss+1e-6f);
    if(!key) scale=scale*a.scale;
    const float4 nv=make_float4(v.x*scale,v.y*scale,v.z*scale,v.w*scale);
    float* dest=key?&kshared[r][4*lane]:&qshared[r-first][4*lane];
    *(float4*)dest=nv;
    if(!key || tile==0) *(float4*)(pre+(key?K:0)+r*kDim+4*lane)=nv;
  }
  __syncthreads();
  for(int pair=wave;pair<NRW*TM;pair+=NW) {
    const int r=first+pair/TM,c=pair%TM;
    if(r>=T || c>r) continue;
    const unsigned mask=(u32)a.anc[bos+r]|1u|(1u<<r);
    if(!((mask>>c)&1u))continue;
    const float4 qr=*(const float4*)&qshared[r-first][4*lane];
    const float4 kr=*(const float4*)&kshared[r][4*lane];
    const float4 kc=*(const float4*)&kshared[c][4*lane];
    float u=qr.x*kc.x+qr.y*kc.y; u=u+qr.z*kc.z; u=u+qr.w*kc.w;
    float v=kr.x*kc.x+kr.y*kc.y; v=v+kr.z*kc.z; v=v+kr.w*kc.w;
#pragma unroll
    for(int d=16;d>=1;d>>=1) {
      u+=__shfl_xor(u,d,32);
      v+=__shfl_xor(v,d,32);
    }
    if(lane==0) {
      pre[QQ+r*TM+c]=u;
      pre[KK+r*TM+c]=v;
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

extern "C" __global__ __launch_bounds__(256) void p27_gdn_tree16_splitprep_t1(SGDN_TREE_PARAMS) {
  sgdn::split_prep16<1>(SGDN_TREE_ARGS);
}
extern "C" __global__ __launch_bounds__(256) void p27_gdn_tree16_splitprep_t4(SGDN_TREE_PARAMS) {
  sgdn::split_prep16<4>(SGDN_TREE_ARGS);
}
extern "C" __global__ __launch_bounds__(256) void p27_gdn_tree16_splitprep_t8(SGDN_TREE_PARAMS) {
  sgdn::split_prep16<8>(SGDN_TREE_ARGS);
}
SGDN_TREE(16, p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs1_adv1_split16, 1, 0, 1)
SGDN_TREE(16, p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs2_adv1_split16, 2, 0, 1)
SGDN_TREE(16, p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs4_adv1_split16, 4, 0, 1)
SGDN_TREE(16, p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs8_adv1_split16, 8, 0, 1)
