// Saber paged attention, CLEANROOM-2 (CR2-2c): the split-KV decode body and its in-kernel merge.
//
// One workgroup = NW waves x one KV head x one key segment of 16-key tiles (spec sections 0, 2.4-2.8).
// Each wave owns 16 query rows (row r -> query r / GQA, head-in-group r % GQA); K goes from global memory
// straight into the A operand of the score WMMA, V is transposed through LDS for the PV WMMA. The output is
// an f16 normalised partial and (m, l) per row per segment, or -- in the merge tail -- the merged bf16 row
// written by the last segment block to arrive.
#include "sattn_prims.h"

namespace sattn {

enum class KeyMap { Pages, List };          // block table, or the QSA union slot list
enum class RowSrc { Uniform, Ragged };      // q_len for every sequence, or cu_q prefix
enum class MaskBy { Causal, Ancestor, Bits };
enum class Tail { Partial, Merge, MergeDrop };

template <int GQA_, int NW_, bool KV8_, KeyMap KM_, RowSrc RS_, MaskBy MB_, Tail TL_>
struct DecodeSpec {
  static constexpr int GQA = GQA_, NW = NW_;
  static constexpr bool KV8 = KV8_;
  static constexpr KeyMap KM = KM_;
  static constexpr RowSrc RS = RS_;
  static constexpr MaskBy MB = MB_;
  static constexpr Tail TL = TL_;
  static constexpr int kEB = KV8 ? 1 : 2;                  // bytes per KV element
  static constexpr int kVJobs = KV8 ? 8 : 16;              // transposing loads per 16-key tile
  static constexpr int kVPerWave = (kVJobs + NW - 1) / NW;
  static constexpr bool kVAhead = kVPerWave <= 2;          // prefetch the next tile's V
  static constexpr int kVRow = 24;                         // LDS row stride of V^T, in halves
  static constexpr int kVBatch = KV8 ? 2 : 4;
  // K fetch schedule across the four 4-chunk groups of the score chain (bits identical in every form):
  //   fp8: two groups in flight (group g + 1 issued before group g is widened);
  //   bf16: group g widened, then g + 1 issued, then g's WMMAs (one group of raw registers live).
  static constexpr bool kKTwoInFlight = KV8;
  static constexpr bool kVAll = kVPerWave * NW == kVJobs;  // every wave has exactly kVPerWave jobs
  static constexpr bool kVContig = kVAll && NW % 2 == 0;  // see vjob()                        // un-prefetched V loads in flight per wave
};

// The KV address of a slot splits into a wave-uniform part (kv head, page: SGPRs) and a 32-bit lane offset.
template <int EB>
__device__ __forceinline__ const uint8_t* head_base(const Params& a, int kvh) {
  return (const uint8_t*)a.cache + (long)kvh * a.head_stride * EB;
}
template <int EB>
__device__ __forceinline__ const uint8_t* page_base(const Params& a, const uint8_t* hb, int page) {
  return hb + (long)page * a.page_stride * EB;
}

template <class S>
struct DecodeShared {
  _Float16 vt[kHead * S::kVRow];
};

template <class S>
struct MergeShared {
  _Float16 vt[kHead * S::kVRow];
  float wrow[S::NW][128];
  int last;
};

// One wave merges the partials of one (token, q head) job exactly as the pf1 combine does: lane L holds
// leaves L + 32 i of the 256-leaf weight tree and output columns L + 32 i.
__device__ __forceinline__ float lane_read(float v, int src) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(src << 2, __builtin_bit_cast(int, v)));
}

__device__ __forceinline__ void wave_lds_order() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
}

__device__ __forceinline__ void merge_one_job(const _Float16* __restrict__ po, const float* __restrict__ pm,
                                              const float* __restrict__ pl, int base, int used, float vd,
                                              float* wrow, uint16_t* __restrict__ orow, int lane) {
  float mv[8], lv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int s = lane + 32 * i;
    mv[i] = s < used ? pm[base + s] : kFloor;
    lv[i] = s < used ? pl[base + s] : 0.0f;
  }
  // the max is order-free (every m >= the floor)
  float gmax = mv[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) gmax = fmaxf(gmax, mv[i]);
#pragma unroll
  for (int x = 16; x >= 1; x >>= 1) gmax = fmaxf(gmax, lane_read(gmax, lane ^ x));
  float e[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int s = lane + 32 * i;
    float t = 0.0f;
    if (s < used) {
      float w = (mv[i] <= kFloor) ? 0.0f : ex2(mv[i] - gmax);
      w = w * lv[i];
      wrow[s] = w;
      t = t + w;
    }
    e[i] = t;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) e[i] = e[i] + e[i + 4];
#pragma unroll
  for (int i = 0; i < 2; ++i) e[i] = e[i] + e[i + 2];
  float t = e[0] + e[1];
#pragma unroll
  for (int step = 16; step >= 1; step >>= 1) t = t + lane_read(t, lane + step < 32 ? lane + step : lane);
  const float tot = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, t)));
  wave_lds_order();

  // four fma chains per column over the segments in steps of four; the remainder goes to chain 0
  float ch[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int k = 0; k < 4; ++k) ch[i][k] = 0.0f;
  const _Float16* col = po + (long)base * kHead + lane;
  int s = 0;
  for (; s + 4 <= used; s += 4) {
    float p[4][8];
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int i = 0; i < 8; ++i) p[k][i] = (float)col[(long)(s + k) * kHead + 32 * i];
    const float w0 = wrow[s], w1 = wrow[s + 1], w2 = wrow[s + 2], w3 = wrow[s + 3];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ch[i][0] = fmaf(w0, p[0][i], ch[i][0]);
      ch[i][1] = fmaf(w1, p[1][i], ch[i][1]);
      ch[i][2] = fmaf(w2, p[2][i], ch[i][2]);
      ch[i][3] = fmaf(w3, p[3][i], ch[i][3]);
    }
  }
  for (; s < used; ++s) {
    const float w = wrow[s];
#pragma unroll
    for (int i = 0; i < 8; ++i) ch[i][0] = fmaf(w, (float)col[(long)s * kHead + 32 * i], ch[i][0]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float av = 0.0f;
    av = av + ch[i][0];
    av = av + ch[i][1];
    av = av + ch[i][2];
    av = av + ch[i][3];
    const float r = tot > 0.0f ? (av * vd) / tot : 0.0f;
    orow[lane + 32 * i] = (uint16_t)bf16_bits(r);
  }
  wave_lds_order();
}

template <class S>
__device__ __forceinline__ void decode_body(const Params& a, int splits, const int* __restrict__ cu_q,
                                            const uint16_t* __restrict__ anc, const UnionKeys& uk,
                                            unsigned* __restrict__ counter) {
  constexpr int GQA = S::GQA, NW = S::NW, EB = S::kEB, VR = S::kVRow;
  constexpr bool kMerge = S::TL != Tail::Partial;
  typedef typename pick<kMerge, MergeShared<S>, DecodeShared<S>>::type Smem;
  __shared__ Smem sm;

  static_assert((NW == 6 || NW == 12) && GQA == 6 && S::TL == Tail::Partial
                && S::RS == RowSrc::Ragged && S::MB == MaskBy::Ancestor,
                "32-row probe requires the paged ragged partial shape");
  constexpr int qtiles = 12 / NW;
  const int sp = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z / qtiles;
  const int qtile = blockIdx.z % qtiles;
  const int tid = threadIdx.x, lane = tid & 31;
  // wave index: an SGPR from 4 waves up; at 2-3 waves a VGPR (those bodies are SGPR-bound, not VGPR-bound)
  const int wv = NW == 1 ? 0 : (NW >= 4 ? __builtin_amdgcn_readfirstlane(tid >> 5) : (tid >> 5));
  const int c = lane & 15, hh = lane >> 4;

  int q0, ql;
  if constexpr (S::RS == RowSrc::Ragged) {
    q0 = cu_q[seq];
    ql = cu_q[seq + 1] - q0;
    if (ql <= qtile * (NW * 16 / GQA) || ql > 32) return;
  } else {
    q0 = seq * a.qrows;
    ql = a.qrows;
  }
  int nctx;
  if constexpr (S::KM == KeyMap::List) {
    nctx = a.ctx_len[seq];
  } else {
    nctx = a.ctx_len[seq];
  }
  if (nctx <= 0) {
    if constexpr (kMerge) {
      if (sp == 0) {
        uint16_t* o = (uint16_t*)a.dst;
        const int n = ql * GQA * kHead;
        for (int i = tid; i < n; i += NW * 32) {
          int qi = i / (GQA * kHead), rem = i - qi * (GQA * kHead);
          o[((long)(q0 + qi) * a.nqh + kvh * GQA) * kHead + rem] = 0;
        }
      }
    }
    return;
  }
  const int ntl = (nctx + 15) >> 4;
  const int tps = (int)((unsigned)(ntl + splits - 1) / (unsigned)splits);
  const int t_lo = sp * tps;
  if (t_lo >= ntl) return;
  const int t_hi = min(t_lo + tps, ntl);

  // ---- this lane's row
  const int r = (qtile * NW + wv) * 16 + c;
  const int qi = r / GQA, hi = r - (r / GQA) * GQA;
  const bool live = qi < ql;
  const int qrow = live ? qi : ql - 1;
  const int qhead = kvh * GQA + hi;
  const long tok = (long)q0 + qrow;

  const float kd = a.kscale ? a.kscale[seq * a.nkvh + kvh] : 1.0f;
  const float mul = (a.sm_scale * kd) * kLog2e;

  // Q as the B operand of the score WMMA: lane (c, h) holds dims 16 t + 8 h .. +7 of row c, f16 RTZ (the raw
  // bf16 chunks are converted in place, after the first tile's V fetch has been issued)
  u32x4 qv[16];
  {
    const uint16_t* qp = (const uint16_t*)a.query + (tok * a.nqh + qhead) * kHead + 8 * hh;
#pragma unroll
    for (int t = 0; t < 16; ++t) qv[t] = ld16(qp + 16 * t);
  }
  auto q_convert = [&]() {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const u32x4 w = qv[t];
      u32x4 o;
      o.x = rtz_pair(bf_lo(w.x) * mul, bf_hi(w.x) * mul);
      o.y = rtz_pair(bf_lo(w.y) * mul, bf_hi(w.y) * mul);
      o.z = rtz_pair(bf_lo(w.z) * mul, bf_hi(w.z) * mul);
      o.w = rtz_pair(bf_lo(w.w) * mul, bf_hi(w.w) * mul);
      qv[t] = o;
    }
  };
#define SATTN_QF(t) __builtin_bit_cast(f16x8, qv[t])

  // mask inputs
  int klimit = 0, tail0 = 0;
  uint32_t aw = 0;
  const uint8_t* bits = nullptr;
  if constexpr (S::MB == MaskBy::Causal) {
    klimit = nctx - ql + qrow;
  } else if constexpr (S::MB == MaskBy::Ancestor) {
    tail0 = nctx - ql;
    // Two little-endian halfwords per row preserve the probe kernarg pointer ABI.
    const long word = 2 * (long)(q0 + qrow);
    const uint32_t bits32 = (uint32_t)anc[word] | ((uint32_t)anc[word + 1] << 16);
    const uint32_t prefix = ~0u >> (31 - qrow);
    aw = (bits32 & prefix) | (1u << qrow);
  } else {
    bits = uk.kbits + (long)seq * uk.kstride * 4 + qi;  // unclamped query index
  }
  const int* slots = nullptr;
  const int* btrow = nullptr;
  if constexpr (S::KM == KeyMap::List) {
    slots = uk.kslot + (long)seq * uk.kstride;
  } else {
    btrow = a.page_map + (long)seq * a.map_width;
  }
  const int last = nctx - 1;

  // V staging: job j of a tile = (key group j & 1, dim block j >> 1); lane (g = lane >> 3, i = lane & 7)
  const int vg = lane >> 3, vi = lane & 7;
  const uint8_t* hb = head_base<EB>(a, kvh);
  // lane part of a V job's address for key group kg of tile t (the job's dim block adds a constant)
  auto v_lane = [&](int t, int kg) -> uint32_t {
    const int key = min(t * 16 + 8 * kg + vi, last);
    if constexpr (S::KM == KeyMap::List) {
      return (uint32_t)key;
    } else {
      return (uint32_t)(((key & 15) * kSlotElems + kHead + (S::KV8 ? 16 : 8) * vg) * EB);
    }
  };
  auto v_src = [&](int page, const uint32_t (&vl)[2], int job) -> const uint8_t* {
    const uint32_t dconst = (uint32_t)((S::KV8 ? 64 : 32) * (job >> 1) * EB);
    if constexpr (S::KM == KeyMap::List) {
      const int s = slots[vl[job & 1]];
      return page_base<EB>(a, hb, s >> 4) +
             ((uint32_t)(((s & 15) * kSlotElems + kHead + (S::KV8 ? 16 : 8) * vg) * EB) + dconst);
    } else {
      return page_base<EB>(a, hb, page) + (vl[job & 1] + dconst);
    }
  };
  // job id = (dim block << 1) | key group. Contiguous form: wave w takes key group w & 1 and a run of
  // consecutive dim blocks (one 64/128-byte-aligned run per key); else jobs are dealt round-robin.
  auto vjob = [&](int n) -> int {
    if constexpr (S::kVContig) {
      return ((((wv >> 1) * S::kVPerWave + n) << 1) | (wv & 1));
    } else {
      return wv + NW * n;
    }
  };
  auto v_put = [&](int job, u32x4 raw) {
    const int kg = job & 1;
    if constexpr (S::KV8) {
      const int d = 64 * (job >> 1) + 16 * vg + 2 * vi;
      u32x2 ev, od;
      unzip_pairs(raw, ev, od);
      *lds_ptr((f16x8*)&sm.vt[d * VR + 8 * kg]) = e4m3x8_to_f16(ev);
      *lds_ptr((f16x8*)&sm.vt[(d + 1) * VR + 8 * kg]) = e4m3x8_to_f16(od);
    } else {
      const int d = 32 * (job >> 1) + 8 * vg + vi;
      *lds_ptr((f16x8*)&sm.vt[d * VR + 8 * kg]) = bf16x8_to_f16(raw);
    }
  };

  // per-row online-softmax state
  f32x8 acc[16];
#pragma unroll
  for (int g = 0; g < 16; ++g) acc[g] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  float m_ref = kFloor, off = 0.0f - kFloor, l = 0.0f;

  int page = 0, page_next = 0;
  if constexpr (S::KM == KeyMap::Pages) page_next = btrow[t_lo];
  u32x4 vreg[S::kVPerWave];
  if constexpr (S::kVAhead) {
    page = page_next;
#pragma unroll
    for (int n = 0; n < S::kVPerWave; ++n) {
      int job = vjob(n);
      const uint32_t vl[2] = {v_lane(t_lo, 0), v_lane(t_lo, 1)};
      if (S::kVAll || job < S::kVJobs) vreg[n] = load_tr16(v_src(page, vl, job));
    }
  }

  q_convert();

  for (int t = t_lo; t < t_hi; ++t) {
    page = page_next;
    if constexpr (S::KM == KeyMap::Pages) {
      if (t + 1 < t_hi) page_next = btrow[t + 1];
    }
    const int k0 = t * 16;
    const uint8_t* kp;
    {
      int key = min(k0 + c, last);
      if constexpr (S::KM == KeyMap::List) {
        int s = slots[key];
        kp = page_base<EB>(a, hb, s >> 4) + (uint32_t)(((s & 15) * kSlotElems + 8 * hh) * EB);
      } else {
        kp = page_base<EB>(a, hb, page) + (uint32_t)(((key & 15) * kSlotElems + 8 * hh) * EB);
      }
    }
    typedef typename pick<S::KV8, u32x2, u32x4>::type KRaw;
    lds_sync();
    if constexpr (!S::kVAhead) {
      const uint32_t vl[2] = {v_lane(t, 0), v_lane(t, 1)};
      // fetch and store in batches: at most kVBatch transposed loads in flight per wave
#pragma unroll
      for (int n0 = 0; n0 < S::kVPerWave; n0 += S::kVBatch) {
#pragma unroll
        for (int n = n0; n < n0 + S::kVBatch && n < S::kVPerWave; ++n) {
          int job = vjob(n);
          if (S::kVAll || job < S::kVJobs) vreg[n] = load_tr16(v_src(page, vl, job));
        }
#pragma unroll
        for (int n = n0; n < n0 + S::kVBatch && n < S::kVPerWave; ++n) {
          int job = vjob(n);
          if (S::kVAll || job < S::kVJobs) v_put(job, vreg[n]);
        }
        sched_fence();
      }
    } else {
#pragma unroll
      for (int n = 0; n < S::kVPerWave; ++n) {
        int job = vjob(n);
        if (S::kVAll || job < S::kVJobs) v_put(job, vreg[n]);
      }
    }
    lds_sync();
    if constexpr (S::kVAhead) {
      if (t + 1 < t_hi) {
        const uint32_t vl[2] = {v_lane(t + 1, 0), v_lane(t + 1, 1)};
#pragma unroll
        for (int n = 0; n < S::kVPerWave; ++n) {
          int job = vjob(n);
          if (S::kVAll || job < S::kVJobs) vreg[n] = load_tr16(v_src(page_next, vl, job));
        }
      }
    }

    // ---- scores: A = K (row = key c of the tile, k = dims 16 t + 8 h ..), B = Q, chain over t ascending
    f32x8 sc = {0, 0, 0, 0, 0, 0, 0, 0};
    auto k_load = [&](KRaw (&r)[4], int g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) r[i] = *(const KRaw*)(kp + 16 * EB * (4 * g + i));
    };
    auto k_widen = [&](const KRaw (&r)[4], f16x8 (&kf)[4]) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (S::KV8) {
          kf[i] = e4m3x8_to_f16(r[i]);
        } else {
          kf[i] = bf16x8_to_f16(r[i]);
        }
      }
    };
    if constexpr (S::kKTwoInFlight) {
      KRaw kr[2][4];
      k_load(kr[0], 0);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        if (g + 1 < 4) k_load(kr[(g + 1) & 1], g + 1);
        sched_fence();
        f16x8 kf[4];
        k_widen(kr[g & 1], kf);
#pragma unroll
        for (int i = 0; i < 4; ++i) sc = mma16(kf[i], SATTN_QF(4 * g + i), sc);
        sched_fence();
      }
    } else {
      KRaw kr[4];
      k_load(kr, 0);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        sched_fence();
        f16x8 kf[4];
        k_widen(kr, kf);
        sched_fence();
        if (g + 1 < 4) k_load(kr, g + 1);
        sched_fence();
#pragma unroll
        for (int i = 0; i < 4; ++i) sc = mma16(kf[i], SATTN_QF(4 * g + i), sc);
      }
      sched_fence();
    }

    // ---- mask: lane (c, h) holds keys kb .. kb + 7 of row c
    const int kb = k0 + 8 * hh;
    if constexpr (S::MB == MaskBy::Causal) {
      if (wave_any(kb + 7 > klimit)) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (kb + e > klimit) sc[e] = -__builtin_inff();
      }
    } else if constexpr (S::MB == MaskBy::Ancestor) {
      if (wave_any(kb + 7 >= tail0)) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int kr = kb + e - tail0;
          bool vis = kr < 0 || (kr < 32 && ((aw >> kr) & 1u));
          if (!vis) sc[e] = -__builtin_inff();
        }
      }
    } else {
      uint32_t b = bits[(kb >> 3) * 32];
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (!((b >> e) & 1u)) sc[e] = -__builtin_inff();
    }

    // ---- lazy rescale (wave-wide decision), GROW 14
    float smax = sc[0];
#pragma unroll
    for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
    if (wave_any(smax > m_ref + 14.0f)) {
      float rmax = fmaxf(smax, other_half(smax));
      float mnew = fmaxf(fmaxf(m_ref, rmax), kFloor);
      float alpha = ex2(m_ref - mnew);
#pragma unroll
      for (int g = 0; g < 16; ++g) acc[g] *= alpha;
      l *= alpha;
      m_ref = mnew;
      off = 0.0f - mnew;
    }

    // ---- probabilities, denominator over the RTZ f16 values, PV
    uint32_t pw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) pw[i] = rtz_pair(ex2(sc[2 * i] + off), ex2(sc[2 * i + 1] + off));
#pragma unroll
    for (int i = 0; i < 4; ++i) l = dot2_ones(pw[i], l);
    const f16x8 pb = __builtin_bit_cast(f16x8, (u32x4){pw[0], pw[1], pw[2], pw[3]});
    // A = V^T (row = dim 16 g + c, k = keys 8 h ..), B = P
    const SATTN_LDS _Float16* vrow = lds_ptr(&sm.vt[c * VR + 8 * hh]);
#pragma unroll
    for (int g0 = 0; g0 < 16; g0 += 4) {
      f16x8 vf[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) vf[i] = *(const SATTN_LDS f16x8*)(vrow + (g0 + i) * 16 * VR);
      sched_fence();
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[g0 + i] = mma16(vf[i], pb, acc[g0 + i]);
      sched_fence();
    }
  }

  // ---- partial store (PF16 form)
  const float lt = l + other_half(l);
  const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
  const long nrow = (long)a.nseq * a.qrows * a.nqh * splits;
  _Float16* po = (_Float16*)a.work;
  float* pm = (float*)(po + nrow * kHead);
  float* pl = pm + nrow;
  if (live) {
    const long idx = (tok * a.nqh + qhead) * splits + sp;
    _Float16* dst = po + idx * kHead + 8 * hh;
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      f32x8 v = acc[g];
      u32x4 o;
      o.x = rtz_pair(v[0] * inv, v[1] * inv);
      o.y = rtz_pair(v[2] * inv, v[3] * inv);
      o.z = rtz_pair(v[4] * inv, v[5] * inv);
      o.w = rtz_pair(v[6] * inv, v[7] * inv);
      *(u32x4*)(dst + 16 * g) = o;
    }
    if (hh == 0) {
      pm[idx] = m_ref;
      pl[idx] = lt;
    }
  }

  if constexpr (kMerge) {
    // ---- last arriver merges the item (seq, kvh)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    const int used = (int)((unsigned)(ntl + tps - 1) / (unsigned)tps);
    if (tid == 0) {
      unsigned* cnt = counter + seq * a.nkvh + kvh;
      unsigned old = atomicAdd(cnt, 1u);
      int is_last = old == (unsigned)(used - 1);
      if (is_last) *cnt = 0u;
      sm.last = is_last;
    }
    __syncthreads();
    if (!sm.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const int uc = S::TL == Tail::MergeDrop ? used - 1 : used;
    const float vd = a.vscale ? a.vscale[seq * a.nkvh + kvh] : 1.0f;
    uint16_t* o = (uint16_t*)a.dst;
    const int jobs = ql * GQA;
    for (int j = wv; j < jobs; j += NW) {
      const int jq = j / GQA, jh = j - (j / GQA) * GQA;
      const long jt = (long)q0 + jq;
      const int jhead = kvh * GQA + jh;
      const int base = (int)((jt * a.nqh + jhead) * splits);
      merge_one_job(po, pm, pl, base, uc, vd, sm.wrow[wv], o + (jt * a.nqh + jhead) * kHead, lane);
    }
  }
}

#undef SATTN_QF
}  // namespace sattn

namespace {
using namespace sattn;
template<int NW, bool FP8> using Tree32 = DecodeSpec<6, NW, FP8, KeyMap::Pages, RowSrc::Ragged, MaskBy::Ancestor, Tail::Partial>;
}

extern "C" __global__ void __launch_bounds__(192)
p27_attn_decode_anc_probe32_h256_gqa6_fp8kv_nw6(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<Tree32<6, true>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(192)
p27_attn_decode_anc_probe32_h256_gqa6_bf16kv_nw6(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<Tree32<6, false>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(384)
p27_attn_decode_anc_probe32_h256_gqa6_fp8kv_nw12(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<Tree32<12, true>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(384)
p27_attn_decode_anc_probe32_h256_gqa6_bf16kv_nw12(sattn::Params args, int splits, const int* cu_q, const uint16_t* anc) {
  decode_body<Tree32<12, false>>(args, splits, cu_q, anc, UnionKeys{}, nullptr);
}
