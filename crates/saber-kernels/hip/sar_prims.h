// Saber two-rank all-reduce kernels, CLEANROOM-2 (CR2-4c): conversions and protocol primitives.
//
// Written from the CR2-4a specification (saber-cache/cr2-4/spec/allreduce.md), not from any earlier all-reduce
// source. Every helper is the operation the spec's arithmetic contract (section 2.1) or synchronisation protocol
// (section 3.3) pins: the conversions must not be swapped for "equal" library forms, and the scopes / fences of
// the hand-off are part of the contract (a weaker one makes the peer read a slot that has not landed).
//
// Includability (spec 1.5.1): the megakernel prototype pastes `saber_ar_exact.hip` in as text after remapping
// __global__ / __launch_bounds__ / blockIdx / gridDim / blockDim / __syncthreads. So everything reachable from
// that file lives in namespace sar, reads work indices only through those names, spells every workgroup
// barrier `__syncthreads()`, and holds no host-only code.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

// the bounded spin of every cross-rank wait (polls); `ar_race_check.h` reads this name
#ifndef SABER_AR_SPIN_MAX
#define SABER_AR_SPIN_MAX 4000000000ull
#endif

namespace sar {

using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
using u32x4 = u32 __attribute__((ext_vector_type(4)));

// ---- conversions (spec 2.1)

// bf16 bits -> f32, exact
__device__ __forceinline__ float bf(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ float bf_lo(u32 w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(u32 w) { return __uint_as_float(w & 0xFFFF0000u); }
// f16 bits -> f32, exact
__device__ __forceinline__ float hf(u32 h) { return (float)__builtin_bit_cast(_Float16, (u16)h); }

// f32 -> bf16 round to nearest even; a NaN keeps sign and payload with the quiet bit forced (the compiler's
// __bf16 conversion on gfx1201 is exactly this integer sequence + the unordered select)
__device__ __forceinline__ u32 rne16(float x) { return (u32)__builtin_bit_cast(u16, (__bf16)x); }
__device__ __forceinline__ u32 rne16x2(float lo, float hi) { return rne16(lo) | (rne16(hi) << 16); }

// f32 -> f16 (v_cvt_f16_f32: RNE, subnormals, overflow to inf)
__device__ __forceinline__ u32 to_h(float x) { return (u32)__builtin_bit_cast(u16, (_Float16)x); }
__device__ __forceinline__ u32 to_h2(float lo, float hi) { return to_h(lo) | (to_h(hi) << 16); }

// the group-scale rounding: the same integer RNE with no NaN arm, kept as an f32 with a zero low half
__device__ __forceinline__ float scl16(float x) {
  const u32 u = __float_as_uint(x);
  return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// v_rsq_f32 (approximate); never constant-folded, never emulated
__device__ __forceinline__ float rsq(float x) { return __builtin_amdgcn_rsqf(x); }

// ---- element types of the exact sum: 16-byte words, lane by lane conv(f32(a) + f32(b)), `a` = the
// rank's own value (the left operand everywhere: spec 2.2 / 7.1)

enum Elem { kBf16 = 0, kF16 = 1, kF32 = 2 };

template <int E>
__device__ __forceinline__ u32 add_lane(u32 a, u32 b) {
  if constexpr (E == kBf16) {
    return rne16x2(bf_lo(a) + bf_lo(b), bf_hi(a) + bf_hi(b));
  } else if constexpr (E == kF16) {
    return to_h2(hf(a & 0xFFFFu) + hf(b & 0xFFFFu), hf(a >> 16) + hf(b >> 16));
  } else {
    return __float_as_uint(__uint_as_float(a) + __uint_as_float(b));
  }
}

template <int E>
__device__ __forceinline__ int4 add_word(int4 a, int4 b) {
  return make_int4((int)add_lane<E>((u32)a.x, (u32)b.x), (int)add_lane<E>((u32)a.y, (u32)b.y),
                   (int)add_lane<E>((u32)a.z, (u32)b.z), (int)add_lane<E>((u32)a.w, (u32)b.w));
}

// ---- the one-shot hand-off (spec 3.3)

// a relaxed device-scope take of this block's counter, broadcast through one LDS word
__device__ __forceinline__ u32 take_seq(u32* seq, int b, u32* bcast) {
  if (threadIdx.x == 0) *bcast = atomicAdd(seq + b, 1u) + 1u;
  __syncthreads();
  return *bcast;
}

// gp / ll: every counter c = b + k nb (k >= 1, c < nc) advances once, so all nc counters move by one a call
__device__ __forceinline__ void advance_idle(u32* seq, int b, int nb, int nc) {
  for (int c = b + ((int)threadIdx.x + 1) * nb; c < nc; c += (int)blockDim.x * nb) atomicAdd(seq + c, 1u);
}

__device__ __forceinline__ void wait_stores() { asm volatile("s_wait_storecnt 0x0" ::: "memory"); }
__device__ __forceinline__ void fence_sys() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, ""); }
__device__ __forceinline__ void writeback_sys() { asm volatile("global_wb scope:SCOPE_SYS" ::: "memory"); }

// the push drain by the kernarg: 1 = every thread a system fence, 3 = every thread waits its stores; then the
// barrier; 2 = thread 0 a system fence after it
__device__ __forceinline__ void drain_and_sync(int drain) {
  if (drain == 1) fence_sys();
  if (drain == 3) wait_stores();
  __syncthreads();
  if (drain == 2 && threadIdx.x == 0) fence_sys();
}

__device__ __forceinline__ void flag_release(u32* f, u32 v) {
  __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ u32 flag_acquire(const u32* f) {
  return __hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// thread 0: raise my flag word in the peer, spin on mine until it reaches s (bounded); returns the polls
__device__ __forceinline__ u64 signal_wait(u32* peer_flag, const u32* my_flag, u32 s) {
  flag_release(peer_flag, s);
  u64 z = 0;
  while (flag_acquire(my_flag) < s) {
    if (++z > SABER_AR_SPIN_MAX) {
#ifdef SABER_AR_TRAP_TIMEOUT
      __builtin_trap();
#endif
      break;
    }
  }
  return z;
}

__device__ __forceinline__ void acquire_opt(int acq) {
  if (acq == 1) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

}  // namespace sar
