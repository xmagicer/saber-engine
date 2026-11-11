// Advance every GDN layer for a 16-row target tree. The single-layer kernel
// uses the same TreeRec<16> arithmetic and state format.
#include "sgdn_tree.h"

extern "C" __global__ __launch_bounds__(256)
void p27_gdn_tree_advance_layers_t16_f16s_k128_v128(
    const unsigned long long* __restrict__ state_base,
    const unsigned long long* __restrict__ ts_base,
    long st_slot, long st_head,
    const int* __restrict__ slot, const int* __restrict__ slot_out,
    long ts_seq, long ts_head, const int* __restrict__ acc) {
  auto* state = reinterpret_cast<sgdn::u16*>(state_base[blockIdx.x]);
  const auto* record = reinterpret_cast<const float*>(ts_base[blockIdx.x]);
  sgdn::tree_advance_kernel<16>(state, st_slot, st_head, slot, slot_out, record,
                                ts_seq, ts_head, acc, blockIdx.z, blockIdx.y);
}
