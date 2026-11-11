// Standalone exact transport over registered host memory. Every call uses the
// same grid and both GPUs participate, including blocks with an empty run.
#define SABER_AR_SPIN_MAX 1000000ull
#define SABER_AR_TRAP_TIMEOUT 1
#include "sar_exact.h"

extern "C" __global__ __launch_bounds__(1024) void fn_ar_host_exact_bf16(SAR_EXACT_PARAMS) {
  // Each writer drains at system scope before block release; acquire the data
  // after the matching system-scope epoch. A missing peer fails the HIP launch.
  sar::exact_oneshot<sar::kBf16, sar::kPerBlock, sar::kNoCheck>(
      in, peer_scratch, my_scratch, out, peer_flags, my_flags, seq_ctrs, n16, slot_stride16, 1, 1);
}
