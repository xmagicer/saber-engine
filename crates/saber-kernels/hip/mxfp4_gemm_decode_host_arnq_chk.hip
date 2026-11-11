// Same GEMM/reduce arithmetic, with a short bound for shared-host communication.
#define SABER_AR_SPIN_MAX 1000000ull
#include "mxfp4_gemm_decode_arnq_chk.hip"
