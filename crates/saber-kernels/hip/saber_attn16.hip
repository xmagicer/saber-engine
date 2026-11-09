// Six 16-row waves cover 16 uniform causal queries with six query heads per KV head.
#include "sattn_decode.h"

namespace {
using namespace sattn;
template <bool KV8>
using Uniform16 = DecodeSpec<6, 6, KV8, KeyMap::Pages, RowSrc::Uniform, MaskBy::Causal, Tail::Partial>;
}  // namespace

extern "C" __global__ void __launch_bounds__(192)
saber_attn_decode_uniform16_h256_gqa6_fp8kv_nw6(sattn::Params args, int splits) {
  decode_body<Uniform16<true>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(192)
saber_attn_decode_uniform16_h256_gqa6_bf16kv_nw6(sattn::Params args, int splits) {
  decode_body<Uniform16<false>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}
