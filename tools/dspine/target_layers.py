"""DSpine zero-based post-decoder ids to Saber's pre-layer residual edges.

Tencent's target adapter wraps decoder layers and records their outputs.
SGLang 0.5.22 dflash_utils.build_target_layer_ids documents the same convention.
Saber lN.resid.a is the input residual of layer N, after decoder block N-1.
"""


def capture_layers_from_dspine(target_layer_ids):
    if (not target_layer_ids or any(type(v) is not int or v<0 for v in target_layer_ids)
            or any(a>=b for a,b in zip(target_layer_ids,target_layer_ids[1:]))):
        raise ValueError('DSpine decoder ids must be nonnegative, unique and increasing')
    return [v+1 for v in target_layer_ids]


def validate_dspine_capture(metadata,target_layer_ids):
    expected=capture_layers_from_dspine(target_layer_ids)
    if metadata.get('target_layer_ids')!=expected:
        raise ValueError(f'DSpine capture requires Saber pre-layer ids {expected}; checkpoint decoder ids are zero-based post-layer outputs')
    return expected
