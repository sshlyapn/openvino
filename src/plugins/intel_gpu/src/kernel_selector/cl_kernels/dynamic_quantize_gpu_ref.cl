// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_ref.cl: Unsupported output dimension"
#endif

inline uint FUNC(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint y, uint x) {
    return OUTPUT1_GET_INDEX(b, f, y, x);
}

inline uint FUNC(get_scales_offset)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint y, uint x) {
#ifdef SCALES_OUTPUT_ORDER
    return FUNC_CALL(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_TENSOR SCALES_OUTPUT_ORDER);
#else
    return FUNC_CALL(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, x);
#endif
}

KERNEL(dynamic_quantize_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale)
{
    const uint bf = (uint)get_global_id(0);
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;
    const uint y = (uint)get_global_id(1);
    const uint x = (uint)get_global_id(2);
#ifdef SCALES_OUTPUT_ORDER
    const uint scale_idx = FUNC_CALL(get_scales_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, x);
#else
    const uint scale_idx = OUTPUT1_GET_INDEX_SAFE(b, f, y, x);
#endif

    half max_val = 0.0001h;
    for (int b_off = 0; b_off < (GROUP_SIZE_DIM0 == 1 ? 1 : INPUT0_BATCH_NUM); b_off++) {
    for (int f_off = 0; f_off < (GROUP_SIZE_DIM1 == 1 ? 1 : INPUT0_FEATURE_NUM); f_off++) {
    for (int y_off = 0; y_off < (GROUP_SIZE_DIM2 == 1 ? 1 : INPUT0_SIZE_Y); y_off++) {
#if GROUP_SIZE_DIM3 == 1
        const uint offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, x);
        half val = input[offset];
        half abs_val = fabs(val);
        max_val = fmax(max_val, abs_val);
#else
        const uint offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, 0);
        int x;
        for (x = 0; x < INPUT0_SIZE_X / 8; x++) {
            half8 val = as_half8(vload8(0, (ushort*)input + offset + x * 8));
            half8 abs_val = fabs(val);

            for (int j = 0; j < 8; j++)
                max_val = fmax(max_val, abs_val[j]);
        }
        x *= 8;
        for (; x < INPUT0_SIZE_X; x++)
            max_val = fmax(max_val, fabs(input[offset + x]));
#endif
    }
    }
    }

    half scale = 127.0h / max_val;
    for (int b_off = 0; b_off < (GROUP_SIZE_DIM0 == 1 ? 1 : INPUT0_BATCH_NUM); b_off++) {
    for (int f_off = 0; f_off < (GROUP_SIZE_DIM1 == 1 ? 1 : INPUT0_FEATURE_NUM); f_off++) {
    for (int y_off = 0; y_off < (GROUP_SIZE_DIM2 == 1 ? 1 : INPUT0_SIZE_Y); y_off++) {
#if GROUP_SIZE_DIM3 == 1
        const uint in_offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, x);
        const uint out_offset = OUTPUT_GET_INDEX(b + b_off, f + f_off, y + y_off, x);

        half val = input[in_offset];
        val *= scale;
        output[out_offset] = convert_char(val);
#else
        const uint in_offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, 0);
        const uint out_offset = OUTPUT_GET_INDEX(b + b_off, f + f_off, y + y_off, 0);
        int x;
        for (x = 0; x < INPUT0_SIZE_X / 8; x++) {
            half8 val = as_half8(vload8(0, (ushort*)input + in_offset + x * 8));
            val *= scale;
            vstore8(convert_char8(val), 0, output + out_offset + x * 8);
        }
        x *= 8;
        for (; x < INPUT0_SIZE_X; x++)
            output[out_offset + x] = convert_char(input[in_offset + x] * scale);
#endif
    }
    }
    }

    output_scale[scale_idx] = 1.0h / scale;
}
