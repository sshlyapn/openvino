// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"


#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT_TYPE_N(x) AS_TYPE_N(INPUT0_TYPE, VEC_SIZE, x)


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

#define SUBGROUP_SIZE 16
#define HEAD_SIZE 128
#define NUM_HEADS 32
#define INPUT_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)

__attribute__((reqd_work_group_size(1, NUM_HEADS * SUBGROUP_SIZE, 1)))
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale)
{
    const uint batch_indexes = get_global_id(0);
    const uint head_idx = get_global_id(1) / SUBGROUP_SIZE;
    const uint sglid = get_sub_group_local_id();
    // const uint data_indexes = get_global_id(1);

    DECLARE_BATCHED_DIMS_INDEXES(batch_indexes);
    const uint f = head_idx;
    const uint x = 0;

    half max_value = 0.0001h;
    half val[HEAD_SIZE / SUBGROUP_SIZE];

    const uint data_offset = INPUT0_GET_INDEX(b, f, y, x);
    unroll_for (uint i = 0; i < HEAD_SIZE / SUBGROUP_SIZE; i++) {
        // val[i] = input[data_offset + i * SUBGROUP_SIZE + sglid];
        val[i] = INPUT_BLOCK_READ(input, data_offset + i * SUBGROUP_SIZE);
        max_value = fmax(max_value, fabs(val[i]));
    }

    max_value = work_group_reduce_max(max_value);

    half scale = 127.0h / max_value;

    unroll_for (uint i = 0; i < HEAD_SIZE / SUBGROUP_SIZE; i++) {
        OUTPUT_BLOCK_WRITE(output, data_offset + i * SUBGROUP_SIZE, convert_char(val[i] * scale));
        // output[data_offset + i * SUBGROUP_SIZE + sglid] = convert_char(val[i] * scale);
    }

#ifdef SCALES_OUTPUT_ORDER
    const uint scale_idx = FUNC_CALL(get_scales_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, x);
#else
    const uint scale_idx = OUTPUT1_GET_INDEX_SAFE(b, f, y, x);
#endif

    if (get_global_id(1) == 0 && get_global_id(2) == 0)
        output_scale[scale_idx] = 1.0h / scale;
}
