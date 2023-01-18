// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/fetch_data.cl"

#define SIMD_SIZE 16

#define ARRAY_TO_VEC_2(vec, arr, offset)                \
    (vec).s0 = (arr)[(offset)];                         \
    (vec).s1 = (arr)[(offset) + 1]
#define ARRAY_TO_VEC_4(vec, arr, offset)                \
    ARRAY_TO_VEC_2((vec).s01, arr, offset);             \
    ARRAY_TO_VEC_2((vec).s23, arr, (offset) + 2)
#define ARRAY_TO_VEC_8(vec, arr, offset)                \
    ARRAY_TO_VEC_4((vec).s0123, arr, offset);           \
    ARRAY_TO_VEC_4((vec).s4567, arr, (offset) + 4)
#define ARRAY_TO_VEC_16(vec, arr, offset)               \
    ARRAY_TO_VEC_8((vec).s01234567, arr, offset);       \
    ARRAY_TO_VEC_8((vec).s89abcdef, arr, (offset) + 8)

#define ARRAY_TO_VEC(vec, arr, offset) CAT(ARRAY_TO_VEC_, BLOCK_SIZE)(vec, arr, offset)

#define VEC_TO_ARRAY_2(arr, vec, offset)                \
    (arr)[(offset) + 0] = (vec).s0;                     \
    (arr)[(offset) + 1] = (vec).s1
#define VEC_TO_ARRAY_4(arr, vec, offset)                \
    VEC_TO_ARRAY_2(arr, (vec).s01, offset);             \
    VEC_TO_ARRAY_2(arr, (vec).s23, (offset) + 2)
#define VEC_TO_ARRAY_8(arr, vec, offset)                \
    VEC_TO_ARRAY_4(arr, (vec).s0123, offset);           \
    VEC_TO_ARRAY_4(arr, (vec).s4567, (offset) + 4)
#define VEC_TO_ARRAY_16(arr, vec, offset)               \
    VEC_TO_ARRAY_8(arr, (vec).s01234567, offset);       \
    VEC_TO_ARRAY_8(arr, (vec).s89abcdef, (offset) + 8)

#define VEC_TO_ARRAY(arr, vec, offset) CAT(VEC_TO_ARRAY_, BLOCK_SIZE)(arr, vec, offset)

#define OUTPUT_TYPE_BLOCK               MAKE_VECTOR_TYPE(OUTPUT_TYPE, BLOCK_SIZE)
#define TO_TYPE(type, val)              CAT(convert_, type)(val)

#if BLOCK_SIZE != 1
    #define READ_FUNC(ptr, offset) CAT(DT_INPUT_BLOCK_READ, BLOCK_SIZE)(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, BLOCK_SIZE)(ptr, offset, val)
#else
    #define READ_FUNC(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#endif

#if ELTWISE_BROADCAST
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#else
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

REQD_SUB_GROUP_SIZE(SIMD_SIZE)
KERNEL(eltwise_b_fs_yx_fsv16)(INPUTS_DECLS
                              __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
, FUSED_OPS_DECLS
#endif
)
{
    const uint f_block = (uint)get_group_id(0);
    const uint zyx = (uint)get_global_id(1);
    const uint b = (uint)get_global_id(2);
    const uint x = (zyx % BLOCKS_COUNT) * BLOCK_SIZE_X;
#if OUTPUT_DIMS == 5
    const uint zy = zyx / BLOCKS_COUNT;
    const uint z = zy / OUTPUT_SIZE_Y;
    const uint y = zy % OUTPUT_SIZE_Y;
#else
    const uint z = 0;
    const uint y = zyx / BLOCKS_COUNT;
#endif

    // Output offset calculations:
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
#if OUTPUT_DIMS == 5
    const uint output_z_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_fs_pitch = output_z_pitch * (OUTPUT_PAD_BEFORE_SIZE_Z + OUTPUT_SIZE_Z + OUTPUT_PAD_AFTER_SIZE_Z);
#else
    const uint output_z_pitch = 0;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
#endif
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset = b * output_b_pitch +
                               (f_block + output_fs_pad_before) * output_fs_pitch +
                               (z + OUTPUT_PAD_BEFORE_SIZE_Z) * output_z_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

#if BLOCK_SIZE != 1
    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE) res;
#else
    ACCUMULATOR_TYPE res;
#endif

    DO_ELTWISE

#if HAS_FUSED_OPS
#if FEATURE_SLICE_SIZE == 32
    OUTPUT_TYPE out_arr[BLOCK_SIZE];
    OUTPUT_TYPE_BLOCK out;
    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 2) tmp_res;
    for (int block_x = 0; block_x < BLOCK_SIZE_X; block_x++) {
        tmp_res.s0 = res[block_x * 2 + 0];
        tmp_res.s1 = res[block_x * 2 + 1];
        FUSED_OPS;
        out[block_x * 2 + 0] = TO_TYPE(OUTPUT_TYPE, (FUSED_OPS_RESULT).s0);
        out[block_x * 2 + 1] = TO_TYPE(OUTPUT_TYPE, (FUSED_OPS_RESULT).s1);
    }
#else
    FUSED_OPS;
    OUTPUT_TYPE_BLOCK out = TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, BLOCK_SIZE), FUSED_OPS_RESULT);
#endif
#else
#if BLOCK_SIZE != 1
    OUTPUT_TYPE_BLOCK out = ACTIVATION_TYPED(TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, BLOCK_SIZE), res), ACTIVATION_PARAMS_TYPED);
#else
    OUTPUT_TYPE out = ACTIVATION_TYPED(TO_OUTPUT_TYPE(res), ACTIVATION_PARAMS_TYPED);
#endif
#endif

#ifdef LEFTOVERS
    if ((f_block + 1) * FEATURE_SLICE_SIZE > OUTPUT_FEATURE_NUM) {
        const uint sglid = get_sub_group_local_id();
        if (sglid < OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE) {
            for (uint block_x = 0; block_x < BLOCK_SIZE_X; block_x++) {
#if BLOCK_SIZE_X != 1 && FEATURE_SLICE_SIZE == 16
                output[output_offset + block_x * output_x_pitch + sglid] = out[block_x];
#elif BLOCK_SIZE_X != 1 && FEATURE_SLICE_SIZE == 32
                output[output_offset + block_x * output_x_pitch + sglid] = out[block_x * 2];
                if (f_block * FEATURE_SLICE_SIZE + sglid + SIMD_SIZE < OUTPUT_FEATURE_NUM)
                    output[output_offset + block_x * output_x_pitch + sglid + SIMD_SIZE] = out[block_x * 2 + 1];
#else
                output[output_offset + block_x * output_x_pitch + sglid] = out;
#endif
            }
        }
    } else
#endif
    {
        WRITE_FUNC(output, output_offset, out);
    }

}

#undef VEC_TO_ARRAY_2
#undef VEC_TO_ARRAY_4
#undef VEC_TO_ARRAY_8
#undef VEC_TO_ARRAY_16
#undef VEC_TO_ARRAY

#undef ARRAY_TO_VEC_2
#undef ARRAY_TO_VEC_4
#undef ARRAY_TO_VEC_8
#undef ARRAY_TO_VEC_16
#undef ARRAY_TO_VEC

#undef SIMD_SIZE
#undef OUTPUT_TYPE_BLOCK
#undef TO_TYPE
#undef READ_FUNC
#undef WRITE_FUNC
#undef GET_INDEX
