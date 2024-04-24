// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// tmp_buf       [batch, heads_num, q_len, kv_len]

// When dealing with long sequences and FP16 execution, accuracy may significantly vary depending on two factors:
// 1) The order of scale application (which can be controlled with the APPLY_SCALE_TO_QUERY macro).
// 2) The type of SoftMax accumulator; in most cases, an FP32 accumulator shows better accuracy,
//    but there are situations where FP16 appears more correct from a generated text perspective.

KERNEL(sdpa_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
    const __global INPUT3_TYPE* attn_mask,
    __global OUTPUT_TYPE* output,
    __global OUTPUT_TYPE* tmp_buf
)
{
    uint dim0 = get_global_id(0);
    uint batch_idx = dim0 / INPUT0_FEATURE_NUM;
    uint head_num_idx = dim0 % INPUT0_FEATURE_NUM;
    uint seq_idx = get_global_id(1);
    uint head_size_idx = get_global_id(2);

#ifndef INPUT4_TYPE
    #define APPLY_SCALE_TO_QUERY 1
    const OUTPUT_TYPE scale = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(INPUT1_SIZE_X));
#endif

    if (get_local_id(2) == 0) {
        for (uint s = 0; s < INPUT1_SIZE_Y /* seq_len */; s++) {
            OUTPUT_TYPE acc = 0;
            for (uint h = 0; h < INPUT0_SIZE_X /* head_size */; h++) {
                uint query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx, h);
                uint key_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, s, h);

#if APPLY_SCALE_TO_QUERY
                INPUT0_TYPE q_val = query_input[query_offset] * scale;
#else
                INPUT0_TYPE q_val = query_input[query_offset];
#endif
                INPUT1_TYPE k_val = key_input[key_offset];
                acc += q_val * k_val;
            }

#if !APPLY_SCALE_TO_QUERY
            const OUTPUT_TYPE scale = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(INPUT1_SIZE_X));
            acc *= scale;
#endif

            uint tmp_buf_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  head_num_idx * (INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  seq_idx * (INPUT1_SIZE_Y) + s;
            tmp_buf[tmp_buf_offset] = acc;
        }

        ACCUMULATOR_TYPE qk_max = ACCUMULATOR_VAL_MIN;
        for (uint s = 0; s < INPUT1_SIZE_Y /* seq_len */; s++) {
            uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(batch_idx, head_num_idx, seq_idx, s);
            uint tmp_buf_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  head_num_idx * (INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  seq_idx * (INPUT1_SIZE_Y) + s;

            OUTPUT_TYPE qk_val = tmp_buf[tmp_buf_offset] + attn_mask[attn_mask_offset];
            tmp_buf[tmp_buf_offset] = qk_val;

            qk_max = ACCUMULATOR_MAX_FUNC(qk_max, TO_ACCUMULATOR_TYPE(qk_val));
        }

        ACCUMULATOR_TYPE exp_sum = ACCUMULATOR_VAL_ZERO;
        for (uint s = 0; s < INPUT1_SIZE_Y /* seq_len */; s++) {
            uint tmp_buf_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  head_num_idx * (INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  seq_idx * (INPUT1_SIZE_Y) + s;

            OUTPUT_TYPE qk_val = tmp_buf[tmp_buf_offset];
            ACCUMULATOR_TYPE val = native_exp(TO_ACCUMULATOR_TYPE(qk_val) - qk_max);
            exp_sum += val;

            tmp_buf[tmp_buf_offset] = TO_OUTPUT_TYPE(val);
        }

        const ACCUMULATOR_TYPE inv_sum = ACCUMULATOR_VAL_ONE / exp_sum;
        for (uint s = 0; s < INPUT1_SIZE_Y /* seq_len */; s++) {
            uint tmp_buf_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  head_num_idx * (INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                                  seq_idx * (INPUT1_SIZE_Y) + s;

            OUTPUT_TYPE qk_val = tmp_buf[tmp_buf_offset];
            ACCUMULATOR_TYPE val = TO_ACCUMULATOR_TYPE(qk_val) * inv_sum;
            tmp_buf[tmp_buf_offset] = TO_OUTPUT_TYPE(val);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    OUTPUT_TYPE acc = 0;
    for (uint s = 0; s < INPUT2_SIZE_Y /* seq_len */; s++) {
        uint tmp_buf_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                              head_num_idx * (INPUT0_SIZE_Y * INPUT1_SIZE_Y) +
                              seq_idx * (INPUT1_SIZE_Y) + s;
        uint value_offset = INPUT2_GET_INDEX(batch_idx, head_num_idx, s, head_size_idx);

        acc += tmp_buf[tmp_buf_offset] * value_input[value_offset];
    }

    uint output_offset = OUTPUT_GET_INDEX(batch_idx, head_num_idx, seq_idx, head_size_idx);
    output[output_offset] = acc;
}
