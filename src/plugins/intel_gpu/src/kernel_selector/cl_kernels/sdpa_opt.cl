// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// tmp_buf       [batch, heads_num, q_len, kv_len]

#if OUTPUT_TYPE_SIZE == 4
    #define VLOAD(offset, ptr) CAT(vload, SUBGROUP_SIZE)(offset, ptr)
#else
    #define VLOAD(offset, ptr) CAT(vload, SUBGROUP_SIZE)(offset, (__global ushort*)(ptr))
#endif
#define KEY_VEC_TYPE MAKE_VECTOR_TYPE(INPUT1_TYPE, SUBGROUP_SIZE)
#define AS_VALUE_VEC(val) CAT(as_, KEY_VEC_TYPE)(val)

#define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define VALUE_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT2_TYPE, 1, ptr, offset)

#define TOTAL_SEQ_LEN INPUT1_SIZE_Y

#define SUBGROUPS_PER_WG (HEAD_SIZE / SUBGROUP_SIZE)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
    const __global INPUT3_TYPE* attn_mask,
    __global OUTPUT_TYPE* output,
    __global ACCUMULATOR_TYPE* exp_sums,
    __global ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
)
{
    uint dim0 = get_global_id(0);
    uint batch_idx = dim0 / INPUT0_FEATURE_NUM;
    uint head_num_idx = dim0 % INPUT0_FEATURE_NUM;
    uint seq_idx = get_global_id(1);
    uint head_size_idx = get_global_id(2);

    const uint lid = get_local_id(2);
    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint partition_id = get_group_id(2);
    const uint num_of_partitions = get_num_groups(2);
    const uint wi_num_per_partition = get_local_size(2);

    const uint partition_seq_len =
        ((partition_id + 1) < num_of_partitions) ? (SEQ_LEN_PARTITION_SIZE)
                                                : (TOTAL_SEQ_LEN % SEQ_LEN_PARTITION_SIZE);

    __local OUTPUT_TYPE qk_vals_local[SLM_SIZE];
    ACCUMULATOR_TYPE qk_max = ACCUMULATOR_VAL_MIN;

#ifndef INPUT4_TYPE
    const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(HEAD_SIZE));
#endif

    /* Calculate Gemm1 */
    for (uint seq_len = lid; seq_len < partition_seq_len; seq_len += wi_num_per_partition) {
        uint query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx, 0);
        uint key_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, /* TODO: start_partition_idx + seq_len */ seq_len, 0);

        INPUT0_TYPE acc = INPUT0_VAL_ZERO;
        unroll_for (uint h = 0; h < HEAD_SIZE; h += SUBGROUP_SIZE) {
            INPUT0_TYPE query_val = QUERY_BLOCK_READ(query_input, query_offset);
            KEY_VEC_TYPE key_vec = AS_VALUE_VEC(VLOAD(0, key_input + key_offset));

            unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                acc = mad(sub_group_broadcast(query_val, i), key_vec[i], acc);
            }

            query_offset += SUBGROUP_SIZE;
            key_offset += SUBGROUP_SIZE;
        }

        // Apply scale
        acc *= scale_val;

        // Apply attention mask
        uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(batch_idx, head_num_idx, seq_idx, /* TODO: start_partition_idx + seq_len */ seq_len);
        acc += attn_mask[attn_mask_offset];

        // Update qk_max value
        qk_max = ACCUMULATOR_MAX_FUNC(qk_max, TO_ACCUMULATOR_TYPE(acc));

        qk_vals_local[seq_len] = acc;
    }

    /* Apply SoftMax */
    __local ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG];
    {
        // Find the maximum value of qk in the subgroup
        qk_max = sub_group_reduce_max(qk_max);

        // Find the maximum value of qk across all subgroups in the workgroup
        if (sglid == 0)
            qk_max_vals[sgid] = qk_max;

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = ACCUMULATOR_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG)
            qk_max = qk_max_vals[sglid];

        // Final maximum value of qk after reduction across all subgroups
        qk_max = sub_group_reduce_max(qk_max);

        ACCUMULATOR_TYPE exp_sum = ACCUMULATOR_VAL_ZERO;
        const uint qk_num_per_wi = CEIL_DIV(partition_seq_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
        for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            if (local_data_idx < partition_seq_len) {
                ACCUMULATOR_TYPE qk_new = native_exp(TO_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) - qk_max);
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                exp_sum += qk_new;
            }
        }

        exp_sum = sub_group_reduce_add(exp_sum);

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = ACCUMULATOR_VAL_ZERO;

        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Find the final sum of all exp_sum values in workgroup
        exp_sum = sub_group_reduce_add(exp_sum);

        const ACCUMULATOR_TYPE inv_sum = ACCUMULATOR_VAL_ONE / exp_sum;
        for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            if (local_data_idx < partition_seq_len) {
                ACCUMULATOR_TYPE qk_new = TO_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) * inv_sum;
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(qk_new);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        {
            // Save temporary exm_sums and max_logits values for each portion
            if (num_of_partitions > 1 && sgid == 0) {
                const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_partitions +
                                             head_num_idx * num_of_partitions +
                                             partition_id;
                exp_sums[exp_sums_offset] = exp_sum;

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = qk_max;
            }
        }
    }

    /* Calculate Gemm2 */
    {
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;
        for (uint seq_len = 0; seq_len < partition_seq_len; seq_len++) {
            const uint value_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, /* TODO: start_partition_idx + seq_len */ seq_len, head_size_idx);

            /* Load seq_len / 16 + sglid */
            OUTPUT_TYPE qk_val = qk_vals_local[seq_len];
            INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);

            acc = mad(qk_val, value_val, acc);
        }

        if (num_of_partitions > 1) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * num_of_partitions) +
                                        head_num_idx * (HEAD_SIZE * num_of_partitions) +
                                        partition_id * HEAD_SIZE +
                                        sgid * SUBGROUP_SIZE +
                                        sglid;

            // tmp_output data layout [num_seqs, num_heads, num_portions, head_size]
            tmp_out[tmp_out_offset] = acc;
        } else {
            const uint output_offset = OUTPUT_GET_INDEX(batch_idx, head_num_idx, seq_idx, head_size_idx);

            output[output_offset] = acc;
        }
    }
}
