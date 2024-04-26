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
// exp_sums      [batch, heads_num, q_len, partition_idx]
// max_logits    [batch, heads_num, q_len, partition_idx]
// tmp_out       [batch, heads_num, q_len, partition_idx, head_size]

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

#ifdef SDPA_STAGE_0

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
    const __global INPUT3_TYPE* attn_mask,
    __global OUTPUT_TYPE* output,
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
)
{
    const uint batch_head_num_idx = get_global_id(0);
    const uint batch_idx = batch_head_num_idx / INPUT0_FEATURE_NUM;

    /* RENAME HEAD_NUM_IDX TO HEAD_IDX */

    const uint head_num_idx = batch_head_num_idx % INPUT0_FEATURE_NUM;

    /* RENAME HEAD_NUM_IDX TO HEAD_IDX */

    const uint seq_idx = get_global_id(1);
    const uint head_size_idx = get_local_id(2);
    // uint head_size_idx = get_global_id(2);
    const uint lid = get_local_id(2);

    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint partition_idx = get_group_id(2);
    const uint num_of_partitions = get_num_groups(2);
    const uint wi_num_per_partition = get_local_size(2);

    const uint partition_seq_len =
        ((partition_idx + 1) < num_of_partitions) ? (SEQ_LEN_PARTITION_SIZE)
                                                  : (TOTAL_SEQ_LEN - partition_idx * SEQ_LEN_PARTITION_SIZE);

    // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_local_id(2) == 0) {
    //     printf("Main kernel partition_idx=%d, partition_seq_len=%d\n", partition_idx, partition_seq_len);
    // }

    __local OUTPUT_TYPE qk_vals_local[SLM_SIZE];
    SOFTMAX_ACCUMULATOR_TYPE qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;

#ifndef INPUT4_TYPE
    const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(HEAD_SIZE));
#endif

    // __local INPUT0_TYPE query_vals_local[HEAD_SIZE];
    const uint start_partition_idx = partition_idx * SEQ_LEN_PARTITION_SIZE;

    { // start Gemm1
    #define QUERY_BLOCK_SIZE 8
    #define QUERY_BLOCK_READ_NEW(ptr, offset) BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, ptr, offset)
    #define QUERY_BLOCK_NEW MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERY_BLOCK_SIZE)

    const uint query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx, 0);
    QUERY_BLOCK_NEW query_vals = QUERY_BLOCK_READ_NEW(query_input, query_offset);
    // query_vals_local[head_size_idx] = QUERY_BLOCK_READ(query_input, query_offset);

    // barrier(CLK_LOCAL_MEM_FENCE);

    /* Calculate Gemm1 */
    for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE / SUBGROUP_SIZE)) {
        uint key_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + seq_len, 0);

        INPUT0_TYPE acc = INPUT0_VAL_ZERO;

#define MULS_NUM 2
#define KEY_BLOCK_READ_NEW(ptr, offset) BLOCK_READN(INPUT1_TYPE, MULS_NUM, ptr, offset)
#define KEY_BLOCK_NEW MAKE_VECTOR_TYPE(INPUT1_TYPE, MULS_NUM)

        unroll_for (uint h = 0; h < HEAD_SIZE / SUBGROUP_SIZE / MULS_NUM; h++) {
            KEY_BLOCK_NEW key_vec = KEY_BLOCK_READ_NEW(key_input, key_offset);

            unroll_for (uint i = 0; i < MULS_NUM; i++) {
#if MULS_NUM == 1
                acc = mad(query_vals[h * MULS_NUM + i], key_vec, acc);
#else
                acc = mad(query_vals[h * MULS_NUM + i], key_vec[i], acc);
#endif
            }

            key_offset += SUBGROUP_SIZE * MULS_NUM;
        }

        acc = sub_group_reduce_add(acc);

        if (sglid == 0) {
            // Apply scale
            acc *= scale_val;

            // Apply attention mask
            uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(batch_idx, head_num_idx, seq_idx, start_partition_idx + seq_len);
            acc += attn_mask[attn_mask_offset];

            // Update qk_max value
            qk_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max, TO_SOFTMAX_ACCUMULATOR_TYPE(acc));

            qk_vals_local[seq_len] = acc;
        }
    }
    } // finish Gemm1

    /* Apply SoftMax */
    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG];
    {
        // Find the maximum value of qk in the subgroup
        qk_max = sub_group_reduce_max(qk_max);

        // Find the maximum value of qk across all subgroups in the workgroup
        if (sglid == 0)
            qk_max_vals[sgid] = qk_max;

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG)
            qk_max = qk_max_vals[sglid];

        // Final maximum value of qk after reduction across all subgroups
        qk_max = sub_group_reduce_max(qk_max);

        SOFTMAX_ACCUMULATOR_TYPE exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        const uint qk_num_per_wi = CEIL_DIV(partition_seq_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
        for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            if (local_data_idx < partition_seq_len) {
                SOFTMAX_ACCUMULATOR_TYPE qk_new = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) - qk_max);
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                exp_sum += qk_new;
            }
        }

        exp_sum = sub_group_reduce_add(exp_sum);

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;

        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Find the final sum of all exp_sum values in workgroup
        exp_sum = sub_group_reduce_add(exp_sum);

        const SOFTMAX_ACCUMULATOR_TYPE inv_sum = SOFTMAX_ACCUMULATOR_VAL_ONE / exp_sum;
        for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            if (local_data_idx < partition_seq_len) {
                SOFTMAX_ACCUMULATOR_TYPE qk_new = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) * inv_sum;
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(qk_new);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        {
            // Save temporary exm_sums and max_logits values for each partition
            if (num_of_partitions > 1 && lid == 0) {
                const uint exp_sums_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * num_of_partitions) +
                                             head_num_idx * (INPUT0_SIZE_Y * num_of_partitions) +
                                             seq_idx * (num_of_partitions) +
                                             partition_idx;
                exp_sums[exp_sums_offset] = exp_sum;

                // if (batch_idx == 0 && head_num_idx < 2 && get_global_id(1) < 2) {
                //     printf("head_num_idx=%d seq_id=%d, partition_idx=%d(len=%d) Main kernel save exp_sum %f, max_val %f; at offset %d\n",
                //           head_num_idx, seq_idx, partition_idx, partition_seq_len, exp_sum, qk_max, exp_sums_offset);
                // }

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = qk_max;
            }
        }
    }

    /* Calculate Gemm2 */
    {
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;
        for (uint seq_len = 0; seq_len < partition_seq_len / SUBGROUP_SIZE; seq_len++) {
            uint value_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);

            OUTPUT_TYPE qk_val = qk_vals_local[seq_len * SUBGROUP_SIZE + sglid];

            for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);
                acc = mad(sub_group_broadcast(qk_val, i), value_val, acc);

                value_offset += HEAD_SIZE;
            }
        }

        const uint seq_len_leftover_start = partition_seq_len / SUBGROUP_SIZE * SUBGROUP_SIZE;
        if (seq_len_leftover_start != partition_seq_len) {
            for (uint seq_len = seq_len_leftover_start; seq_len < partition_seq_len; seq_len++) {
                const uint value_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + seq_len, head_size_idx);

                /* Load seq_len / 16 + sglid and broadcast */
                OUTPUT_TYPE qk_val = qk_vals_local[seq_len];
                INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);

                acc = mad(qk_val, value_val, acc);
            }
        }

#ifdef OPTIMIZE_READ_OF_LEFTOVERS
        /* This shows worse performace then above approach */
        const uint seq_len_leftover_start = partition_seq_len / SUBGROUP_SIZE * SUBGROUP_SIZE;
        if (seq_len_leftover_start != partition_seq_len) {
            OUTPUT_TYPE qk_val = seq_len_leftover_start + sglid < partition_seq_len ? qk_vals_local[seq_len_leftover_start + sglid] : 0;
            for (uint seq_len = seq_len_leftover_start; seq_len < partition_seq_len; seq_len++) {
                const uint value_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + seq_len, head_size_idx);

                /* Load seq_len / 16 + sglid and broadcast */
                INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);

                acc = mad(sub_group_broadcast(qk_val, seq_len % SUBGROUP_SIZE), value_val, acc);
            }
        }
#endif

        if (num_of_partitions > 1) {
            // tmp_output data layout  [batch, heads_num, q_len, partition_idx, head_size]
            const uint tmp_out_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * num_of_partitions * HEAD_SIZE) +
                                        head_num_idx * (INPUT0_SIZE_Y * num_of_partitions * HEAD_SIZE) +
                                        seq_idx * (num_of_partitions * HEAD_SIZE) +
                                        partition_idx * (HEAD_SIZE) +
                                        head_size_idx;
            tmp_out[tmp_out_offset] = acc;
        } else {
            const uint output_offset = OUTPUT_GET_INDEX(batch_idx, head_num_idx, seq_idx, head_size_idx);

            output[output_offset] = acc;
        }
    }
}

#endif

#ifdef SDPA_STAGE_1

#if SOFTMAX_ACCUMULATOR_TYPE_SIZE == 4
#define REG_VERSION_MAX_VALUES_PER_WI 24
#elif SOFTMAX_ACCUMULATOR_TYPE_SIZE == 2
#define REG_VERSION_MAX_VALUES_PER_WI 48
#else
#error Unexpected SOFTMAX_ACCUMULATOR data type size
#endif

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// exp_sums      [batch, heads_num, q_len, partition_idx]
// max_logits    [batch, heads_num, q_len, partition_idx]
// tmp_out       [batch, heads_num, q_len, partition_idx, head_size]

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt_finalization_stage)(
    OPTIONAL_SHAPE_INFO_ARG
    __global OUTPUT_TYPE* output,
    const __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    const __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint num_of_partitions) {
    const uint batch_head_num_idx = get_global_id(0);
    const uint batch_idx = batch_head_num_idx / INPUT0_FEATURE_NUM;
    const uint head_num_idx = batch_head_num_idx % INPUT0_FEATURE_NUM;
    const uint seq_idx = get_global_id(1);
    const uint head_size_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();

    // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
    //     printf("Num of partitions is %d\n", num_of_partitions);
    // }

    if (num_of_partitions == 1) {
        /* Short path, just copies input to output */
        const uint out_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * HEAD_SIZE) +
                                head_num_idx * (INPUT0_SIZE_Y * HEAD_SIZE) +
                                seq_idx * (HEAD_SIZE) +
                                head_size_idx;
        output[out_offset] = tmp_out[out_offset];
    } else if (num_of_partitions <= SUBGROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI) {
        /* Registers kernel version, can handle up to SEQ_LEN_PARTITION_SIZE(256) * SUBGROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI(24/48) = 98304/196608 tokens */
        SOFTMAX_ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        SOFTMAX_ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * num_of_partitions) +
                                         head_num_idx * (INPUT0_SIZE_Y * num_of_partitions) +
                                         seq_idx * (num_of_partitions) +
                                         partition_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (partition_idx < num_of_partitions) {
                exp_sum[i] = exp_sums[exp_sums_offset];
                max_logit[i] = max_logits[max_logit_offset];
                local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logit[i]);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Update exp_sum with respect to the global maximum
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            if (partition_idx < num_of_partitions) {
                exp_sum[i] = exp_sum[i] * native_exp(max_logit[i] - global_max);
                local_exp_sum += exp_sum[i];
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
        for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
            const uint tmp_out_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * num_of_partitions * HEAD_SIZE) +
                                        head_num_idx * (INPUT0_SIZE_Y * num_of_partitions * HEAD_SIZE) +
                                        seq_idx * (num_of_partitions * HEAD_SIZE) +
                                        partition_idx * (HEAD_SIZE) +
                                        head_size_idx;
            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) *
                   TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[partition_idx / SUBGROUP_SIZE], partition_idx % SUBGROUP_SIZE)) /
                   TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * HEAD_SIZE) +
                                head_num_idx * (INPUT0_SIZE_Y * HEAD_SIZE) +
                                seq_idx * (HEAD_SIZE) +
                                head_size_idx;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    } else {
        /* Global memory kernel version, can handle any number of tokens */
        // SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        // SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        // const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        // for (uint i = 0; i < iters_num; i++) {
        //     const uint partition_idx = i * SUBGROUP_SIZE + sglid;
        //     const uint max_logit_offset = seq_offset * HEADS_NUM * num_of_partitions +
        //                                  head_num_idx * num_of_partitions + partition_idx;

        //     if (partition_idx < num_of_partitions) {
        //         local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logits[max_logit_offset]);
        //     }
        // }

        // SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // // Calculate global sum
        // for (uint i = 0; i < iters_num; i++) {
        //     const uint partition_idx = i * SUBGROUP_SIZE + sglid;
        //     const uint exp_sums_offset = seq_offset * HEADS_NUM * num_of_partitions +
        //                                  head_num_idx * num_of_partitions + partition_idx;
        //     const uint max_logit_offset = exp_sums_offset;

        //     if (partition_idx < num_of_partitions) {
        //         local_exp_sum += exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);
        //     }
        // }

        // SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        // SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
        // for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
        //     const uint tmp_out_offset = seq_offset * (HEADS_NUM * num_of_partitions * HEAD_SIZE) +
        //                                 head_num_idx * (num_of_partitions * HEAD_SIZE) +
        //                                 partition_idx * HEAD_SIZE +
        //                                 head_size_idx;

        //     const uint exp_sums_offset = seq_offset * HEADS_NUM * num_of_partitions +
        //                                  head_num_idx * num_of_partitions + partition_idx;
        //     const uint max_logit_offset = exp_sums_offset;

        //     SOFTMAX_ACCUMULATOR_TYPE new_exp_sum = exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);

        //     OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
        //     acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * new_exp_sum / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
        // }
        // const uint out_offset = seq_offset * (HEADS_NUM * HEAD_SIZE) +
        //                         head_num_idx * HEAD_SIZE +
        //                         head_size_idx;

        // output[out_offset] = TO_OUTPUT_TYPE(acc);
    }
}

#endif
