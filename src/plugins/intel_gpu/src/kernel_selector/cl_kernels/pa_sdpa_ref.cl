// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"





// constexpr size_t HEAD_SIZE = 64;
// constexpr size_t HEADS_NUM = 32;
// constexpr size_t KV_HEADS_NUM = 4;
// constexpr size_t BLOCK_SIZE = 16;
// constexpr size_t X_SIZE = 4;

// constexpr size_t MAX_SEQUENCE_LENGTH = 1024;



#define SUB_GROUP_SIZE 16

// The size of portion of HEAD_SIZE each WI process
#define HEAD_ITEMS_PER_WI (HEAD_SIZE / SUB_GROUP_SIZE)

// How much QK outputs each subgroup calculates per cycle
#define QK_PER_SG 4

#define KV_CACHE_BLOCK_STRIDE (HEAD_SIZE * HEADS_NUM * BLOCK_SIZE)

#define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)

#define SUBGROUPS_PER_WG HEAD_SIZE / SUB_GROUP_SIZE

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(pa_sdpa_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* query,
    __global const INPUT1_TYPE* key_cache,
    __global const INPUT2_TYPE* value_cache,
    __global const INPUT3_TYPE* max_context_len,
    __global const INPUT4_TYPE* context_lens,
    __global const INPUT5_TYPE* block_tables,
    __global OUTPUT_TYPE* output)
{
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();

    const uint batch_idx = seq_idx / INPUT0_FEATURE_NUM;
    const uint token_idx = seq_idx % INPUT0_FEATURE_NUM;

    const uint context_len = context_lens[batch_idx];

    const uint blocks_num = INPUT5_FEATURE_NUM;

    // sgid0: 0..3
    // sgid1: 4..7
    // sgid2: 8..11
    // sgid3: 12..15

    // sgid0: 16..19
    // sgid1: 20..23
    // sgid2: 24..27
    // sgid3: 28..31

    // TODO: Need to make blocks division more flexible. Current approach suggests
    // to have 4 SG per WG, where each SG process 4 QK outputs, so 16 in total per WG

    __local OUTPUT_TYPE qk_vals[SHARED_MEM_SIZE];

    OUTPUT_TYPE qk_max = OUTPUT_VAL_MIN;

    for (uint block = 0; block < blocks_num; block++) {
        const uint block_idx = batch_idx * blocks_num + block;
        const uint block_offset = block_tables[block_idx] * KV_CACHE_BLOCK_STRIDE;

        OUTPUT_TYPE qk[QK_PER_SG] = {0};

        for (uint hs = 0; hs < HEAD_ITEMS_PER_WI; hs++) {
            const uint query_idx = seq_idx * HEAD_SIZE * HEADS_NUM + hs * SUB_GROUP_SIZE;

            // TODO: can be preloaded outside HEAD_ITEMS_PER_WI loop - need to check perf
            INPUT0_TYPE q = QUERY_BLOCK_READ(query, query_idx);
            for (uint qk_idx = 0; qk_idx < QK_PER_SG; qk_idx++) {
                uint current_token = block * BLOCK_SIZE + sgid * QK_PER_SG + qk_idx;
                if (current_token >= context_len)
                    continue;

                const uint key_idx = block_offset +
                                     (X_SIZE * QK_PER_SG) * sgid +
                                     (HEAD_ITEMS_PER_WI * BLOCK_SIZE * X_SIZE) * hs +
                                     (sglid / X_SIZE) * X_SIZE * BLOCK_SIZE +
                                     (sglid % X_SIZE) + qk_idx * X_SIZE;
                // TODO1: try block loading and shuffling
                // TODO2: try to load k*4 times and then calculate
                // TODO3: try bigger X block
                INPUT1_TYPE k = key_cache[key_idx];

                qk[qk_idx] = mad(q, k, qk[qk_idx]);
            }
        }

        // Summurize qk calculation across all WIs
        for (uint qk_idx = 0; qk_idx < QK_PER_SG; qk_idx++) {
            qk[QK_PER_SG] = sub_group_reduce_add(qk[QK_PER_SG]);
            qk_max = OUTPUT_MAX_FUNC(qk_max, qk[QK_PER_SG]);
        }

        // Save QK results to local memory
        if (sglid < QK_PER_SG) {
            const uint qk_local_idx = block * BLOCK_SIZE * sgid * QK_PER_SG + sglid;
            qk_vals[qk_local_idx] = qk[sglid];
        }
    }

    /* WARNING NEED TO ADD BIAS BEFORE SOFTMAX */

    // Apply SoftMax operation
    __local OUTPUT_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local OUTPUT_TYPE qk_sum_vals[SUBGROUPS_PER_WG];
    {
        if (sglid == 0)
            qk_max_vals[sgid] = qk_max;

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = OUTPUT_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG)
            qk_max = qk_max_vals[sglid];

        // Final max value after reduction across of all SG and WI
        qk_max = sub_group_reduce_max(qk_max);

        OUTPUT_TYPE exp_sum = OUTPUT_VAL_ZERO;
        for (uint qk_idx = 0; qk_idx < CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE); qk_idx++) {
            const uint data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            if (data_idx < context_len) {
                OUTPUT_TYPE val = native_exp(qk_vals[data_idx] - qk_max);
                exp_sum += val;
                qk_vals[data_idx] = val;
            }
        }

        exp_sum = sub_group_reduce_add(exp_sum);

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Final sum of all values
        exp_sum = sub_group_reduce_add(exp_sum);

        const OUTPUT_TYPE inv_sum = OUTPUT_VAL_ONE / exp_sum;

        for (uint qk_idx = 0; qk_idx < CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE); qk_idx++) {
            const uint data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            if (data_idx < context_len) {
                OUTPUT_TYPE val = qk_vals[data_idx] * inv_sum;
                qk_vals[data_idx] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    output[seq_idx + sglid] = qk_vals[sglid % context_len];
}
