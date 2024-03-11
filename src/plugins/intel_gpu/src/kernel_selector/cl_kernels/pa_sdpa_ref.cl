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
// constexpr NUM_QUERIES_PER_KV_HEAD (HEADS_NUM / KV_HEADS_NUM)
// constexpr size_t BLOCK_SIZE = 16;
// constexpr size_t X_BLOCK_SIZE = 4;

// constexpr size_t MAX_SEQUENCE_LENGTH = 1024;



#define SUB_GROUP_SIZE 16
#define SUBGROUPS_PER_WG (HEAD_SIZE / SUB_GROUP_SIZE)

// The size of portion of HEAD_SIZE each WI process
#define Q_LOAD_ITERS (HEAD_SIZE / SUB_GROUP_SIZE)

// How much QK outputs each subgroup calculates per cycle
#define QK_VALS_PER_SG_PER_ITER (BLOCK_SIZE / SUBGROUPS_PER_WG)

#define KV_CACHE_BLOCK_STRIDE (HEAD_SIZE * KV_HEADS_NUM * BLOCK_SIZE)

#define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)


REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, HEAD_SIZE)))
KERNEL(pa_sdpa_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* query,
    __global const INPUT1_TYPE* key_cache,
    __global const INPUT2_TYPE* value_cache,
    __global const INPUT3_TYPE* max_context_len,
    __global const INPUT4_TYPE* context_lens,
    __global const INPUT5_TYPE* block_tables,
    __global const INPUT6_TYPE* scale,
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

    // if (seq_idx < 2 && head_num_idx < 2 && sgid < 2 && sglid < 2) {
    //     if (INPUT5_FEATURE_NUM == 0) {
    //         printf("Empty blocks. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n",
    //         seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //     } else if (INPUT5_FEATURE_NUM == 1) {
    //         printf("Blocks table[b=0]: %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0],
    //         seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //     } else if (INPUT5_FEATURE_NUM == 2) {
    //         printf("Blocks table[b=0]: %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1],
    //         seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //     } else if (INPUT5_FEATURE_NUM == 3) {
    //         printf("Blocks table[b=0]: %d %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1], block_tables[2],
    //         seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //     } else if (INPUT5_FEATURE_NUM == 4) {
    //         printf("Blocks table[b=0]: %d %d %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1], block_tables[2], block_tables[3],
    //         seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //     }

    //     if (seq_idx == 0 && head_num_idx == 0 && sgid == 0 && sglid == 0) {
    //         printf("key_cache[405504]=%f\n", key_cache[405504]);
    //         printf("value_cache[405504]=%f\n", value_cache[405504]);
    //     }
    // }

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

        OUTPUT_TYPE qk[QK_VALS_PER_SG_PER_ITER] = {0};

        for (uint hs = 0; hs < Q_LOAD_ITERS; hs++) {
            const uint query_idx = seq_idx * HEAD_SIZE * HEADS_NUM +
                                   head_num_idx * HEAD_SIZE +
                                   hs * SUB_GROUP_SIZE;

            // TODO: can be preloaded outside Q_LOAD_ITERS loop - need to check perf
            INPUT0_TYPE q = QUERY_BLOCK_READ(query, query_idx);
            for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                uint current_token = block * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
                if (current_token >= context_len)
                    continue;

                const uint key_idx = block_offset +
                                     (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) +
                                     (X_BLOCK_SIZE * QK_VALS_PER_SG_PER_ITER) * sgid +
                                     (SUB_GROUP_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) * hs +
                                     (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE * BLOCK_SIZE +
                                     (sglid % X_BLOCK_SIZE) + qk_idx * X_BLOCK_SIZE;

                // TODO1: try block loading and shuffling
                // TODO2: try to load k*4 times and then calculate
                // TODO3: try bigger X block
                INPUT1_TYPE k = key_cache[key_idx];


                // if (seq_idx == 0 && head_num_idx == 0) {
                //     printf("main_calc: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d, block=%d, hs=%d, qk_idx=%d, current_token=%d, query_idx=%d, key_idx=%d (block_offset=%d): %f * %f\n",
                //         seq_idx, head_num_idx, sgid, sglid, block, hs, qk_idx, current_token, query_idx, key_idx - block_offset, block_offset, q, k);
                // }

                qk[qk_idx] = mad(q, k, qk[qk_idx]);
            }
        }

        // Summurize qk calculation across all WIs and apply scale
        for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
            const uint current_token = block * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
            if (current_token < context_len) {
                OUTPUT_TYPE tmp_print = qk[qk_idx];
                qk[qk_idx] = sub_group_reduce_add(qk[qk_idx]);
                // if (head_num_idx < 4)
                //     printf("final_calc: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: before qk[%d]=%f, after=%f\n",
                //             seq_idx, head_num_idx, sgid, sglid, qk_idx, tmp_print, qk[qk_idx]);
                qk[qk_idx] = scale[0] * qk[qk_idx];
                qk_max = OUTPUT_MAX_FUNC(qk_max, qk[qk_idx]);
            }
        }

        // Save QK results to local memory
        if (sglid < QK_VALS_PER_SG_PER_ITER) {
            const uint current_token = block * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
            // Fixed -> // const uint qk_local_idx = block * BLOCK_SIZE * sgid * QK_VALS_PER_SG_PER_ITER + sglid;
            // OUTPUT_TYPE tmp_print = (current_token >= context_len ? 0 : qk[sglid]);
            // if (head_num_idx < 4 || head_num_idx == 31)
            //     printf("slm save: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: qk_vals[%d]=%f. Max=%f\n",
            //             seq_idx, head_num_idx, sgid, sglid, current_token, tmp_print, qk_max);
            qk_vals[current_token] = current_token >= context_len ? 0 : qk[sglid];
        }
    }

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

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        //     printf("QK max value = %f\n", qk_max);
        // }

        OUTPUT_TYPE exp_sum = OUTPUT_VAL_ZERO;
        for (uint qk_idx = 0; qk_idx < CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE); qk_idx++) {
            const uint data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            if (data_idx < context_len) {
                OUTPUT_TYPE val = native_exp(qk_vals[data_idx] - qk_max);
                exp_sum += val;
                qk_vals[data_idx] = val;
                // if (head_num_idx < 4 || head_num_idx == 31)
                //     printf("head_num %d, sgid = %d, sglid = %d, exp_sum = %f\n", head_num_idx, sgid, sglid, exp_sum);
            }
        }

        exp_sum = sub_group_reduce_add(exp_sum);

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        //     printf("exp_sum final value = %f\n", exp_sum);
        // }

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = OUTPUT_VAL_ZERO;

        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Final sum of all values
        exp_sum = sub_group_reduce_add(exp_sum);

        const OUTPUT_TYPE inv_sum = OUTPUT_VAL_ONE / exp_sum;


        // TODO: replace CEIL_DIV with ALIGN and use += SUBGROUPS_PER_WG * SUB_GROUP_SIZE increment
        for (uint qk_idx = 0; qk_idx < CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE); qk_idx++) {
            const uint data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            if (data_idx < context_len) {
                OUTPUT_TYPE val = qk_vals[data_idx] * inv_sum;
                qk_vals[data_idx] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // if (seq_idx == 0 && sgid == 0 && sglid == 0) {
    //     for (uint i = 0; i < context_len; i++) {
    //         printf("Softmax res for %d head: %d. %f\n", head_num_idx, i, qk_vals[i]);
    //     }
    // }

    {
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;

        for (uint qk_idx = 0; qk_idx < ALIGN(context_len, SUB_GROUP_SIZE); qk_idx += SUB_GROUP_SIZE) {
            const uint qk_offset = qk_idx + sglid;

            OUTPUT_TYPE qk = qk_offset < context_len ? qk_vals[qk_offset] : OUTPUT_VAL_ZERO;

            const uint block_idx = block_tables[batch_idx * blocks_num + (qk_idx / BLOCK_SIZE)];
            if (block_idx == 0)
                continue;

            const uint value_cache_offset = block_idx * KV_CACHE_BLOCK_STRIDE +
                                            (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE * BLOCK_SIZE) +
                                            sgid * (SUB_GROUP_SIZE * BLOCK_SIZE) +
                                            sglid * BLOCK_SIZE +
                                            ((qk_idx / SUB_GROUP_SIZE) % (BLOCK_SIZE / SUB_GROUP_SIZE)) * SUB_GROUP_SIZE;

            #define VALUE_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, BLOCK_SIZE)
            #define VALUE_VLOAD(offset, ptr) CAT(vload, BLOCK_SIZE)(offset, ptr)

            ushort16 v_tmp = vload16(0, (__global ushort*)(value_cache + value_cache_offset));
            OUTPUT_TYPE* v = (OUTPUT_TYPE*)&v_tmp;

            // VALUE_VEC_TYPE* tmp_print = v;

            // if (seq_idx == 0 && head_num_idx == 0) {
            //     printf("gemm2: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d, block_idx=%d, qk_idx=%d, qk_offset=%d, value_offset=%d (block_offset=%d): %v8f\n",
            //         seq_idx, head_num_idx, sgid, sglid, block_idx, qk_idx, qk_offset, value_cache_offset - (block_idx * KV_CACHE_BLOCK_STRIDE), block_idx * KV_CACHE_BLOCK_STRIDE, *tmp_print);
            // }

            for (uint token = 0; token < SUB_GROUP_SIZE; token++) {
                OUTPUT_TYPE qk_tmp = sub_group_broadcast(qk, token);
                if (qk_idx + token < context_len) {
                    acc = mad(qk_tmp, v[token], acc);
                }
            }
        }


        const uint output_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                   head_num_idx * HEAD_SIZE +
                                   sgid * SUB_GROUP_SIZE +
                                   sglid;

        // if (seq_idx == 0 && head_num_idx < 2 || head_num_idx == 31) {
        //     printf("output res: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: output[%d] = %f\n",
        //         seq_idx, head_num_idx, sgid, sglid, output_offset, acc);
        // }

        output[output_offset] = acc;
    }
}
