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
// define Q_LOAD_PER_ITER 4
#define Q_LOAD_ITERS (HEAD_SIZE / SUB_GROUP_SIZE)

// How much QK outputs each subgroup calculates per cycle
#define QK_VALS_PER_SG_PER_ITER (BLOCK_SIZE / SUBGROUPS_PER_WG)

#define KV_CACHE_BLOCK_STRIDE (HEAD_SIZE * KV_HEADS_NUM * BLOCK_SIZE)

#define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)

ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

#ifdef SDPA_STAGE_0

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, HEAD_SIZE)))
KERNEL(pa_sdpa_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* max_context_len,
    const __global INPUT4_TYPE* context_lens,
    const __global INPUT5_TYPE* block_tables,
    const __global INPUT6_TYPE* scale,
#ifdef USE_SPLIT_ACROSS_SEQ_LEN
    const __global OUTPUT_TYPE* output,
    __global ACCUMULATOR_TYPE* exp_sums,
    __global ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out,
    const uint num_of_portions
#else
    __global OUTPUT_TYPE* output
#endif
) {
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();

    const uint batch_idx = seq_idx / INPUT0_FEATURE_NUM;
    const uint token_idx = seq_idx % INPUT0_FEATURE_NUM;

    const uint context_len = context_lens[batch_idx];

    const uint blocks_pitch = INPUT5_FEATURE_NUM;

#ifdef USE_SPLIT_ACROSS_SEQ_LEN
    const uint portion_id = get_group_id(2);
    const uint block_start_idx = portion_id * SEQ_LEN_PORTION_SIZE / BLOCK_SIZE;

    if (portion_id * SEQ_LEN_PORTION_SIZE >= context_len) {
        return;
    }
#else
    const uint block_start_idx = 0;
#endif

    const uint total_blocks_num = CEIL_DIV(context_len, BLOCK_SIZE);

    // if (seq_idx < 2 && head_num_idx < 2 && sgid < 2 && sglid < 2) {
    //     if (INPUT5_BATCH_NUM == 2) {
    //         if (INPUT5_FEATURE_NUM == 0) {
    //             printf("Empty blocks. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n",
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 1) {
    //             printf("Blocks table[b=0..1]: %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 2) {
    //             printf("Blocks table[b=0..3]: %d %d %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1], block_tables[2], block_tables[3],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 3) {
    //             printf("Blocks table[b=0..6]: %d %d %d %d %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1], block_tables[2], block_tables[3], block_tables[4], block_tables[5],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 4) {
    //             printf("Blocks table[b=0]: %d %d %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1], block_tables[2], block_tables[3],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         }
    //     } else {
    //         if (INPUT5_FEATURE_NUM == 0) {
    //             printf("Empty blocks. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n",
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 1) {
    //             printf("Blocks table[b=0]: %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 2) {
    //             printf("Blocks table[b=0]: %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 3) {
    //             printf("Blocks table[b=0]: %d %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1], block_tables[2],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         } else if (INPUT5_FEATURE_NUM == 4) {
    //             printf("Blocks table[b=0]: %d %d %d %d. Seq_idx=%d, head_num_idx=%d, head_idx=%d, sglid=%d, sgid=%d, batch_idx=%d, token_idx=%d, context_len=%d, scale=%f\n", block_tables[0], block_tables[1], block_tables[2], block_tables[3],
    //             seq_idx, head_num_idx, head_idx, sglid, sgid, batch_idx, token_idx, context_len, scale[0]);
    //         }
    //     }

    //     // if (seq_idx == 0 && head_num_idx == 0 && sgid == 0 && sglid == 0) {
    //     //     printf("key_cache[0]=%f  key_cache[%d]=%f\n", key_cache[0], KV_CACHE_BLOCK_STRIDE * 2399, key_cache[KV_CACHE_BLOCK_STRIDE * 2399]);
    //     //     printf("value_cache[0]=%f value_cache[%d]=%f\n", value_cache[0], KV_CACHE_BLOCK_STRIDE * 2399, value_cache[KV_CACHE_BLOCK_STRIDE * 2399]);
    //     // }
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

    ACCUMULATOR_TYPE qk_max = ACCUMULATOR_VAL_MIN;


    {
        ulong timer_start = intel_get_cycle_counter();
        INPUT0_TYPE q[HEAD_SIZE / SUB_GROUP_SIZE];
        unroll_for (uint i = 0; i < HEAD_SIZE / SUB_GROUP_SIZE; i++) {
            const uint query_idx = seq_idx * HEAD_SIZE * HEADS_NUM +
                                    head_num_idx * HEAD_SIZE +
                                    i * SUB_GROUP_SIZE;
            q[i] = QUERY_BLOCK_READ(query, query_idx);
        }

#ifdef USE_SPLIT_ACROSS_SEQ_LEN
        // FINAL: Compile time restriction: devisible SEQ_LEN_PORTION_SIZE / BLOCK_SIZE
        const uint blocks_num = (portion_id == num_of_portions - 1) ? (total_blocks_num - (portion_id * SEQ_LEN_PORTION_SIZE / BLOCK_SIZE))
                                                                    : (SEQ_LEN_PORTION_SIZE / BLOCK_SIZE);
#else
        const uint blocks_num = total_blocks_num;
#endif
        for (uint block_num = 0; block_num < blocks_num; block_num++) {
            const uint block_idx = batch_idx * blocks_pitch + block_start_idx + block_num;
            const uint block_offset = block_tables[block_idx] * KV_CACHE_BLOCK_STRIDE;

            OUTPUT_TYPE qk[QK_VALS_PER_SG_PER_ITER] = {0};

            ulong timer2 = intel_get_cycle_counter();
            for (uint hs = 0; hs < Q_LOAD_ITERS; hs++) {
                for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                    uint current_token = (block_start_idx + block_num) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
                    if (current_token >= context_len)
                        continue;

                    const uint key_idx = block_offset +
                                        (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) +
                                        (X_BLOCK_SIZE * QK_VALS_PER_SG_PER_ITER) * sgid +
                                        (SUB_GROUP_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) * hs +
                                        (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE * BLOCK_SIZE +
                                        (sglid % X_BLOCK_SIZE) + qk_idx * X_BLOCK_SIZE;


                    // if (get_global_id(1) == 0 && get_global_id(2) == 0 && hs == 0) {
                    //     printf("test=%d, %d seq_idx=%d (b=%d, t=%d), %d %d: q_idx = %d, k_idx= %d, block_num=%d\n", get_global_size(0), INPUT0_FEATURE_NUM, get_global_id(0),
                    //     batch_idx, token_idx,
                    //     get_global_id(1), get_global_id(2), seq_idx * HEAD_SIZE * HEADS_NUM + head_num_idx * HEAD_SIZE, key_idx, block_num);
                    // }

                    // TODO1: try block_num loading and shuffling
                    // TODO2: try to load k*4 times and then calculate
                    // TODO3: try bigger X block_num
                    #if X_BLOCK_SIZE == 16
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset)
                    INPUT1_TYPE k = KEY_BLOCK_READ(key_cache, key_idx);
                    #else
                    INPUT1_TYPE k = key_cache[key_idx];
                    #endif


                    // if (seq_idx == 0 && head_num_idx == 0) {
                    //     printf("main_calc: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d, block_num=%d, hs=%d, qk_idx=%d, current_token=%d, query_idx=%d, key_idx=%d (block_offset=%d): %f * %f\n",
                    //         seq_idx, head_num_idx, sgid, sglid, block_num, hs, qk_idx, current_token, query_idx, key_idx - block_offset, block_offset, q, k);
                    // }

                    qk[qk_idx] = mad(q[hs], k, qk[qk_idx]);
                }
            }
            ulong timer3 = intel_get_cycle_counter();

            // Summurize qk calculation across all WIs and apply scale
            for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                const uint current_token = (block_start_idx + block_num) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
                if (current_token < context_len) {
                    OUTPUT_TYPE tmp_print = qk[qk_idx];
                    qk[qk_idx] = sub_group_reduce_add(qk[qk_idx]);
                    // if (head_num_idx < 4)
                    //     printf("final_calc: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: before qk[%d]=%f, after=%f\n",
                    //             seq_idx, head_num_idx, sgid, sglid, qk_idx, tmp_print, qk[qk_idx]);
                    qk[qk_idx] = scale[0] * qk[qk_idx];

                    // FINAL: Apply attention mask at prefill stage
                    if (INPUT0_FEATURE_NUM > 1 && current_token > token_idx) {
                        qk[qk_idx] = qk[qk_idx] + OUTPUT_VAL_MIN;
                        // printf("Before: qk[%d]=%f. qk_max=%f, max_res=%f\n", qk_idx, qk[qk_idx], qk_max, ACCUMULATOR_MAX_FUNC(qk_max, convert_float(qk[qk_idx])));
                    }

                    // if (get_global_id(0) == 0 && get_global_id(1) == 0 && max_context_len[0] == 17) {
                    //     printf("%d %d %d. block_num %d. QK max value = %f, qk[%d] = %f\n", batch_idx, head_num_idx, head_idx, block_num, qk_max, qk_idx, qk[qk_idx]);
                    // }
                    qk_max = ACCUMULATOR_MAX_FUNC(qk_max, TO_ACCUMULATOR_TYPE(qk[qk_idx]));
                }
            }
            ulong timer4 = intel_get_cycle_counter();

            // Save QK results to local memory
            if (sglid < QK_VALS_PER_SG_PER_ITER) {
                const uint current_token_global_idx = (block_start_idx + block_num) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
#ifdef USE_SPLIT_ACROSS_SEQ_LEN
                const uint current_token_local = block_num * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
#else
                const uint current_token_local = current_token_global_idx;
#endif
                // Fixed -> // const uint qk_local_idx = block_num * BLOCK_SIZE * sgid * QK_VALS_PER_SG_PER_ITER + sglid;
                // OUTPUT_TYPE tmp_print = (current_token_local >= context_len ? 0 : qk[sglid]);
                // if (head_num_idx < 4 || head_num_idx == 31)
                //     printf("slm save: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: qk_vals[%d]=%f. Max=%f\n",
                //             seq_idx, head_num_idx, sgid, sglid, current_token_local, tmp_print, qk_max);
                qk_vals[current_token_local] = current_token_global_idx >= context_len ? 0 : qk[sglid];
            }
            ulong timer5 = intel_get_cycle_counter();

            // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
            //     printf("SDPA kernel GEMM1 block_num %d: main_loop=%d, summarization=%d, saving=%d\n", block_num,
            //         (uint)(timer3 - timer2),
            //         (uint)(timer4 - timer3),
            //         (uint)(timer5 - timer4));
            // }
        }
        ulong timer_end = intel_get_cycle_counter();
        ulong total_time = timer_end - timer_start;

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_local_id(2) == 0 && context_len >= 496)
        //     printf("%d. %d. SDPA kernel GEMM1: %d; qk_max=%f, blocks_num=%d, total_blocks_num=%d, portion_id=%d, num_of_portions=%d\n",
        //         context_len, get_global_id(2), (uint)total_time, qk_max, blocks_num, total_blocks_num, portion_id, num_of_portions);
    }

    // barrier(CLK_LOCAL_MEM_FENCE);
    // if (get_global_id(1) == 0 && get_global_id(2) == 0) {
    //     const uint block_idx = batch_idx * blocks_num + 0;
    //     const uint block_offset = block_tables[block_idx] * KV_CACHE_BLOCK_STRIDE;
    //     uint key_idx =  block_offset +
    //                                     (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) +
    //                                     (X_BLOCK_SIZE * QK_VALS_PER_SG_PER_ITER) * sgid +
    //                                     (SUB_GROUP_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) * 0 +
    //                                     (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE * BLOCK_SIZE +
    //                                     (sglid % X_BLOCK_SIZE) + 0 * X_BLOCK_SIZE;
    //     printf("Intermidiate results for b=%d, seq=%d: %f %f %f %f %f. Q_idx=%d, q_val=%f. K_idx=%d, k_val=%f, block_idx=%d, block_tables[block_idx]=%d\n", batch_idx, token_idx, qk_vals[0], qk_vals[1], qk_vals[2], qk_vals[3], qk_vals[4],
    //     seq_idx * HEAD_SIZE * HEADS_NUM + head_num_idx * HEAD_SIZE, query[seq_idx * HEAD_SIZE * HEADS_NUM + head_num_idx * HEAD_SIZE],
    //     key_idx, key_cache[key_idx], block_idx, block_tables[block_idx]);
    // }

    // Apply SoftMax operation
    __local ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG];
    {
        ulong timer_start = intel_get_cycle_counter();
        if (sglid == 0)
            qk_max_vals[sgid] = qk_max;

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = ACCUMULATOR_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG)
            qk_max = qk_max_vals[sglid];

        // Final max value after reduction across of all SG and WI
        qk_max = sub_group_reduce_max(qk_max);

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        //     printf("%d %d %d. QK max value = %f\n", batch_idx, head_num_idx, head_idx, qk_max);
        // }

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && max_context_len[0] == 17 && sglid == 0 && sgid == 0) {
        //     for (uint i = 0; i < SHARED_MEM_SIZE; i++) {
        //         printf("Before %d %d %d, portion %d. qk_vals[%d] = %f\n", batch_idx, head_num_idx, head_idx, portion_id, i, qk_vals[i]);
        //     }
        // }

        // // temp test
        // barrier(CLK_LOCAL_MEM_FENCE);
        ulong timer_start2 = intel_get_cycle_counter();

        ACCUMULATOR_TYPE exp_sum = ACCUMULATOR_VAL_ZERO;
#ifdef USE_SPLIT_ACROSS_SEQ_LEN
        const uint qk_num = (num_of_portions == 1) ? CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE)
                                                   : CEIL_DIV(SEQ_LEN_PORTION_SIZE, SUBGROUPS_PER_WG * SUB_GROUP_SIZE);
#else
        const uint qk_num = CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE);
#endif
        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
#ifdef USE_SPLIT_ACROSS_SEQ_LEN
            if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PORTION_SIZE) {
#else
            if (global_data_idx < context_len) {
#endif
                // if (get_global_id(0) == 0 && get_global_id(1) && max_context_len[0] == 17) {
                //     printf("Calculation %d %d %d, portion %d, global_id2=%d. qk_vals[%d] = %f, qk_max = %f. Res= %f\n",
                //     batch_idx, head_num_idx, head_idx, portion_id, get_global_id(2), local_data_idx, qk_vals[local_data_idx], qk_max, native_exp(TO_ACCUMULATOR_TYPE(qk_vals[local_data_idx]) - qk_max));
                // }
                ACCUMULATOR_TYPE val = native_exp(TO_ACCUMULATOR_TYPE(qk_vals[local_data_idx]) - qk_max);
                exp_sum += val;
                qk_vals[local_data_idx] = TO_OUTPUT_TYPE(val);
                // if (head_num_idx < 4 || head_num_idx == 31)
                //     printf("head_num %d, sgid = %d, sglid = %d, exp_sum = %f\n", head_num_idx, sgid, sglid, exp_sum);
            }
        }

        ulong timer_start3 = intel_get_cycle_counter();

        // // temp test
        // barrier(CLK_LOCAL_MEM_FENCE);

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && max_context_len[0] == 17 && sglid == 0 && sgid == 0) {
        //     for (uint i = 0; i < SHARED_MEM_SIZE; i++) {
        //         printf("After %d %d %d, portion %d. qk_vals[%d] = %f\n", batch_idx, head_num_idx, head_idx, portion_id, i, qk_vals[i]);
        //     }
        // }


        // // temp test
        // barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = sub_group_reduce_add(exp_sum);

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        //     printf("exp_sum final value = %f\n", exp_sum);
        // }

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = ACCUMULATOR_VAL_ZERO;

        ulong timer_start4 = intel_get_cycle_counter();

        // FINAL FIX: Compile time restiction SUBGROUPS_PER_WG <= SG_SIZE
        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Final sum of all values
        exp_sum = sub_group_reduce_add(exp_sum);

        const ACCUMULATOR_TYPE inv_sum = ACCUMULATOR_VAL_ONE / exp_sum;


        // TODO: replace CEIL_DIV with ALIGN and use += SUBGROUPS_PER_WG * SUB_GROUP_SIZE increment
        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
#ifdef USE_SPLIT_ACROSS_SEQ_LEN
            if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PORTION_SIZE) {
#else
            if (global_data_idx < context_len) {
#endif
                ACCUMULATOR_TYPE val = TO_ACCUMULATOR_TYPE(qk_vals[local_data_idx]) * inv_sum;
                qk_vals[local_data_idx] = TO_OUTPUT_TYPE(val);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        ulong timer_start5 = intel_get_cycle_counter();

#ifdef USE_SPLIT_ACROSS_SEQ_LEN
        {
            // Save temporary exm_sums and max_logits values for each portion
            if (sgid == 0) {
                // if (get_global_id(0) == 0 && get_global_id(1) == 0 && max_context_len[0] == 17 && sglid == 0 && sgid == 0) {
                //     printf("Save exp_sum = %f, qk_max = %f\n", exp_sum, qk_max);
                // }
                const uint num_of_portions = get_num_groups(2);
                const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_portions +
                                             head_num_idx * num_of_portions +
                                             portion_id;
                exp_sums[exp_sums_offset] = exp_sum;

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = qk_max;

                // if (get_global_id(0) < 2 && get_global_id(1) < 2 && get_global_id(2) == 0)
                //     printf("%d, %d. Saved exp_sums=%f, max_logits=%f\n", seq_idx, head_num_idx, exp_sum, qk_max);
            }
        }
#endif

        ulong timer_end = intel_get_cycle_counter();

        ulong total_time1 = timer_start2 - timer_start;
        ulong total_time2 = timer_start3 - timer_start2;
        ulong total_time3 = timer_start4 - timer_start3;
        ulong total_time4 = timer_start5 - timer_start4;
        ulong total_time5 = timer_end - timer_start5;

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_local_id(2) == 0)
        //     printf("%d. SDPA kernel Softmax: qk_max calc: %d, exp_sum_loc calc: %d, exp_sum calc: %d, qk_vals recalc: %d, save: %d\n",
        //     get_global_id(2), (uint)total_time1, (uint)total_time2, (uint)total_time3, (uint)total_time4, (uint)total_time5);
    }

    // if (seq_idx == 0 && sgid == 0 && sglid == 0) {
    //     for (uint i = 0; i < context_len; i++) {
    //         printf("Softmax res for %d head: %d. %f\n", head_num_idx, i, qk_vals[i]);
    //     }
    // }

    {
        ulong timer_start = intel_get_cycle_counter();
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;

#ifdef USE_SPLIT_ACROSS_SEQ_LEN
        // FINAL: Compile time restriction: devisible SEQ_LEN_PORTION_SIZE / BLOCK_SIZE
        const uint qk_num = (portion_id == num_of_portions - 1) ? (context_len - (portion_id * SEQ_LEN_PORTION_SIZE))
                                                                : (SEQ_LEN_PORTION_SIZE);
#else
        const uint qk_num = context_len;
#endif
        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx += SUB_GROUP_SIZE) {
            const uint qk_offset_local = qk_idx + sglid;
            const uint qk_offset_global = block_start_idx * BLOCK_SIZE + qk_offset_local;

            OUTPUT_TYPE qk = qk_offset_global < context_len ? qk_vals[qk_offset_local] : OUTPUT_VAL_ZERO;

            const uint block_idx = block_tables[batch_idx * blocks_pitch + block_start_idx + (qk_idx / BLOCK_SIZE)];
            // if (block_idx == 0)
            //     continue;

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

            // FINAL: rename token -> value_idx
            if (block_start_idx * BLOCK_SIZE + qk_idx + SUB_GROUP_SIZE <= context_len) {
                unroll_for (uint token = 0; token < SUB_GROUP_SIZE; token++) {
                    OUTPUT_TYPE qk_tmp = sub_group_broadcast(qk, token);
                    acc = mad(qk_tmp, v[token], acc);
                }
            } else {
                for (uint token = 0; token < SUB_GROUP_SIZE; token++) {
                    OUTPUT_TYPE qk_tmp = sub_group_broadcast(qk, token);
                    if (block_start_idx * BLOCK_SIZE + qk_idx + token < context_len) {
                        acc = mad(qk_tmp, v[token], acc);
                    }
                }
            }
        }

#ifdef USE_SPLIT_ACROSS_SEQ_LEN
        // tmp_output data layout [num_seqs, num_heads, num_portions, head_size]
        const uint num_of_portions = get_num_groups(2);
        const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * num_of_portions) +
                                    head_num_idx * (HEAD_SIZE * num_of_portions) +
                                    portion_id * HEAD_SIZE +
                                    sgid * SUB_GROUP_SIZE +
                                    sglid;

        tmp_out[tmp_out_offset] = acc;
#else
        const uint output_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                   head_num_idx * HEAD_SIZE +
                                   sgid * SUB_GROUP_SIZE +
                                   sglid;

        output[output_offset] = acc;
#endif

        ulong timer_end = intel_get_cycle_counter();
        ulong total_time = timer_end - timer_start;

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_local_id(2) == 0)
        //     printf("%d. SDPA kernel GEMM2: %d\n", get_global_id(2), (uint)total_time);
    }
}

#endif

#ifdef SDPA_STAGE_1

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(pa_sdpa_finalization_stage)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* max_context_len,
    const __global INPUT4_TYPE* context_lens,
    const __global INPUT5_TYPE* block_tables,
    const __global INPUT6_TYPE* scale,
    __global OUTPUT_TYPE* output,
    const __global ACCUMULATOR_TYPE* exp_sums,
    const __global ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint num_of_portions) {
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_size_idx = get_group_id(2) * SUB_GROUP_SIZE;
    const uint sglid = get_sub_group_local_id();

    if (num_of_portions == 1) {
        const uint out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx + sglid;
        output[out_offset] = tmp_out[out_offset];
    } else if (num_of_portions <= SUB_GROUP_SIZE) {
        const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_portions +
                                     head_num_idx * num_of_portions + sglid;
        const uint max_logit_offset = exp_sums_offset;

        ACCUMULATOR_TYPE exp_sum = sglid >= num_of_portions ? ACCUMULATOR_VAL_ZERO : exp_sums[exp_sums_offset];
        ACCUMULATOR_TYPE max_logit = sglid >= num_of_portions ? ACCUMULATOR_VAL_MIN : max_logits[max_logit_offset];

        ACCUMULATOR_TYPE global_max = sub_group_reduce_max(max_logit);

        // Update exp_sum with respect to the global maximum
        if (sglid < num_of_portions)
            exp_sum = exp_sum * native_exp(max_logit - global_max);

        ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(exp_sum);

        float acc = 0.0f;
        for (uint portion = 0; portion < num_of_portions; portion++) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * num_of_portions * HEAD_SIZE) +
                                        head_num_idx * (num_of_portions * HEAD_SIZE) +
                                        portion * HEAD_SIZE +
                                        head_size_idx +
                                        sglid;
            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_ACCUMULATOR_TYPE(out_val) * TO_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum, portion)) / TO_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx +
                                sglid;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    } else {
        /* Need to update this part */
        const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_portions +
                                     head_num_idx * num_of_portions + sglid + 1;
        const uint max_logit_offset = exp_sums_offset + 1;

        ACCUMULATOR_TYPE exp_sum = sglid >= num_of_portions ? ACCUMULATOR_VAL_ZERO : exp_sums[exp_sums_offset];
        ACCUMULATOR_TYPE max_logit = sglid >= num_of_portions ? ACCUMULATOR_VAL_MIN : max_logits[max_logit_offset];

        ACCUMULATOR_TYPE global_max = sub_group_reduce_max(max_logit);

        // Update exp_sum with respect to the global maximum
        if (sglid < num_of_portions)
            exp_sum = exp_sum * native_exp(max_logit - global_max);

        ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(exp_sum);

        float acc = 0.0f;
        for (uint portion = 0; portion < num_of_portions; portion++) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * num_of_portions * HEAD_SIZE) +
                                        head_num_idx * (num_of_portions * HEAD_SIZE) +
                                        portion * HEAD_SIZE +
                                        head_size_idx +
                                        sglid;
            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_ACCUMULATOR_TYPE(out_val) * TO_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum, portion)) / TO_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx +
                                sglid;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    }
}

#endif
