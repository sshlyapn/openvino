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
    __global const INPUT0_TYPE* query,
    __global const INPUT1_TYPE* key_cache,
    __global const INPUT2_TYPE* value_cache,
    __global const INPUT3_TYPE* max_context_len,
    __global const INPUT4_TYPE* context_lens,
    __global const INPUT5_TYPE* block_tables,
    __global const INPUT6_TYPE* scale,
    __global OUTPUT_TYPE* output,
    __global OUTPUT_TYPE* exp_sums,
    __global OUTPUT_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out,
    uint num_of_portions)
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

    const uint portion_id = get_group_id(2);
    const uint block_start_idx = portion_id * SEQ_LEN_PORTION_SIZE / BLOCK_SIZE;
    const uint block_end_idx = min(block_start_idx + (SEQ_LEN_PORTION_SIZE / BLOCK_SIZE), blocks_num);


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

    OUTPUT_TYPE qk_max = OUTPUT_VAL_MIN;


    {
        ulong timer_start = intel_get_cycle_counter();
        INPUT0_TYPE q[HEAD_SIZE / SUB_GROUP_SIZE];
        unroll_for (uint i = 0; i < HEAD_SIZE / SUB_GROUP_SIZE; i++) {
            const uint query_idx = seq_idx * HEAD_SIZE * HEADS_NUM +
                                    head_num_idx * HEAD_SIZE +
                                    i * SUB_GROUP_SIZE;
            q[i] = QUERY_BLOCK_READ(query, query_idx);
        }

        // JIT: Compile time restriction: devisible SEQ_LEN_PORTION_SIZE / BLOCK_SIZE
        for (uint block = 0; block < SEQ_LEN_PORTION_SIZE / BLOCK_SIZE; block++) {
            const uint block_idx = batch_idx * blocks_num + block + block_start_idx;
            const uint block_offset = block_tables[block_idx] * KV_CACHE_BLOCK_STRIDE;

            OUTPUT_TYPE qk[QK_VALS_PER_SG_PER_ITER] = {0};

            ulong timer2 = intel_get_cycle_counter();
            for (uint hs = 0; hs < Q_LOAD_ITERS; hs++) {
                for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                    uint current_token = (block + block_start_idx) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
                    if (current_token >= context_len)
                        continue;

                    const uint key_idx = block_offset +
                                        (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) +
                                        (X_BLOCK_SIZE * QK_VALS_PER_SG_PER_ITER) * sgid +
                                        (SUB_GROUP_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) * hs +
                                        (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE * BLOCK_SIZE +
                                        (sglid % X_BLOCK_SIZE) + qk_idx * X_BLOCK_SIZE;


                    // if (get_global_id(1) == 0 && get_global_id(2) == 0 && hs == 0) {
                    //     printf("test=%d, %d seq_idx=%d (b=%d, t=%d), %d %d: q_idx = %d, k_idx= %d, block=%d\n", get_global_size(0), INPUT0_FEATURE_NUM, get_global_id(0),
                    //     batch_idx, token_idx,
                    //     get_global_id(1), get_global_id(2), seq_idx * HEAD_SIZE * HEADS_NUM + head_num_idx * HEAD_SIZE, key_idx, block);
                    // }

                    // TODO1: try block loading and shuffling
                    // TODO2: try to load k*4 times and then calculate
                    // TODO3: try bigger X block
                    #if X_BLOCK_SIZE == 16
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset)
                    INPUT1_TYPE k = KEY_BLOCK_READ(key_cache, key_idx);
                    #else
                    INPUT1_TYPE k = key_cache[key_idx];
                    #endif


                    // if (seq_idx == 0 && head_num_idx == 0) {
                    //     printf("main_calc: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d, block=%d, hs=%d, qk_idx=%d, current_token=%d, query_idx=%d, key_idx=%d (block_offset=%d): %f * %f\n",
                    //         seq_idx, head_num_idx, sgid, sglid, block, hs, qk_idx, current_token, query_idx, key_idx - block_offset, block_offset, q, k);
                    // }

                    qk[qk_idx] = mad(q[hs], k, qk[qk_idx]);
                }
            }
            ulong timer3 = intel_get_cycle_counter();

            // Summurize qk calculation across all WIs and apply scale
            for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                const uint current_token = (block + block_start_idx) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
                if (current_token < context_len) {
                    OUTPUT_TYPE tmp_print = qk[qk_idx];
                    qk[qk_idx] = sub_group_reduce_add(qk[qk_idx]);
                    // if (head_num_idx < 4)
                    //     printf("final_calc: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: before qk[%d]=%f, after=%f\n",
                    //             seq_idx, head_num_idx, sgid, sglid, qk_idx, tmp_print, qk[qk_idx]);
                    qk[qk_idx] = scale[0] * qk[qk_idx];

                    // Apply attention mask at prefill stage
                    if (INPUT0_FEATURE_NUM > 1 && current_token > token_idx) {
                        qk[qk_idx] = qk[qk_idx] + OUTPUT_VAL_MIN;
                    }
                    qk_max = OUTPUT_MAX_FUNC(qk_max, qk[qk_idx]);
                }
            }
            ulong timer4 = intel_get_cycle_counter();

            // Save QK results to local memory
            if (sglid < QK_VALS_PER_SG_PER_ITER) {
                const uint current_token = block * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
                const uint current_token_global_idx = (block + block_start_idx) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
                // Fixed -> // const uint qk_local_idx = block * BLOCK_SIZE * sgid * QK_VALS_PER_SG_PER_ITER + sglid;
                // OUTPUT_TYPE tmp_print = (current_token >= context_len ? 0 : qk[sglid]);
                // if (head_num_idx < 4 || head_num_idx == 31)
                //     printf("slm save: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: qk_vals[%d]=%f. Max=%f\n",
                //             seq_idx, head_num_idx, sgid, sglid, current_token, tmp_print, qk_max);
                qk_vals[current_token] = current_token_global_idx >= context_len ? 0 : qk[sglid];
            }
            ulong timer5 = intel_get_cycle_counter();

            // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
            //     printf("SDPA kernel GEMM1 block %d: main_loop=%d, summarization=%d, saving=%d\n", block,
            //         (uint)(timer3 - timer2),
            //         (uint)(timer4 - timer3),
            //         (uint)(timer5 - timer4));
            // }
        }
        ulong timer_end = intel_get_cycle_counter();
        ulong total_time = timer_end - timer_start;

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
        //     printf("SDPA kernel GEMM1: %d; qk_max=%f\n", (uint)total_time, qk_max);
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
    __local OUTPUT_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local OUTPUT_TYPE qk_sum_vals[SUBGROUPS_PER_WG];
    {
        ulong timer_start = intel_get_cycle_counter();
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
        for (uint qk_idx = 0; qk_idx < CEIL_DIV(SEQ_LEN_PORTION_SIZE, SUBGROUPS_PER_WG * SUB_GROUP_SIZE); qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            if (global_data_idx < context_len) {
                OUTPUT_TYPE val = native_exp(qk_vals[local_data_idx] - qk_max);
                exp_sum += val;
                qk_vals[local_data_idx] = val;
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


        // JIT: Compile time restiction SUBGROUPS_PER_WG <= SG_SIZE
        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Final sum of all values
        exp_sum = sub_group_reduce_add(exp_sum);

        const OUTPUT_TYPE inv_sum = OUTPUT_VAL_ONE / exp_sum;


        // TODO: replace CEIL_DIV with ALIGN and use += SUBGROUPS_PER_WG * SUB_GROUP_SIZE increment
        for (uint qk_idx = 0; qk_idx < CEIL_DIV(SEQ_LEN_PORTION_SIZE, SUBGROUPS_PER_WG * SUB_GROUP_SIZE); qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            if (global_data_idx < context_len) {
                OUTPUT_TYPE val = qk_vals[local_data_idx] * inv_sum;
                qk_vals[local_data_idx] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        ulong timer_end = intel_get_cycle_counter();
        ulong total_time = timer_end - timer_start;

        {
            // Save temporary exm_sums and max_logits values for each portion
            if (sgid == 0) {
                const uint num_of_portions = get_num_groups(2);
                const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_portions +
                                             head_num_idx * num_of_portions +
                                             portion_id;
                exp_sums[exp_sums_offset] = exp_sum;

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = qk_max;
            }
        }

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
        //     printf("SDPA kernel Softmax: %d\n", (uint)total_time);
    }

    // if (seq_idx == 0 && sgid == 0 && sglid == 0) {
    //     for (uint i = 0; i < context_len; i++) {
    //         printf("Softmax res for %d head: %d. %f\n", head_num_idx, i, qk_vals[i]);
    //     }
    // }

    {
        ulong timer_start = intel_get_cycle_counter();
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;


        for (uint qk_idx = 0; qk_idx < SEQ_LEN_PORTION_SIZE / BLOCK_SIZE * SUB_GROUP_SIZE; qk_idx += SUB_GROUP_SIZE) {
            const uint qk_offset_local = qk_idx + sglid;
            const uint qk_offset_global = block_start_idx * BLOCK_SIZE + qk_offset_local;

            OUTPUT_TYPE qk = qk_offset_global < context_len ? qk_vals[qk_offset_local] : OUTPUT_VAL_ZERO;

            const uint block_idx = block_tables[batch_idx * blocks_num + block_start_idx + (qk_idx / BLOCK_SIZE)];
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


        // const uint output_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
        //                            head_num_idx * HEAD_SIZE +
        //                            sgid * SUB_GROUP_SIZE +
        //                            sglid;

        // if (seq_idx == 0 && head_num_idx < 2 || head_num_idx == 31) {
        //     printf("output res: seq_idx=%d, head_num_idx=%d, sgid=%d, sglid=%d: output[%d] = %f\n",
        //         seq_idx, head_num_idx, sgid, sglid, output_offset, acc);
        // }

        // output[output_offset] = acc;

        {
            // [num_seqs, num_heads, max_num_partitions, head_size]
            const uint num_of_portions = get_num_groups(2);
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * num_of_portions) +
                                       head_num_idx * (HEAD_SIZE * num_of_portions) +
                                       portion_id * HEAD_SIZE +
                                       sgid * SUB_GROUP_SIZE +
                                       sglid;

            // if (output_offset != tmp_out_offset)
            //     printf("Different tmp_out_offset index!! %d vs %d, for portion_id %d\n", output_offset, tmp_out_offset, portion_id);

            tmp_out[tmp_out_offset] = acc;
        }

        ulong timer_end = intel_get_cycle_counter();
        ulong total_time = timer_end - timer_start;

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
        //     printf("SDPA kernel GEMM2: %d\n", (uint)total_time);
    }
}

#endif

#ifdef SDPA_STAGE_1

//   exp_sums,        // [num_seqs, num_heads, max_num_partitions]
//   max_logits,      // [num_seqs, num_heads, max_num_partitions]
//   tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(pa_sdpa_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* query,
    __global const INPUT1_TYPE* key_cache,
    __global const INPUT2_TYPE* value_cache,
    __global const INPUT3_TYPE* max_context_len,
    __global const INPUT4_TYPE* context_lens,
    __global const INPUT5_TYPE* block_tables,
    __global const INPUT6_TYPE* scale,
    __global OUTPUT_TYPE* output,
    __global OUTPUT_TYPE* exp_sums,
    __global OUTPUT_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out,
    uint num_of_portions) {
    if (num_of_portions <= SUB_GROUP_SIZE) {
        const uint seq_idx = get_global_id(0);
        const uint head_num_idx = get_global_id(1);
        const uint head_idx = get_global_id(2);
        const uint sglid = get_sub_group_local_id();

        const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_portions +
                                     head_num_idx * num_of_portions;
        const uint max_logit_offset = exp_sums_offset;

        OUTPUT_TYPE exp_sum = BLOCK_READN(OUTPUT_TYPE, 1, exp_sums, exp_sums_offset);
        OUTPUT_TYPE max_logit = BLOCK_READN(OUTPUT_TYPE, 1, max_logits, max_logit_offset);
        if (sglid >= num_of_portions) {
            exp_sum = 0;
            max_logit = OUTPUT_VAL_MIN;
        }

        OUTPUT_TYPE global_max = sub_group_reduce_max(max_logit);

        // Update exp_sum with respect to the global maximum
        OUTPUT_TYPE test_exp_sum = exp_sum;
        if (sglid < num_of_portions)
            exp_sum = exp_sum * native_exp(max_logit - global_max);

        OUTPUT_TYPE global_sum = sub_group_reduce_add(exp_sum);

        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
            printf("Run second kernel for reduction: num_of_portions=%d: max_logit=%f, exp_sum = %f, global_sum = %f, global_max=%f, test = %f, %f, %f\n", num_of_portions,
            max_logit, exp_sum, global_sum, global_max, test_exp_sum, native_exp(max_logit - global_max), test_exp_sum * native_exp(max_logit - global_max));

        for (uint i = 0; i < HEAD_SIZE / SUB_GROUP_SIZE; i++) {
            OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;
            for (uint portion = 0; portion < num_of_portions; portion++) {
                const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * num_of_portions) +
                                            head_num_idx * (HEAD_SIZE * num_of_portions) +
                                            portion * HEAD_SIZE;
                OUTPUT_TYPE out_val = BLOCK_READN(OUTPUT_TYPE, 1, tmp_out, tmp_out_offset);
                acc += out_val * sub_group_broadcast(exp_sum, portion) / global_sum;
            }
            const uint out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                    head_num_idx * HEAD_SIZE +
                                    i * SUB_GROUP_SIZE;
            output[out_offset] = acc;
        }
    } else {
        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
            printf("run second kernel for portion >= 16\n");
    }
}

#endif
