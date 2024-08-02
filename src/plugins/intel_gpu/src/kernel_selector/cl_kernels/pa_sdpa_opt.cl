// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#define SUBGROUP_SIZE 16
#define SUBGROUPS_PER_WG (HEAD_SIZE / SUBGROUP_SIZE)

// The size of portion of HEAD_SIZE each WI process
#define Q_LOAD_ITERS (HEAD_SIZE / SUBGROUP_SIZE)

// How much QK outputs each subgroup calculates per block
#define QK_VALS_PER_SG_PER_ITER CEIL_DIV(VLLM_BLOCK_SIZE, SUBGROUPS_PER_WG)

#define KV_CACHE_BLOCK_STRIDE (HEAD_SIZE * KV_HEADS_NUM * VLLM_BLOCK_SIZE)

#define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)

ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

#ifdef SDPA_STAGE_0

// key shape   [num_blocks, NUM_HEADS, head_size, block_size]
// value shape [num_blocks, NUM_HEADS, block_size, head_size]

#if SEQ_LEN_PARTITION_SIZE % VLLM_BLOCK_SIZE != 0
    #error pa_sdpa_opt.cl
#endif

#if SUBGROUP_SIZE != VLLM_BLOCK_SIZE
    #error pa_sdpa_opt.cl
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, HEAD_SIZE)))
KERNEL(pa_sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* past_lens,
    const __global INPUT4_TYPE* subsequence_begins,
    const __global INPUT5_TYPE* block_indices,
    const __global INPUT6_TYPE* block_indices_begins,
    __global OUTPUT_TYPE* output
#ifdef USE_SEQ_LEN_SPLIT
    , __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums
    , __global SOFTMAX_ACCUMULATOR_TYPE* max_logits
    , __global OUTPUT_TYPE* tmp_out
#endif
) {
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_size_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();
    const uint num_of_portions = get_num_groups(2);

    const uint batch_idx = seq_idx;

    // const uint context_len = context_lens[batch_idx];
    const uint seq_len = past_lens[seq_idx] + 1;

    const uint partition_idx = get_group_id(2);
    const uint block_start_idx = partition_idx * SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE;

    if (partition_idx * SEQ_LEN_PARTITION_SIZE >= seq_len) {
        return;
    }

    const uint total_blocks_num = CEIL_DIV(seq_len, VLLM_BLOCK_SIZE);

    __local OUTPUT_TYPE qk_vals_local[SHARED_MEM_SIZE];
    SOFTMAX_ACCUMULATOR_TYPE qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;

    {
        INPUT0_TYPE q_val[HEAD_SIZE / SUBGROUP_SIZE];
        unroll_for (uint i = 0; i < HEAD_SIZE / SUBGROUP_SIZE; i++) {
            const uint query_idx = seq_idx * HEAD_SIZE * HEADS_NUM +
                                   head_num_idx * HEAD_SIZE +
                                   i * SUBGROUP_SIZE;
            q_val[i] = QUERY_BLOCK_READ(query, query_idx);
        }

        // uint blocks_num = ((partition_idx + 1) * SEQ_LEN_PARTITION_SIZE > context_len) ? (total_blocks_num - (partition_idx * SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE))
                                                                                    //    : (SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE);

        // blocks_num = blocks_num / SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE;

        const uint blocks_per_sg = SEQ_LEN_PARTITION_SIZE / SUBGROUP_SIZE / SUBGROUPS_PER_WG;
        const uint blocks_num =

        uint token_idx_debug = 0;

        uint seq_start_block_idx = block_indices_begins[seq_idx] + partition_idx * (SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE) + ;
        for (uint block_num = 0; block_num < blocks_num; block_num++) {
            const uint block_offset = block_indices[seq_start_block_idx + block_num] * KV_CACHE_BLOCK_STRIDE;

            INPUT0_TYPE qk_acc = INPUT0_VAL_ZERO;

            #define KEY_VEC_SIZE 8
            #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_VEC_SIZE, ptr, offset);
            #define KEY_VEC MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_VEC_SIZE)
            unroll_for (uint qk_idx = 0; qk_idx < HEAD_SIZE / KEY_VEC_SIZE; qk_idx++) {
                KEY_VEC key_vals = 0;
                key_vals = BLOCK_READN(key_cache, block_offset + qk_idx * SUBGROUP_SIZE * KEY_VEC_SIZE);

                unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
                    qk_acc = mad(sub_group_broadcast(q_val[qk_idx / 2], (qk_idx & 1) * KEY_VEC_SIZE + i), key_vals, qk_acc);
                }
            }

        }

            for (uint q_idx = 0; q_idx < Q_LOAD_ITERS; q_idx++) {
                for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                    uint current_token = (block_start_idx + block_num) * VLLM_BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
#if VLLM_BLOCK_SIZE % SUBGROUPS_PER_WG != 0
                    // TODO: Optimize for VLLM_BLOCK_SIZE % SUBGROUPS_PER_WG != 0 case
                    if (current_token >= context_len || sgid >= VLLM_BLOCK_SIZE / QK_VALS_PER_SG_PER_ITER)
#else
                    if (current_token >= context_len)
#endif
                        continue;

                    const uint key_idx = block_offset +
                                        (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE / X_BLOCK_SIZE * VLLM_BLOCK_SIZE * X_BLOCK_SIZE) +
                                        (X_BLOCK_SIZE * QK_VALS_PER_SG_PER_ITER) * sgid +
                                        (SUBGROUP_SIZE / X_BLOCK_SIZE * VLLM_BLOCK_SIZE * X_BLOCK_SIZE) * q_idx +
                                        (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE * VLLM_BLOCK_SIZE +
                                        (sglid % X_BLOCK_SIZE) + qk_idx * X_BLOCK_SIZE;

#if X_BLOCK_SIZE == 16
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset)
                    INPUT1_TYPE k_val = KEY_BLOCK_READ(key_cache, key_idx);
#else
                    INPUT1_TYPE k_val = key_cache[key_idx];
#endif

                    qk[qk_idx] = mad(q_val[q_idx], k_val, qk[qk_idx]);
                }
            }

            // Summurize qk calculation across all WIs and apply scale
            for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                const uint current_token = (block_start_idx + block_num) * VLLM_BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
#if VLLM_BLOCK_SIZE % SUBGROUPS_PER_WG != 0
                if (current_token < context_len && sgid < VLLM_BLOCK_SIZE / QK_VALS_PER_SG_PER_ITER) {
#else
                if (current_token < context_len) {
#endif
                    qk[qk_idx] = sub_group_reduce_add(qk[qk_idx]);

                    // Apply scale
                    qk[qk_idx] = scale[0] * qk[qk_idx];

                    // Apply attention mask for context processing stage
                    const bool is_prefill_stage = INPUT0_FEATURE_NUM > 1;
                    if (is_prefill_stage && current_token > token_idx) {
                        qk[qk_idx] = qk[qk_idx] + OUTPUT_VAL_MIN;
                    }

                    qk_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max, TO_SOFTMAX_ACCUMULATOR_TYPE(qk[qk_idx]));
                }
            }

            // Save QK results to local memory
#if VLLM_BLOCK_SIZE % SUBGROUPS_PER_WG != 0
            if (sglid < QK_VALS_PER_SG_PER_ITER && sgid < VLLM_BLOCK_SIZE / QK_VALS_PER_SG_PER_ITER) {
#else
            if (sglid < QK_VALS_PER_SG_PER_ITER) {
#endif
                const uint current_token_global_idx = (block_start_idx + block_num) * VLLM_BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
#ifdef USE_SEQ_LEN_SPLIT
                const uint current_token_local = block_num * VLLM_BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
#else
                const uint current_token_local = current_token_global_idx;
#endif
                qk_vals_local[current_token_local] = current_token_global_idx >= context_len ? 0 : qk[sglid];
            }
        }
    }

    // Apply SoftMax operation
    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG];
    {
        if (sglid == 0)
            qk_max_vals[sgid] = qk_max;

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG)
            qk_max = qk_max_vals[sglid];

        // Final max value after reduction across of all SG and WI
        qk_max = sub_group_reduce_max(qk_max);

        SOFTMAX_ACCUMULATOR_TYPE exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
#ifdef USE_SEQ_LEN_SPLIT
        const uint qk_num = (num_of_portions == 1) ? CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE)
                                                   : CEIL_DIV(SEQ_LEN_PARTITION_SIZE, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
#else
        const uint qk_num = CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
#endif
        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * VLLM_BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
#ifdef USE_SEQ_LEN_SPLIT
            if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PARTITION_SIZE) {
#else
            if (global_data_idx < context_len) {
#endif
                SOFTMAX_ACCUMULATOR_TYPE val = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) - qk_max);
                exp_sum += val;
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(val);
            }
        }

        exp_sum = sub_group_reduce_add(exp_sum);

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;

        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Final sum of all exp_sum values
        exp_sum = sub_group_reduce_add(exp_sum);

        const SOFTMAX_ACCUMULATOR_TYPE inv_sum = SOFTMAX_ACCUMULATOR_VAL_ONE / exp_sum;

        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * VLLM_BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
#ifdef USE_SEQ_LEN_SPLIT
            if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PARTITION_SIZE) {
#else
            if (global_data_idx < context_len) {
#endif
                SOFTMAX_ACCUMULATOR_TYPE val = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) * inv_sum;
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(val);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

#ifdef USE_SEQ_LEN_SPLIT
        {
            // Save temporary exm_sums and max_logits values for each portion
            if (num_of_portions > 1 && sgid == 0) {
                const uint num_of_portions = get_num_groups(2);
                const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_portions +
                                             head_num_idx * num_of_portions +
                                             partition_idx;
                exp_sums[exp_sums_offset] = exp_sum;

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = qk_max;
            }
        }
#endif
    }

    {
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;

#ifdef USE_SEQ_LEN_SPLIT
        const uint qk_num = ((partition_idx + 1) * SEQ_LEN_PARTITION_SIZE > context_len) ? (context_len - (partition_idx * SEQ_LEN_PARTITION_SIZE))
                                                                                    : (SEQ_LEN_PARTITION_SIZE);
#else
        const uint qk_num = context_len;
#endif
        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx += SUBGROUP_SIZE) {
            const uint qk_offset_local = qk_idx + sglid;
            const uint qk_offset_global = block_start_idx * VLLM_BLOCK_SIZE + qk_offset_local;

            OUTPUT_TYPE qk = qk_offset_global < context_len ? qk_vals_local[qk_offset_local] : OUTPUT_VAL_ZERO;

            const uint block_idx = block_tables[batch_idx * blocks_pitch + block_start_idx + (qk_idx / VLLM_BLOCK_SIZE)];

            const uint value_cache_offset = block_idx * KV_CACHE_BLOCK_STRIDE +
                                            (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE * VLLM_BLOCK_SIZE) +
                                            sgid * (SUBGROUP_SIZE * VLLM_BLOCK_SIZE) +
                                            sglid * VLLM_BLOCK_SIZE +
                                            ((qk_idx / SUBGROUP_SIZE) % (VLLM_BLOCK_SIZE / SUBGROUP_SIZE)) * SUBGROUP_SIZE;

            #define VALUE_VEC_TYPE MAKE_VECTOR_TYPE(INPUT2_TYPE, SUBGROUP_SIZE)
            #define AS_VALUE_VEC(val) CAT(as_, VALUE_VEC_TYPE)(val)
#if INPUT2_TYPE_SIZE == 4
            #define VALUE_VLOAD(offset, ptr) CAT(vload, SUBGROUP_SIZE)(offset, ptr)
#else
            #define VALUE_VLOAD(offset, ptr) CAT(vload, SUBGROUP_SIZE)(offset, (__global ushort*)(ptr))
#endif
            VALUE_VEC_TYPE v_val = AS_VALUE_VEC(VALUE_VLOAD(0, value_cache + value_cache_offset));

            if (block_start_idx * VLLM_BLOCK_SIZE + qk_idx + SUBGROUP_SIZE <= context_len) {
                unroll_for (uint v_idx = 0; v_idx < SUBGROUP_SIZE; v_idx++) {
                    OUTPUT_TYPE qk_val = sub_group_broadcast(qk, v_idx);
                    acc = mad(qk_val, v_val[v_idx], acc);
                }
            } else {
                for (uint v_idx = 0; v_idx < SUBGROUP_SIZE; v_idx++) {
                    OUTPUT_TYPE qk_val = sub_group_broadcast(qk, v_idx);
                    if (block_start_idx * VLLM_BLOCK_SIZE + qk_idx + v_idx < context_len) {
                        acc = mad(qk_val, v_val[v_idx], acc);
                    }
                }
            }
        }

#ifdef USE_SEQ_LEN_SPLIT
        if (num_of_portions > 1) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * num_of_portions) +
                                        head_num_idx * (HEAD_SIZE * num_of_portions) +
                                        partition_idx * HEAD_SIZE +
                                        sgid * SUBGROUP_SIZE +
                                        sglid;

            // tmp_output data layout [num_seqs, num_heads, num_portions, head_size]
            tmp_out[tmp_out_offset] = acc;
        }
        else
#endif
        {
            const uint output_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                       head_num_idx * HEAD_SIZE +
                                       sgid * SUBGROUP_SIZE +
                                       sglid;

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

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(pa_sdpa_finalization_stage)(
    const __global INPUT4_TYPE* context_lens,
    __global OUTPUT_TYPE* output,
    const __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    const __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint total_num_of_portions) {
    const uint batch_idx = get_global_id(0);
    const uint token_idx = get_global_id(1);
    const uint head_dim = get_global_id(2);
    const uint head_num_idx = head_dim / HEAD_SIZE;
    const uint head_size_idx = head_dim % HEAD_SIZE;
    const uint sglid = get_sub_group_local_id();

    const uint seq_offset = batch_idx * get_global_size(1) + token_idx;

    const uint num_of_portions = CEIL_DIV(context_lens[batch_idx], SEQ_LEN_PARTITION_SIZE);

    if (total_num_of_portions == 1) {
        /* Short path, just copies input to output */
        const uint out_offset = seq_offset * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;
        output[out_offset] = tmp_out[out_offset];
    } else if (num_of_portions <= SUBGROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI) {
        /* Registers kernel version, can handle up to SEQ_LEN_PARTITION_SIZE(256) * SUBGROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI(24) = 98304 tokens */
        SOFTMAX_ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        SOFTMAX_ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_portions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + partition_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (partition_idx < num_of_portions) {
                exp_sum[i] = exp_sums[exp_sums_offset];
                max_logit[i] = max_logits[max_logit_offset];
                local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logit[i]);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Update exp_sum with respect to the global maximum
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            if (partition_idx < num_of_portions) {
                exp_sum[i] = exp_sum[i] * native_exp(max_logit[i] - global_max);
                local_exp_sum += exp_sum[i];
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
        for (uint portion = 0; portion < num_of_portions; portion++) {
            const uint tmp_out_offset = seq_offset * (HEADS_NUM * total_num_of_portions * HEAD_SIZE) +
                                        head_num_idx * (total_num_of_portions * HEAD_SIZE) +
                                        portion * HEAD_SIZE +
                                        head_size_idx;
            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[portion / SUBGROUP_SIZE], portion)) / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_offset * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    } else {
        /* Global memory kernel version, can handle any number of tokens */
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_portions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint max_logit_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + partition_idx;

            if (partition_idx < num_of_portions) {
                local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logits[max_logit_offset]);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Calculate global sum
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + partition_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (partition_idx < num_of_portions) {
                local_exp_sum += exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
        for (uint portion = 0; portion < num_of_portions; portion++) {
            const uint tmp_out_offset = seq_offset * (HEADS_NUM * total_num_of_portions * HEAD_SIZE) +
                                        head_num_idx * (total_num_of_portions * HEAD_SIZE) +
                                        portion * HEAD_SIZE +
                                        head_size_idx;

            const uint exp_sums_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + portion;
            const uint max_logit_offset = exp_sums_offset;

            SOFTMAX_ACCUMULATOR_TYPE new_exp_sum = exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);

            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * new_exp_sum / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_offset * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    }
}

#endif
