// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

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

#if SEQ_ID_BLOCK_SIZE > 1
    const uint seq_idx = (uint)get_global_id(1) * SEQ_ID_BLOCK_SIZE;
#else
    const uint seq_idx = get_global_id(1);
#endif
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

    #if SEQ_ID_BLOCK_SIZE > 1
        #define MULTI_TOKENS_OPT 1
    #else
        #define SINGLE_TOKEN_OPT 1
    #endif

    #if HEAD_SIZE > 256 || MULTI_TOKENS_OPT
        #define QUERY_IN_SLM 1
        #if FIRST_APPROACH
        __local INPUT0_TYPE query_vals[HEAD_SIZE * SEQ_ID_BLOCK_SIZE];
        #else
        __local INPUT0_TYPE query_vals[HEAD_SIZE];
        #endif
    #else
        #define QUERY_IN_REGS 1
    #endif

    __local OUTPUT_TYPE qk_vals_local[SLM_SIZE * SEQ_ID_BLOCK_SIZE];
    SOFTMAX_ACCUMULATOR_TYPE qk_max[SEQ_ID_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_MIN};

#ifndef INPUT4_TYPE
    const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(HEAD_SIZE));
#endif

    // __local INPUT0_TYPE query_vals_local[HEAD_SIZE];
    const uint start_partition_idx = partition_idx * SEQ_LEN_PARTITION_SIZE;

    { // start Gemm1
    ulong timer_start1 = intel_get_cycle_counter();

#if (HEAD_SIZE % SUBGROUP_SIZE == 0) && (HEAD_SIZE / SUBGROUP_SIZE == 16) && !defined(MULTI_TOKENS_OPT)
    /* Optimized case for HEAD_SIZE == {256} */
    #define QUERY_VEC_SIZE 16
    #define QUERY_BLOCK_SIZE 8

    MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERY_VEC_SIZE) query_vals;

    uint query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx, 0);
    query_vals.lo = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

    query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx, HEAD_SIZE / 2);
    query_vals.hi = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

#elif (HEAD_SIZE % SUBGROUP_SIZE == 0) && ((HEAD_SIZE / SUBGROUP_SIZE == 8) || \
                                           (HEAD_SIZE / SUBGROUP_SIZE == 4) || \
                                           (HEAD_SIZE / SUBGROUP_SIZE == 2)) && !defined(MULTI_TOKENS_OPT)
    /* Optimized case for HEAD_SIZE == {128, 64, 32} */

    uint query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx, 0);

#if HEAD_SIZE / SUBGROUP_SIZE == 8
    #define QUERY_BLOCK_SIZE 8
#elif HEAD_SIZE / SUBGROUP_SIZE == 4
    #define QUERY_BLOCK_SIZE 4
#elif HEAD_SIZE / SUBGROUP_SIZE == 2
    #define QUERY_BLOCK_SIZE 2
#endif

    MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERY_BLOCK_SIZE) query_vals;
    query_vals = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

#else
    /* Optimized case for any HEAD_SIZE % SUBGROUP_SIZE == 0 */

    #if defined(FIRST_APPROACH) || defined(SINGLE_TOKEN_OPT)  /* temprorary disable loading */

    #ifdef QUERY_IN_SLM
        #if MULTI_TOKENS_OPT
            #define QUERY_LOCAL_STEP SUBGROUP_SIZE * SUBGROUPS_PER_WG
            uint query_local_offset = sgid * SUBGROUP_SIZE + sglid;
        #else
            #define QUERY_LOCAL_STEP SUBGROUP_SIZE
            uint query_local_offset = sglid;
        #endif
        // TODO: Optimize the SLM version as follows:
        // 1. Load a portion of the query input from SLM to registers.
        // 2. Partially calculate GEMM1 using the loaded queries and save the intermediate result to qk_vals_local.
        // 3. Load the next part of the query input from SLM to registers, calculate GEMM1, add the result to qk_vals_local.
        // 4. Repeat the process for subsequent query parts.
    #else
        #define QUERY_LOCAL_STEP 1
        INPUT0_TYPE query_vals[HEAD_SIZE / SUBGROUP_SIZE];
        uint query_local_offset = 0;
    #endif

#if MULTI_TOKENS_OPT
    const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
#else
    const uint seq_idx_index_end = 1;
#endif
    for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {

    uint query_head_offset = 0;
    uint query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx + seq_idx_index, 0);
    const uint query_head_pitch = 1;

    #ifdef SINGLE_TOKEN_OPT
    #define QUERY_BLOCK_SIZE 8
    unroll_for(; query_head_offset + (QUERY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; query_head_offset += QUERY_BLOCK_SIZE * SUBGROUP_SIZE) {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERY_BLOCK_SIZE) vec = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

        unroll_for (uint i = 0; i < QUERY_BLOCK_SIZE; i++) {
            query_vals[query_local_offset] = vec[i];
            query_local_offset += QUERY_LOCAL_STEP;
        }
        query_offset += query_head_pitch * QUERY_BLOCK_SIZE * SUBGROUP_SIZE;
    }
    #undef QUERY_BLOCK_SIZE

    #define QUERY_BLOCK_SIZE 4
    unroll_for(; query_head_offset + (QUERY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; query_head_offset += QUERY_BLOCK_SIZE * SUBGROUP_SIZE) {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERY_BLOCK_SIZE) vec = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

        unroll_for (uint i = 0; i < QUERY_BLOCK_SIZE; i++) {
            query_vals[query_local_offset] = vec[i];
            query_local_offset += QUERY_LOCAL_STEP;
        }
        query_offset += query_head_pitch * QUERY_BLOCK_SIZE * SUBGROUP_SIZE;
    }
    #undef QUERY_BLOCK_SIZE

    #define QUERY_BLOCK_SIZE 2
    unroll_for(; query_head_offset + (QUERY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; query_head_offset += QUERY_BLOCK_SIZE * SUBGROUP_SIZE) {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERY_BLOCK_SIZE) vec = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

        unroll_for (uint i = 0; i < QUERY_BLOCK_SIZE; i++) {
            query_vals[query_local_offset] = vec[i];
            query_local_offset += QUERY_LOCAL_STEP;
        }
        query_offset += query_head_pitch * QUERY_BLOCK_SIZE * SUBGROUP_SIZE;
    }
    #undef QUERY_BLOCK_SIZE

    #define QUERY_BLOCK_SIZE 1
    unroll_for(; query_head_offset + (QUERY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; query_head_offset += QUERY_BLOCK_SIZE * SUBGROUP_SIZE) {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERY_BLOCK_SIZE) val = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

        query_vals[query_local_offset] = val;
    }
    #undef QUERY_BLOCK_SIZE

    #else /* MULTI_TOKENS_OPT */

    #define QUERY_BLOCK_SIZE 1
    query_offset += sgid * query_head_pitch * SUBGROUP_SIZE;

    INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

    query_vals[query_local_offset] = val;
    query_local_offset += QUERY_LOCAL_STEP;

    #endif
    } // New loop over seq

    #ifdef QUERY_IN_SLM
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    #endif

#endif

    /* Calculate Gemm1 */

#if defined(MULTI_TOKENS_OPT) && defined(FIRST_APPROACH)

    #define KEY_BLOCK_SIZE MULS_NUM
    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset)
    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)

    for (uint h = 0; h < HEAD_SIZE / SUBGROUP_SIZE / KEY_BLOCK_SIZE; h++) {
        uint key_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + sgid, h * KEY_BLOCK_SIZE * SUBGROUP_SIZE);
        MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE) query_vals_regs[SEQ_ID_BLOCK_SIZE];

        unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
            unroll_for (uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                const uint query_local_offset = seq_idx_index * HEAD_SIZE + (h * KEY_BLOCK_SIZE + i) * SUBGROUP_SIZE + sglid;
                query_vals_regs[seq_idx_index][i] = query_vals[query_local_offset]; /* READ FROM SLM TO REG */
            }
        }

        for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE / SUBGROUP_SIZE)) {
            INPUT0_TYPE acc[SEQ_ID_BLOCK_SIZE] = {INPUT0_VAL_ZERO};
            KEY_BLOCK key_vec = KEY_BLOCK_READ(key_input, key_offset);
            unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
                unroll_for (uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                    INPUT0_TYPE before_acc = acc[seq_idx_index];
                    acc[seq_idx_index] = mad(query_vals_regs[seq_idx_index][i], key_vec[i], acc[seq_idx_index]);

                }
            }

            unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
                acc[seq_idx_index] = sub_group_reduce_add(acc[seq_idx_index]);
            }

            unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
                INPUT0_TYPE prev_val = h == 0 ? 0 : qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len];
                qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx_index] + prev_val;
                // TODO: try another approach
                // if (h == 0) {
                //     qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx_index];
                // } else {
                //     INPUT0_TYPE prev_val = qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len];
                //     qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len] += acc[seq_idx_index];
                // }
            }

            key_offset += (HEAD_SIZE / SUBGROUP_SIZE) * HEAD_SIZE;
        }
    }

    {
        #ifndef FIRST_APPROACH_OPTION2
        barrier(CLK_LOCAL_MEM_FENCE);

        INPUT0_TYPE acc[SEQ_ID_BLOCK_SIZE];
        const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
        for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
            for (uint seq_len = sgid * SUBGROUP_SIZE + sglid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE)) {
                // Apply scale
                acc[seq_idx_index] = qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len];


                acc[seq_idx_index] *= scale_val;

                // Apply attention mask
                uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(batch_idx, head_num_idx, seq_idx + seq_idx_index, start_partition_idx + seq_len);
                acc[seq_idx_index] += attn_mask[attn_mask_offset];

                // Update qk_max value
                qk_max[seq_idx_index] = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max[seq_idx_index], TO_SOFTMAX_ACCUMULATOR_TYPE(acc[seq_idx_index]));

                qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx_index];
            }
        }

        #else

        if (sglid == 0) {
            INPUT0_TYPE acc[SEQ_ID_BLOCK_SIZE];
            for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE / SUBGROUP_SIZE)) {
                const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
                for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
                    // Apply scale
                    acc[seq_idx_index] = qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len];
                    acc[seq_idx_index] *= scale_val;


                    // Apply attention mask
                    uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(batch_idx, head_num_idx, seq_idx + seq_idx_index, start_partition_idx + seq_len);


                    acc[seq_idx_index] += attn_mask[attn_mask_offset];


                    // Update qk_max value
                    qk_max[seq_idx_index] = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max[seq_idx_index], TO_SOFTMAX_ACCUMULATOR_TYPE(acc[seq_idx_index]));

                    qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx_index];
                }
            }
        }
        #endif
    }


    /* TODO: H leftovers processing */
#if HEAD_SUZE % (SUBGROUP_SIZE * KEY_BLOCK_SIZE) != 0
#define KEY_BLOCK_SIZE 1
#define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset)
        {
            #error Shouldn't be here[1]
            INPUT1_TYPE key_val = KEY_BLOCK_READ(key_input, key_offset);

#ifdef QUERY_IN_SLM
            const uint query_local_offset = HEAD_SIZE - SUBGROUP_SIZE + sglid;
#else
            const uint query_local_offset = HEAD_SIZE / SUBGROUP_SIZE - 1;
#endif
            acc[0] = mad(query_vals[HEAD_SIZE / SUBGROUP_SIZE - 1], key_val, acc[0]);
        }
#endif

#elif defined(MULTI_TOKENS_OPT)
/* Second try */

    for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE / SUBGROUP_SIZE)) {
        uint key_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + seq_len, 0);

        INPUT0_TYPE acc[SEQ_ID_BLOCK_SIZE] = {INPUT0_VAL_ZERO};

    #define KEY_BLOCK_SIZE MULS_NUM
    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset)
    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)


    #define QUERY_BLOCK_SIZE MULS_NUM
    #define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, ptr, offset)
    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, QUERY_BLOCK_SIZE)

        unroll_for (uint h = 0; h < HEAD_SIZE / SUBGROUP_SIZE / KEY_BLOCK_SIZE; h++) {
            KEY_BLOCK key_vec = KEY_BLOCK_READ(key_input, key_offset);


            // TODO: try not unrolled loop
            // const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
            // for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
            // const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
            unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
                const uint query_offset = INPUT0_GET_INDEX(batch_idx, head_num_idx, seq_idx + seq_idx_index, h * KEY_BLOCK_SIZE * SUBGROUP_SIZE);
                QUERY_BLOCK query_vec = QUERY_BLOCK_READ(query_input, query_offset);

                unroll_for (uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                    #if KEY_BLOCK_SIZE == 1
                        acc[seq_idx_index] = mad(query_vec, key_vec, acc[seq_idx_index]);
                    #else
                        acc[seq_idx_index] = mad(query_vec[i], key_vec[i], acc[seq_idx_index]);
                    #endif
                }
            }


            key_offset += SUBGROUP_SIZE * KEY_BLOCK_SIZE;
        }

#if HEAD_SUZE % (SUBGROUP_SIZE * KEY_BLOCK_SIZE) != 0
#define KEY_BLOCK_SIZE 1
#define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset)
        {
            #error Shouldn't be here[2]
            INPUT1_TYPE key_val = KEY_BLOCK_READ(key_input, key_offset);

            const uint query_local_offset = HEAD_SIZE - SUBGROUP_SIZE + sglid;
            acc[0] = mad(query_vals[HEAD_SIZE / SUBGROUP_SIZE - 1], key_val, acc[0]);
        }
#endif

        unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
            acc[seq_idx_index] = sub_group_reduce_add(acc[seq_idx_index]);
        }


        if (sglid == 0) {
            // Apply scale

            const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
            for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
                acc[seq_idx_index] *= scale_val;

                // Apply attention mask
                uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(batch_idx, head_num_idx, seq_idx + seq_idx_index, start_partition_idx + seq_len);

                acc[seq_idx_index] += attn_mask[attn_mask_offset];

                // Update qk_max value
                qk_max[seq_idx_index] = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max[seq_idx_index], TO_SOFTMAX_ACCUMULATOR_TYPE(acc[seq_idx_index]));

                qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx_index];
            }
        }
    }


#else /* SINGLE_TOKEN_OPT */
    for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE / SUBGROUP_SIZE)) {
        uint key_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + seq_len, 0);

        INPUT0_TYPE acc[SEQ_ID_BLOCK_SIZE] = {INPUT0_VAL_ZERO};

#define KEY_BLOCK_SIZE 2
#define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset)
#define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)

        unroll_for (uint h = 0; h < HEAD_SIZE / SUBGROUP_SIZE / KEY_BLOCK_SIZE; h++) {
            KEY_BLOCK key_vec = KEY_BLOCK_READ(key_input, key_offset);

            unroll_for (uint i = 0; i < KEY_BLOCK_SIZE; i++) {
#ifdef QUERY_IN_SLM
                const uint query_local_offset = (h * KEY_BLOCK_SIZE + i) * SUBGROUP_SIZE + sglid;
#else
                const uint query_local_offset = (h * KEY_BLOCK_SIZE + i);
#endif

                INPUT0_TYPE tmp_acc = acc[0];
                acc[0] = mad(query_vals[query_local_offset], key_vec[i], acc[0]);
            }

            key_offset += SUBGROUP_SIZE * KEY_BLOCK_SIZE;
        }

#if HEAD_SUZE % (SUBGROUP_SIZE * KEY_BLOCK_SIZE) != 0
#define KEY_BLOCK_SIZE 1
#define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset)
        {
            INPUT1_TYPE key_val = KEY_BLOCK_READ(key_input, key_offset);

#ifdef QUERY_IN_SLM
            const uint query_local_offset = HEAD_SIZE - SUBGROUP_SIZE + sglid;
#else
            const uint query_local_offset = HEAD_SIZE / SUBGROUP_SIZE - 1;
#endif
            acc[0] = mad(query_vals[HEAD_SIZE / SUBGROUP_SIZE - 1], key_val, acc[0]);
        }
#endif

        INPUT0_TYPE tmp_acc = acc[0];

        acc[0] = sub_group_reduce_add(acc[0]);

        if (sglid == 0) {
            // Apply scale
            acc[0] *= scale_val;

            // Apply attention mask
            uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(batch_idx, head_num_idx, seq_idx, start_partition_idx + seq_len);

            acc[0] += attn_mask[attn_mask_offset];

            // Update qk_max value
            qk_max[0] = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max[0], TO_SOFTMAX_ACCUMULATOR_TYPE(acc[0]));

            qk_vals_local[seq_len] = acc[0];
        }
    }
#endif

    ulong timer_start2 = intel_get_cycle_counter();
    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        ulong diff = timer_start2 - timer_start1;
        printf("Gemm1 time: %lu\n", diff);
    }

    } // finish Gemm1


    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG * SEQ_ID_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG * SEQ_ID_BLOCK_SIZE];
    { // Start softamx

    /* Apply SoftMax */
#if MULTI_TOKENS_OPT
    const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
#else
    const uint seq_idx_index_end = 1;
#endif
        ulong timer_start1 = intel_get_cycle_counter();
        // Find the maximum value of qk in the subgroup
        for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
            qk_max[seq_idx_index] = sub_group_reduce_max(qk_max[seq_idx_index]);
        }

        // Find the maximum value of qk across all subgroups in the workgroup
        if (sglid == 0) {
            for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
                qk_max_vals[seq_idx_index * SUBGROUPS_PER_WG + sgid] = qk_max[seq_idx_index];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
            qk_max[seq_idx_index] = SOFTMAX_ACCUMULATOR_VAL_MIN;

            if (sglid < SUBGROUPS_PER_WG)
                qk_max[seq_idx_index] = qk_max_vals[seq_idx_index * SUBGROUPS_PER_WG + sglid];

            // Final maximum value of qk after reduction across all subgroups
            qk_max[seq_idx_index] = sub_group_reduce_max(qk_max[seq_idx_index]);
        }

        SOFTMAX_ACCUMULATOR_TYPE exp_sum[SEQ_ID_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        const uint qk_num_per_wi = CEIL_DIV(partition_seq_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
        for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            if (local_data_idx < partition_seq_len) {
                for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
                    SOFTMAX_ACCUMULATOR_TYPE qk_new = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + local_data_idx]) - qk_max[seq_idx_index]);
                    qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                    exp_sum[seq_idx_index] += qk_new;
                }
            }
        }

        for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
            exp_sum[seq_idx_index] = sub_group_reduce_add(exp_sum[seq_idx_index]);

            if (sglid == 0)
                qk_sum_vals[seq_idx_index * SUBGROUPS_PER_WG + sgid] = exp_sum[seq_idx_index];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
            exp_sum[seq_idx_index] = SOFTMAX_ACCUMULATOR_VAL_ZERO;

            if (sglid < SUBGROUPS_PER_WG)
                exp_sum[seq_idx_index] = qk_sum_vals[seq_idx_index * SUBGROUPS_PER_WG + sglid];

            // Find the final sum of all exp_sum[seq_idx_index] values in workgroup
            exp_sum[seq_idx_index] = sub_group_reduce_add(exp_sum[seq_idx_index]);
        }

        // const SOFTMAX_ACCUMULATOR_TYPE inv_sum = SOFTMAX_ACCUMULATOR_VAL_ONE / exp_sum;
        for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            if (local_data_idx < partition_seq_len) {
                for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
                    SOFTMAX_ACCUMULATOR_TYPE qk_new = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + local_data_idx]) / exp_sum[seq_idx_index];
                    qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);
                }
            }
        }

        /* TODO: can move this barrier outside the loop? */
        barrier(CLK_LOCAL_MEM_FENCE);

        {
            // Save temporary exm_sums and max_logits values for each partition
            if (num_of_partitions > 1 && lid == 0) {
                for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
                    const uint exp_sums_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * num_of_partitions) +
                                                head_num_idx * (INPUT0_SIZE_Y * num_of_partitions) +
                                                (seq_idx_index + seq_idx) * (num_of_partitions) +
                                                partition_idx;
                    exp_sums[exp_sums_offset] = exp_sum[seq_idx_index];

                    const uint max_logits_offset = exp_sums_offset;
                    max_logits[max_logits_offset] = qk_max[seq_idx_index];
                }
            }
        }


        ulong timer_start2 = intel_get_cycle_counter();
        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
            ulong diff = timer_start2 - timer_start1;
            printf("%d. Softmax time: %lu\n", 0, diff);
        }

    }

    /* Calculate Gemm2 */
    {
    ulong timer_start1 = intel_get_cycle_counter();

    OUTPUT_TYPE acc[SEQ_ID_BLOCK_SIZE] = {OUTPUT_VAL_ZERO};
    for (uint seq_len = 0; seq_len < partition_seq_len / SUBGROUP_SIZE; seq_len++) {
        uint value_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);

        OUTPUT_TYPE qk_val[SEQ_ID_BLOCK_SIZE];
        unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
            qk_val[seq_idx_index] = qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len * SUBGROUP_SIZE + sglid];
        }

        /* TODO: try to unroll it */
        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
            INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);
            unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
                acc[seq_idx_index] = mad(sub_group_broadcast(qk_val[seq_idx_index], i), value_val, acc[seq_idx_index]);
            }

            value_offset += HEAD_SIZE;
        }
    }

    const uint seq_len_leftover_start = (partition_seq_len / SUBGROUP_SIZE) * SUBGROUP_SIZE;
    /* TODO: Remove if */
    if (seq_len_leftover_start != partition_seq_len) {
        for (uint seq_len = seq_len_leftover_start; seq_len < partition_seq_len; seq_len++) {
            const uint value_offset = INPUT1_GET_INDEX(batch_idx, head_num_idx, start_partition_idx + seq_len, head_size_idx);

            OUTPUT_TYPE qk_val[SEQ_ID_BLOCK_SIZE];
            unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
                qk_val[seq_idx_index] = qk_vals_local[seq_idx_index * SEQ_LEN_PARTITION_SIZE + seq_len];
            }

            INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);

            unroll_for (uint seq_idx_index = 0; seq_idx_index < SEQ_ID_BLOCK_SIZE; seq_idx_index++) {
                acc[seq_idx_index] = mad(qk_val[seq_idx_index], value_val, acc[seq_idx_index]);
            }
        }
    }

        if (num_of_partitions > 1) {
#if MULTI_TOKENS_OPT
    const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
#else
    const uint seq_idx_index_end = 1;
#endif
    for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
            // tmp_output data layout  [batch, heads_num, q_len, partition_idx, head_size]
            const uint tmp_out_offset = batch_idx * (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * num_of_partitions * HEAD_SIZE) +
                                        head_num_idx * (INPUT0_SIZE_Y * num_of_partitions * HEAD_SIZE) +
                                        (seq_idx + seq_idx_index) * (num_of_partitions * HEAD_SIZE) +
                                        partition_idx * (HEAD_SIZE) +
                                        head_size_idx;
            tmp_out[tmp_out_offset] = acc[seq_idx_index];
    }
        } else {
#if MULTI_TOKENS_OPT
    const uint seq_idx_index_end = min(INPUT0_SIZE_Y - seq_idx, (uint)SEQ_ID_BLOCK_SIZE);
#else
    const uint seq_idx_index_end = 1;
#endif
    for (uint seq_idx_index = 0; seq_idx_index < seq_idx_index_end; seq_idx_index++) {
            const uint output_offset = OUTPUT_GET_INDEX(batch_idx, head_num_idx, seq_idx + seq_idx_index, head_size_idx);

            output[output_offset] = acc[seq_idx_index];
    }
        }


    ulong timer_start2 = intel_get_cycle_counter();
    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        ulong diff = timer_start2 - timer_start1;
        printf("Gemm2 time: %lu\n", diff);
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
