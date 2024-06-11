// bad version
// Copyright (C) 2024 Intel Corporation
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


inline uint FUNC(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#else
#if INPUT0_DIMS == 4
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX(b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported input 0 format
#endif
#endif
}

inline uint FUNC(get_input0_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT0_DIMS_ORDER
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT0_DIMS_ORDER);
#else
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

inline uint FUNC(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef DO_BROADCAST_KEY_VALUE
    DO_BROADCAST_KEY_VALUE;
#endif
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D(INPUT1, b, f, w, z, y, x);
#else
#if INPUT1_DIMS == 4
    return INPUT1_GET_INDEX(b, f, y, x);
#elif INPUT1_DIMS == 5
    return INPUT1_GET_INDEX(b, f, z, y, x);
#elif INPUT1_DIMS == 6
    return INPUT1_GET_INDEX(b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input1_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT1_DIMS_ORDER
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
#else
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

inline uint FUNC(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef DO_BROADCAST_KEY_VALUE
    DO_BROADCAST_KEY_VALUE;
#endif
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, y, x);
#else
#if INPUT2_DIMS == 4
    return INPUT2_GET_INDEX(b, f, y, x);
#elif INPUT2_DIMS == 5
    return INPUT2_GET_INDEX(b, f, z, y, x);
#elif INPUT2_DIMS == 6
    return INPUT2_GET_INDEX(b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input2_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT2_DIMS_ORDER
    return FUNC_CALL(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT2_DIMS_ORDER);
#else
    return FUNC_CALL(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

#ifdef BEAM_TABLE_TYPE
inline uint FUNC(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if BEAM_TABLE_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(BEAM_TABLE, b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported beam table format
#endif
}

inline uint FUNC(get_bt_index_key)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
}

inline uint FUNC(get_bt_index_value)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT2_DIMS_ORDER);
}
#endif

#define OUTPUT_BLOCK_READ(ptr, offset) BLOCK_READN(OUTPUT_TYPE, 1, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define VALUE_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT2_TYPE, 1, ptr, offset)
#define SUBGROUPS_PER_WG (HEAD_SIZE / SUBGROUP_SIZE)

#ifdef SDPA_STAGE_0

#if TARGET_SEQ_LEN_BLOCK_SIZE == 1
/* This version is used for 2nd token */

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
#if HAS_ATTN_MASK_INPUT
    const __global INPUT3_TYPE* attn_mask,
#endif
#if HAS_SCALE_INPUT
    const __global INPUT4_TYPE* scale,
#endif
    __global OUTPUT_TYPE* output,
#ifdef BEAM_TABLE_TYPE
    const __global BEAM_TABLE_TYPE* beam_table,
#endif
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
)
{
    const uint batch_idx = get_global_id(0);
    const uint b0_idx = batch_idx / NUM_HEADS; /* BATCH dim */
    const uint b1_idx = batch_idx % NUM_HEADS; /* HEADS_NUM dim */
    const uint target_seq_idx = get_global_id(1);
    const uint lid = get_local_id(2);
    const uint head_size_idx = lid;

    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint partition_idx = get_group_id(2);
    const uint num_of_partitions = get_num_groups(2);
    const uint wi_num_per_partition = get_local_size(2);

    const uint start_partition_idx = partition_idx * SEQ_LEN_PARTITION_SIZE;
    const uint partition_seq_len =
        ((partition_idx + 1) < num_of_partitions) ? (SEQ_LEN_PARTITION_SIZE)
                                                  : (SOURCE_SEQ_LEN - partition_idx * SEQ_LEN_PARTITION_SIZE);

    // SLM for query inputs
    __local INPUT0_TYPE query_local[HEAD_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM for intermediate QK results
    __local OUTPUT_TYPE qk_local[SEQ_LEN_PARTITION_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM buffers for SoftMax calculation and qk_max/qk_sums results aggregation across all WG
    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];

    {
        // Gemm1 and SoftMax calculation

        SOFTMAX_ACCUMULATOR_TYPE qk_max[TARGET_SEQ_LEN_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
            qk_max[i] = SOFTMAX_ACCUMULATOR_VAL_MIN;
        }

        {
            // Gemm1 calculation
#if HAS_SCALE_INPUT
            const OUTPUT_TYPE scale_val = *scale;
#else
            const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(HEAD_SIZE));
#endif
            {
                // Query input loading to SLM
                #define QUERY_STEP_LOCAL SUBGROUP_SIZE * SUBGROUPS_PER_WG
                uint query_local_offset = sgid * SUBGROUP_SIZE + sglid;
                const uint seq_idx_end = 1;
#ifdef INPUT0_DIMS_ORDER
                uint query_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx, (sgid * SUBGROUP_SIZE));
                uint query_offset_next_seq = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx + 1, (sgid * SUBGROUP_SIZE));
                const uint query_pitch = query_offset_next_seq - query_offset;
#else
                uint query_offset = INPUT0_GET_INDEX(b0_idx, b1_idx, target_seq_idx, (sgid * SUBGROUP_SIZE));
                const uint query_pitch = QUERY_STEP_LOCAL;
#endif
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    #define QUERY_BLOCK_SIZE 1

                    INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

                    query_local[query_local_offset] = val;
                    query_local_offset += QUERY_STEP_LOCAL;
                    query_offset += query_pitch;
                }
                #undef QUERY_BLOCK_SIZE
                #undef QUERY_STEP

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // Main Gemm1 calculation loop
            // Each SG performs element-wise multiplications of Q[HEAD_SIZE]xK[HEAD_SIZE] values
            // HEAD_SIZE / SUBGROUPS_PER_WG times in the loop and saves the result to the qk_local SLM buffer
            for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE / SUBGROUP_SIZE)) {
#ifdef INPUT1_DIMS_ORDER
#ifdef BEAM_TABLE_TYPE
                const uint b_idx = beam_table[FUNC_CALL(get_bt_index_key)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + seq_len, 0)];
#else
                const uint b_idx = b0_idx;
#endif
                const uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + seq_len, 0);
#else
                const uint key_offset = INPUT1_GET_INDEX(b0_idx, b1_idx, start_partition_idx + seq_len, 0);
#endif

                INPUT0_TYPE acc[TARGET_SEQ_LEN_BLOCK_SIZE] = {INPUT0_VAL_ZERO};

                uint head_idx_index = 0;
                #define KEY_BLOCK_SIZE 8
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(query_vals_reg[i], key_vals[i], acc[seq_idx]);
                        }

                        query_offset += HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 4
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(query_vals_reg[i], key_vals[i], acc[seq_idx]);
                        }

                        query_offset += HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 2
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(query_vals_reg[i], key_vals[i], acc[seq_idx]);
                        }

                        query_offset += HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 1
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        acc[seq_idx] = mad(query_vals_reg, key_vals, acc[seq_idx]);
                        query_offset += HEAD_SIZE;
                    }
                }

                // Sum up all accumulators accross single SG and save result to SLM
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                    acc[seq_idx] = sub_group_reduce_add(acc[seq_idx]);
                    qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx];
                }
            }

            {
                // Wait until all SG finishes their calculations and apply scale and attention mask to the results
                barrier(CLK_LOCAL_MEM_FENCE);

                INPUT0_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
                const uint seq_idx_end = 1;
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    // Iterate over all values QK values in SLM and apply scale and attention mask
                    for (uint seq_len = sgid * SUBGROUP_SIZE + sglid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE)) {
                        // Read value from SLM and apply scale
                        qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len];
                        qk_val[seq_idx] *= scale_val;

                        // Apply attention mask
#if IS_CAUSAL
                        if (start_partition_idx + seq_len > target_seq_idx + seq_idx)
                            qk_val[seq_idx] += INPUT0_VAL_MIN;
#elif !IS_CAUSAL && HAS_ATTN_MASK_INPUT
                        const uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(b0_idx, b1_idx, target_seq_idx + seq_idx, start_partition_idx + seq_len);
                        qk_val[seq_idx] += attn_mask[attn_mask_offset];
#endif

                        // Update qk_max value
                        qk_max[seq_idx] = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max[seq_idx], TO_SOFTMAX_ACCUMULATOR_TYPE(qk_val[seq_idx]));

                        // Save modified qk value back to SLM
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len] = qk_val[seq_idx];
                    }
                }
            }
        } // Gemm1 calculation end

        {
            // SoftMax calculation
            const uint seq_idx_end = 1;
            // Find the maximum value of qk in the subgroup
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
            }

            // Find the maximum value of qk across all subgroups in the workgroup
            if (sglid == 0) {
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    qk_max_vals[seq_idx * SUBGROUPS_PER_WG + sgid] = qk_max[seq_idx];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_max[seq_idx] = SOFTMAX_ACCUMULATOR_VAL_MIN;

                if (sglid < SUBGROUPS_PER_WG)
                    qk_max[seq_idx] = qk_max_vals[seq_idx * SUBGROUPS_PER_WG + sglid];

                // Final maximum value of qk after reduction across all subgroups
                qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
            }

            SOFTMAX_ACCUMULATOR_TYPE exp_sum[TARGET_SEQ_LEN_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
            const uint qk_num_per_wi = CEIL_DIV(partition_seq_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
            for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
                const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + head_size_idx;
                if (local_data_idx < partition_seq_len) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        SOFTMAX_ACCUMULATOR_TYPE qk_new = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx]) - qk_max[seq_idx]);
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                        exp_sum[seq_idx] += qk_new;
                    }
                }
            }

            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);

                if (sglid == 0)
                    qk_sum_vals[seq_idx * SUBGROUPS_PER_WG + sgid] = exp_sum[seq_idx];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                exp_sum[seq_idx] = SOFTMAX_ACCUMULATOR_VAL_ZERO;

                if (sglid < SUBGROUPS_PER_WG)
                    exp_sum[seq_idx] = qk_sum_vals[seq_idx * SUBGROUPS_PER_WG + sglid];

                // Find the final sum of all exp_sum[seq_idx] values in workgroup
                exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);
            }

            // const SOFTMAX_ACCUMULATOR_TYPE inv_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ONE / exp_sum[seq_idx];
            for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
                const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
                if (local_data_idx < partition_seq_len) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        SOFTMAX_ACCUMULATOR_TYPE qk_new = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx]) / exp_sum[seq_idx];
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            {
                // If the number of partitions is greater than 1, save exm_sums and max_logits to the temporary buffers
                // Use single WI in the WG, since all the WIs have the same value
                if (num_of_partitions > 1 && head_size_idx == 0) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        const uint exp_sums_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                                     b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                                     (seq_idx + target_seq_idx) * (num_of_partitions) +
                                                     partition_idx;
                        exp_sums[exp_sums_offset] = exp_sum[seq_idx];

                        const uint max_logits_offset = exp_sums_offset;
                        max_logits[max_logits_offset] = qk_max[seq_idx];
                    }
                }
            }
        } // SoftMax calculation end
    } // Gemm1 + SoftMax calculations end

    {
        // Gemm2 calculation
        OUTPUT_TYPE acc[TARGET_SEQ_LEN_BLOCK_SIZE] = {OUTPUT_VAL_ZERO};
#ifndef BEAM_TABLE_TYPE
#ifdef INPUT2_DIMS_ORDER
        uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 0, 0);
        uint value_offset_next_seq = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 1, 0);
        const uint value_pitch = value_offset_next_seq - value_offset;
#else
        const uint value_pitch = HEAD_SIZE;
#endif
#endif

        for (uint seq_len = 0; seq_len < partition_seq_len / SUBGROUP_SIZE; seq_len++) {
#ifdef BEAM_TABLE_TYPE
            uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE)];
            uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE);
#else
#ifdef INPUT2_DIMS_ORDER
            uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);
#else
            uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);
#endif
#endif

            OUTPUT_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len * SUBGROUP_SIZE + sglid];
            }

            unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
#ifdef BEAM_TABLE_TYPE
                INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, sub_group_broadcast(value_offset, i));
#else
                INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);
#endif
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                    acc[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc[seq_idx]);
                }

#ifndef BEAM_TABLE_TYPE
                value_offset += value_pitch;
#endif
            }
        }

        const uint seq_len_leftovers_start = (partition_seq_len / SUBGROUP_SIZE) * SUBGROUP_SIZE;
        for (uint seq_len = seq_len_leftovers_start; seq_len < partition_seq_len; seq_len++) {
#ifdef INPUT2_DIMS_ORDER
#ifdef BEAM_TABLE_TYPE
            const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + seq_len, head_size_idx)];
#else
            const uint b_idx = b0_idx;
#endif
            const uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + seq_len, head_size_idx);
#else
            const uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, start_partition_idx + seq_len, head_size_idx);
#endif

            OUTPUT_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len];
            }

            INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);

            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                acc[seq_idx] = mad(qk_val[seq_idx], value_val, acc[seq_idx]);
            }
        }

        // If the number of partitions is greater than 1, save results to the temporary buffer;
        // otherwise, save results directly to the main output.
        if (num_of_partitions > 1) {
            const uint seq_idx_end = 1;
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                // Data layout of tmp_output buf: [batch, heads_num, q_len, partition_idx, head_size]
                const uint tmp_out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            (target_seq_idx + seq_idx) * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            head_size_idx;
                tmp_out[tmp_out_offset] = acc[seq_idx];
            }
        } else {
            const uint seq_idx_end = 1;
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                const uint output_offset = OUTPUT_GET_INDEX(b0_idx, b1_idx, target_seq_idx + seq_idx, head_size_idx);

                output[output_offset] = acc[seq_idx];
            }
        }
    } // Gemm2 calculation end
}

#else
/* This version is used for 1st token */


__attribute__((noinline))
uint FUNC(test_get_q_idx)(OPTIONAL_SHAPE_INFO_ARG uint b0_idx, uint b1_idx, uint seq_len, uint* key_pitch) {
#ifdef BEAM_TABLE_TYPE
        const uint b_idx = b0_idx;
        const uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, seq_len, 0);
#else
#ifdef INPUT1_DIMS_ORDER
        uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len, 0);
        uint key_offset_next_seq = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len + 1, 0);
        key_pitch = key_offset_next_seq - key_offset;
#else
        uint key_offset = INPUT1_GET_INDEX(b0_idx, b1_idx, seq_len, 0);
        key_pitch = HEAD_SIZE;
#endif
#endif
}

#if HAS_ATTN_MASK_INPUT
    #define ATTN_MASK_BUFFER , attn_mask
    #define ATTN_MASK_BUFFER_ARG , const __global INPUT3_TYPE* attn_mask
#else
    #define ATTN_MASK_BUFFER
    #define ATTN_MASK_BUFFER_ARG
#endif

#if HAS_SCALE_INPUT
    #define SCALE_BUFFER
#endif

inline MAKE_VECTOR_TYPE(INPUT0_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) FUNC(load_attn_mask)(OPTIONAL_SHAPE_INFO_ARG
                                                                                     uint b0_idx,
                                                                                     uint b1_idx,
                                                                                     uint target_seq_idx,
                                                                                     uint source_seq_idx
                                                                                     ATTN_MASK_BUFFER_ARG) {
    MAKE_VECTOR_TYPE(INPUT0_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) mask_vec;
#if !IS_CAUSAL && HAS_ATTN_MASK_INPUT
    // TODO: move inside else?
    const uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(b0_idx, b1_idx, target_seq_idx, source_seq_idx);
    if (target_seq_idx >= (uint)TARGET_SEQ_LEN) {
        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
            mask_vec[i] = NAN;
        }
    } else {
        if (source_seq_idx + SUBGROUP_SIZE <= (uint)SOURCE_SEQ_LEN) {
            unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                const INPUT3_TYPE mask_val = attn_mask[attn_mask_offset + i];
                mask_vec[i] = mask_val;
            }
        } else {
            // TODO: shoudl be SOURCE_SEQ_LEN
            const uint max_mask_offset = min(source_seq_idx + SUBGROUP_SIZE, (uint)SOURCE_SEQ_LEN);
            for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                const INPUT3_TYPE mask_val = source_seq_idx + i < max_mask_offset ? attn_mask[attn_mask_offset + i] : NAN;
                mask_vec[i] = mask_val;
            }
        }
    }
#endif

#if IS_CAUSAL
    for (uint i = 0; i < SUBGROUP_SIZE; i++) {
        if (source_seq_idx + i > target_seq_idx)
            // TODO: should be NaN
            mask_vec[i] = INPUT0_VAL_MIN;
    }
#endif

#if HAS_SCALE_INPUT
    const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / *scale;
#else
    const INPUT0_TYPE scale_val = TO_INPUT0_TYPE(STATIC_SCALE_VALUE_INV);
#endif

    // Apply scale to attn_mask
    // mask_vec *= scale_val;

    return mask_vec;
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
#if HAS_ATTN_MASK_INPUT
    const __global INPUT3_TYPE* attn_mask,
#endif
#if HAS_SCALE_INPUT
    const __global INPUT4_TYPE* scale,
#endif
    __global OUTPUT_TYPE* output,
#ifdef BEAM_TABLE_TYPE
    const __global BEAM_TABLE_TYPE* beam_table,
#endif
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
)
{
    const uint batch_idx = get_global_id(0);
    const uint b0_idx = batch_idx / NUM_HEADS; /* BATCH dim */
    const uint b1_idx = batch_idx % NUM_HEADS; /* HEADS_NUM dim */

#if TARGET_SEQ_LEN_BLOCK_SIZE != 1 && TARGET_SEQ_LEN_BLOCK_SIZE != 16
    #error TARGET_SEQ_LEN_BLOCK_SIZE unexpected size
#endif

#if TARGET_SEQ_LEN_BLOCK_SIZE > 1
    const uint target_seq_idx = (uint)get_global_id(1) * TARGET_SEQ_LEN_BLOCK_SIZE;
#else
    const uint target_seq_idx = get_global_id(1);
#endif
#if 0
    const uint lid = get_local_id(2);
#else
    #define lid (uint)get_local_id(2)
    #define head_size_idx (uint)get_local_id(2)
#endif

#if 0
    const uint sgid = get_sub_group_id();
#else
    #define sgid (uint)get_sub_group_id()
#endif
    const uint sglid = get_sub_group_local_id();

    #define CUSTOM_SEQ_LEN_PARTITION_SIZE (SUBGROUPS_PER_WG * SUBGROUP_SIZE)


    // SLM for query inputs
    __local INPUT0_TYPE query_local[HEAD_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // __local OUTPUT_TYPE slm_tmp_output[HEAD_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM for intermediate QK results
    __local OUTPUT_TYPE qk_local[CUSTOM_SEQ_LEN_PARTITION_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM buffers for SoftMax calculation and qk_max/qk_sums results aggregation across all WG
    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sum_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];
    // __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals_prev[TARGET_SEQ_LEN_BLOCK_SIZE];
    // __local SOFTMAX_ACCUMULATOR_TYPE qk_exp_sum_vals_prev[TARGET_SEQ_LEN_BLOCK_SIZE];
    // __local SOFTMAX_ACCUMULATOR_TYPE qk_exp_sum_prev_updated_slm[TARGET_SEQ_LEN_BLOCK_SIZE];


    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sum_cur[TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_max_val_cur[TARGET_SEQ_LEN_BLOCK_SIZE];

    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sum_prev[TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_max_val_prev[TARGET_SEQ_LEN_BLOCK_SIZE];

    // MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) output_acc_res_final = OUTPUT_VAL_ZERO;

    // Gemm1 calculation
    {
        // Load Query input to SLM and transpose it
#ifdef INPUT0_DIMS_ORDER
        uint query_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx, (sgid * SUBGROUP_SIZE));
        uint query_offset_next_seq = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx + 1, (sgid * SUBGROUP_SIZE));
        const uint query_pitch = query_offset_next_seq - query_offset;
#else
        uint query_offset = INPUT0_GET_INDEX(b0_idx, b1_idx, target_seq_idx, (sgid * SUBGROUP_SIZE));
        const uint query_pitch = SUBGROUP_SIZE * SUBGROUPS_PER_WG;
#endif
        uint query_local_offset = (sgid * SUBGROUP_SIZE + sglid) * TARGET_SEQ_LEN_BLOCK_SIZE;
        const uint target_seq_len_bs = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);

        if (target_seq_len_bs != TARGET_SEQ_LEN_BLOCK_SIZE) {
            for (uint seq_idx = 0; seq_idx < target_seq_len_bs; seq_idx++) {
                INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);

                query_local[query_local_offset] = val;
                query_offset += query_pitch;
                query_local_offset++;
            }
        } else {
            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);

                query_local[query_local_offset] = val;
                query_offset += query_pitch;
                query_local_offset++;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        #if TARGET_SEQ_LEN_BLOCK_SIZE <= SUBGROUP_SIZE
            // Initialize qk_max_vals_prev with MIN values
            if (sgid == 0 && sglid < TARGET_SEQ_LEN_BLOCK_SIZE) {
                slm_max_val_prev[sglid] = SOFTMAX_ACCUMULATOR_VAL_MIN;
                slm_exp_sum_prev[sglid] = SOFTMAX_ACCUMULATOR_VAL_ZERO;
            }
        #else
            #error sdpa_opt.cl: unsupported TARGET_SEQ_LEN_BLOCK_SIZE
        #endif
    }

    // Main Gemm1 calculation loop
    const uint aligned_seq_len = CEIL_DIV(SOURCE_SEQ_LEN, CUSTOM_SEQ_LEN_PARTITION_SIZE);

    MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) qk_acc_total = OUTPUT_VAL_ZERO;

    __attribute__((opencl_unroll_hint(1)))
    for (uint start_partition_idx = 0; start_partition_idx < aligned_seq_len; start_partition_idx++) {
        SOFTMAX_ACCUMULATOR_TYPE qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint seq_len = start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE + sgid * SUBGROUP_SIZE;
        const uint partition_seq_len = min((uint)SOURCE_SEQ_LEN - (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE), (uint)CUSTOM_SEQ_LEN_PARTITION_SIZE);

#ifdef BEAM_TABLE_TYPE
            const uint b_idx = beam_table[FUNC_CALL(get_bt_index_key)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len + sglid, 0)];
            const uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, seq_len + sglid, 0);
#else
#ifdef INPUT1_DIMS_ORDER
            uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len, 0);
            uint key_offset_next_seq = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len + 1, 0);
            const uint key_pitch = key_offset_next_seq - key_offset;
#else
            uint key_offset = INPUT1_GET_INDEX(b0_idx, b1_idx, seq_len, 0);
            const uint key_pitch = HEAD_SIZE;
#endif
#endif
            int seq_len_calc_size = min((int)(SOURCE_SEQ_LEN) - (int)seq_len, (int)SUBGROUP_SIZE);
            MAKE_VECTOR_TYPE(INPUT0_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) qk_acc = INPUT0_VAL_ZERO;

            qk_acc = FUNC_CALL(load_attn_mask)(OPTIONAL_SHAPE_INFO_TENSOR
                            b0_idx,
                            b1_idx,
                            target_seq_idx + sglid,
                            // TODO: pass seq_len_calc_size here
                            seq_len
                            ATTN_MASK_BUFFER);

            if (seq_len_calc_size >= SUBGROUP_SIZE) {
                __attribute__((opencl_unroll_hint(1)))
                for (uint head_idx_index = 0; head_idx_index < HEAD_SIZE; head_idx_index += SUBGROUP_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset);
                    #define QUERY_VEC MAKE_VECTOR_TYPE(INPUT1_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE)

                    QUERY_VEC queries_vec;
                    uint query_local_offset = (head_idx_index * TARGET_SEQ_LEN_BLOCK_SIZE) + sglid;
                    unroll_for (uint q_row_idx = 0; q_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; q_row_idx++) {
                        queries_vec[q_row_idx] = query_local[query_local_offset];
                        query_local_offset += TARGET_SEQ_LEN_BLOCK_SIZE;
                    }

                    unroll_for (uint key_row_idx = 0; key_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; key_row_idx++) {
#ifdef BEAM_TABLE_TYPE
                        INPUT1_TYPE key_vals = KEY_BLOCK_READ(key_input, sub_group_broadcast(key_offset, key_row_idx) + head_idx_index);
#else
                        INPUT1_TYPE key_vals = KEY_BLOCK_READ(key_input, key_offset + key_row_idx * key_pitch + head_idx_index);
#endif

                        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                            qk_acc[key_row_idx] = mad(sub_group_broadcast(key_vals, i), queries_vec[i], qk_acc[key_row_idx]);
                        }
                    }
                }
            } else if (seq_len_calc_size > 0) {
                __attribute__((opencl_unroll_hint(1)))
                for (uint head_idx_index = 0; head_idx_index < HEAD_SIZE; head_idx_index += SUBGROUP_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset);
                    #define QUERY_VEC MAKE_VECTOR_TYPE(INPUT1_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE)

                    QUERY_VEC queries_vec;
                    uint query_local_offset = (head_idx_index * TARGET_SEQ_LEN_BLOCK_SIZE) + sglid;
                    unroll_for (uint q_row_idx = 0; q_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; q_row_idx++) {
                        queries_vec[q_row_idx] = query_local[query_local_offset];
                        query_local_offset += TARGET_SEQ_LEN_BLOCK_SIZE;
                    }

                    unroll_for (uint key_row_idx = 0; key_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; key_row_idx++) {
#ifdef BEAM_TABLE_TYPE
                        INPUT1_TYPE key_vals = 0;
                        if (key_row_idx < seq_len_calc_size)
                            key_vals = KEY_BLOCK_READ(key_input, sub_group_broadcast(key_offset, key_row_idx) + head_idx_index);
#else
                        INPUT1_TYPE key_vals = 0;
                        // TODO: can be replaced with seq_len + key_row_idx < SEQ_LEN
                        if (key_row_idx < seq_len_calc_size)
                            key_vals = KEY_BLOCK_READ(key_input, key_offset + key_row_idx * key_pitch + head_idx_index);
#endif

                        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                            qk_acc[key_row_idx] = mad(sub_group_broadcast(key_vals, i), queries_vec[i], qk_acc[key_row_idx]);
                        }
                    }
                }
            }

            {
                unroll_for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
#if HAS_SCALE_INPUT
                    const OUTPUT_TYPE scale_val = *scale;
#else
                    const OUTPUT_TYPE scale_val = TO_OUTPUT_TYPE(STATIC_SCALE_VALUE);
#endif
                    qk_acc[i] *= scale_val;

                    /* Adding this clamp improves performance for some reason */
                    qk_acc[i] = INPUT0_MIN_FUNC(INPUT0_MAX_FUNC(qk_acc[i], INPUT0_VAL_MIN), INPUT0_VAL_MAX);
                    // if (seq_len + i >= partition_seq_len) {
                    //     qk_acc[i] = INPUT0_VAL_MIN;
                    // }

                    // if (b0_idx == 0 && b1_idx == 0 && target_seq_idx == 3072 && (head_size_idx == 0 || head_size_idx == 10) && (start_partition_idx == 12 || start_partition_idx == 2)) {
                    //     printf("%d %d %d %d; %d; %d. GEMM1 CLAMP(qk_acc*scale): %f\n",
                    //         b0_idx, b1_idx, target_seq_idx, head_size_idx, start_partition_idx, i, qk_acc[i]);
                    // }

                    qk_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max, TO_SOFTMAX_ACCUMULATOR_TYPE(qk_acc[i]));
                    // qk_local[sglid * CUSTOM_SEQ_LEN_PARTITION_SIZE + sgid * SUBGROUP_SIZE + i] = qk_acc[i];
                }
            }

            // if (b0_idx == 0 && b1_idx == 0 && target_seq_idx == 0 && start_partition_idx == 36)
            //     printf("%d %d %d %d; p_idx=%d. GEMM1 qk_acc: %f %f  %f %f  %f %f  %f %f; qk_max=%f\n",
            //         b0_idx, b1_idx, target_seq_idx, head_size_idx, start_partition_idx,
            //         qk_acc[0], qk_acc[1], qk_acc[2], qk_acc[3], qk_acc[4], qk_acc[5], qk_acc[6], qk_acc[7], qk_max);


            {
                qk_max_vals[sgid * SUBGROUP_SIZE + sglid] = qk_max;
                qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
            }

        // TODO: need this barrier?
        barrier(CLK_LOCAL_MEM_FENCE);

        // SoftMax calculation
        {
            #define QK_MAX_NUMS_PER_SG CEIL_DIV(TARGET_SEQ_LEN_BLOCK_SIZE, SUBGROUPS_PER_WG)
            #if (TARGET_SEQ_LEN_BLOCK_SIZE % SUBGROUPS_PER_WG != 0)
                /* If TARGET_SEQ_LEN_BLOCK_SIZE is not divisible by SUBGROUPS_PER_WG, then some subgroups will have to handle more QK rows than others */
                #define QK_ITERS_END \
                    (TARGET_SEQ_LEN_BLOCK_SIZE / SUBGROUPS_PER_WG + (sgid < TARGET_SEQ_LEN_BLOCK_SIZE % SUBGROUPS_PER_WG ? 1 : 0))
            #else
                #define QK_ITERS_END QK_MAX_NUMS_PER_SG
            #endif

            SOFTMAX_ACCUMULATOR_TYPE qk_max_new = SOFTMAX_ACCUMULATOR_VAL_MIN;

            for (uint i = 0; i < SUBGROUPS_PER_WG; i++) {
                SOFTMAX_ACCUMULATOR_TYPE qk_max_val = qk_max_vals[i * SUBGROUP_SIZE + sglid];
                qk_max_new = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max_new, qk_max_val);
            }

            if (sgid == 0) {
                slm_max_val_cur[sglid] = qk_max_new;
            }

            SOFTMAX_ACCUMULATOR_TYPE exp_sum_new = SOFTMAX_ACCUMULATOR_VAL_ZERO;

            for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
                qk_acc[i] = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_acc[i]) - qk_max_new);
                exp_sum_new += qk_acc[i];
            }

            {
                slm_exp_sum_vals[sgid * SUBGROUP_SIZE + sglid] = exp_sum_new;
            }

            exp_sum_new = SOFTMAX_ACCUMULATOR_VAL_ZERO;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = 0; i < SUBGROUPS_PER_WG; i++) {
                SOFTMAX_ACCUMULATOR_TYPE exp_sum = slm_exp_sum_vals[i * SUBGROUP_SIZE + sglid];
                exp_sum_new += exp_sum;
            }

            // TODO: Need to add SG barrier?
            // sub_group_barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
                qk_acc[i] = qk_acc[i] / exp_sum_new;
            }

            if (sgid == 0) {
                slm_exp_sum_cur[sglid] = exp_sum_new;
            }

            for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
                // TODO: try transposed
                // qk_local[sgid * TARGET_SEQ_LEN_BLOCK_SIZE * SUBGROUP_SIZE + i * TARGET_SEQ_LEN_BLOCK_SIZE + sglid] = qk_acc[i];
                qk_local[sglid * CUSTOM_SEQ_LEN_PARTITION_SIZE + sgid * TARGET_SEQ_LEN_BLOCK_SIZE + i] = qk_acc[i];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }


        {
            const uint seq_len_leftovers_start = (partition_seq_len / SUBGROUP_SIZE) * SUBGROUP_SIZE;
            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) acc_output_res = OUTPUT_VAL_ZERO;
            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) qk_val;
            if (/* seq_len_leftovers_start != partition_seq_len */ 1) {
                // Gemm2 calculation

#ifdef INPUT2_DIMS_ORDER
                uint value_offset_base = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 0, 0);
                uint value_offset_next_seq = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 1, 0);
                const uint value_pitch = value_offset_next_seq - value_offset_base;
#else
                const uint value_pitch = HEAD_SIZE;
#endif
                for (uint seq_len = 0; seq_len < partition_seq_len / SUBGROUP_SIZE; seq_len++) {
#ifdef BEAM_TABLE_TYPE
                    const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE)];
                    const uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE);
#else
#ifdef INPUT2_DIMS_ORDER
                    uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + (seq_len * SUBGROUP_SIZE), head_size_idx);
#else
                    uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + (seq_len * SUBGROUP_SIZE), head_size_idx);
#endif
#endif

                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        qk_val[seq_idx] = qk_local[seq_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE + seq_len * SUBGROUP_SIZE + sglid];
                    }

                    unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
#ifdef BEAM_TABLE_TYPE
                        INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, sub_group_broadcast(value_offset, i));
#else
                        INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);
#endif
                        unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                            acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc_output_res[seq_idx]);
                        }

#ifndef BEAM_TABLE_TYPE
                        value_offset += value_pitch;
#endif
                    }
                }

                // TODO: try to add if?
                {
                    // TODO: shouldn't be (partition_seq_len - 1) ?
                    uint qk_offset = min(seq_len_leftovers_start + sglid, partition_seq_len);
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        qk_val[seq_idx] = qk_local[qk_offset];
                        qk_offset += CUSTOM_SEQ_LEN_PARTITION_SIZE;
                    }
#ifdef BEAM_TABLE_TYPE
                    const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + seq_len_leftovers_start + sglid, sgid * SUBGROUP_SIZE)];
                    const uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + seq_len_leftovers_start + sglid, sgid * SUBGROUP_SIZE);
#else
#ifdef INPUT2_DIMS_ORDER
                    uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + seq_len_leftovers_start, head_size_idx);
#else
                    uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, (start_partition_idx * CUSTOM_SEQ_LEN_PARTITION_SIZE) + seq_len_leftovers_start, head_size_idx);
#endif
#endif

                    for (uint seq_len_idx = 0; seq_len_idx < partition_seq_len - seq_len_leftovers_start; seq_len_idx++) {
#ifdef BEAM_TABLE_TYPE
                        INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, sub_group_broadcast(value_offset, seq_len_idx));
#else
                        INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);
#endif

                        for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                            acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], seq_len_idx), value_val, acc_output_res[seq_idx]);
                        }

#ifndef BEAM_TABLE_TYPE
                        value_offset += value_pitch;
#endif
                    }
                }

                // Rescale acc_output_res values
                {
                    SOFTMAX_ACCUMULATOR_TYPE exp_sum_prev = slm_exp_sum_prev[sglid];
                    SOFTMAX_ACCUMULATOR_TYPE exp_sum_cur = slm_exp_sum_cur[sglid];
                    SOFTMAX_ACCUMULATOR_TYPE max_val_prev = slm_max_val_prev[sglid];
                    SOFTMAX_ACCUMULATOR_TYPE max_val_cur = slm_max_val_cur[sglid];

                    // TODO: add this barrier
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // if (b0_idx == 0 && b1_idx == 0 && target_seq_idx == 0 && head_size_idx == 0)
                    //     printf("%d %d %d %d; p_idx=%d. Current acc: %f %f  %f %f  %f %f  %f %f\n",
                    //         b0_idx, b1_idx, target_seq_idx, head_size_idx, start_partition_idx,
                    //         acc_output_res[0], acc_output_res[1], acc_output_res[2], acc_output_res[3], acc_output_res[4], acc_output_res[5], acc_output_res[6], acc_output_res[7]);

                    // if (b0_idx == 0 && b1_idx == 0 && target_seq_idx == 0 && head_size_idx == 0) {
                    //     printf("slm_exp_sum_prev: %f %f  %f %f  %f %f  %f %f\n",
                    //         slm_exp_sum_prev[0], slm_exp_sum_prev[1], slm_exp_sum_prev[2], slm_exp_sum_prev[3], slm_exp_sum_prev[4], slm_exp_sum_prev[5], slm_exp_sum_prev[6], slm_exp_sum_prev[7]);
                    //     printf("slm_exp_sum_cur: %f %f  %f %f  %f %f  %f %f\n",
                    //         slm_exp_sum_cur[0], slm_exp_sum_cur[1], slm_exp_sum_cur[2], slm_exp_sum_cur[3], slm_exp_sum_cur[4], slm_exp_sum_cur[5], slm_exp_sum_cur[6], slm_exp_sum_cur[7]);
                    //     printf("slm_max_val_prev: %f %f  %f %f  %f %f  %f %f\n",
                    //         slm_max_val_prev[0], slm_max_val_prev[1], slm_max_val_prev[2], slm_max_val_prev[3], slm_max_val_prev[4], slm_max_val_prev[5], slm_max_val_prev[6], slm_max_val_prev[7]);
                    //     printf("slm_max_val_cur: %f %f  %f %f  %f %f  %f %f\n",
                    //         slm_max_val_cur[0], slm_max_val_cur[1], slm_max_val_cur[2], slm_max_val_cur[3], slm_max_val_cur[4], slm_max_val_cur[5], slm_max_val_cur[6], slm_max_val_cur[7]);
                    // }

                    const uint seq_idx_end = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        SOFTMAX_ACCUMULATOR_TYPE total_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(sub_group_broadcast(max_val_prev, seq_idx), sub_group_broadcast(max_val_cur, seq_idx));
                        SOFTMAX_ACCUMULATOR_TYPE updated_exp_sum_prev = sub_group_broadcast(exp_sum_prev, seq_idx) * native_exp(sub_group_broadcast(max_val_prev, seq_idx) - total_max);
                        SOFTMAX_ACCUMULATOR_TYPE updated_exp_sum_cur = sub_group_broadcast(max_val_cur, seq_idx) * native_exp(sub_group_broadcast(max_val_cur, seq_idx) - total_max);
                        SOFTMAX_ACCUMULATOR_TYPE updated_total_exp_sum = updated_exp_sum_prev + updated_exp_sum_cur;

                        const uint output_offset = OUTPUT_GET_INDEX(b0_idx, b1_idx, target_seq_idx + seq_idx, sgid * SUBGROUP_SIZE);
                        // const uint output_offset = OUTPUT_GET_INDEX(b0_idx, b1_idx, target_seq_idx + seq_idx, head_size_idx);


                        if (start_partition_idx > 0) {
                            // OUTPUT_TYPE tmp_prev = OUTPUT_BLOCK_READ(output, output_offset);
                            // OUTPUT_TYPE updated_prev_res = TO_SOFTMAX_ACCUMULATOR_TYPE(OUTPUT_BLOCK_READ(output, output_offset)) * updated_exp_sum_prev / updated_total_exp_sum;
                            OUTPUT_TYPE updated_prev_res = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_acc_total[seq_idx]) * updated_exp_sum_prev / updated_total_exp_sum;;
                            // OUTPUT_TYPE updated_prev_res = slm_tmp_output[seq_idx * HEAD_SIZE + sgid * SUBGROUP_SIZE + sglid] * updated_exp_sum_prev / updated_total_exp_sum;
                            // OUTPUT_TYPE updated_prev_res = output[output_offset] * updated_exp_sum_prev / updated_total_exp_sum;
                            OUTPUT_TYPE tmp_cur = acc_output_res[seq_idx];
                            acc_output_res[seq_idx] *= updated_exp_sum_cur / updated_total_exp_sum;
                            OUTPUT_TYPE tmp_cur_updated = acc_output_res[seq_idx];
                            acc_output_res[seq_idx] += updated_prev_res;


                            // if (b0_idx == 0 && b1_idx == 0 && target_seq_idx == 0 && head_size_idx == 0)
                            //     printf("%d %d %d %d; seq_idx=%d. prev: %f current: %f; prev_new: %f, current_new: %f. Save %f (%f, %f, %f)\n",
                            //         b0_idx, b1_idx, target_seq_idx, head_size_idx, seq_idx,
                            //         tmp_prev, tmp_cur, updated_prev_res, tmp_cur_updated, acc_output_res[seq_idx], updated_exp_sum_prev, updated_exp_sum_cur, updated_total_exp_sum);
                        }

                        // slm_tmp_output[seq_idx * HEAD_SIZE + sgid * SUBGROUP_SIZE + sglid] = acc_output_res[seq_idx];
                        qk_acc_total[seq_idx] = acc_output_res[seq_idx];
                        // OUTPUT_BLOCK_WRITE(output, output_offset, acc_output_res[seq_idx]);
                        // output[output_offset] = acc_output_res[seq_idx];

                        if (sgid == 0 && sglid == 0) {
                            slm_exp_sum_prev[seq_idx] = updated_total_exp_sum;
                            slm_max_val_prev[seq_idx] = total_max;
                        }
                    }

                    // barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        }
    }

    const uint seq_idx_end = min((uint)TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
        const uint output_offset = OUTPUT_GET_INDEX(b0_idx, b1_idx, target_seq_idx + seq_idx, sgid * SUBGROUP_SIZE);
        OUTPUT_BLOCK_WRITE(output, output_offset, qk_acc_total[seq_idx]);
    }
}

#endif // TARGET_SEQ_LEN_BLOCK_SIZE != 1

#endif  // SDPA_STAGE_0

#ifdef SDPA_STAGE_1

// MTL iGPU faces high register pressure issue with a higher number of REG_VERSION_MAX_VALUES_PER_WI.
// To mitigate this, add an additional level of SDPA results processing
// with lower register pressure (REG_VERSION_MAX_VALUES_PER_WI_LOWER).

#if SOFTMAX_ACCUMULATOR_TYPE_SIZE == 4
#define REG_VERSION_MAX_VALUES_PER_WI 24
#define REG_VERSION_MAX_VALUES_PER_WI_LOWER 8
#elif SOFTMAX_ACCUMULATOR_TYPE_SIZE == 2
#define REG_VERSION_MAX_VALUES_PER_WI 48
#define REG_VERSION_MAX_VALUES_PER_WI_LOWER 16
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
    const uint batch_idx = get_global_id(0);
    const uint b0_idx = batch_idx / NUM_HEADS;
    const uint b1_idx = batch_idx % NUM_HEADS;
    const uint target_seq_idx = get_global_id(1);
    const uint sglid = get_sub_group_local_id();

    if (num_of_partitions <= SUBGROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI_LOWER) {
        /* Registers kernel version, can handle up to SEQ_LEN_PARTITION_SIZE(256) * SUBGROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI_LOWER(8/16) = 32768/65536 tokens */
        SOFTMAX_ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI_LOWER] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        SOFTMAX_ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI_LOWER] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                         b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                         target_seq_idx * (num_of_partitions) +
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

        for (uint head_size_idx = 0; head_size_idx < HEAD_SIZE / SUBGROUP_SIZE; head_size_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
            for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
                const uint tmp_out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            target_seq_idx * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            (head_size_idx * SUBGROUP_SIZE + sglid);
                OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
                acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) *
                    TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[partition_idx / SUBGROUP_SIZE], partition_idx % SUBGROUP_SIZE)) /
                    TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
            }
            const uint out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * HEAD_SIZE) +
                                    b1_idx * (TARGET_SEQ_LEN * HEAD_SIZE) +
                                    target_seq_idx * (HEAD_SIZE) +
                                    (head_size_idx * SUBGROUP_SIZE + sglid);

            output[out_offset] = TO_OUTPUT_TYPE(acc);
        }
    } else if (num_of_partitions <= SUBGROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI) {
        /* Registers kernel version, can handle up to SEQ_LEN_PARTITION_SIZE(256) * SUBGROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI(24/48) = 98304/196608 tokens */
        SOFTMAX_ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        SOFTMAX_ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                         b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                         target_seq_idx * (num_of_partitions) +
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

        for (uint head_size_idx = 0; head_size_idx < HEAD_SIZE / SUBGROUP_SIZE; head_size_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
            for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
                const uint tmp_out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            target_seq_idx * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            (head_size_idx * SUBGROUP_SIZE + sglid);
                OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
                acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) *
                    TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[partition_idx / SUBGROUP_SIZE], partition_idx % SUBGROUP_SIZE)) /
                    TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
            }
            const uint out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * HEAD_SIZE) +
                                    b1_idx * (TARGET_SEQ_LEN * HEAD_SIZE) +
                                    target_seq_idx * (HEAD_SIZE) +
                                    (head_size_idx * SUBGROUP_SIZE + sglid);

            output[out_offset] = TO_OUTPUT_TYPE(acc);
        }
    } else {
        /* Global memory kernel version, can handle any number of tokens, but could be very slow. */
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint max_logit_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                          b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                          target_seq_idx * (num_of_partitions) +
                                          partition_idx;


            if (partition_idx < num_of_partitions) {
                local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logits[max_logit_offset]);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Calculate global sum
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                         b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                         target_seq_idx * (num_of_partitions) +
                                         partition_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (partition_idx < num_of_partitions) {
                local_exp_sum += exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        for (uint head_size_idx = 0; head_size_idx < HEAD_SIZE / SUBGROUP_SIZE; head_size_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
            for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
                const uint tmp_out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            target_seq_idx * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            (head_size_idx * SUBGROUP_SIZE + sglid);

                const uint exp_sums_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                            target_seq_idx * (num_of_partitions) +
                                            partition_idx;
                const uint max_logit_offset = exp_sums_offset;

                SOFTMAX_ACCUMULATOR_TYPE new_exp_sum = exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);

                OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
                acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * new_exp_sum / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
            }

            const uint out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * HEAD_SIZE) +
                                    b1_idx * (TARGET_SEQ_LEN * HEAD_SIZE) +
                                    target_seq_idx * (HEAD_SIZE) +
                                    (head_size_idx * SUBGROUP_SIZE + sglid);

            output[out_offset] = TO_OUTPUT_TYPE(acc);
        }
    }
}

#endif
