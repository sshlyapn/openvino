// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// JIT Parameters:
// SIMD         - sub-group size/simd width, one of {8, 16};
// TILE_B       - number of batches processed by each work-item;
// TILE_OFM     - number of output features calculated by work-item, one of {1, 2, 4, 8};
// TILE_IFM     - number of input features loaded from input by work-item, one of {1, 2, 4, 8};
// TILE_K       - number of input features loaded from weights, one of {1, 2, 4, 8};
// TILE_K_OFM   - must be equal to TILE_OFM * TILE_K and less or equal to 8;
// DISPATCH_FSV - output coordinates for each sub-group are calculated from linearized coordinates
// DISPATCH_BSV   as if they laid in bs_fs_bsv_fsv format, these macros describe fsv and bsv factors;

// Verify JIT parameters.
#if SIMD != 8 && SIMD != 16
#   error "fully_connected_gpu_bf_tiled.cl - SIMD must be one of {8, 16}"
#endif

#if TILE_OFM != 1 && TILE_OFM != 2 && TILE_OFM != 4 && TILE_OFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_OFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_IFM != 1 && TILE_IFM != 2 && TILE_IFM != 4 && TILE_IFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_IFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_K != 1 && TILE_K != 2 && TILE_K != 4 && TILE_K != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4, 8}"
#endif

#if TILE_K_OFM != (TILE_K * TILE_OFM) || TILE_K_OFM > 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be equal to TILE_K * TILE_OFM and at most 8"
#endif

// Macros for vectorized types.
#define INPUT_VEC_TYPE            MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_IFM)
#define ACCUMULATOR_VEC_TYPE      MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define FILTER_VEC_TYPE           MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_K_OFM)
#define BIAS_VEC_TYPE             MAKE_VECTOR_TYPE(BIAS_TYPE, TILE_OFM)
#define OUTPUT_VEC_TYPE           MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_OFM)
#define ACTIVATION_VEC_TYPE       MAKE_VECTOR_TYPE(ACTIVATION_TYPE, TILE_OFM)
#define TO_OUTPUT_VEC_TYPE(x)     CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_ACTIVATION_VEC_TYPE(x) CAT(convert_, ACTIVATION_VEC_TYPE)(x)

#define INPUT_BLOCK_READ(ptr, offset)        BLOCK_READN(INPUT0_TYPE, TILE_IFM, ptr, offset)
#define FILTER_BLOCK_READ(ptr, offset)       BLOCK_READN(FILTER_TYPE, TILE_K_OFM, ptr, offset)
#define BIAS_BLOCK_READ(ptr, offset)         BLOCK_READN(BIAS_TYPE, TILE_OFM, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, TILE_OFM, ptr, offset, val)

// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * TILE_OUT_B_PITCH) % 16 == 0 && (OUTPUT_TYPE_SIZE * OUTPUT_OFFSET) % 16 == 0)

#if !REALIGN_FP16_OFFSET
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_SIZE_Y
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_ELEMENTS_COUNT
#   endif
#else
// For REALIGN_FP16_OFFSET one feature is processed separately before entering main loop to correct alignment.
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_SIZE_Y - 1)
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_ELEMENTS_COUNT - 1)
#   endif
#endif

#if OUTPUT_3D
#   define INPUT_ELEMENTS_COUNT INPUT0_SIZE_Y
#else
#   define INPUT_ELEMENTS_COUNT INPUT0_ELEMENTS_COUNT
#endif

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(fc)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
#ifdef HAS_DYNAMIC_PARAMS
    , DYNAMIC_PARAMS_INPUT_DECL
#endif
) {
    uint gid = (uint)get_group_id(0);
    uint sglid = (uint)get_sub_group_local_id();

#ifdef HAS_DYNAMIC_PARAMS
    const uint DISPATCH_FSV = DISPATCH_FSV1;
    const uint DISPATCH_BSV = DISPATCH_BSV1;
    const int temp_bi = TILE_B1;
#endif

    if (get_global_id(0) == 0) {
        // printf("Empty\n");
        printf("Kernel has: %d\n", DISPATCH_BSV);
        printf("Kernel has2: %d\n", DISPATCH_FSV);
    }

    // Dispatch as bs_fs_bsv_fsv, where bsv = DISPATCH_BSV and fsv = DISPATCH_FSV.
    // This allows more fine grained control over dispatch order than using work-groups and
    // avoids requirement of threads being available for whole work-group.
    // It could hovewer have some drawbacks like not providing physical locality or not using
    // full dispatch pipeline.
    uint feature_mini_block = gid % DISPATCH_FSV;
    uint batch_mini_block = gid / DISPATCH_FSV % DISPATCH_BSV;
    uint feature_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV) % (CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);
    uint batch_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV * CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);

    uint out_f = (feature_mega_block * DISPATCH_FSV + feature_mini_block) * (TILE_OFM * SIMD);
    uint out_b = ((batch_mega_block * DISPATCH_BSV + batch_mini_block) * temp_bi);

    ACCUMULATOR_VEC_TYPE acc[TILE_B] = { };
    INPUT_VEC_TYPE       in_0[TILE_B] = { };

    FILTER_VEC_TYPE wei = 0;
    uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
    uint weights_offset = out_f * INPUT_ELEMENTS_COUNT;

#if REALIGN_FP16_OFFSET
    // For fp16 we need to ensure that all block reads are aligned to 4 byte (2 words) boundary.
    // To do this solve first input feature separately.
    {
        INPUT0_TYPE tmp_input = input[input_offset + get_sub_group_local_id() % temp_bi * TILE_IN_B_PITCH];
        MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_OFM) tmp_wei = BLOCK_READN(FILTER_TYPE, TILE_OFM, weights, weights_offset);

        unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
            acc[bi] = _sub_group_shuffle(tmp_input, bi) * tmp_wei;
        }

        weights_offset += TILE_OFM * SIMD;
        input_offset += 1;
    }
#endif
    // =====================================================================================================================================
    // Main computation loop
    uint iterations = MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD);

    INPUT0_TYPE in_val;

    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // IDEA:
        // bool bi_iter = bi < TILE_B;
        // if (bi_iter) LOAD_IN_0(bi)
        // Load input.


#ifndef CONST_LOOP_8
#define CONST_LOOP_8(macro)
#endif

#ifndef CONST_LOOP_7
#define CONST_LOOP_7(macro)
#endif

#ifndef CONST_LOOP_7
#define CONST_LOOP_7(macro)
#endif

#ifndef CONST_LOOP_6
#define CONST_LOOP_6(macro)
#endif

#ifndef CONST_LOOP_5
#define CONST_LOOP_5(macro)
#endif

#ifndef CONST_LOOP_4
#define CONST_LOOP_4(macro)
#endif

#ifndef CONST_LOOP_3
#define CONST_LOOP_3(macro)
#endif

#ifndef CONST_LOOP_2
#define CONST_LOOP_2(macro)
#endif

#ifndef CONST_LOOP_1
#define CONST_LOOP_1(macro)
#endif

        // if (/* bi + out_b < OUTPUT_BATCH_NUM */ /* bi >= temp_bi */ false)
        #define LOAD_IN_0(bi) {                           \
                in_0[bi] = 0;     \
                input_offset += TILE_IN_B_PITCH; \
            }

            // if (temp_bi == 8) { CONST_LOOP_8(LOAD_IN_0) }
            // else if (temp_bi == 7) { CONST_LOOP_7(LOAD_IN_0) }
            // else if (temp_bi == 6) { CONST_LOOP_6(LOAD_IN_0) }
            // else if (temp_bi == 5) { CONST_LOOP_5(LOAD_IN_0) }
            // else if (temp_bi == 4) { CONST_LOOP_4(LOAD_IN_0) }
            // else if (temp_bi == 3) { CONST_LOOP_3(LOAD_IN_0) }
            // else if (temp_bi == 2) { CONST_LOOP_2(LOAD_IN_0) }
            // else  { CONST_LOOP_1(LOAD_IN_0) }

        CONST_LOOP(TILE_B, LOAD_IN_0);

        #undef LOAD_IN_0

        // input_offset -= (TILE_B - temp_bi) * TILE_IN_B_PITCH;
        // input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * temp_bi;
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * TILE_B;

        // input_offset += - TILE_IN_B_PITCH * TILE_B;
        // input_offset += TILE_IFM * SIMD;

        // unroll_for (int i = TILE_B - 1; out_b + i >= OUTPUT_BATCH_NUM; i-- )
        //     in_0[i] = 0;
        // NOTE: Manually unrolling multiplication loop leads to lower register pressure and allows for bigger block sizes,
        //       but significantly degrades readability and generality of code.
        //       It doesn't also show noticable performance improvement on tested configurations.
        unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
            wei = ki;
            weights_offset += TILE_K_OFM * SIMD;

            // unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
            //     unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
            //         unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
            //             const uint total_k = ki * TILE_K + kii;
            //             INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
            //             ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((FILTER_TYPE*)(&wei))[kii * TILE_OFM + fi];
            //         }
            //     }
            // }

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                    // if (bi + out_b >= OUTPUT_BATCH_NUM)
                    //     continue;
                    const uint total_k = ki * TILE_K + kii;
                    in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                    unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                        ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((FILTER_TYPE*)(&wei))[kii * TILE_OFM + fi];
                    }
                }
            }
        }
    }
    // =====================================================================================================================================
    // Leftovers
#if MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    // Handle leftovers in normal case without alignment correction.
    #define LEFTOVER_IFM               (MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD))
    {
        #define LOAD_IN_0(bi) do {                                  \
                if (bi >- temp_bi) \
                    break; \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                input_offset += TILE_IN_B_PITCH;                    \
            } while (false)

        CONST_LOOP(TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * temp_bi;
        unroll_for(uint ki = 0; ki < CEIL_DIV(LEFTOVER_IFM, TILE_K); ++ki) {
            wei = FILTER_BLOCK_READ(weights, weights_offset);
            weights_offset += TILE_K_OFM * SIMD;

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                    #define MUL_OP2(bi) if (bi < temp_bi) { \
                            const uint total_k = ki * TILE_K + kii; \
                            if (total_k < LEFTOVER_IFM) { \
                            unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) { \
                                INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD); \
                            } \
                    }
                    CONST_LOOP(TILE_B, MUL_OP2);
                    #undef MUL_OP2

                    // unroll_for (uint bi = 0; bi < temp_bi; ++bi) {
                    //     const uint total_k = ki * TILE_K + kii;
                    //     if (total_k < LEFTOVER_IFM) {
                    //         INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                    //         ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((FILTER_TYPE*)(&wei))[kii * TILE_OFM + fi];
                    //     }
                    // }
                }
            }
        }
    }
    #undef LEFTOVER_IFM
#endif // MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    // =====================================================================================================================================
    // Post-processing: bias, activation, fused-ops
    ACTIVATION_VEC_TYPE activated[TILE_B] = {0};
    for (uint bi = 0; bi < temp_bi; ++bi) {
        activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
    }

#if BIAS_TERM
    #if TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0
        BIAS_VEC_TYPE bias = BIAS_BLOCK_READ(biases, out_f);
    #else
        BIAS_VEC_TYPE bias = 0;
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            ((BIAS_TYPE*)(&bias))[fi] = biases[out_f + sglid + fi * SIMD];
        }
    #endif
    unroll_for (uint bi = 0; bi < temp_bi; ++bi) {
        activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
    }
#endif

    OUTPUT_VEC_TYPE result[TILE_B] = { };
#if HAS_FUSED_OPS
    unroll_for (uint bi = 0; bi < temp_bi; ++bi) {
    #if TILE_OFM > 1
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            FUSED_OPS_VEC;
            result[bi][fi] = FUSED_OPS_RESULT_VEC;
        }
    #else
        FUSED_OPS_SCALAR;
        result[bi] = FUSED_OPS_RESULT_SCALAR;
    #endif // TILE_OFM > 1
    }
#else
    unroll_for (uint bi = 0; bi < temp_bi; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
    }
#endif
    // =====================================================================================================================================
    // Write results
    uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;

    if (temp_bi != TILE_B) {
        // for (int bi = 0; bi < temp_bi; bi++) {
#define WRITE_OP(bi) if (bi < temp_bi) {OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]); output_offset += TILE_OUT_B_PITCH;}
            // OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);
            // output_offset += TILE_OUT_B_PITCH;

            // if (temp_bi == 8) { CONST_LOOP_8(WRITE_OP) }
            // else if (temp_bi == 7) { CONST_LOOP_7(WRITE_OP) }
            // else if (temp_bi == 6) { CONST_LOOP_6(WRITE_OP) }
            // else if (temp_bi == 5) { CONST_LOOP_5(WRITE_OP) }
            // else if (temp_bi == 4) { CONST_LOOP_4(WRITE_OP) }
            // else if (temp_bi == 3) { CONST_LOOP_3(WRITE_OP) }
            // else if (temp_bi == 2) { CONST_LOOP_2(WRITE_OP) }
            // else  { CONST_LOOP_1(WRITE_OP) }

            CONST_LOOP(TILE_B, WRITE_OP)
#undef WRITE_OP
        // }
    } else if (USE_BLOCK_WRITE && (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 || out_f + (TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
        #define WRITE_OUTPUT(bi) do {                                       \
                if (/* bi + out_b < BATCH_SIZE */ bi >= temp_bi)                          \
                    break; \
                    OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);  \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)

        CONST_LOOP(TILE_B, WRITE_OUTPUT);
        #undef WRITE_OUTPUT
    } else {
        output_offset += sglid;

        // TODO: Investigate why below code doesn't compile and check how it affects performance.
        //#define WRITE_OUTPUT_FEATURE(fi) do {                                                   \
        //        const bool should_write =                                                       \
        //            TILE_OUT_F_NUM %  (TILE_OFM * SIMD) == 0 ||                                 \
        //            out_f + (fi) * SIMD + sglid < TILE_OUT_F_NUM;                               \
        //        if (should_write) {                                                             \
        //            output[output_offset] = result[out_bi][fi];                                 \
        //        }                                                                               \
        //        output_offset += SIMD;                                                          \
        //    } while (false)
        //
        //#define WRITE_OUTPUT(bi) do {                                                           \
        //        const uint out_bi = bi;                                                         \
        //        CONST_LOOP(TILE_OFM, WRITE_OUTPUT_FEATURE);                                     \
        //        output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;                            \
        //    } while (false)
        //
        //CONST_LOOP(TILE_B, WRITE_OUTPUT);
        //#undef WRITE_OUTPUT
        //#undef WRITE_OUTPUT_FEATURE

        for (uint bi = 0; bi < temp_bi; ++bi) {
            for (uint fi = 0; fi < TILE_OFM; ++fi) {
                const bool should_write =
                    TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 ||
                    out_f + fi * SIMD + sglid < TILE_OUT_F_NUM;
                if (should_write) {
                    output[output_offset] = ((OUTPUT_TYPE*)(&result[bi]))[fi];
                }
                output_offset += SIMD;
            }
            output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
        }
    }
    // =====================================================================================================================================
}

#undef INPUT_VEC_TYPE
#undef ACCUMULATOR_VEC_TYPE
#undef FILTER_VEC_TYPE
#undef BIAS_VEC_TYPE
#undef OUTPUT_VEC_TYPE
#undef ACTIVATION_VEC_TYPE
#undef TO_OUTPUT_VEC_TYPE
#undef TO_ACTIVATION_VEC_TYPE

#undef INPUT_BLOCK_READ
#undef FILTER_BLOCK_READ
#undef BIAS_BLOCK_READ
#undef OUTPUT_BLOCK_WRITE

#undef USE_BLOCK_WRITE

#undef MAIN_LOOP_ELEMENTS_COUNT
