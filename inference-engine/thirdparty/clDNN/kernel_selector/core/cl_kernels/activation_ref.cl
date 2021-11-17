// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#ifdef PARAMETERIZED
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#else
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

// TODO: move it from layout based to memory based
KERNEL(activation)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
#ifdef PARAMETERIZED
    , __global ADDITIONAL_PARAMS_TYPE* params
#endif
    )
{
    printf("Activation %p\n", input);
#if OUTPUT_DIMS == 5
    #define ORDER batch,feature,z,y,x
#elif OUTPUT_DIMS == 4
    #define ORDER batch,feature,y,x
#endif

#if OUTPUT_DIMS == 5
    const unsigned x = get_global_id(0);
    const uint y = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
    const uint z = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
    #if OUTPUT_BATCH_NUM == 1
        const unsigned feature = (uint)get_global_id(2);
        const unsigned batch = 0;
    #else
        const unsigned feature = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
        const unsigned batch = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #endif
#else
    #if defined OUTPUT_LAYOUT_YXFB || defined OUTPUT_LAYOUT_B_FS_YX_FSV16 || defined OUTPUT_LAYOUT_B_FS_YX_FSV32
        const unsigned x = (uint)get_global_id(1);
        const unsigned y = (uint)get_global_id(2);
        #define z 0
        #if OUTPUT_BATCH_NUM == 1
            const unsigned feature = (uint)get_global_id(0);
            const unsigned batch = 0;
        #else
            const unsigned feature = (uint)get_global_id(0) % OUTPUT_FEATURE_NUM;
            const unsigned batch = (uint)get_global_id(0) / OUTPUT_FEATURE_NUM;
        #endif
    #elif defined OUTPUT_LAYOUT_BS_FS_YX_BSV32_FSV32 || defined OUTPUT_LAYOUT_BS_FS_YX_BSV32_FSV16
        const unsigned x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
        const unsigned y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
        const unsigned feature = (uint)get_global_id(1);
        const unsigned batch = (uint)get_global_id(2);
    #else
        #define z 0
            const unsigned x = (uint)get_global_id(0);
            const unsigned y = (uint)get_global_id(1);
        #if OUTPUT_BATCH_NUM == 1
            const unsigned feature = (uint)get_global_id(2);
            const unsigned batch = 0;
        #else
            const unsigned feature = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
            const unsigned batch = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
        #endif
    #endif
#endif

// GWS.feature and GWS.batch is aligned to 16. Otherwise, there are some idling WIs.
#if (defined(OUTPUT_LAYOUT_B_FS_YX_FSV16) || defined(OUTPUT_LAYOUT_B_FS_YX_FSV32)) \
    && OUTPUT_FEATURE_NUM % 16 != 0
    if (feature >= OUTPUT_FEATURE_NUM)
        return;
#elif (defined(OUTPUT_LAYOUT_BS_FS_YX_BSV32_FSV16) || defined(OUTPUT_LAYOUT_BS_FS_YX_BSV32_FSV32)) \
    && (OUTPUT_FEATURE_NUM % 16 != 0 || OUTPUT_BATCH_NUM % 16 != 0)
    if (batch >= OUTPUT_BATCH_NUM || feature >= OUTPUT_FEATURE_NUM)
        return;
#endif

    const unsigned src_index = GET_INDEX(INPUT,0,ORDER);
    const unsigned dst_index = GET_INDEX(OUTPUT,,ORDER);

#if defined PARAMETERIZED
    #if PARAMS_NUM > 2
        #error Too many params
    #elif PARAMS_NUM == 2
        #define NL_M_PARAMETERIZED (float)params[2*feature + 0]
        #define NL_N_PARAMETERIZED (float)params[2*feature + 1]
    #elif PARAMS_NUM == 1
        #define NL_M_PARAMETERIZED (float)params[feature]
        #define NL_N_PARAMETERIZED (float)NL_N
    #else
        #define NL_M_PARAMETERIZED (float)NL_M
        #define NL_N_PARAMETERIZED (float)NL_N
    #endif
    #define PARAMETERIZED_ACTIVATION_PARAMS NL_M_PARAMETERIZED, NL_N_PARAMETERIZED

    INPUT0_TYPE dst = ACTIVATION_KERNEL(input[src_index], PARAMETERIZED_ACTIVATION_PARAMS);
    #if HAS_FUSED_OPS
        FUSED_OPS;
        output[dst_index] = FUSED_OPS_RESULT;
    #else
        output[dst_index] = dst;
    #endif
#else
    INPUT0_TYPE dst = ACTIVATION_KERNEL(input[src_index], ACTIVATION_PARAMS);
    #if HAS_FUSED_OPS
        FUSED_OPS;
        output[dst_index] = FUSED_OPS_RESULT;
    #else
        output[dst_index] = dst;
    #endif
#endif
}
