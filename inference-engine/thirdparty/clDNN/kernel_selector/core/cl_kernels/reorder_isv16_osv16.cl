// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// #include "include/fetch.cl"
#include "include/reshape_dims.cl"
#include "include/data_types.cl"
#include "include/common.cl"


#define GET_FILTER_INDEX(prefix, g, o, i, y, x) GET_FILTER_GOIYX(prefix, g, o, i, y, x)

#define GET_FILTER_INDEX_SAFE(prefix, g, o, i, y, x) GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x)

#define GET_FILTER_INDEX_5D(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x)

#define GET_FILTER_INDEX_5D_SAFE(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x)


#define GET_FILTER_GOIYX(prefix, g, o, i, y, x) \
    CAT(prefix, _OFFSET) +                      \
    (x)*CAT(prefix, _X_PITCH) +                 \
    (y)*CAT(prefix, _Y_PITCH) +                 \
    (i)*CAT(prefix, _IFM_PITCH) +               \
    (o)*CAT(prefix, _OFM_PITCH) +               \
    (g)*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x)        \
    CAT(prefix, _OFFSET) +                                  \
    (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) +     \
    (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) +     \
    (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) +   \
    (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) +   \
    (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)


#define GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x) \
    CAT(prefix, _OFFSET) +                          \
    (x)*CAT(prefix, _X_PITCH) +                     \
    (y)*CAT(prefix, _Y_PITCH) +                     \
    (z)*CAT(prefix, _Z_PITCH) +                     \
    (i)*CAT(prefix, _IFM_PITCH) +                   \
    (o)*CAT(prefix, _OFM_PITCH) +                   \
    (g)*CAT(prefix, _GROUPS_PITCH)

#define GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x)    \
    CAT(prefix, _OFFSET) +                                  \
    (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) +     \
    (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) +     \
    (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) +     \
    (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) +   \
    (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) +   \
    (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)


#define GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(prefix, o, i, y, x, sub_group_size) \
    CAT(prefix, _OFFSET) +                                                        \
    ((o) % (sub_group_size)) +                                                    \
    (sub_group_size)*(                                                            \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) +                              \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) +                              \
        ((i) % (sub_group_size)) +                                                \
        ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) +       \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                          \
    )

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint g, uint o, uint i, uint z, uint y, uint x)
{
#if   INPUT0_SIMPLE && INPUT0_DIMS <= 4
    return GET_FILTER_INDEX(INPUT0, 0, o, i, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_FILTER_INDEX_5D(INPUT0, 0, o, i, z, y, x);
#else
#error reorder_weights.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint g, uint o, uint i, uint z, uint y, uint x)
{
#if   OUTPUT_SIMPLE && OUTPUT_DIMS <= 4
    return GET_FILTER_INDEX(OUTPUT, 0, o, i, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 5
    return GET_FILTER_INDEX_5D(OUTPUT, 0, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISV16_OSV16
    return GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#else
#error reorder_weights.cl: output format - not supported
#endif
}

#if OUTPUT_TYPE_SIZE == 2
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write_us((__global ushort*)(ptr) + (offset), as_ushort(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write_us2((__global ushort*)(ptr) + (offset), as_ushort2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write_us4((__global ushort*)(ptr) + (offset), as_ushort4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write_us8((__global ushort*)(ptr) + (offset), as_ushort8(val))
#elif OUTPUT_TYPE_SIZE == 4
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write((__global uint*)(ptr) + (offset), as_uint(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write2((__global uint*)(ptr) + (offset), as_uint2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write4((__global uint*)(ptr) + (offset), as_uint4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write8((__global uint*)(ptr) + (offset), as_uint8(val))
#else
#   error convolution_gpu_bfyx_f16.cl - unsupported output type.
#endif
#define INPUT_TYPE_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_IFM_BLOCK_SIZE)
KERNEL (reorder_isv16_osv16)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if OUTPUT_GROUPS_NUM > 1
    const unsigned g = (uint)get_group_id(2) / OUTPUT_OFM_NUM;
    const unsigned o = (uint)get_group_id(2) % OUTPUT_OFM_NUM;
    #error Groups is unsupported now
#else
    const unsigned g = 0;
    const unsigned o_block = (uint)get_group_id(2) * OSV;
#endif
    const unsigned lid = get_sub_group_local_id();
    const unsigned o = o_block + lid;
    const unsigned i = (uint)get_global_id(1) * OUTPUT_IFM_BLOCK_SIZE;

#if   OUTPUT_DIMS == 2 || (OUTPUT_DIMS == 3 && OUTPUT_GROUPED)
    const unsigned x = 0;
    const unsigned y = 0;
    const unsigned z = 0;
#elif OUTPUT_DIMS == 4 || (OUTPUT_DIMS == 5 && OUTPUT_GROUPED)
    const unsigned x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
    const unsigned y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
    const unsigned z = 0;
#elif OUTPUT_DIMS == 5 || (OUTPUT_DIMS == 6 && OUTPUT_GROUPED)
    const unsigned zyx = get_global_id(0);
    const unsigned x = zyx % OUTPUT_SIZE_X;
    const unsigned y = (zyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const unsigned z = (zyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
#endif

    uint input_idx = FUNC_CALL(get_input_index)(g, o, i, z, y, x);
    uint output_idx = FUNC_CALL(get_output_index)(g, o, i, z, y, x);
    output[output_idx] = input[input_idx];
    // if (o == 2 && i == 0 && x == 0 && y == 0)
    //     printf("%d -> %d\n", input_idx, output_idx);
// #if !REORDER_ROTATE
//     uint output_idx = FUNC_CALL(get_output_index)(g, o_block, i, z, y, x);
// #else
//     uint output_idx = FUNC_CALL(get_output_index)(g, o_block, i, OUTPUT_SIZE_Z - z - 1, OUTPUT_SIZE_Y - y - 1, OUTPUT_SIZE_X - x - 1);
// #endif

//     INPUT_TYPE_VEC in_data;
//     for (int i = 0; i < OUTPUT_IFM_BLOCK_SIZE; i++) {
// #if OUTPUT_IFM_BLOCK_SIZE == 1
//         in_data = input[input_idx];
// #else
//         in_data[i] = input[input_idx];
//         input_idx += INPUT0_IFM_PITCH;
// #endif
//     }
//     // if (o_block == 0 && i == 0 && xy == 0)
//     //     printf("%f\n", in_data);
// #if OUTPUT_IFM_BLOCK_SIZE == 8
//     OUTPUT_BLOCK_WRITE8(output, output_idx, in_data);
// #elif OUTPUT_IFM_BLOCK_SIZE == 4
//     OUTPUT_BLOCK_WRITE4(output, output_idx, in_data);
// #elif OUTPUT_IFM_BLOCK_SIZE == 2
//     OUTPUT_BLOCK_WRITE2(output, output_idx, in_data);
// #elif OUTPUT_IFM_BLOCK_SIZE == 1
//     OUTPUT_BLOCK_WRITE(output, output_idx, in_data);
// #else
// #   error reorder_osv16.cl: Unsupported output x block size.
// #endif
}
