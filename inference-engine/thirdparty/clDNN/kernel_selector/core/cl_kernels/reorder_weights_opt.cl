// Copyright (c) 2020 Intel Corporation
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

#include "include/common.cl"
#include "include/data_types.cl"

INIT_INPUT0_INDEX_FUNC_HERE
INIT_OUTPUT_INDEX_FUNC_HERE

#if OUTPUT_GROUPED
#   if OUTPUT_DIMS == 5
#       define IDX_ORDER g, o, i, y, x
#       define BLOCK_IDX_ORDER g, o_blocked, i_blocked, y, x
#   elif OUTPUT_DIMS == 6
#       define IDX_ORDER g, o, i, z, y, x
#       define BLOCK_IDX_ORDER g, o_blocked, i_blocked, z, y, x
#   endif
#else
#   if OUTPUT_DIMS == 4
#       define IDX_ORDER o, i, y, x
#       define BLOCK_IDX_ORDER o_blocked, i_blocked, y, x
#   elif OUTPUT_DIMS == 5
#       define IDX_ORDER o, i, z, y, x
#       define BLOCK_IDX_ORDER o_blocked, i_blocked, z, y, x
#   endif
#endif
#define GET_INDEX(macro, ...) macro(__VA_ARGS__)

#if OUTPUT_TYPE_SIZE == 1
#   define OUTPUT_BLOCK_WRITE1(ptr, offset, val)   intel_sub_group_block_write_uc1((__global uchar*)(ptr) + (offset), as_uchar(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write_uc2((__global uchar*)(ptr) + (offset), as_uchar2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write_uc4((__global uchar*)(ptr) + (offset), as_uchar4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write_uc8((__global uchar*)(ptr) + (offset), as_uchar8(val))
#elif OUTPUT_TYPE_SIZE == 2
#   define OUTPUT_BLOCK_WRITE1(ptr, offset, val)   intel_sub_group_block_write_us((__global ushort*)(ptr) + (offset), as_ushort(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write_us2((__global ushort*)(ptr) + (offset), as_ushort2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write_us4((__global ushort*)(ptr) + (offset), as_ushort4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write_us8((__global ushort*)(ptr) + (offset), as_ushort8(val))
#elif OUTPUT_TYPE_SIZE == 4
#   define OUTPUT_BLOCK_WRITE1(ptr, offset, val)   intel_sub_group_block_write((__global uint*)(ptr) + (offset), as_uint(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write2((__global uint*)(ptr) + (offset), as_uint2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write4((__global uint*)(ptr) + (offset), as_uint4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write8((__global uint*)(ptr) + (offset), as_uint8(val))
#else
#   error reorder_weights_opt.cl - unsupported output type.
#endif

#if OSV_FIRST
#   define FIRST_BLOCK_SIZE OFM_BLOCK_SIZE
#   define SECOND_BLOCK_SIZE IFM_BLOCK_SIZE
#   define PITCH INPUT0_IFM_PITCH
#   define SECOND_SIZE IFM_SIZE
#else
#   define FIRST_BLOCK_SIZE IFM_BLOCK_SIZE
#   define SECOND_BLOCK_SIZE OFM_BLOCK_SIZE
#   define PITCH INPUT0_OFM_PITCH
#   define SECOND_SIZE OFM_SIZE
#endif

#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, SECOND_BLOCK_SIZE)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) CAT(OUTPUT_BLOCK_WRITE, SECOND_BLOCK_SIZE)(ptr, offset, val)

KERNEL(reorder_weights_blocked_opt)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const int lid = get_sub_group_local_id();
    const int g_io = get_global_id(0);
#if OSV_FIRST
#if OUTPUT_GROUPED
    const int i = (g_io % (OUTPUT_IFM_NUM / SECOND_BLOCK_SIZE)) * SECOND_BLOCK_SIZE;
    const int g = (g_io / (OUTPUT_IFM_NUM / SECOND_BLOCK_SIZE));
#else
    const int i = g_io * SECOND_BLOCK_SIZE;
#endif  // OUTPUT_GROUPED
    const int o_blocked = (int)get_group_id(2) * FIRST_BLOCK_SIZE;
    const int o = o_blocked + lid;
    const int i_blocked = i;
#else  // OSV_FIRST
#if OUTPUT_GROUPED
    const int o = (g_io % (OUTPUT_OFM_NUM / SECOND_BLOCK_SIZE)) * SECOND_BLOCK_SIZE;
    const int g = (g_io / (OUTPUT_OFM_NUM / SECOND_BLOCK_SIZE));
#else
    const int o = g_io * SECOND_BLOCK_SIZE;
#endif  // OUTPUT_GROUPED
    const int i_blocked = (int)get_group_id(2) * FIRST_BLOCK_SIZE;
    const int i = i_blocked + lid;
    const int o_blocked = o;
#endif  // OSV_FIRST

    const int zyx = get_global_id(1);
    const int x = zyx % OUTPUT_SIZE_X;
#if (OUTPUT_DIMS - OUTPUT_GROUPED) == 5
    const int y = zyx / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const int z = zyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y;
#else
    const int y = zyx / OUTPUT_SIZE_X;
#endif  // (OUTPUT_DIMS - OUTPUT_GROUPED) == 5

    int input_idx = GET_INDEX(INPUT0_GET_INDEX, IDX_ORDER);
    const int output_idx = GET_INDEX(OUTPUT_GET_INDEX, BLOCK_IDX_ORDER);

#if SECOND_BLOCK_SIZE == 1
    const OUTPUT_TYPE val = TO_OUTPUT_TYPE(input[input_idx]);
#else
    OUTPUT_VEC_TYPE val = 0;
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < SECOND_BLOCK_SIZE; i++) {
        val[i] = TO_OUTPUT_TYPE(input[input_idx]);
        input_idx += PITCH;
    }
#endif  // SECOND_BLOCK_SIZE == 1
#if OUTPUT_LEFTOVERS
#if OSV_FIRST
    const bool doWrite = o < OUTPUT_OFM_NUM;
    if (o_blocked >= OUTPUT_OFM_NUM - FIRST_BLOCK_SIZE) {
#else
    const bool doWrite = i < OUTPUT_IFM_NUM;
    if (i_blocked >= OUTPUT_IFM_NUM - FIRST_BLOCK_SIZE) {
#endif  // OSV_FIRST
#if SECOND_BLOCK_SIZE > 1
        __attribute__((opencl_unroll_hint))
        for (int b = 0; b < SECOND_BLOCK_SIZE; b++)
            if (doWrite)
                output[output_idx + b * SECOND_SIZE + lid] = val[b];
#else
            if (doWrite)
                output[output_idx + lid] = val;
#endif  // SECOND_BLOCK_SIZE > 1
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        OUTPUT_BLOCK_WRITE(output, output_idx, val);
    }
}

#undef OUTPUT_VEC_TYPE
#undef OSV_FIRST
#undef FIRST_BLOCK_SIZE
#undef SECOND_BLOCK_SIZE
#undef PITCH
#undef SECOND_SIZE
#undef OUTPUT_BLOCK_WRITE8
#undef OUTPUT_BLOCK_WRITE4
#undef OUTPUT_BLOCK_WRITE2
#undef OUTPUT_BLOCK_WRITE1
#undef OUTPUT_BLOCK_WRITE
#undef GET_INDEX
#undef BLOCK_IDX_ORDER
#undef IDX_ORDER
