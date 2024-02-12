// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"


// printf("in0 shape[%dx%d], in1 shape[%dx%d], out shape[%dx%d]", INPUT0_BATCH_NUM, INPUT0_BATCH_PITCH, INPUT1_BATCH_NUM, INPUT1_BATCH_PITCH, OUTPUT_BATCH_NUM, OUTPUT_BATCH_PITCH);

KERNEL(beam_table_update)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* state_prev,
    __global const INPUT1_TYPE* beam_idx,
    __global OUTPUT_TYPE* state_new,
    uchar is_state_set)
{
    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0 && INPUT1_BATCH_NUM == 2) {
        // printf("Bean content: %d %d\n", beam_idx[0], beam_idx[1]);
    }
    const unsigned int b = (uint)get_global_id(0);
    const unsigned int s = (uint)get_global_id(1);

    const unsigned int out_offset = b * OUTPUT_BATCH_PITCH + s;
    const unsigned int in_offset = beam_idx[b] * INPUT0_BATCH_PITCH + s;

    if (s >= OUTPUT_BATCH_PITCH)
        return;

    if (!is_state_set) {
        // printf("%d %d. in0 shape[%dx%d], in1 shape[%dx%d], out shape[%dx%d]. Init state_new[%d]=%d\n",
        //     b, s, INPUT0_BATCH_NUM, INPUT0_BATCH_PITCH, INPUT1_BATCH_NUM, INPUT1_BATCH_PITCH, OUTPUT_BATCH_NUM, OUTPUT_BATCH_PITCH, out_offset, b);
        state_new[out_offset] = TO_OUTPUT_TYPE(b);
    } else {
        if (s < INPUT0_BATCH_PITCH) {
            // printf("%d %d. in0 shape[%dx%d], in1 shape[%dx%d], out shape[%dx%d]. Reuse state_new[%d]=state_prev[%d](%d)\n",
            //     b, s, INPUT0_BATCH_NUM, INPUT0_BATCH_PITCH, INPUT1_BATCH_NUM, INPUT1_BATCH_PITCH, OUTPUT_BATCH_NUM, OUTPUT_BATCH_PITCH, out_offset, in_offset, state_prev[in_offset]);
            state_new[out_offset] = state_prev[in_offset];
        } else {
            // printf("%d %d. in0 shape[%dx%d], in1 shape[%dx%d], out shape[%dx%d]. New state_new[%d]=%d\n",
            //     b, s, INPUT0_BATCH_NUM, INPUT0_BATCH_PITCH, INPUT1_BATCH_NUM, INPUT1_BATCH_PITCH, OUTPUT_BATCH_NUM, OUTPUT_BATCH_PITCH, out_offset, b);
            state_new[out_offset] = b;
        }
    }
}
