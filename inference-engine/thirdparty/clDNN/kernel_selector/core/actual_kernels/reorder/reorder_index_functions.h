// Copyright (c) 2016 Intel Corporation
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


#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
    inline size_t get_index_oiyx(size_t o, size_t i, size_t y, size_t x, size_t ifm_size, size_t y_size, size_t x_size) {
        const size_t x_pitch = 1;
        const size_t y_pitch = x_size * x_pitch;
        const size_t ifm_pitch = y_size * y_pitch;
        const size_t ofm_pitch = ifm_size * ifm_pitch;
        const size_t idx = o * ofm_pitch + i * ifm_pitch + y * y_pitch + x * x_pitch;
        return idx;
    }

    inline size_t get_index_goizyx(size_t g, size_t o, size_t i, size_t z, size_t y, size_t x, size_t ofm_size, size_t ifm_size, size_t z_size, size_t y_size, size_t x_size) {
        const size_t x_pitch = 1;
        const size_t y_pitch = x_size * x_pitch;
        const size_t z_pitch = y_size * y_pitch;
        const size_t ifm_pitch = z_size * z_pitch;
        const size_t ofm_pitch = ifm_size * ifm_pitch;
        const size_t g_pitch = ofm_size * ofm_pitch;
        const size_t idx = g * g_pitch + o * ofm_pitch + i * ifm_pitch + z * z_pitch + y * y_pitch + x * x_pitch;
        return idx;
    }

    inline size_t get_index_os_iyx_osv16(size_t o, size_t i, size_t y, size_t x, size_t ofm_pitch, size_t ifm_pitch, size_t y_pitch, size_t x_pitch) {
        const size_t block_size = 16;
        const size_t idx = (o % block_size) + block_size * (x * x_pitch +
                                                            y * y_pitch +
                                                            i * ifm_pitch +
                                                            (o / block_size) * ofm_pitch);
        return idx;
    }

    inline size_t get_index_os_is_yx_isv16_osv16(size_t o, size_t i, size_t y, size_t x, size_t ofm_pitch, size_t ifm_pitch, size_t y_pitch, size_t x_pitch) {
        const size_t block_size = 16;
        const size_t idx = (o % block_size) + block_size * (x * block_size * x_pitch +
                                                            y * block_size * y_pitch +
                                                            (i % block_size) +
                                                            (i / block_size) * block_size * ifm_pitch +
                                                            (o / block_size) * ofm_pitch);
        return idx;
    }

    inline size_t get_index_oi(size_t o, size_t i, size_t ifm_size) {
        const size_t ofm_pitch = ifm_size;
        const size_t ifm_pitch = 1;
        const size_t idx = o * ofm_pitch + i * ifm_pitch;
        return idx;
    }

    inline size_t get_index_io(size_t o, size_t i, size_t ofm_size) {
        const size_t ofm_pitch = 1;
        const size_t ifm_pitch = ofm_size * ofm_pitch;
        const size_t idx = o * ofm_pitch + i * ifm_pitch;
        return idx;
    }

    inline size_t get_index(WeightsLayout l, size_t g, size_t o, size_t i, size_t z, size_t y, size_t x,  size_t g_size, size_t o_size, size_t i_size, size_t z_size, size_t y_size, size_t x_size) {
        switch (l) {
            case WeightsLayout::oi:
                return get_index_oi(o, i, i_size);
            case WeightsLayout::io:
                return get_index_io(o, i, o_size);
            case WeightsLayout::oiyx:
                return get_index_goizyx(0, o, i, 0, y, x, o_size, i_size, 0, y_size, x_size);
            case WeightsLayout::oizyx:
                return get_index_goizyx(0, o, i, z, y, x, o_size, i_size, z_size, y_size, x_size);
            case WeightsLayout::goizyx:
                return get_index_goizyx(g, o, i, z, y, x, o_size, i_size, z_size, y_size, x_size);
            default:
                return 0;
        }
        // if (tensor.GetLayout() == WeightsLayout::oi) {
        //     size_t ifm_pitch = 1;
        //     size_t ofm_pitch = tensor.IFM().v * ifm_pitch;
        //     return o * ofm_pitch + i * ifm_pitch;
        // } else if (tensor.GetLayout() == WeightsLayout::io) {
        //     size_t ofm_pitch = 1;
        //     size_t ifm_pitch = tensor.OFM().v * ofm_pitch;
        //     return o * ofm_pitch + i * ifm_pitch;
        // } else if (tensor.GetLayout() == WeightsLayout::oiyx) {
        //     size_t x_pitch = 1;
        //     size_t y_pitch = tensor.X().v * x_pitch;
        //     size_t ifm_pitch = tensor.Y().v * y_pitch;
        //     size_t ofm_pitch = tensor.IFM().v * ifm_pitch;
        //     return o * ofm_pitch + i * ifm_pitch + y * y_pitch + x * x_pitch; 
        // } else if (tensor.GetLayout() == WeightsLayout::os_is_yx_isv16_osv16) {
        //     const size_t block_size = 16;
        //     const size_t idx = o%block_size + (o / block_size)*tensor.IFM().v*tensor.X().v*tensor.Y().v*block_size +
        //                        block_size *(i+ x*tensor.IFM().v + y*tensor.IFM().v*tensor.X().v);
        //     return idx;
        // } else {
        //     return 0;
        // }
    }
}  // namespace kernel_selector
