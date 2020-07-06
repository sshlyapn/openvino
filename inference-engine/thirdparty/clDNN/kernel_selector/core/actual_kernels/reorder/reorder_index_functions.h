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

    inline size_t get_index(size_t ofm, size_t ifm, size_t y, size_t x, WeightsTensor tensor) {
        if (tensor.GetLayout() == WeightsLayout::oi) {
            size_t ifm_pitch = 1;
            size_t ofm_pitch = tensor.IFM().v * ifm_pitch;
            return ofm * ofm_pitch + ifm * ifm_pitch;
        } else if (tensor.GetLayout() == WeightsLayout::io) {
            size_t ofm_pitch = 1;
            size_t ifm_pitch = tensor.OFM().v * ofm_pitch;
            return ofm * ofm_pitch + ifm * ifm_pitch;
        } else if (tensor.GetLayout() == WeightsLayout::oiyx) {
            size_t x_pitch = 1;
            size_t y_pitch = tensor.X().v * x_pitch;
            size_t ifm_pitch = tensor.Y().v * y_pitch;
            size_t ofm_pitch = tensor.IFM().v * ifm_pitch;
            return ofm * ofm_pitch + ifm * ifm_pitch + y * y_pitch + x * x_pitch; 
        } else if (tensor.GetLayout() == WeightsLayout::os_is_yx_isv16_osv16) {
            const size_t block_size = 16;
            const size_t idx = ofm%block_size + (ofm / block_size)*tensor.IFM().v*tensor.X().v*tensor.Y().v*block_size +
                               block_size *(ifm+ x*tensor.IFM().v + y*tensor.IFM().v*tensor.X().v);
            return idx;
        } else {
            return 0;
        }
    }
}  // namespace kernel_selector
