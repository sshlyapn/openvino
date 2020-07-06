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


#include "kernel_selector_common.h"
#include "reorder_weights_kernel.h"
#include "kernel_selector_utils.h"
#include "reorder_index_functions.h"
#include "inference-engine/include/ie_parallel.hpp"
#include <chrono>
#include <iostream>

namespace kernel_selector {
ParamsKey ReorderWeightsKernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::INT8);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableAllInputWeightsLayout();
    k.EnableAllOutputWeightsLayout();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableRotateReorder();
    return k;
}

KernelsData ReorderWeightsKernel::GetKernelsData(const Params& params, const optional_params& options) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
}

WeightsType ReorderWeightsKernel::GetExpectedInputType() {
    return WeightsType::F32;
}

WeightsLayout ReorderWeightsKernel::GetExpectedInputLayout() const {
    return WeightsLayout::os_is_yx_isv16_osv16;
}

void ReorderWeightsKernel::Execute(void* input, size_t input_size, void* output, size_t output_size) const {
    // printf("Hi from specific Reroder input %d and outptu %d: %lu -> %lu\n", (int)this->input.GetLayout(), (int)this->output.GetLayout(), 
    // get_index(10,0,0,0, this->input), get_index(10,0,0,0, this->output));
    // std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
    size_t g_in = this->input.G().v;
    size_t ofm_in = this->input.OFM().v;
    size_t ifm_in = this->input.IFM().v;
    size_t y_in = this->input.Y().v;
    size_t x_in = this->input.X().v;
    auto input_ptr = static_cast<float*>(input);
    auto output_ptr = static_cast<float*>(output);
    // if (this->input.GetLayout() == WeightsLayout::io || this->output.GetLayout() == WeightsLayout::io)
    //     printf("Input or output is ::io: %lu %lu %lu %lu %lu\n", g_in, ofm_in, ifm_in, y_in, x_in);
    for (size_t g = 0; g < g_in; g++) {
        for (size_t ofm = 0; ofm < ofm_in; ofm++) {
            for (size_t ifm = 0; ifm < ifm_in; ifm++) {
                for (size_t y = 0; y < y_in; y++) {
                    for (size_t x = 0; x < x_in; x++) {
                        size_t input_idx = get_index(ofm, ifm, y, x, this->input);
                        size_t output_idx = get_index(ofm, ifm, y, x, this->output);
                        output_ptr[output_idx] = input_ptr[input_idx];
                    }
                }
            }
        }
    }
    // std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> execute_time = t0 - t;
    // std::cout << "Kernel execution took " << execute_time.count() << " ms\n";
}

}  // namespace kernel_selector
