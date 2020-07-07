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
#include "ie_parallel.hpp"
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
    std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
    size_t g_in = this->input.G().v;
    size_t ofm_in = this->input.OFM().v;
    size_t ifm_in = this->input.IFM().v;
    size_t y_in = this->input.Y().v;
    size_t x_in = this->input.X().v;
    auto input_ptr = static_cast<float*>(input);
    auto output_ptr = static_cast<float*>(output);

    // InferenceEngine::parallel_for2d(g_in, 1, [&](size_t i, size_t j) {
    //         printf("Hi from parallel\n");
    //     });

        // ie::parallel_for4d(H, W, _num_priors, 4, [&](int h, int w, int i, int j) {
        //     dst_data[j + 4 * (i + _num_priors * (w + W * h))] = ie::PrecisionUtils::f32tof16(_variance[j]);
        // });        

    // if (this->input.GetLayout() == WeightsLayout::io || this->output.GetLayout() == WeightsLayout::io)
    //     printf("Input or output is ::io: %lu %lu %lu %lu %lu\n", g_in, ofm_in, ifm_in, y_in, x_in);
    if (this->output.GetLayout() == WeightsLayout::os_is_yx_isv16_osv16 && this->input.GetLayout() == WeightsLayout::oiyx) {
        const size_t block_size = 16;
        size_t ofm_blocks_num = CeilDiv(ofm_in, block_size);
        size_t ifm_blocks_num = CeilDiv(ifm_in, block_size);
        // printf("ifm %lu (%lu) ofm %lu (%lu)\n", ifm_in, ifm_blocks_num, ofm_in, ofm_blocks_num);

        // printf("Loop params: %lu %lu %lu %lu(%lu) %lu(%lu)\n", g_in, y_in, x_in, ifm_blocks_num -1, ifm_in, ofm_blocks_num - 1, ofm_in);
        for (size_t g = 0; g < g_in; g++) {
            for (size_t y = 0; y < y_in; y++) {
                for (size_t x = 0; x < x_in; x++) {
                    for (size_t ifm_block = 0; ifm_block < ifm_blocks_num - 1; ifm_block++) {
                        for (size_t ofm_block = 0; ofm_block < ofm_blocks_num - 1; ofm_block++) {
                            InferenceEngine::parallel_for(block_size, [&](size_t ifm) {
                                size_t ifm_out = ifm_block * block_size + ifm;
                                size_t ofm_idx = ofm_block * block_size;
                                size_t output_idx = get_index(ofm_idx, ifm_out, y, x, this->output);
                                size_t input_idx = get_index(ofm_idx, ifm_out, y, x, this->input);
                                for (size_t ofm_out = 0; ofm_out < block_size; ofm_out++) {
                                    output_ptr[output_idx + ofm_out] = input_ptr[input_idx];
                                    input_idx += this->input.OFM().pitch;
                                }
                            });
                        }
                    }
                }
            }
        }
        size_t ofm_leftover = (ofm_in % block_size == 0) ? block_size : ofm_in % block_size;
        size_t ifm_leftover = (ifm_in % block_size == 0) ? block_size : ifm_in % block_size;
        for (size_t g = 0; g < g_in; g++) {
            for (size_t y = 0; y < y_in; y++) {
                for (size_t x = 0; x < x_in; x++) {
                    InferenceEngine::parallel_for(ifm_leftover, [&](size_t ifm) {
                        size_t ifm_out = (ifm_blocks_num - 1) * block_size + ifm;
                        size_t ofm_idx = (ofm_blocks_num - 1) * block_size;
                        size_t output_idx = get_index(ofm_idx, ifm_out, y, x, this->output);
                        for (size_t ofm_out = 0; ofm_out < ofm_leftover; ofm_out++) {
                            size_t input_idx = get_index(ofm_idx + ofm_out, ifm_out, y, x, this->input);
                            output_ptr[output_idx + ofm_out] = input_ptr[input_idx];
                        }
                    });
                }
            }
        }
    } else if (this->output.GetLayout() == WeightsLayout::io && this->input.GetLayout() == WeightsLayout::oiyx) {
        printf("IO OI HERE\n");
        for (size_t g = 0; g < g_in; g++) {
            for (size_t y = 0; y < y_in; y++) {
                for (size_t x = 0; x < x_in; x++) {
                    InferenceEngine::parallel_for(ifm_in, [&](size_t ifm) {
                        size_t input_idx = get_index(0, ifm, y, x, this->input);
                        size_t output_idx = get_index(0, ifm, y, x, this->output);
                        for (size_t ofm = 0; ofm < ofm_in; ofm++) {
                            output_ptr[output_idx] = input_ptr[input_idx];
                            output_idx++;
                            input_idx += this->input.OFM().pitch;
                        }
                    });
                }
            }
        }
    } else {
        printf("IO OI %d %d\n", (int)this->output.GetLayout(), (int)this->input.GetLayout());
        for (size_t g = 0; g < g_in; g++) {
            for (size_t y = 0; y < y_in; y++) {
                for (size_t x = 0; x < x_in; x++) {
                    for (size_t ifm = 0; ifm < ifm_in; ifm++) {
                        InferenceEngine::parallel_for(ofm_in, [&](size_t ofm) {
                            size_t input_idx = get_index(ofm, ifm, y, x, this->input);
                            size_t output_idx = get_index(ofm, ifm, y, x, this->output);
                            output_ptr[output_idx] = input_ptr[input_idx];
                        });
                    }
                }
            }
        }
    }
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execute_time = t0 - t;
    std::cout << "Kernel execution took " << execute_time.count() << " ms\n";
}

}  // namespace kernel_selector
