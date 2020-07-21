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
#include <sstream>
#include <bits/stdc++.h> 
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
    // std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
    auto input_ptr = static_cast<float*>(input);
    auto output_ptr = static_cast<float*>(output);

    // for (size_t c = 0; c < static_cast<size_t>(Tensor::WeightsChannelName::COUNT); c++) {
    //     if (Tensor::WeightsChannelName::X == static_cast<Tensor::WeightsChannelName>(c))
    //         printf("X = %lu\n", c);
    //     else if (Tensor::WeightsChannelName::Y == static_cast<Tensor::WeightsChannelName>(c))
    //         printf("Y = %lu\n", c);
    //     else if (Tensor::WeightsChannelName::Z == static_cast<Tensor::WeightsChannelName>(c))
    //         printf("Z = %lu\n", c);
    //     else if (Tensor::WeightsChannelName::IFM == static_cast<Tensor::WeightsChannelName>(c))
    //         printf("IFM = %lu\n", c);
    //     else if (Tensor::WeightsChannelName::OFM == static_cast<Tensor::WeightsChannelName>(c))
    //         printf("OFM = %lu\n", c);
    //     else if (Tensor::WeightsChannelName::LX == static_cast<Tensor::WeightsChannelName>(c))
    //         printf("LX = %lu\n", c);
    //     else if (Tensor::WeightsChannelName::LY == static_cast<Tensor::WeightsChannelName>(c))
    //         printf("LY = %lu\n", c);
    // }

    const size_t input_g_size = this->input.G().v;
    const size_t input_ofm_size = this->input.OFM().v;
    const size_t input_ifm_size = this->input.IFM().v;
    // if (this->input.OFM().blockedPitch.size())
    //     printf("OFM i_blockedPitch = %lu\n", this->input.OFM().blockedPitch[0]);
    // if (this->input.IFM().blockedPitch.size())
    //     printf("IFM i_blockedPitch = %lu\n", this->input.IFM().blockedPitch[0]);
    const size_t input_z_size = this->input.Z().v;
    const size_t input_y_size = this->input.Y().v;
    const size_t input_x_size = this->input.X().v;
    const size_t input_ofm_pitch = this->input.OFM().Pitch();
    const size_t input_ifm_pitch = this->input.IFM().Pitch();
    const size_t input_y_pitch = this->input.Y().Pitch();
    const size_t input_x_pitch = this->input.X().Pitch();

    const size_t output_ofm_pitch = this->output.OFM().Pitch();
    const size_t output_ifm_pitch = this->output.IFM().Pitch();
    size_t output_ifm16_pitch = 0;
    size_t output_ofm16_pitch = 0;
    // if (this->output.OFM().blockedPitch.size())
    //     output_ofm16_pitch = this->output.OFM().blockedPitch[0];
    // if (this->output.IFM().blockedPitch.size())
    //     output_ifm16_pitch = this->output.IFM().blockedPitch[0];
    const size_t output_y_pitch = this->output.Y().Pitch();
    const size_t output_x_pitch = this->output.X().Pitch();

    auto input_order = WeightsTensor::GetSortedDimsIndices(this->output.GetLayout());

    std::stringstream s;
    auto blockedDims = this->output.GetDims();
    // auto blockedDims = this->output.GetBlockedDims();
    for (size_t i = 0; i < blockedDims.size(); i++) {
        s << blockedDims[i].v << "[";
        for (auto& p : blockedDims[i].pitches)
            s << p.first << "(" << p.second << ") ";
        s << "] ";
        // s << blockedDims[i].v << "[" << blockedDims[i].Pitch() << "] ";
    }
    
    s << "__";
    for (size_t i = 0; i < input_order.size(); i++)
        s << input_order[i] << " ";

    printf("Logical size: %lu (%d -> %d). Input sizes: %lu %lu %lu %lu %lu [%lu %lu %lu %lu] out [%lu %lu %lu %lu [%lu %lu]] blockedDims %s\n", this->input.LogicalSize(), (int)this->input.GetLayout(), (int)this->output.GetLayout(),
           input_g_size, input_ofm_size, input_ifm_size, input_y_size, input_x_size, input_ofm_pitch, input_ifm_pitch, input_y_pitch, input_x_pitch,
           output_ofm_pitch, output_ifm_pitch, output_y_pitch, output_x_pitch, output_ifm16_pitch, output_ofm16_pitch, s.str().c_str());

    const size_t input_layout = this->input.GetLayout();
    const size_t output_layout = this->output.GetLayout();

    if (output_layout == WeightsLayout::os_is_yx_isv16_osv16 && input_layout == WeightsLayout::oiyx) {
        const size_t block_size = 16;

        const size_t ofm_blocks_num = CeilDiv(input_ofm_size, block_size);
        const size_t ifm_blocks_num = CeilDiv(input_ifm_size, block_size);

        size_t ofm_leftover = input_ofm_size % block_size;
        size_t ifm_leftover = input_ifm_size % block_size;
        printf("Leftovers os_is_yx_isv16_osv16 %lu %lu %lu %lu (%lu %lu)\n", input_ofm_size, input_ifm_size, input_y_size, input_x_size, ifm_leftover, ofm_leftover);
        if (ofm_leftover == 0 && ifm_leftover == 0) {
            InferenceEngine::parallel_for4d(ofm_blocks_num, ifm_blocks_num, input_y_size, input_x_size, [&](size_t ofm_block, size_t ifm_block, size_t y, size_t x) {
                const size_t ifm_idx = ifm_block * block_size;
                const size_t ofm_idx = ofm_block * block_size;
                for (size_t ifm = ifm_idx; ifm < ifm_idx + block_size; ifm++) {
                    size_t output_idx = get_index_os_is_yx_isv16_osv16(ofm_idx, ifm, y, x, output_ofm_pitch, output_ifm_pitch, output_y_pitch, output_x_pitch);
                    size_t input_idx = get_index_oiyx(ofm_idx, ifm, y, x, input_ifm_size, input_y_size, input_x_size);
                    for (size_t ofm = 0; ofm < block_size; ofm++, input_idx += input_ofm_pitch) {
                        output_ptr[0 + output_idx - output_idx] = input_ptr[0];
                        // output_ptr[output_idx + ofm] = input_ptr[input_idx];
                    }
                }
            });
        } else {
            InferenceEngine::parallel_for4d(ofm_blocks_num, ifm_blocks_num, input_y_size, input_x_size, [&](size_t ofm_block, size_t ifm_block, size_t y, size_t x) {
                const size_t ifm_idx = ifm_block * block_size;
                const size_t ofm_idx = ofm_block * block_size;
                const size_t ofm_block_size = (ofm_leftover != 0 && ofm_block == ofm_blocks_num - 1) ? ofm_leftover : block_size;
                const size_t ifm_block_size = (ifm_leftover != 0 && ifm_block == ifm_blocks_num - 1) ? ifm_leftover : block_size;
                for (size_t ifm = ifm_idx; ifm < ifm_idx + ifm_block_size; ifm++) {
                    size_t output_idx = get_index_os_is_yx_isv16_osv16(ofm_idx, ifm, y, x, output_ofm_pitch, output_ifm_pitch, output_y_pitch, output_x_pitch);
                    size_t input_idx = get_index_oiyx(ofm_idx, ifm, y, x, input_ifm_size, input_y_size, input_x_size);
                    for (size_t ofm = 0; ofm < ofm_block_size; ofm++, input_idx += input_ofm_pitch) {
                        output_ptr[0 + output_idx - output_idx] = input_ptr[0];
                        // output_ptr[output_idx + ofm] = input_ptr[input_idx];
                    }
                }
            });
        }
    } else if (output_layout == WeightsLayout::os_iyx_osv16 && input_layout == WeightsLayout::oiyx) {
        const size_t block_size = 16;

        const size_t ofm_blocks_num = CeilDiv(input_ofm_size, block_size);
        const size_t ifm_blocks_num = CeilDiv(input_ifm_size, block_size);

        size_t ofm_leftover = input_ofm_size % block_size;
        size_t ifm_leftover = input_ifm_size % block_size;
        printf("Leftovers os_iyx_osv16 %lu %lu %lu %lu (%lu %lu)\n", input_ofm_size, input_ifm_size, input_y_size, input_x_size, ifm_leftover, ofm_leftover);
        if (ofm_leftover == 0 && ifm_leftover == 0) {
            InferenceEngine::parallel_for4d(ofm_blocks_num, ifm_blocks_num, input_y_size, input_x_size, [&](size_t ofm_block, size_t ifm_block, size_t y, size_t x) {
                const size_t ifm_idx = ifm_block * block_size;
                const size_t ofm_idx = ofm_block * block_size;
                for (size_t ifm = ifm_idx; ifm < ifm_idx + block_size; ifm++) {
                    size_t output_idx = get_index_os_iyx_osv16(ofm_idx, ifm, y, x, output_ofm_pitch, output_ifm_pitch, output_y_pitch, output_x_pitch);
                    size_t input_idx = get_index_oiyx(ofm_idx, ifm, y, x, input_ifm_size, input_y_size, input_x_size);
                    for (size_t ofm = 0; ofm < block_size; ofm++, input_idx += input_ofm_pitch) {
                        output_ptr[output_idx + ofm] = input_ptr[input_idx];
                    }
                }
            });
        } else {
            InferenceEngine::parallel_for4d(ofm_blocks_num, ifm_blocks_num, input_y_size, input_x_size, [&](size_t ofm_block, size_t ifm_block, size_t y, size_t x) {
                const size_t ifm_idx = ifm_block * block_size;
                const size_t ofm_idx = ofm_block * block_size;
                const size_t ofm_block_size = (ofm_leftover != 0 && ofm_block == ofm_blocks_num - 1) ? ofm_leftover : block_size;
                const size_t ifm_block_size = (ifm_leftover != 0 && ifm_block == ifm_blocks_num - 1) ? ifm_leftover : block_size;
                for (size_t ifm = ifm_idx; ifm < ifm_idx + ifm_block_size; ifm++) {
                    size_t output_idx = get_index_os_iyx_osv16(ofm_idx, ifm, y, x, output_ofm_pitch, output_ifm_pitch, output_y_pitch, output_x_pitch);
                    size_t input_idx = get_index_oiyx(ofm_idx, ifm, y, x, input_ifm_size, input_y_size, input_x_size);
                    for (size_t ofm = 0; ofm < ofm_block_size; ofm++, input_idx += input_ofm_pitch) {
                        output_ptr[output_idx + ofm] = input_ptr[input_idx];
                    }
                }
            });
        }
    } else if (output_layout == WeightsLayout::io && input_layout == WeightsLayout::oiyx) {
        const size_t block_size = 16;

        const size_t ofm_blocks_num = CeilDiv(input_ofm_size, block_size);
        const size_t ifm_blocks_num = CeilDiv(input_ifm_size, block_size);

        size_t ofm_leftover = input_ofm_size % block_size;
        size_t ifm_leftover = input_ifm_size % block_size;

        printf("Leftovers io %lu %lu %lu %lu (%lu %lu)\n", input_ofm_size, input_ifm_size, input_y_size, input_x_size, ifm_leftover, ofm_leftover);
        if (ofm_leftover == 0 && ifm_leftover == 0) {
            InferenceEngine::parallel_for2d(ofm_blocks_num, ifm_blocks_num, [&](size_t ofm_block, size_t ifm_block) {
                const size_t ifm_idx = ifm_block * block_size;
                const size_t ofm_idx = ofm_block * block_size;
                for (size_t ifm = ifm_idx; ifm < ifm_idx + block_size; ifm++) {
                    size_t output_idx = get_index_io(ofm_idx, ifm, input_ofm_size);
                    size_t input_idx = get_index_oiyx(ofm_idx, ifm, 0, 0, input_ifm_size, input_y_size, input_x_size);
                    for (size_t ofm = 0; ofm < block_size; ofm++, input_idx += input_ofm_pitch) {
                        output_ptr[output_idx + ofm] = input_ptr[input_idx];
                    }
                }
            });
        } else {
            InferenceEngine::parallel_for2d(ofm_blocks_num, ifm_blocks_num, [&](size_t ofm_block, size_t ifm_block) {
                const size_t ifm_idx = ifm_block * block_size;
                const size_t ofm_idx = ofm_block * block_size;
                const size_t ofm_block_size = (ofm_leftover != 0 && ofm_block == ofm_blocks_num - 1) ? ofm_leftover : block_size;
                const size_t ifm_block_size = (ifm_leftover != 0 && ifm_block == ifm_blocks_num - 1) ? ifm_leftover : block_size;
                for (size_t ifm = ifm_idx; ifm < ifm_idx + ifm_block_size; ifm++) {
                    size_t output_idx = get_index_io(ofm_idx, ifm, input_ofm_size);
                    size_t input_idx = get_index_oiyx(ofm_idx, ifm, 0, 0, input_ifm_size, input_y_size, input_x_size);
                    for (size_t ofm = 0; ofm < ofm_block_size; ofm++, input_idx += input_ofm_pitch) {
                        output_ptr[output_idx + ofm] = input_ptr[input_idx];
                    }
                }
            });
        }
    } else {
        printf("Default value %d -> %d\n", (int)this->input.GetLayout(), (int)this->output.GetLayout());
        
        auto inputDims = this->input.GetDims();
        std::vector<size_t> inputDimsVec(inputDims.size());
        std::transform(inputDims.begin(), inputDims.end(), inputDimsVec.begin(), [](const Tensor::Dim& d) { return d. v; });

        auto outputDims = this->output.GetDims();
        std::vector<size_t> outputDimsVec(outputDims.size());
        std::transform(outputDims.begin(), outputDims.end(), outputDimsVec.begin(), [](const Tensor::Dim& d) { return d. v; });

        const auto blockSizes = this->output.GetBlockSizes();
        const auto dim0 = inputDimsVec[0];
        const auto dims_num = inputDims.size();
        const auto dim0_block = 32;
        const auto dim0_blocks_num = CeilDiv(dim0, dim0_block);

        if (dims_num == 4) {
            const size_t dim1 = inputDimsVec[1];
            const size_t dim2 = inputDimsVec[2];
            const size_t dim3 = inputDimsVec[3];

            InferenceEngine::parallel_for4d(dim3, dim2, dim1, dim0_blocks_num, [&](size_t d3, size_t d2, size_t d1, size_t d0_blocks) {
                // size_t d0 = d0_blocks * dim0_block;
                // // d0 d1 d2 d3
                // size_t input_idx = 0;
                // input_idx += ;
            // const size_t ifm_idx = ifm_block * block_size;
            // const size_t ofm_idx = ofm_block * block_size;
            // const size_t ofm_block_size = (ofm_leftover != 0 && ofm_block == ofm_blocks_num - 1) ? ofm_leftover : block_size;
            // const size_t ifm_block_size = (ifm_leftover != 0 && ifm_block == ifm_blocks_num - 1) ? ifm_leftover : block_size;
            // for (size_t ifm = ifm_idx; ifm < ifm_idx + ifm_block_size; ifm++) {
            //     size_t output_idx = get_index_io(ofm_idx, ifm, input_ofm_size);
            //     size_t input_idx = get_index_oiyx(ofm_idx, ifm, 0, 0, input_ifm_size, input_y_size, input_x_size);
            //     for (size_t ofm = 0; ofm < ofm_block_size; ofm++, input_idx += input_ofm_pitch) {
            //         output_ptr[output_idx + ofm] = input_ptr[input_idx];
            //     }
            // }
        });
        }


        InferenceEngine::parallel_for5d(input_g_size, input_ofm_size, input_ifm_size, input_z_size, input_y_size, dim0_blocks_num, [&](size_t g, size_t ofm, size_t ifm, size_t z, size_t y, size_t x_blocks) {
            // const size_t ifm_idx = ifm_block * block_size;
            // const size_t ofm_idx = ofm_block * block_size;
            // const size_t ofm_block_size = (ofm_leftover != 0 && ofm_block == ofm_blocks_num - 1) ? ofm_leftover : block_size;
            // const size_t ifm_block_size = (ifm_leftover != 0 && ifm_block == ifm_blocks_num - 1) ? ifm_leftover : block_size;
            // for (size_t ifm = ifm_idx; ifm < ifm_idx + ifm_block_size; ifm++) {
            //     size_t output_idx = get_index_io(ofm_idx, ifm, input_ofm_size);
            //     size_t input_idx = get_index_oiyx(ofm_idx, ifm, 0, 0, input_ifm_size, input_y_size, input_x_size);
            //     for (size_t ofm = 0; ofm < ofm_block_size; ofm++, input_idx += input_ofm_pitch) {
            //         output_ptr[output_idx + ofm] = input_ptr[input_idx];
            //     }
            // }
        });
        // for (size_t g = 0; g < input_g_size; g++) {
        //     for (size_t y = 0; y < input_y_size; y++) {
        //         for (size_t x = 0; x < input_x_size; x++) {
        //             for (size_t ifm = 0; ifm < input_ifm_size; ifm++) {
        //                 InferenceEngine::parallel_for(input_ofm_size, [&](size_t ofm) {
        //                     size_t input_idx = get_index(ofm, ifm, y, x, this->input);
        //                     size_t output_idx = get_index(ofm, ifm, y, x, this->output);
        //                     output_ptr[output_idx] = input_ptr[input_idx];
        //                 });
        //             }
        //         }
        //     }
        // }
    }
    // std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> execute_time = t0 - t;
    // std::cout << "Kernel execution took " << execute_time.count() << " ms\n";
}

}  // namespace kernel_selector
