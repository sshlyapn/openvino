﻿// Copyright (c) 2016-2020 Intel Corporation
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

#include "convolution_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <algorithm>

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

size_t ConvolutionKernel_b_fs_yx_fsv16::GetInputWidth(const convolution_params& params, size_t blockWidth) const {
    return std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1)*params.dilation.x + 1,
                    params.inputs[0].X().v + params.inputs[0].X().pad.Total());
}

ConvolutionKernel_b_fs_yx_fsv16::ConvolutionMode ConvolutionKernel_b_fs_yx_fsv16::GetConvolutionMode(const convolution_params& params) const {
    const auto& input = params.inputs[0];
    const auto& output = params.output;

    auto outFeaturesPerGroup = output.Feature().v / params.groups;
    auto inFeaturesPerGroup = input.Feature().v / params.groups;
    auto multipleGroupsInputPreload = (feature_block_size % outFeaturesPerGroup == 0) &&
                                      (feature_block_size % inFeaturesPerGroup == 0) &&
                                      (feature_block_size / outFeaturesPerGroup > 1) &&
                                      (feature_block_size / inFeaturesPerGroup > 1) &&
                                      (outFeaturesPerGroup != 1) &&
                                      (inFeaturesPerGroup != 1);

    auto grouped = inFeaturesPerGroup % sub_group_size == 0 &&
                   (outFeaturesPerGroup % sub_group_size == 0 || sub_group_size % outFeaturesPerGroup == 0);

    if (params.groups == 1)
        return ConvolutionMode::SIMPLE;
    else if (grouped)
        return ConvolutionMode::SIMPLE_GROUPED;
    else if (multipleGroupsInputPreload)
        return ConvolutionMode::GROUPED_WITH_PRELOAD;
    else
        return ConvolutionMode::UNSUPPORTED;
}

size_t ConvolutionKernel_b_fs_yx_fsv16::GetMinRegisterUsage(const convolution_params& params, size_t blockWidth, size_t /*blockHeight*/) const {
    // size_t weightsRegisters = (GetConvolutionMode(params) == ConvolutionMode::GROUPED_WITH_PRELOAD) ? 8 : 16;
    // size_t inputRegisters = GetInputWidth(params, blockWidth);
    // size_t outputRegisters = blockWidth;

    // return weightsRegisters + inputRegisters + outputRegisters;
    const size_t weightsElements = (GetConvolutionMode(params) == ConvolutionMode::GROUPED_WITH_PRELOAD) ? 128 : 256;
    const size_t inputElements = GetInputWidth(params, blockWidth) * 16; // 80el, 126el, 192el : 5, 8, 17
    const size_t outputElements = blockWidth * 16; // 32, 64, 128 : 2 4 8
    const size_t elementSize = Datatype::F32  == params.inputs[0].GetDType() ? 4 : 2;
    const size_t totalBytes = (weightsElements + inputElements + outputElements) * elementSize;
    // printf("(%lu + %lu + %lu) * %lu = %lu\n", weightsElements, inputElements, outputElements, elementSize, totalBytes);
    const size_t registerSize = 32;

    return CeilDiv(totalBytes, registerSize);
}

ConvolutionKernel_b_fs_yx_fsv16::ConvolutionKernel_b_fs_yx_fsv16() : ConvolutionKernelBase("convolution_gpu_bfyx_f16") {
    std::vector<size_t> outputBlockWidths = {2, 4, 8};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : outputBlockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{w, exeMode});
        }
    }
}

ConvolutionKernel_b_fs_yx_fsv16::AutoTuneOption ConvolutionKernel_b_fs_yx_fsv16::GetAutoTuneOptions(const Params& params,
                                                                                          int /*autoTuneIndex*/) const {
    std::vector<size_t> block_sizes {2, 4, 8};
    auto block_size = GetOptimalBlockSize(params, block_sizes);
    return { block_size , AGE_BASED };
}

ParamsKey ConvolutionKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);

    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    // TODO Add bias per output support to kernel
    // k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableSplitSupport();
    k.EnableBatching();
    k.EnableDepthwiseSeparableOpt();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableGroupedConvolution();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv16::SetDefault(const convolution_params& params,
                                                                           int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;
    const auto& in = params.inputs[0];

    auto xi = in.X().v;
    auto yi = in.Y().v;
    auto fi = in.Feature().v;
    auto bi = in.Batch().v;

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    kd.cldnnStyle.blockWidth = autoTune.blockWidth;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;
    printf("%s (%lux%lux%lux%lu -> %lux%lux%lux%lu)\n%lu/%lu: \n", params.layerID.c_str(), bi, fi, yi, xi, b, f, y, x, autoTune.blockWidth, GetMinRegisterUsage(params, autoTune.blockWidth));

    kd.gws0 = CeilDiv(x, autoTune.blockWidth) * y;
    kd.gws1 = Align(f, sub_group_size);
    kd.gws2 = b;

    kd.lws0 = 1;
    kd.lws1 = sub_group_size;
    kd.lws2 = 1;

    if (b == 1)
        kd.efficiency = FORCE_PRIORITY_2;
    else
        kd.efficiency = FORCE_PRIORITY_7;

    return kd;
}

bool ConvolutionKernel_b_fs_yx_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    if (params.groups > 1) {
        auto outFeaturesPerGroup = output.Feature().v / params.groups;
        auto inFeaturesPerGroup = input.Feature().v / params.groups;
        auto multipleGroupsInputPreload = (feature_block_size % outFeaturesPerGroup == 0) &&
                                          (feature_block_size % inFeaturesPerGroup == 0) &&
                                          (feature_block_size / outFeaturesPerGroup > 1) &&
                                          (feature_block_size / inFeaturesPerGroup > 1) &&
                                          (outFeaturesPerGroup != 1) &&
                                          (inFeaturesPerGroup != 1);
        auto grouped = inFeaturesPerGroup % sub_group_size == 0 &&
                       (outFeaturesPerGroup % sub_group_size == 0 || sub_group_size % outFeaturesPerGroup == 0);

        if (!multipleGroupsInputPreload && !grouped)
            return false;
    }

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0)
        return false;

    if (!params.bias.empty() && params.bias[0].GetDType() != input.GetDType())
        return false;

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16::GetJitConstants(const convolution_params& params,
                                                              const DispatchData& runInfo) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params, runInfo);

    auto blockWidth = runInfo.cldnnStyle.blockWidth;
    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           {"b", "(f_block*16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           blockWidth,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              {"b", "(f_block*16)", "y", "(x+i)"},
                                              "dst[i]",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec, conf_scalar}));
    }

    size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1)*params.dilation.x + 1,
                                      input.X().v + input.X().pad.Total());

    auto outFeaturesPerGroup = output.Feature().v / params.groups;
    auto inFeaturesPerGroup = input.Feature().v / params.groups;
    auto multipleGroupsInputPreload = (feature_block_size % outFeaturesPerGroup == 0) &&
                                      (feature_block_size % inFeaturesPerGroup == 0) &&
                                      (feature_block_size / outFeaturesPerGroup > 1) &&
                                      (feature_block_size / inFeaturesPerGroup > 1);

    if (multipleGroupsInputPreload)
        jit.AddConstant(MakeJitConstant("MULTIPLE_GROUPS_INPUT_PRELOAD", 1));

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, blockWidth)));
    jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(inFeaturesPerGroup, feature_block_size)));
    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }
    if (inFeaturesPerGroup % feature_block_size != 0 && !multipleGroupsInputPreload) {
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetTunedKernelsDataByIndex(const Params& params,
                                                                   const optional_params& options,
                                                                   const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, options, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetKernelsDataForAutoTune(const Params& params,
                                                                  const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

size_t ConvolutionKernel_b_fs_yx_fsv16::ComputeWorkGroupsNumber(const convolution_params& params, size_t block_size) const {
    const auto& out = params.output;
    auto wg_num = 1;

    wg_num *= out.Batch().v;
    wg_num *= CeilDiv(out.Feature().v, 16);
    wg_num *= out.Y().v * CeilDiv(out.X().v, block_size);

    return wg_num;
}

size_t ConvolutionKernel_b_fs_yx_fsv16::GetOptimalBlockSize(const Params& params, const std::vector<size_t>& block_sizes) const {
    auto& p = static_cast<const convolution_params&>(params);
    const auto width = p.inputs[0].X().v;
    const auto max_effective_bs_usage = 3;
    const auto threads_per_exec_unit = 7;
    const auto registers_num = 128;
    const auto register_bytes = 32;
    const auto max_registers_bytes = registers_num * register_bytes;
    const auto threads = p.engineInfo.computeUnitsCount * threads_per_exec_unit;
    auto sorted_block_sizes(block_sizes);

    std::sort(sorted_block_sizes.begin(), sorted_block_sizes.end());

    const auto max_block_size = sorted_block_sizes[sorted_block_sizes.size() - 1];

    auto selected_block_size = 0;
    if (ComputeWorkGroupsNumber(p, max_block_size) <= 0.5 * threads) {
        return sorted_block_sizes[0];
    } else {
        if (p.stride.x > 2) {
            for (auto& b : block_sizes)
                if (b > width) {
                    selected_block_size = b;
                    break;
                }
        } else {
            auto rated_block_sizes = GetRatedBlockSizes(width, sorted_block_sizes);
            for (size_t bs = 0; bs < rated_block_sizes.size(); bs++) {
                if (bs == 0 && CeilDiv(width, rated_block_sizes[bs].second) <= max_effective_bs_usage) {
                    selected_block_size = rated_block_sizes[bs].second;
                    break;
                } else if (CeilDiv(width, rated_block_sizes[bs].second) <= max_effective_bs_usage + 1) {
                    selected_block_size = rated_block_sizes[bs].second;
                    break;
                }
            }
            selected_block_size = max_block_size;
        }
    }
    auto block_size_idx = std::distance(block_sizes.begin(), std::find(block_sizes.begin(), block_sizes.end(), selected_block_size));
    while (block_size_idx > 0 && GetMinRegisterUsage(p, block_sizes[block_size_idx]) > 0.75 * max_registers_bytes)
        block_size_idx--;
    return block_sizes[block_size_idx];
}

}  // namespace kernel_selector
