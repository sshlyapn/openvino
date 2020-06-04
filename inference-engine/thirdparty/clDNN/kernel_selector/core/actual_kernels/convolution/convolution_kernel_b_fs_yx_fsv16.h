﻿// Copyright (c) 2016-2019 Intel Corporation
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

#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_b_fs_yx_fsv16 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    ConvolutionKernel_b_fs_yx_fsv16();
    virtual ~ConvolutionKernel_b_fs_yx_fsv16() {}

    KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                           const optional_params& options,
                                           int autoTuneIndex = -1) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &p) const override {
        return (p.groups > 1) ? WeightsLayout::g_os_is_yx_isv16_osv16 : WeightsLayout::os_is_yx_isv16_osv16;
    }
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

    bool NeedPaddedInput() const override { return false; }
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& kd) const override;
    size_t GetMinRegisterUsage(const convolution_params& params, size_t blockWidth = 1, size_t blockHeight = 1) const;
    size_t ComputeWorkGroupsNumber(const convolution_params& params, size_t block_size) const override;
    size_t GetOptimalBlockSize(const Params& params, const std::vector<size_t>& block_sizes) const override;

private:
    enum ConvolutionMode { SIMPLE, SIMPLE_GROUPED, GROUPED_WITH_PRELOAD, UNSUPPORTED };
    struct AutoTuneOption {
        size_t blockWidth;
        std::string exeMode;
    };

    std::vector<AutoTuneOption> autoTuneOptions;
    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;
    size_t GetInputWidth(const convolution_params& params, size_t blockWidth) const;
    ConvolutionMode GetConvolutionMode(const convolution_params& params) const;
};
}  // namespace kernel_selector
