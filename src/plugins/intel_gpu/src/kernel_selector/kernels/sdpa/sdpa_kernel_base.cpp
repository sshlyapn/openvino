// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

bool SDPAKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SDPA) {
        return false;
    }

    const sdpa_params& params = static_cast<const sdpa_params&>(p);

    if (params.inputs[0].Dimentions() != 4)
        return false;

    return true;
}

CommonDispatchData SDPAKernelBase::SetDefault(const sdpa_params& params) const {
    CommonDispatchData dispatchData;

    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y,
                                                                       Tensor::DataChannelName::Z, Tensor::DataChannelName::W }};

    const auto& output = params.outputs[0];
    dispatchData.gws = { output.Batch().v, output.Feature().v, output.W().v * output.Z().v * output.Y().v * output.X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

JitConstants SDPAKernelBase::GetJitConstants(const sdpa_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    return jit;
}

KernelsData SDPAKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<sdpa_params>(params);

    return { kd };
}
}  // namespace kernel_selector
