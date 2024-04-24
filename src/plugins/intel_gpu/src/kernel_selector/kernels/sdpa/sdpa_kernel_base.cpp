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

JitConstants SDPAKernelBase::GetJitConstants(const sdpa_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    return jit;
}

KernelsData SDPAKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<sdpa_params>(params);

    return { kd };
}
}  // namespace kernel_selector
