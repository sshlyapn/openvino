// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_selector.h"
#include "dynamic_quantize_kernel_ref.h"
#include "dynamic_quantize_kernel_opt.h"
#include "dynamic_quantize_kernel_opt_generic.h"

namespace kernel_selector {
dynamic_quantize_kernel_selector::dynamic_quantize_kernel_selector() {
    Attach<DynamicQuantizeKernelRef>();
    int USE_REF_DQ = 0;
    if (const auto env_var = std::getenv("USE_REF_DQ")) {
        std::istringstream ss(env_var);
        ss >> USE_REF_DQ;
    }

    if (!USE_REF_DQ) {
        Attach<DynamicQuantizeKernelOptGeneric>();
    }
    // Attach<DynamicQuantizeKernelOpt>();
}

KernelsData dynamic_quantize_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::DYNAMIC_QUANTIZE);
}
}  // namespace kernel_selector
