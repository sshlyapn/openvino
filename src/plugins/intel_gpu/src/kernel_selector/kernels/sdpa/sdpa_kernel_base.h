// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
struct sdpa_configuration {
    int64_t head_size = -1;
    int64_t heads_num = -1;
    int64_t kv_heads_num = -1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// sdpa_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct sdpa_params : public base_params {
    sdpa_params() : base_params(KernelType::SDPA) {}

    sdpa_configuration conf;
};

struct sdpa_fuse_params : fuse_params {
    sdpa_fuse_params() : fuse_params(KernelType::SDPA) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SDPAKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SDPAKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~SDPAKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params&) const override;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
