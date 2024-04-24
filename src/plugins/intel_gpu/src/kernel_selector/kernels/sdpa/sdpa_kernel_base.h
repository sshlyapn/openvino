// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// sdpa_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct sdpa_params : public base_params {
    sdpa_params() : base_params(KernelType::SDPA) {}
    DimTensor<uint32_t> block_shape;
    DimTensor<uint32_t> crops_begin;
    DimTensor<uint32_t> crops_end;

    base_params::ArgType block_type = base_params::ArgType::Input;
    base_params::ArgType begin_type = base_params::ArgType::Input;
    base_params::ArgType end_type = base_params::ArgType::Input;

    size_t block_dims = 0;
    size_t begin_dims = 0;
    size_t end_dims = 0;

    size_t block_input_index = 0;
    size_t begin_input_index = 0;
    size_t end_input_index = 0;
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
    virtual JitConstants GetJitConstants(const sdpa_params& params) const;
    virtual CommonDispatchData SetDefault(const sdpa_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
