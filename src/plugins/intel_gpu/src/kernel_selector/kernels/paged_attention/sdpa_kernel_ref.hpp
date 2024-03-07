// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct sdpa_params : base_params {
    sdpa_params() : base_params(KernelType::PA_SDPA) {}

    struct sdpa_configuration {
        // Configures layout of key/value cache as follow:
        // key_cache: [preallocated_blocks_num, kv_heads_num, head_size / x, block_size, x]
        // value_cache: [preallocated_blocks_num, kv_heads_num, head_size, block_size]
        size_t head_size;
        size_t heads_num;
        size_t kv_heads_num;
        size_t block_size;
        size_t x_size;
    };

    sdpa_configuration configuration;
};

class SDPAKernelRef : public KernelBaseOpenCL {
public:
    struct DispatchData : CommonDispatchData {
        size_t max_sequence_length;
    };

    SDPAKernelRef() : KernelBaseOpenCL{"pa_sdpa_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const sdpa_params& kernel_params) const;
    static CommonDispatchData SetDefault(const sdpa_params& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
