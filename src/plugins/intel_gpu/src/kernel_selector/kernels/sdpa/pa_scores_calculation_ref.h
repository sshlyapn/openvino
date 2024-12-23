// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "sdpa_kernel_base.h"

namespace kernel_selector {

struct pa_scores_calculation : base_params {
    pa_scores_calculation() : base_params(KernelType::PA_SCORES_CALCULATION) {}

    sdpa_configuration conf;
};

class PAScoresCalculation : public KernelBaseOpenCL {
public:
    PAScoresCalculation() : KernelBaseOpenCL{"pa_scores_calc"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    virtual ~PAScoresCalculation() {}

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const pa_scores_calculation& kernel_params) const;
    static CommonDispatchData SetDefault(const pa_scores_calculation& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
