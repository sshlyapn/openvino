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
#include "reorder_weights_kernel_isv16_osv16.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
static const size_t osv = 16;
static const size_t sub_group_size = 16;
static const std::vector<size_t> optimal_ifm_sizes = { 8, 4, 2, 1 };

static size_t GetOptimalSize(size_t val, const std::vector<size_t>& optimal_sizes) {
    return 1;
    for (auto& s : optimal_sizes)
        if (val % s == 0)
            return s;
    return 1;
}

ParamsKey ReorderWeightsKernel_isv16_osv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::INT8);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_isv16_osv16);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableRotateReorder();
    return k;
}

JitConstants ReorderWeightsKernel_isv16_osv16::GetJitConstants(const reorder_weights_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);

    const size_t ifm_block = GetOptimalSize(params.output.IFM().v, optimal_ifm_sizes);
    jit.AddConstant(MakeJitConstant("OUTPUT_IFM_BLOCK_SIZE", ifm_block));
    jit.AddConstant(MakeJitConstant("OSV", osv));

    if (params.output.OFM().v % sub_group_size != 0) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS", params.output.OFM().v % sub_group_size));
    }

    return jit;
}

ReorderKernelBase::DispatchData ReorderWeightsKernel_isv16_osv16::SetDefault(const reorder_weights_params& params) const {
    DispatchData kd;
    const auto& out = params.output;

    std::vector<size_t> global(3);
    std::vector<size_t> local(3);

    const size_t ifm_block = GetOptimalSize(out.IFM().v, optimal_ifm_sizes); 

    global = { out.X().v * out.Y().v,
               out.IFM().v / ifm_block,
               Align(out.OFM().v, sub_group_size)};
    local = { 1, 1, sub_group_size };

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData ReorderWeightsKernel_isv16_osv16::GetKernelsData(const Params& params, const optional_params& options) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
}
}  // namespace kernel_selector
