// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

constexpr size_t seq_len_partition_size = 256;
constexpr size_t subgroup_size = 16;

ParamsKey SDPAKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

bool SDPAKernelOpt::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SDPA) {
        return false;
    }

    const sdpa_params& params = static_cast<const sdpa_params&>(p);

    if (params.inputs[0].Dimentions() != 4)
        return false;

    if (params.conf.head_size < 1)
        return false;

    return true;
}

JitConstants SDPAKernelOpt::GetJitConstants(const sdpa_params& params) const {
    auto jit = Parent::GetJitConstants(params);

    const auto softmax_acc_dt = params.inputs[0].GetDType();
    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "ACCUMULATOR"));

    const auto& config = params.conf;
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", config.head_size));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", config.heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", config.kv_heads_num));

    jit.AddConstant(MakeJitConstant("USE_SEQ_LEN_SPLIT", 1));
    jit.AddConstant(MakeJitConstant("SEQ_LEN_PARTITION_SIZE", seq_len_partition_size));
    jit.AddConstant(MakeJitConstant("SLM_SIZE", seq_len_partition_size));

    return jit;
}

CommonDispatchData SDPAKernelOpt::SetDefault(const sdpa_params& params, size_t kernel_idx) const {
    CommonDispatchData dispatch_data;

    const auto& query_input = params.inputs[0];
    const auto& key_input = params.inputs[1];
    const auto& output = params.outputs[0];
    if (!query_input.is_dynamic()) {
        const size_t seq_len = key_input.Y().v;
        const size_t num_of_partitions = CeilDiv(seq_len, seq_len_partition_size);
        const size_t head_size = static_cast<size_t>(params.conf.head_size);

        if (kernel_idx == 0) {
            dispatch_data.gws = { output.Batch().v * output.Feature().v,
                                  output.Y().v,
                                  head_size * num_of_partitions };
            dispatch_data.lws = { 1, 1, head_size };
        } else {
            dispatch_data.gws = { output.Batch().v * output.Feature().v,
                                  output.Y().v,
                                  head_size };
            dispatch_data.lws = { 1, 1, subgroup_size };
        }
    }

    return dispatch_data;
}

KernelsData SDPAKernelOpt::GetKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<sdpa_params>(params);
    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(prim_params, 0);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto cldnn_jit = GetJitConstants(prim_params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    GetUpdateDispatchDataFunc(kd);

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(prim_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params), 1, prim_params.is_shape_agnostic);

    auto num_of_partitions = 1;

    auto& output = prim_params.outputs[0];
    auto buf_dt_size = 4;
    // auto buf_elements_count = tokens_num * prim_params.configuration.heads_num * num_of_portions;
    auto buf_elements_count = output.LogicalSize() / output.X().v * num_of_partitions;
    auto buf_size = buf_elements_count * buf_dt_size;

    auto tmp_out_dt_size = 4;
    auto tmp_out_elements_count = output.LogicalSize() / output.X().v * num_of_partitions * prim_params.conf.head_size;
    auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});

    kd.internalBufferSizes.clear();
    kd.internalBufferSizes.push_back(buf_size);
    kd.internalBufferSizes.push_back(buf_size);
    kd.internalBufferSizes.push_back(tmp_out_size);
    kd.internalBufferDataType = prim_params.inputs[0].GetDType();

    // ScalarDescriptor num_of_partitions_scalar;
    // num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
    // num_of_partitions_scalar.v.u32 = 1;

    // kd.kernels[1].params.scalars.resize(1);
    // kd.kernels[1].params.scalars[0] = num_of_partitions_scalar;

    return { kd };
}

void SDPAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        const auto& prim_params = static_cast<const sdpa_params&>(params);
        auto dispatchData = SetDefault(prim_params, 0);
        OPENVINO_ASSERT(kernel_data.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kernel_data.kernels[0].params.workGroups.global = dispatchData.gws;
        kernel_data.kernels[0].params.workGroups.local = dispatchData.lws;
        kernel_data.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        // auto& in_q = prim_params.inputs[0];
        // auto& in_k = prim_params.inputs[1];

        // ScalarDescriptor num_of_partitions_scalar;
        // num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
        // num_of_partitions_scalar.v.u32 = 1;

        // kernel_data.kernels[0].params.scalars.resize(1);
        // kernel_data.kernels[0].params.scalars[0] = num_of_partitions_scalar;

        auto num_of_partitions = 1;

        auto& output = prim_params.outputs[0];
        auto buf_dt_size = 4;
        // auto buf_elements_count = tokens_num * prim_params.configuration.heads_num * num_of_portions;
        auto buf_elements_count = output.LogicalSize() / output.X().v * num_of_partitions;
        auto buf_size = buf_elements_count * buf_dt_size;

        auto tmp_out_dt_size = 4;
        auto tmp_out_elements_count = output.LogicalSize() / output.X().v * num_of_partitions * prim_params.conf.head_size;
        auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

        kernel_data.internalBufferSizes.clear();
        kernel_data.internalBufferSizes.push_back(buf_size);
        kernel_data.internalBufferSizes.push_back(buf_size);
        kernel_data.internalBufferSizes.push_back(tmp_out_size);
        kernel_data.internalBufferDataType = prim_params.inputs[0].GetDType();
    };
}

KernelsPriority SDPAKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
