// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_update_kernel_ref.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr size_t kv_cache_block_size = 16;

void KVCacheUpdateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const kv_cache_update_update_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 2, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;

        kd.kernels[1].params.workGroups.global = dispatchData.gws;
        kd.kernels[1].params.workGroups.local = dispatchData.lws;
        kd.kernels[1].skip_execution = false;

        ScalarDescriptor block_elem_num;
        const auto& kv_cache_output = prim_params.outputs[0];
        block_elem_num.t = ScalarDescriptor::Types::UINT32;
        block_elem_num.v.u32 = kv_cache_output.LogicalSize() / kv_cache_output.Batch().v;
        kd.kernels[0].params.scalars.resize(1);
        kd.kernels[0].params.scalars[0] = block_elem_num;
        kd.kernels[1].params.scalars.resize(1);
        kd.kernels[1].params.scalars[0] = block_elem_num;
    };
}

KernelsData KVCacheUpdateKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<kv_cache_update_update_params>(params, 2);
    kd.needs_sub_kernels_sync = false;
    GetUpdateDispatchDataFunc(kd);

    const auto& kernel_params = static_cast<const kv_cache_update_update_params&>(params);
    for (size_t i = 0; i < 2; i++) {
        const auto kernel_stage = i == 0 ? KernelMode::value_cache_update : KernelMode::key_cache_update;
        const auto dispatch_data = SetDefault(kernel_params);
        const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params, i);
        const auto jit_constants = GetJitConstants(kernel_params, kernel_stage);
        const auto jit = CreateJit(kernelName, jit_constants, entry_point);

        auto& kernel = kd.kernels[i];
        FillCLKernelData(kernel,
                         dispatch_data,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         {},
                         false,
                         false,
                         static_cast<int>(kernel_params.inputs.size()),
                         GetFusedPrimitiveInputsCount(kernel_params),
                         static_cast<int>(kernel_params.outputs.size()),
                         kernel_params.is_shape_agnostic);

        ScalarDescriptor block_elem_num;
        block_elem_num.t = ScalarDescriptor::Types::UINT32;
        block_elem_num.v.u32 = 0;
        kernel.params.scalars.push_back(block_elem_num);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
    }

    return {kd};
}

ParamsKey KVCacheUpdateKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableInputDataType(Datatype::INT32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::INT32);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfzyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    key.EnableDifferentTypes();
    return key;
}

bool KVCacheUpdateKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PA_KV_CACHE_UPDATE) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const kv_cache_update_update_params&>(params);
    if (kernel_params.inputs.size() != 3)
        return false;

    if (kernel_params.outputs.size() != 2)
        return false;

    // if (kernel_params.inputs[0].GetDType() != kernel_params.outputs[0].GetDType())
    //     return false;

    return true;
}

JitConstants KVCacheUpdateKernelRef::GetJitConstants(const kv_cache_update_update_params& kernel_params, KernelMode mode) const {
    JitConstants jit = MakeBaseParamsJitConstants(kernel_params);

    if (mode == KernelMode::key_cache_update)
        jit.AddConstant(MakeJitConstant("KEY_CACHE_UPDATE", 1));
    else
        jit.AddConstant(MakeJitConstant("VALUE_CACHE_UPDATE", 1));

    jit.AddConstant(MakeJitConstant("KV_CACHE_BLOCK_SIZE", kv_cache_block_size));

    return jit;
}

CommonDispatchData KVCacheUpdateKernelRef::SetDefault(const kv_cache_update_update_params& kernel_params) {
    CommonDispatchData dispatch_data;

    const auto& input = kernel_params.inputs[0];
    const auto& output = kernel_params.outputs[1];
    if (!output.is_dynamic()) {
        const size_t block_size = output.X().v;
        OPENVINO_ASSERT(block_size == kv_cache_block_size, "[GPU] Block size of kv_cache is expected to be 16, got", block_size);

        const size_t batch_size = input.Batch().v;
        const size_t seq_len = input.Feature().v;
        const size_t tokens_num = batch_size * seq_len;
        const size_t hidden_size = input.LogicalSize() / (tokens_num);
        dispatch_data.gws = {batch_size, seq_len, Align(hidden_size, 16)};
        dispatch_data.lws = {1, 1, 16};
    }

    return dispatch_data;
}

}  // namespace kernel_selector
