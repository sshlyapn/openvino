// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_ref.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

constexpr size_t HEAD_SIZE = 64;
constexpr size_t HEADS_NUM = 32;
constexpr size_t KV_HEADS_NUM = 4;
constexpr size_t BLOCK_SIZE = 16;
constexpr size_t X_BLOCK_SIZE = 8;

constexpr size_t MAX_SEQUENCE_LENGTH = 1024;

void SDPAKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const sdpa_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;

        OPENVINO_ASSERT(prim_params.configuration.head_size == HEAD_SIZE,
                        "[GPU] Unexpected HEAD_SIZE in SDPA kernel, expected ", HEAD_SIZE,
                        " got ", prim_params.configuration.head_size);
        OPENVINO_ASSERT(prim_params.configuration.heads_num == HEADS_NUM,
                        "[GPU] Unexpected HEADS_NUM in SDPA kernel, expected ", HEADS_NUM,
                        " got ", prim_params.configuration.heads_num);
        OPENVINO_ASSERT(prim_params.configuration.kv_heads_num == KV_HEADS_NUM,
                        "[GPU] Unexpected KV_HEADS_NUM in SDPA kernel, expected ", KV_HEADS_NUM,
                        " got ", prim_params.configuration.kv_heads_num);
        OPENVINO_ASSERT(prim_params.configuration.block_size == BLOCK_SIZE,
                        "[GPU] Unexpected BLOCK_SIZE in SDPA kernel, expected ", BLOCK_SIZE,
                        " got ", prim_params.configuration.block_size);
        OPENVINO_ASSERT(prim_params.configuration.x_size == X_BLOCK_SIZE,
                        "[GPU] Unexpected X_BLOCK_SIZE in SDPA kernel, expected ", X_BLOCK_SIZE,
                        " got ", prim_params.configuration.x_size);
    };
}

KernelsData SDPAKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<sdpa_params>(params);
    kd.needs_sub_kernels_sync = false;
    GetUpdateDispatchDataFunc(kd);

    const auto& kernel_params = static_cast<const sdpa_params&>(params);

    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);

    auto& kernel = kd.kernels.front();
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

    return {kd};
}

ParamsKey SDPAKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableInputDataType(Datatype::INT32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::INT32);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableInputLayout(DataLayout::bfzyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfzyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    key.EnableDifferentTypes();
    return key;
}

bool SDPAKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PA_SDPA) {
        return false;
    }

    // const auto& kernel_params = dynamic_cast<const sdpa_params&>(params);
    // if (kernel_params.inputs.size() != 3)
    //     return false;

    // if (kernel_params.outputs.size() != 2)
    //     return false;

    return true;
}

JitConstants SDPAKernelRef::GetJitConstants(const sdpa_params& kernel_params) const {
    JitConstants jit = MakeBaseParamsJitConstants(kernel_params);

    jit.AddConstant(MakeJitConstant("HEAD_SIZE", HEAD_SIZE));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", HEADS_NUM));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", KV_HEADS_NUM));
    jit.AddConstant(MakeJitConstant("NUM_QUERIES_PER_KV_HEAD", HEADS_NUM / KV_HEADS_NUM));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", BLOCK_SIZE));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", X_BLOCK_SIZE));

    const auto& output = kernel_params.inputs[0];
    const auto shared_mem_size = MAX_SEQUENCE_LENGTH * BytesPerElement(output.GetDType());
    jit.AddConstant(MakeJitConstant("SHARED_MEM_SIZE", shared_mem_size));

    return jit;
}

CommonDispatchData SDPAKernelRef::SetDefault(const sdpa_params& kernel_params) {
    CommonDispatchData dispatch_data;

    const auto& input = kernel_params.inputs[0];
    if (!input.is_dynamic()) {
        const size_t batch_size = input.Batch().v;
        const size_t seq_len = input.Feature().v;
        const size_t tokens_num = batch_size * seq_len;
        dispatch_data.gws = { tokens_num,
                              kernel_params.configuration.heads_num,
                              kernel_params.configuration.head_size };
        dispatch_data.lws = { 1, 1, kernel_params.configuration.head_size };
    }

    return dispatch_data;
}

}  // namespace kernel_selector
