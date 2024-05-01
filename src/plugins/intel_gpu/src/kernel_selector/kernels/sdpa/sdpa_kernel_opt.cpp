// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

constexpr size_t subgroup_size = 16;

template <typename T>
T convert_to(const std::string &str) {
    std::istringstream ss(str);
    T res;
    ss >> res;
    return res;
}

template <>
std::string convert_to(const std::string &str) {
    return str;
}

static size_t get_seq_id_block_size() {
    static bool called = false;
    size_t block_size = 1;
    if (const auto env_var = std::getenv("BLOCK_SIZE")) {
        block_size = convert_to<size_t>(env_var);
    }

    if (!called) {
        std::cout << "Set block size = " << block_size << "\n";
        called = true;
    }
    return block_size;
}


static size_t get_seq_len_partition_size() {
    static bool called = false;
    size_t seq_len = 128;
    if (const auto env_var = std::getenv("SEQ_LEN")) {
        seq_len = convert_to<size_t>(env_var);
    }

    if (!called) {
        std::cout << "Set seq_len_partition_size = " << seq_len << "\n";
        called = true;
    }
    return seq_len;
}

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

JitConstants SDPAKernelOpt::GetJitConstants(const sdpa_params& params, size_t kernel_idx) const {
    auto jit = MakeBaseParamsJitConstants(params);

    // const auto softmax_acc_dt = Datatype::F32;
    const auto softmax_acc_dt = params.inputs[0].GetDType();
    jit.Merge(MakeTypeJitConstants(softmax_acc_dt, "SOFTMAX_ACCUMULATOR"));

    if (const auto env_var = std::getenv("MULS_NUM")) {
        auto muls_num = convert_to<size_t>(env_var);
        jit.AddConstant(MakeJitConstant("MULS_NUM", muls_num));
    }

    if (const auto env_var = std::getenv("FIRST_APPROACH_OPTION2")) {
        jit.AddConstant(MakeJitConstant("FIRST_APPROACH_OPTION2", 1));
    }

    if (const auto env_var = std::getenv("FIRST_APPROACH")) {
        jit.AddConstant(MakeJitConstant("FIRST_APPROACH", 1));
    }

    static bool printed = false;
    if (!printed) {
        printed = true;
        std::cout << "Input " << static_cast<int>(params.inputs[0].GetDType()) << " " << static_cast<int>(params.outputs[0].GetDType()) << " " << static_cast<int>(softmax_acc_dt) << "\n";

        if (const auto env_var = std::getenv("MULS_NUM")) {
            auto muls_num = convert_to<size_t>(env_var);
            std::cout << "Force MULS_NUM to " << muls_num << "\n";
        }
        if (const auto env_var = std::getenv("FIRST_APPROACH_OPTION2")) {
            std::cout << "Force FIRST_APPROACH_OPTION2\n";
        }
        if (const auto env_var = std::getenv("FIRST_APPROACH")) {
            std::cout << "Force FIRST_APPROACH\n";
        }
    }

    const auto& config = params.conf;
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", config.head_size));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", config.heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", config.kv_heads_num));

    jit.AddConstant(MakeJitConstant("USE_SEQ_LEN_SPLIT", 1));
    jit.AddConstant(MakeJitConstant("SEQ_LEN_PARTITION_SIZE", get_seq_len_partition_size()));
    jit.AddConstant(MakeJitConstant("SLM_SIZE", get_seq_len_partition_size()));
    jit.AddConstant(MakeJitConstant("SDPA_STAGE_" + std::to_string(kernel_idx), 1));

    jit.AddConstant(MakeJitConstant("SEQ_ID_BLOCK_SIZE", get_seq_id_block_size()));

    return jit;
}

CommonDispatchData SDPAKernelOpt::SetDefault(const sdpa_params& params, size_t kernel_idx) const {
    CommonDispatchData dispatch_data;

    const auto& query_input = params.inputs[0];
    const auto& key_input = params.inputs[1];
    const auto& output = params.outputs[0];
    if (!query_input.is_dynamic()) {
        const size_t source_seq_len = key_input.Y().v;
        const size_t target_seq_len = output.Y().v;
        const size_t num_of_partitions = CeilDiv(source_seq_len, get_seq_len_partition_size());
        const size_t head_size = static_cast<size_t>(params.conf.head_size);

        if (kernel_idx == 0) {
            dispatch_data.gws = { output.Batch().v * output.Feature().v,
                                  CeilDiv(target_seq_len, get_seq_id_block_size()),
                                  head_size * num_of_partitions };
            dispatch_data.lws = { 1, 1, head_size };
        } else {
            dispatch_data.gws = { output.Batch().v * output.Feature().v,
                                  target_seq_len,
                                  head_size };
            dispatch_data.lws = { 1, 1, subgroup_size };
        }
    }

    return dispatch_data;
}

KernelsData SDPAKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const size_t kernels_num = 2;
    KernelData kd = KernelData::Default<sdpa_params>(params, kernels_num);
    kd.needs_sub_kernels_sync = true;

    GetUpdateDispatchDataFunc(kd);

    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);
    for (size_t kernel_num = 0; kernel_num < kernels_num; kernel_num++) {
        auto dispatch_data = SetDefault(prim_params, kernel_num);
        auto kernel_name = kernel_num == 0 ? kernelName : "sdpa_opt_finalization";
        auto entry_point = GetEntryPoint(kernel_name, prim_params.layerID, params);
        auto jit_constants = GetJitConstants(prim_params, kernel_num);
        auto jit = CreateJit(kernel_name, jit_constants, entry_point);

        auto& kernel = kd.kernels[kernel_num];

        auto inputs_num = kernel_num == 1 ? 0 : static_cast<int>(prim_params.inputs.size());
        FillCLKernelData(kernel,
                         dispatch_data,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         {},
                         false,
                         false,
                         inputs_num,
                         GetFusedPrimitiveInputsCount(params),
                         static_cast<int>(prim_params.outputs.size()),
                         prim_params.is_shape_agnostic);

        const auto num_of_partitions = 1;
        auto& output = prim_params.outputs[0];
        auto head_size = output.X().v;

        auto buf_dt_size = 4;
        auto buf_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() / head_size * num_of_partitions;
        auto buf_size = buf_elements_count * buf_dt_size;

        auto tmp_out_dt_size = 4;
        auto tmp_out_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() * num_of_partitions;
        auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(buf_size);
        kd.internalBufferSizes.push_back(buf_size);
        kd.internalBufferSizes.push_back(tmp_out_size);
        kd.internalBufferDataType = prim_params.inputs[0].GetDType();

        GPU_DEBUG_TRACE_DETAIL << "configure SDPA " << kernel_num << "th kernel: inputs_num=" << inputs_num << " arguments_num=" << kernel.params.arguments.size() << "\n";

        if (kernel_num == 1) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

            ScalarDescriptor num_of_partitions_scalar;
            num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
            num_of_partitions_scalar.v.u32 = num_of_partitions;

            kernel.params.scalars.clear();
            kernel.params.scalars.push_back(num_of_partitions_scalar);
        }
    }

    return { kd };
}

void SDPAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        const auto& prim_params = static_cast<const sdpa_params&>(params);

        const size_t expected_kernels_num = 2;
        OPENVINO_ASSERT(kernel_data.kernels.size() == expected_kernels_num,
                        "[GPU] Invalid kernels size for update dispatch data func of SDPA kernel");

        auto& output = prim_params.outputs[0];
        auto& key_input = prim_params.inputs[1];

        auto head_size = output.X().v;
        auto source_seq_len = key_input.Y().v;
        auto num_of_partitions = CeilDiv(source_seq_len, get_seq_len_partition_size());

        auto buf_dt_size = 4;
        auto buf_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() / head_size * num_of_partitions;
        auto buf_size = buf_elements_count * buf_dt_size;

        auto tmp_out_dt_size = 4;
        auto tmp_out_elements_count = (num_of_partitions == 1) ? 1 : output.LogicalSize() * num_of_partitions;
        auto tmp_out_size = tmp_out_elements_count * tmp_out_dt_size;

        auto dispatch_data1 = SetDefault(prim_params, 0);
        kernel_data.kernels[0].params.workGroups.global = dispatch_data1.gws;
        kernel_data.kernels[0].params.workGroups.local = dispatch_data1.lws;
        kernel_data.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        ScalarDescriptor num_of_partitions_scalar;
        num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
        num_of_partitions_scalar.v.u32 = num_of_partitions;

        auto dispatch_data2 = SetDefault(prim_params, 1);
        kernel_data.kernels[1].params.workGroups.global = dispatch_data2.gws;
        kernel_data.kernels[1].params.workGroups.local = dispatch_data2.lws;
        kernel_data.kernels[1].skip_execution = num_of_partitions == 1;

        kernel_data.kernels[1].params.scalars.clear();
        kernel_data.kernels[1].params.scalars.push_back(num_of_partitions_scalar);
        GPU_DEBUG_TRACE_DETAIL << "update_dispatch_data_func SDPA 0th kernel: arguments_num=" << kernel_data.kernels[0].params.arguments.size() << "\n";
        GPU_DEBUG_TRACE_DETAIL << "update_dispatch_data_func SDPA 1th kernel: arguments_num=" << kernel_data.kernels[1].params.arguments.size() << "\n";

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
