// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_opt_generic.h"
#include "kernel_selector_utils.h"
#include <string>


static constexpr size_t simd = 16;

namespace kernel_selector {
static Tensor::NDims get_normalized_dims(const DataTensor& tensor) {
    auto dims = tensor.GetDims();
    std::reverse(dims.begin(), dims.end());

    return dims;
}

static size_t get_elements_number_per_batch(const dynamic_quantize_params& params) {
    const auto& group_sizes = params.group_sizes;
    const auto& input_dims = get_normalized_dims(params.inputs[0]);

    auto total_elements_number = 1;
    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] != UINT64_MAX) {
            GPU_DEBUG_TRACE_DETAIL << "Multiply " << input_dims[i].v << "\n";
            total_elements_number *= input_dims[i].v;
        }
    }

    return total_elements_number;
}

static size_t get_elements_number_per_group(const dynamic_quantize_params& params) {
    const auto& group_sizes = params.group_sizes;
    const auto& input_dims = get_normalized_dims(params.inputs[0]);

    auto total_elements_number = 1;
    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] == UINT64_MAX) {
            GPU_DEBUG_TRACE_DETAIL << "-> Multiply " << input_dims[i].v << "\n";
            total_elements_number *= input_dims[i].v;
        } else {
            GPU_DEBUG_TRACE_DETAIL << "=> Multiply " << group_sizes[i] << "\n";
            total_elements_number *= group_sizes[i];
        }
    }

    return total_elements_number;
}

static std::string generate_dims_indexes_calculation(std::vector<std::pair<std::string, std::string>> dims) {
    std::reverse(dims.begin(), dims.end());

    auto generate_calc_function = [&](std::string data_type, std::string index_var, size_t dim_idx) {
        std::string index_calc_str;
        index_calc_str += "const " + data_type + " " + dims[dim_idx].first + " = ";
        index_calc_str += "(" + index_var + " / ";
        index_calc_str += "(1";
        for (size_t i = 0; i < dim_idx; i++) {
            index_calc_str += " * " + dims[i].second;
        }
        index_calc_str += ")) % " + dims[dim_idx].second + ";";

        return index_calc_str;
    };

    std::stringstream indexes_calc_str;
    for (size_t i = 0; i < dims.size(); i++) {
        indexes_calc_str << generate_calc_function("uint", "data_idx", i);
    }

    return indexes_calc_str.str();
}

// static size_t get_innermost_group_size(const dynamic_quantize_params& params) {
//     const auto& group_sizes = params.group_sizes;
//     const auto& input_dims = get_normalized_dims(params.inputs[0]);

//     for (size_t i = group_sizes.size(); i > 0; i--) {
//         if (group_sizes[i - 1] == UINT64_MAX) {
//             return input_dims[i - 1].v;
//         } else if (group_sizes[i - 1] != 1) {
//             return group_sizes[i - 1];
//         }
//     }

//     return 1;
// }

// static size_t get_match_vector_size(const dynamic_quantize_params& params) {
//     // const auto input_dt = BytesPerElement(params.inputs[0].GetDType());
//     auto block_sizes = { 8, 4, 2 };

//     for (auto block_size : block_sizes) {
//         if (((params.inputs[0].X().v * params.inputs[0].Y().v) / simd) % block_size == 0) {
//             return block_size;
//         }
//     }

//     return 1;
// }

static size_t get_per_iter_elements_number(const dynamic_quantize_params& params) {
    const auto maxWorkGroupSize = params.engineInfo.maxWorkGroupSize;
    const auto total_grouped_elements = get_elements_number_per_group(params);

    if (total_grouped_elements % maxWorkGroupSize == 0)
        return maxWorkGroupSize;

    if (total_grouped_elements < maxWorkGroupSize)
        return total_grouped_elements;

    return 0;
}

ParamsKey DynamicQuantizeKernelOptGeneric::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants DynamicQuantizeKernelOptGeneric::GetJitConstants(const dynamic_quantize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const std::vector<std::pair<std::string, std::string>> default_dims = {{"b", "INPUT0_BATCH_NUM"},
                                                                           {"f", "INPUT0_FEATURE_NUM"},
                                                                           {"y", "INPUT0_SIZE_Y"},
                                                                           {"x", "INPUT0_SIZE_X"}};

    const auto& group_sizes = params.group_sizes;
    std::vector<std::pair<std::string, std::string>> batch_dims, grouped_dims;
    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] == 1)
            batch_dims.push_back(default_dims[i]);
        else
            grouped_dims.push_back(default_dims[i]);
    }
    const auto total_grouped_elements = get_elements_number_per_group(params);
    const auto per_iter_elements_number = get_per_iter_elements_number(params);

    jit.AddConstant(MakeJitConstant("DECLARE_BATCHED_DIMS_INDEXES(data_idx)", generate_dims_indexes_calculation(batch_dims)));
    jit.AddConstant(MakeJitConstant("DECLARE_GROUPED_DIMS_INDEXES(data_idx)", generate_dims_indexes_calculation(grouped_dims)));
    jit.AddConstant(MakeJitConstant("LWS_SIZE", per_iter_elements_number));

    const auto iterations_number = total_grouped_elements / per_iter_elements_number;

    jit.AddConstant(MakeJitConstant("ITERATIONS_NUMBER", iterations_number));

    bool rearrange_scales_order = false;
    const auto& scales_output_order = params.scales_output_order;
    if (!scales_output_order.empty()) {
        for (size_t i = 0; i < scales_output_order.size(); i++) {
            if (i != scales_output_order[i]) {
                rearrange_scales_order = true;
                break;
            }
        }
    }

    if (rearrange_scales_order) {
        const std::array<char, 4> default_dim_order = {'b', 'f', 'y', 'x'};

        std::stringstream ss;
        for (size_t i = 0; i < scales_output_order.size(); i++) {
            ss << default_dim_order[scales_output_order[i]];

            if (i + 1 != scales_output_order.size())
                ss << ", ";
        }

        jit.AddConstant(MakeJitConstant("SCALES_OUTPUT_ORDER", ss.str()));
        GPU_DEBUG_TRACE_DETAIL << "SCALES_OUTPUT_ORDER: " << ss.str() << "\n";
    }

    for (size_t i = 0; i < group_sizes.size(); i++) {
        jit.AddConstant(MakeJitConstant("GROUP_SIZE_DIM" + std::to_string(i), group_sizes[i]));
    }

    return jit;
}

CommonDispatchData DynamicQuantizeKernelOptGeneric::SetDefault(const dynamic_quantize_params& params) const {
    CommonDispatchData dispatchData;

    const auto total_batched_elements = get_elements_number_per_batch(params);
    // const auto total_grouped_elements = get_elements_number_per_group(params);
    const auto per_iter_elements_number = get_per_iter_elements_number(params);

    dispatchData.gws = {total_batched_elements, per_iter_elements_number, 1};
    dispatchData.lws = {1, per_iter_elements_number, 1};

    return dispatchData;
}

void DynamicQuantizeKernelOptGeneric::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const dynamic_quantize_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;

        GPU_DEBUG_TRACE_DETAIL << "Update Dispatch data DynamicQuantizeKernelOptGeneric gws : " << dispatchData.gws[0] << ", "
                << dispatchData.gws[1] << ", " << dispatchData.gws[2] << std::endl;
    };
}

KernelsData DynamicQuantizeKernelOptGeneric::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::DYNAMIC_QUANTIZE);

    if (!Validate(params))
        return {};

    const dynamic_quantize_params& prim_params = static_cast<const dynamic_quantize_params&>(params);
    auto dispatchData = SetDefault(prim_params);

    KernelData kd = KernelData::Default<dynamic_quantize_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     static_cast<int>(prim_params.outputs.size()),
                     prim_params.is_shape_agnostic);

    return {kd};
}

KernelsPriority DynamicQuantizeKernelOptGeneric::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

bool DynamicQuantizeKernelOptGeneric::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        return false;

    const auto& dq_params = static_cast<const dynamic_quantize_params&>(params);

    const auto& group_sizes = dq_params.group_sizes;
    const auto& input_dims = get_normalized_dims(dq_params.inputs[0]);
    const size_t non_compressed_dims_number = std::count(group_sizes.begin(), group_sizes.end(), 1);

    if (non_compressed_dims_number == group_sizes.size())
        return false;

    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] != 1 && input_dims[i].is_dynamic) {
            return false;
        }
    }

    if (dq_params.inputs[0].GetPaddedVal() != 0 || dq_params.outputs[0].GetPaddedVal() != 0)
        return false;

    return true;
}
}  // namespace kernel_selector

