// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>

namespace kernel_selector {
namespace {
static inline bool IsBroadcastingPossibleInput(const DataTensor& input, const DataTensor& output) {
    if ((input.LogicalSize() == 1) ||
        (input.LogicalSize() == output.Feature().v && input.Feature().v == output.Feature().v)) {
            return true;
        }
    return false;
}

static inline size_t GetBlockSize(const eltwise_params& params) {
    // Set blocksize 1 when broadcasting X dim
    size_t default_block_size = params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv32 ? 2 : 1;
    for (size_t i = 0; i < params.inputs.size(); i++) {
        if ((params.inputs[i].X().v == 1) && !IsBroadcastingPossibleInput(params.inputs[i], params.outputs[0])) {
            return default_block_size;
        }
    }

    size_t optimal_bs_values[] = {8, 4, 2, 1};

    for (auto bs : optimal_bs_values) {
        size_t block_size = bs / default_block_size;
        if ((params.outputs[0].X().v) % block_size == 0) {
            return bs;
        }
    }

    return 1;
}

static inline size_t GetFsvSize(const eltwise_params& params) {
    return params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv32 ? 32 : 16;
}

static inline bool OpHasFeatureBroadcast(const eltwise_params& params, const size_t op_num) {
    const auto &ew = params.operations[op_num];

    for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
        const auto &input = ew.inputs[input_idx];
        if (input.mode == EltwiseInputMode::INPUT_BUFFER) {
            if (params.inputs[input_idx].LogicalSize() != 1 &&
                params.inputs[input_idx].Feature().v == 1 &&
                params.outputs[0].Feature().v != 1) {
                    return true;
                }
        }
    }

    return false;
}
} // namespace

ParamsKey EltwiseKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableEltwiseBroadcast();
    return k;
}

DeviceFeaturesKey EltwiseKernel_b_fs_yx_fsv16::get_required_device_features_key(const Params& params, const optional_params& options) const {
    return get_common_subgroups_device_features_key(params, options);
}

JitConstants EltwiseKernel_b_fs_yx_fsv16::MakeLoadJitConstants(const eltwise_params& params, bool /*useVload8*/) const {
    JitConstants jit = {};
    std::string vload_decls;
    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = toCodeString(op_num);
        const auto &ew = params.operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + toCodeString(input_idx);

            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                {
                    auto fsv_size = GetFsvSize(params);
                    if (params.inputs[input.index].LogicalSize() == params.outputs[0].Feature().v &&
                        params.inputs[input.index].LogicalSize() == params.inputs[input.index].Feature().v) {
                        size_t fsv_blocks = fsv_size == 32 ? 2 : 1;

                        std::string block_read_str = "BLOCK_READN(INPUT" + toCodeString(input.index) + "_TYPE, " + toCodeString(fsv_blocks) +
                                                     ", input" + toCodeString(input.index) +
                                                     ", INPUT" + toCodeString(input.index);
                        if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 4) {
                            block_read_str = block_read_str + "_GET_INDEX(b, f_block*" + toCodeString(fsv_size) + ", y, x))";
                            if (fsv_size == 32) {
                                // Broadcast values to desired vector size
                                size_t block_size = GetBlockSize(params) / fsv_blocks;
                                if (block_size == 4)
                                    block_read_str = "((" + block_read_str + ").xyxyxyxy)";
                                else if (block_size == 2)
                                    block_read_str = "((" + block_read_str + ").xyxy)";
                            }
                            jit.AddConstant(MakeJitConstant(name, block_read_str));
                        } else {
                            jit.AddConstant(MakeJitConstant(name, block_read_str + "_GET_INDEX(b, f_block*" + toCodeString(fsv_size) + ", z, y, x))"));
                        }
                    } else if (params.inputs[input.index].LogicalSize() == 1) {
                        jit.AddConstant(MakeJitConstant(name,
                                                        "input" + toCodeString(input.index) +
                                                        "[0]"));
                    } else {
                        const std::string idx_order = "INPUT" + toCodeString(input.index) + "_IDX_ORDER";
                        const bool feature_broadcasting = (params.inputs[input_idx].Feature().v == 1 && params.outputs[0].Feature().v != 1);
                        const bool fsv32_layout = fsv_size == 32;

                        if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 4) {
                            jit.AddConstant(MakeJitConstant(idx_order, "b, f_block*" + toCodeString(fsv_size) + ", y, x"));
                        } else {
                            jit.AddConstant(MakeJitConstant(idx_order, "b, f_block*" + toCodeString(fsv_size) + ", z, y, x"));
                        }

                        const std::string block_read_str = "TO_TYPE(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE), BLOCK_READN(INPUT" +
                                                                toCodeString(input.index) + "_TYPE, BLOCK_SIZE, " +
                                                                "input" + toCodeString(input.index) + ", " +
                                                                "GET_INDEX(INPUT, " + toCodeString(input.index) + ", " + idx_order + ")))";
                        if (feature_broadcasting) {
                            const std::string broadcast_name = "DO_FEATURE_BROADCAST" + toCodeString(op_num);
                            std::string sub_group_broadcast;
                            std::string tmp_var = "tmp_b" + toCodeString(op_num);
                            if (GetBlockSize(params) == 1) {
                                sub_group_broadcast = "\\\n\t" + tmp_var +
                                                    " = sub_group_broadcast(" + tmp_var + ", 0);";
                            } else if (fsv32_layout) {
                                sub_group_broadcast = "\\\n\tunroll_for (uint i = 0; i < BLOCK_SIZE; ++i) " + tmp_var +
                                                      "[i] = sub_group_broadcast(" + tmp_var + "[i / 2 * 2], 0);";
                            } else {
                                sub_group_broadcast = "\\\n\tunroll_for (uint i = 0; i < BLOCK_SIZE; ++i) " + tmp_var +
                                                    "[i] = sub_group_broadcast(" + tmp_var + "[i], 0);";
                            }

                            std::string broadcast_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE) " + tmp_var +
                                                        " = " + block_read_str + ";" + sub_group_broadcast;

                            jit.AddConstant(MakeJitConstant(broadcast_name, broadcast_value));
                            jit.AddConstant(MakeJitConstant(name, tmp_var));
                        } else {
                            jit.AddConstant(MakeJitConstant(name, block_read_str));
                        }
                    }
                    break;
                }
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[off]"));
                    break;
                case EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(
                            name,
                            "input" + toCodeString(input.index) + "[(size_t)tmp" + toCodeString(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + toCodeString(input.tmpIndex)));
                    break;
                default:
                    break;
            }
        }
    }

    return jit;
}

JitConstants EltwiseKernel_b_fs_yx_fsv16::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    bool useVload8 = false;

    size_t fsv_size = GetFsvSize(params);
    size_t fsv_blocks = fsv_size == 32 ? 2 : 1;
    size_t block_size = GetBlockSize(params);
    size_t block_size_x = block_size / fsv_blocks;

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", block_size));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE_X", block_size_x));
    jit.AddConstant(MakeJitConstant("BLOCKS_COUNT", CeilDiv(params.outputs[0].X().v, block_size_x)));
    jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", fsv_size));

    jit.Merge(MakeInputDeclsJitConstants(params, useVload8));
    jit.Merge(MakeLoadJitConstants(params, useVload8));
    jit.Merge(GetOperationsJitConstants(params, useVload8, block_size));

    std::string do_eltwise;
    auto& operations = params.operations;
    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        if (OpHasFeatureBroadcast(params, op_num)) {
            do_eltwise += "\\\n\tDO_FEATURE_BROADCAST" + toCodeString(op_num) + ";";
        }
        do_eltwise += "\\\n\tOPERATION" + toCodeString(op_num) + ";";
    }

    do_eltwise += "\\\n\tres = tmp" + toCodeString(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    jit.Merge(MakeActivationJitConstants(params.activations, params.outputs[0].GetDType(), "_TYPED"));

    if (params.outputs[0].Feature().v % fsv_size != 0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", params.outputs[0].Feature().v % fsv_size));

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);
        if (fsv_blocks == 2) {
            std::vector<std::string> idx_order = {"b", "f_block*FEATURE_SLICE_SIZE", "y", "x + block_x"};
            FusedOpsConfiguration conf = {"", idx_order, "tmp_res", input_dt, fsv_blocks};
            conf.load_type = FusedOpsConfiguration::LoadType::LT_ALIGNED_READ;
            conf.vec_axis = Tensor::DataChannelName::FEATURE;

            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        } else {
            std::vector<std::string> idx_order;
            if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
                idx_order = {"b", "f_block*FEATURE_SLICE_SIZE", "y", "x"};
            } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
                idx_order = {"b", "f_block*FEATURE_SLICE_SIZE", "z", "y", "x"};
            }

            FusedOpsConfiguration conf = {"", idx_order, "res", input_dt, block_size_x};
            conf.load_type = FusedOpsConfiguration::LoadType::LT_ALIGNED_READ;
            conf.vec_axis = Tensor::DataChannelName::X;

            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    if (params.broadcast) {
        bool need_idx_safe = true;
        for (size_t i = 0; i < params.inputs.size(); i++) {
            if (IsBroadcastingPossibleInput(params.inputs[i], params.outputs[0])) {
                    need_idx_safe = false;
                    break;
            }
        }
        if (need_idx_safe)
            jit.AddConstant(MakeJitConstant("ELTWISE_BROADCAST", params.broadcast));
    }

    return jit;
}

bool EltwiseKernel_b_fs_yx_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const eltwise_params&>(p);

    const auto count = params.outputs[0].PhysicalSize();

    if (count % 8 != 0)
        return false;

    if (IsUnsupportedModeForVecCode(params))
        return false;

    for (size_t i = 0; i < params.inputs.size(); i++) {
        if ((params.inputs[i].GetLayout() != DataLayout::b_fs_yx_fsv16) &&
            (params.inputs[i].GetLayout() != DataLayout::b_fs_yx_fsv32) &&
            (params.inputs[i].GetLayout() != DataLayout::b_fs_zyx_fsv16) &&
            !IsBroadcastingPossibleInput(params.inputs[i], params.outputs[0])) {
            return false;
        }
    }

    auto input0 = params.inputs[0];

    // Check that padding before features doesn't miss-align the blocks
    auto feature_block_size = GetFsvSize(params);
    if (input0.Feature().pad.before % feature_block_size != 0 || params.outputs[0].Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    auto compareTensors = [](const DataTensor& input0, const DataTensor& input1) -> bool {
        // Check all parameters except DataType
        auto& input0_dims = input0.GetDims();
        auto& input1_dims = input1.GetDims();
        bool same = input0.GetLayout() == input1.GetLayout() &&
                    input0.GetPaddedVal() == input1.GetPaddedVal() &&
                    input0.GetViewOffset() == input1.GetViewOffset() &&
                    input0_dims.size() == input1_dims.size();
        if (same) {
            for (size_t i = 0; i < input0_dims.size(); i++) {
                same &= input0_dims[i].v == input1_dims[i].v &&
                        input0_dims[i].pad.before == input1_dims[i].pad.before &&
                        input0_dims[i].pad.after == input1_dims[i].pad.after &&
                        input0_dims[i].pitch == input1_dims[i].pitch;
            }
        }
        return same;
    };

    for (size_t i = 1; i < params.inputs.size(); i++) {
        if (params.inputs[i].LogicalSize() == input0.LogicalSize() && !(compareTensors(params.inputs[i], input0)))
            return false;
        if (params.inputs[i].Feature().pad.before % feature_block_size != 0) {
            return false;
        }
    }

    return true;
}

EltwiseKernelBase::DispatchData EltwiseKernel_b_fs_yx_fsv16::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    size_t fsv_blocks = GetFsvSize(params) == 32 ? 2 : 1;
    size_t block_size_x = GetBlockSize(params) / fsv_blocks;

    dispatchData.gws[0] = CeilDiv(params.outputs[0].Feature().v, GetFsvSize(params)) * 16;
    dispatchData.gws[1] = CeilDiv(params.outputs[0].X().v, block_size_x) * params.outputs[0].Y().v * params.outputs[0].Z().v;
    dispatchData.gws[2] = params.outputs[0].Batch().v;

    dispatchData.lws[0] = 16;
    dispatchData.lws[1] = 16;
    while (dispatchData.lws[1] > 1) {
        if (dispatchData.gws[1] % dispatchData.lws[1] == 0)
            break;
        dispatchData.lws[1]--;
    }
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority EltwiseKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

KernelsData EltwiseKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);

    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    kernel.params.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(),
                                   false,
                                   false,
                                   GetFusedPrimitiveInputsCount(params));

    return {kd};
}
}  // namespace kernel_selector
