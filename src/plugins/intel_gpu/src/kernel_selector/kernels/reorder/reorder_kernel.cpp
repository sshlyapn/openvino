﻿// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputSurfaceReorder();
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ReorderKernelRef::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    if (params.surface_input) {
        jit.AddConstant(MakeJitConstant("SURFACE_INPUT", true));

        if (params.outputs[0].GetDType() == Datatype::F16)
            jit.AddConstant(MakeJitConstant("SURFACE_DST_TYPE_HALF", true));
        else if (params.outputs[0].GetDType() == Datatype::UINT32)
            jit.AddConstant(MakeJitConstant("SURFACE_DST_TYPE_UINT", true));
        else if (params.outputs[0].GetDType() == Datatype::INT32)
            jit.AddConstant(MakeJitConstant("SURFACE_DST_TYPE_INT", true));
        else if (params.outputs[0].GetDType() == Datatype::F32)
            jit.AddConstant(MakeJitConstant("SURFACE_DST_TYPE_FLOAT", true));
    }

    return jit;
}

bool ReorderKernelRef::Validate(const Params& params, const optional_params& options) const {
    const reorder_params& p = static_cast<const reorder_params&>(params);
    if (p.surface_input) {
        if (p.outputs[0].GetDType() != Datatype::F32 &&
            p.outputs[0].GetDType() != Datatype::F16 &&
            p.outputs[0].GetDType() != Datatype::UINT32 &&
            p.outputs[0].GetDType() != Datatype::INT32)
            return false;
    }
    return true;
}

KernelsData ReorderKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams, options);
}

KernelsPriority ReorderKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
