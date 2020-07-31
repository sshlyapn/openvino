// Copyright (c) 2019-2020 Intel Corporation
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

#include "resample_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>
#include <vector>
#include <unordered_map>

namespace {
struct InterpolateAxisHash {
    template <typename T>
    size_t operator()(T t) const {
        return static_cast<size_t>(t);
    }
};
}  // namespace

namespace kernel_selector {

size_t ResampleKernelBase::GetFeatureBlockSize(const resample_params& params) const {
    const size_t max_size = 32;
    const size_t min_size = 4;
    size_t feature_block_size = 1;
    std::vector<size_t> preferred_sizes = { 32, 16, 8 };
    for (auto& s : preferred_sizes)
        if (params.output.Feature().v % s == 0)
            return s;
    if (params.output.Feature().v < max_size)
        return params.output.Feature().v;
    for (size_t f = 1; f <= params.output.Feature().v && f <= max_size; f++)
        if (params.output.Feature().v % f == 0)
            feature_block_size = f;
    return std::max(feature_block_size, min_size);
}

ResampleKernelBase::DispatchData ResampleKernelBase::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData runInfo;
    std::vector<size_t> global;
    std::vector<size_t> local;
    const auto& out = arg.output;

    if (arg.resampleType == ResampleType::NEAREST_NEIGHBOR)
        global = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};
    else if (arg.resampleType == ResampleType::BILINEAR_INTERP)
        global = {Align(out.X().v, 32), out.Y().v, out.Batch().v};
    else if (arg.resampleType == ResampleType::CAFFE_BILINEAR_INTERP)
        global = {out.X().v * out.Y().v, CeilDiv(out.Feature().v, GetFeatureBlockSize(arg)), out.Batch().v * out.Z().v};
    else
        global = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};

    local = GetOptimalLocalWorkGroupSizes(global, arg.engineInfo);

    if (arg.resampleType == ResampleType::BILINEAR_INTERP) {
        local[0] = 32;
        local[1] = 1;
        local[2] = 1;
    }

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    runInfo.efficiency = FORCE_PRIORITY_7;
    runInfo.fp16UnitUsed = out.GetDType() == Datatype::F16;

    return runInfo;
}

bool ResampleKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::RESAMPLE || o.GetType() != KernelType::RESAMPLE) {
        return false;
    }

    const resample_params& params = static_cast<const resample_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs.size() == 0) {
        return false;
    }

    const auto& input = params.inputs[0];
    if ((input.GetDType() == Datatype::UINT8 || input.GetDType() == Datatype::INT8) &&
        params.resampleType != ResampleType::NEAREST_NEIGHBOR &&
        params.resampleType != ResampleType::BILINEAR_INTERP)
        return false;

    return true;
}

JitConstants ResampleKernelBase::GetJitConstants(const resample_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& input = params.inputs[0];
    const auto& output = params.output;
    const auto align_corners = params.align_corners;
    const auto pad_begin = params.pad_begin;
    const auto pad_end = params.pad_end;
    const auto b_size_padded = pad_begin + input.Batch().v + pad_end;
    const auto f_size_padded = pad_begin + input.Feature().v + pad_end;
    const auto x_size_padded = pad_begin + input.X().v + pad_end;
    const auto y_size_padded = pad_begin + input.Y().v + pad_end;
    const auto z_size_padded = pad_begin + input.Z().v + pad_end;
    const auto out_b_size_padded = pad_begin + output.Batch().v + pad_end;
    const auto out_f_size_padded = pad_begin + output.Feature().v + pad_end;
    const auto out_x_size_padded = pad_begin + output.X().v + pad_end;
    const auto out_y_size_padded = pad_begin + output.Y().v + pad_end;
    const auto out_z_size_padded = pad_begin + output.Z().v + pad_end;
    std::unordered_map<InterpolateAxis, float, InterpolateAxisHash> scales;

    if (align_corners) {
        scales[InterpolateAxis::BATCH] =
            (out_b_size_padded) > 1 ? static_cast<float>(b_size_padded - 1) / static_cast<float>(out_b_size_padded - 1)
                                    : 0.0f;
        scales[InterpolateAxis::FEATURE] =
            (out_f_size_padded) > 1 ? static_cast<float>(f_size_padded - 1) / static_cast<float>(out_f_size_padded - 1)
                                    : 0.0f;
        scales[InterpolateAxis::X] =
            (out_x_size_padded) > 1 ? static_cast<float>(x_size_padded - 1) / static_cast<float>(out_x_size_padded - 1)
                                    : 0.0f;
        scales[InterpolateAxis::Y] =
            (out_y_size_padded) > 1 ? static_cast<float>(y_size_padded - 1) / static_cast<float>(out_y_size_padded - 1)
                                    : 0.0f;
        scales[InterpolateAxis::Z] =
            (out_z_size_padded) > 1 ? static_cast<float>(z_size_padded - 1) / static_cast<float>(out_z_size_padded - 1)
                                    : 0.0f;
    } else {
        scales[InterpolateAxis::BATCH] = static_cast<float>(b_size_padded) / static_cast<float>(out_b_size_padded);
        scales[InterpolateAxis::FEATURE] = static_cast<float>(f_size_padded) / static_cast<float>(out_f_size_padded);
        scales[InterpolateAxis::X] = static_cast<float>(x_size_padded) / static_cast<float>(out_x_size_padded);
        scales[InterpolateAxis::Y] = static_cast<float>(y_size_padded) / static_cast<float>(out_y_size_padded);
        scales[InterpolateAxis::Z] = static_cast<float>(z_size_padded) / static_cast<float>(out_z_size_padded);
    }
    if (params.shapeCalculationMode == kernel_selector::ShapeCalculationMode::SCALES) {
        for (const auto& it : params.axesAndScales) {
            scales[it.first] = 1.f / it.second;
        }
    }

    jit.AddConstants({
        MakeJitConstant(toString(params.resampleType), ""),
        MakeJitConstant(toString(params.nearestMode), "NEAREST_ROUND_PREFER_FLOOR"),
        MakeJitConstant(toString(params.coordTransMode), "COORD_TRANS_MODE_HALF_PIXEL"),
        MakeJitConstant("B_RATIO", scales[InterpolateAxis::BATCH]),
        MakeJitConstant("F_RATIO", scales[InterpolateAxis::FEATURE]),
        MakeJitConstant("X_RATIO", scales[InterpolateAxis::X]),
        MakeJitConstant("Y_RATIO", scales[InterpolateAxis::Y]),
        MakeJitConstant("Z_RATIO", scales[InterpolateAxis::Z]),
        MakeJitConstant("PAD_BEGIN", pad_begin),
        MakeJitConstant("PAD_END", pad_end),
        MakeJitConstant("ALIGN_CORNERS", align_corners),
        MakeJitConstant("KERNEL_W", 2),
        MakeJitConstant("ANTIALIAS", params.antialias),
        MakeJitConstant("CUBE_COEFF", params.cube_coeff),
    });
    if (params.resampleType == ResampleType::CUBIC) {
        jit.AddConstants({
            MakeJitConstant("CUBIC_COEF_COUNT", 4),
            MakeJitConstant("INDICES_B_START", 0),
            MakeJitConstant("INDICES_F_START", 0),
            MakeJitConstant("INDICES_X_START", 0),
            MakeJitConstant("INDICES_Y_START", 0),
            MakeJitConstant("INDICES_Z_START", 0),
            MakeJitConstant("INDICES_B_END", std::max(1, (int)(scales[InterpolateAxis::BATCH] != 1.0) * 4)),
            MakeJitConstant("INDICES_F_END", std::max(1, (int)(scales[InterpolateAxis::FEATURE] != 1.0) * 4)),
            MakeJitConstant("INDICES_X_END", std::max(1, (int)(scales[InterpolateAxis::X] != 1.0) * 4)),
            MakeJitConstant("INDICES_Y_END", std::max(1, (int)(scales[InterpolateAxis::Y] != 1.0) * 4)),
            MakeJitConstant("INDICES_Z_END", std::max(1, (int)(scales[InterpolateAxis::Z] != 1.0) * 4)),
        });
    }

    size_t feature_block_size = GetFeatureBlockSize(params);

    if (params.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        jit.AddConstant(MakeJitConstant("FEATURE_BLOCK_SIZE", feature_block_size));
        if (params.output.Feature().v % feature_block_size != 0) {
            jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));
            jit.AddConstant(MakeJitConstant("FEATURE_LEFTOVER", params.output.Feature().v % feature_block_size));
        }
    }

    if (params.resampleType == ResampleType::BILINEAR_INTERP) {
        if (params.output.X().v % 32 != 0) {
            jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));
        }
    }

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    return jit;
}

KernelsData ResampleKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<resample_params>(params);
    resample_params& newParams = *static_cast<resample_params*>(kd.params.get());

    auto runInfo = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point,
                     DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = runInfo.efficiency;

    return {kd};
}

Datatype ResampleKernelBase::GetAccumulatorType(const resample_params& params) const {
    auto in_dt = params.inputs[0].GetDType();
    auto out_dt = params.output.GetDType();

    if (params.resampleType == ResampleType::NEAREST_NEIGHBOR)
        return in_dt;

    auto smaller_fp_type = [](const Datatype& current, const Datatype& candidate) -> Datatype {
        if (candidate != Datatype::F32 || candidate != Datatype::F16)
            return current;

        return BytesPerElement(candidate) < BytesPerElement(current) ? candidate : current;
    };

    Datatype fp_type = Datatype::F32;
    fp_type = smaller_fp_type(fp_type, in_dt);
    fp_type = smaller_fp_type(fp_type, out_dt);

    return fp_type;
}

}  // namespace kernel_selector
