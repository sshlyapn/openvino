// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_bf_tiled.h"

#include <vector>
#include <functional>
#include "common_types.h"

static constexpr size_t simd = 16;

namespace kernel_selector {

FullyConnected_bf_tiled::FullyConnected_bf_tiled() : FullyConnectedKernelBase("fully_connected_gpu_bf_tiled") {
    for (unsigned tile_b = 1; tile_b <= 32; ++tile_b)
    for (unsigned tile_ofm = 1; tile_ofm <= 4; tile_ofm *= 2)
    for (unsigned tile_ifm = 1; tile_ifm <= 2; tile_ifm *= 2)
    for (unsigned tile_k = 1; tile_k <= 8; tile_k *= 2)
    for (unsigned dispatch_bsv = 1; dispatch_bsv <= 16; ++dispatch_bsv)
    for (unsigned dispatch_fsv = 1; dispatch_fsv <= 16; ++dispatch_fsv)
    for (auto exec : Parent::autoTuneOptions) {
        // Block reads support at most vector size of 8.
        if (tile_k * tile_ofm > 8)
            continue;
        // For bsv == 1 dispatch order reduces to b_fsv, so anything other than fsv == 1 is redundant.
        if (dispatch_bsv == 1 && dispatch_fsv != 1)
            continue;

        auto_tune_params.emplace_back(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, false, exec);
    }
}

ParamsKey FullyConnected_bf_tiled::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::UINT4);
    k.EnableInputWeightsType(WeightsType::INT4);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDynamicShapesSupport();
    k.EnableWeightsCompression();
    return k;
}

DeviceFeaturesKey FullyConnected_bf_tiled::get_required_device_features_key(const Params& params, const optional_params& options) const {
    auto k = get_common_subgroups_device_features_key(params, options);
    k.requires_subgroup_shuffle();

    return k;
}

bool FullyConnected_bf_tiled::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];
    auto& output = fc_params.outputs[0];
    auto& weights = fc_params.weights;

    // Block reads must be aligned to 4 bytes, for fp16 we can correct for offset misalignment,
    // but we need to ensure that batch pitch preserves alignment.
    if (input.GetDType() == Datatype::F16) {
        if (input.Batch().pitch % 2 != 0 && (input.Batch().v > 1 || fc_params.is_shape_agnostic))
            return false;
        // for 3d case we have to check feature alignment as well
        if (output.GetLayout() == DataLayout::bfyx && input.Feature().pitch % 2 != 0 && (input.Feature().v > 1 || fc_params.is_shape_agnostic))
            return false;
    }

    // Dynamic kernel doesn't support dynamic weights yet
    if (fc_params.is_shape_agnostic && input.is_dynamic()) {
        if ((output.GetLayout() == DataLayout::bfyx && input.Y().v == 0) ||
            (output.GetLayout() == DataLayout::bf && input.Feature().v == 0))
            return false;
    }

    if (input.GetLayout() == DataLayout::bfyx) {
        // Padding on input is not supported.
        // TODO: Enable by mirroring the padding in weights.
        if (input.X().pad.Total() != 0)
            return false;
        if (input.Y().pad.Total() != 0)
            return false;
    }

    // We don't support 4d output
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx) {
        if (input.X().v > 1)
            return false;
    }

    auto wt = weights.GetDType();
    if ((wt == WeightsType::UINT4 || wt == WeightsType::INT4) && (weights.IFM().v % 2 != 0 || weights.OFM().v % 2 != 0)) {
        return false;
    }

    return true;
}

namespace {

struct TuneParamsSelector {
    using tune_params = FullyConnected_bf_tiled::tune_params;
    using functional_case = std::function<tune_params(const fully_connected_params&)>;

    TuneParamsSelector(const fully_connected_params& params) : params(params), selected(false) {}

    TuneParamsSelector& Case(const tune_params& tparams) {
        if (!selected && VerifyTuneParams(params, tparams)) {
            result = tparams;
            selected = true;
        }
        return *this;
    }

    TuneParamsSelector& Case(functional_case fun) {
        return Case(fun(params));
    }

    tune_params Default(const tune_params& tparams) {
        if (!selected) {
            selected = true;
            result = tparams;
        }
        return result;
    }

    static bool VerifyTuneParams(const fully_connected_params& params, const tune_params& tparams);

    const fully_connected_params& params;
    bool selected;
    tune_params result;
};

bool TuneParamsSelector::VerifyTuneParams(const fully_connected_params& params, const tune_params& tparams) {
    // Check divisibility by dispatch tile sizes.
    size_t output_f = params.outputs[0].Feature().v;
    size_t output_b = params.outputs[0].Batch().v;
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        output_b *= params.outputs[0].Feature().v;
        output_f = params.outputs[0].Y().v;
    }

    if (params.compressed &&
        (params.weights.GetDType() == WeightsType::INT4 || params.weights.GetDType() == WeightsType::UINT4) &&
        tparams.tile_ofm != 2)
        return false;

    auto batch_size = params.is_shape_agnostic ? Align(output_b, tparams.tile_b) : output_b;
    if (batch_size % (tparams.tile_b * tparams.dispatch_bsv) != 0)
        return false;
    if (CeilDiv(output_f, tparams.tile_ofm * simd) % tparams.dispatch_fsv != 0)
        return false;

    // Same result can be achieved with smaller tile_ofm.
    if (output_f <= (tparams.tile_ofm / 2) * simd)
        return false;
    // No weights layout for such huge tile ofm.
    if (tparams.tile_ofm * simd > 64)
        return false;

    // Reject tile sizes that are guaranteed to spill out of registers.
    unsigned acc_register_bytes = tparams.tile_b * tparams.tile_ofm * simd * BytesPerElement(params.inputs[0].GetDType());
    unsigned in_register_bytes = tparams.tile_b * tparams.tile_ifm * simd * BytesPerElement(params.inputs[0].GetDType());
    unsigned wei_register_bytes = tparams.tile_ofm * tparams.tile_k * simd * BytesPerElement(params.weights.GetDType());

    unsigned total_register_bytes = acc_register_bytes + in_register_bytes + wei_register_bytes;
    unsigned max_register_bytes = 128 * 32;

    if (total_register_bytes > max_register_bytes)
        return false;

    return true;
}

}  // namespace

template <typename T>
T convert_to(const std::string &str) {
    std::istringstream ss(str);
    T res;
    ss >> res;
    return res;
}

FullyConnected_bf_tiled::tune_params
FullyConnected_bf_tiled::GetAutoTuneParams(const fully_connected_params& params, int idx) const {
    if (idx >= 0 && idx < static_cast<int>(auto_tune_params.size())
        && TuneParamsSelector::VerifyTuneParams(params, auto_tune_params[idx]))
        return auto_tune_params[idx];

    size_t batch = params.outputs[0].Batch().v;
    size_t output_f = params.outputs[0].Feature().v;

    // 3d output
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        batch *= params.outputs[0].Feature().v;
        output_f = params.outputs[0].Y().v;
    }
    Datatype dtype = params.inputs[0].GetDType();

    auto selector = TuneParamsSelector(params);

    unsigned max_tile_ofm = 1;
    while (max_tile_ofm * 2 * simd <= output_f && max_tile_ofm < 4)
        max_tile_ofm *= 2;

    if (params.weights.GetDType() == WeightsType::UINT4 || params.weights.GetDType() == WeightsType::INT4) {
        selector.Default(tune_params(8, 2, 1, 4, 1, 1, false, EXE_MODE_DEFAULT));
    } else if (params.compressed && params.engineInfo.supports_immad) {
        return selector.Default(tune_params(1, 1, 1, 4, 1, 1, false, EXE_MODE_DEFAULT));
    } else if (params.is_shape_agnostic) {
        // Use special tuning params for Gen12HP dGPUs, since these parameters demonstrate higher performance
        // due to better HW utilization (reduced TILE_OFM parameter) and better assembler kernel's code
        // generation (extended TILE_K parameter) for both FP16 and FP32 data types
        if (dtype == Datatype::F16) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, exec_options)
            if (params.engineInfo.supports_immad)
                selector.Case(tune_params(8, 1, 1, 4, 1, 1, false, EXE_MODE_AGE_BASED));
            else
                selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1,  1, false, EXE_MODE_AGE_BASED));
        } else if (dtype == Datatype::F32) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, exec_options)
            if (params.engineInfo.supports_immad)
                selector.Case(tune_params(8, 1, 1, 4, 1, 1, false, EXE_MODE_AGE_BASED));
            else
                selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1,  1, false, EXE_MODE_AGE_BASED));
        }
    } else {
        if (dtype == Datatype::F16) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, exec_options)
            selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 16, 2, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 16, 1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 4,  2, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 8,  1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 2,  2, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 4,  1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 1,  1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1,  1, false, EXE_MODE_AGE_BASED));
        } else if (dtype == Datatype::F32) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, exec_options)
            selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 16, 2, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 16, 1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 8,  1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 4,  1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 2,  1, false, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1,  1, false, EXE_MODE_AGE_BASED));
        }

        if (params.compressed && batch == 1)
            selector.Case(tune_params(1,  std::min(max_tile_ofm, 2u), 4, 2, 1, 1, false, EXE_MODE_AGE_BASED));

        selector.Case([&](const fully_connected_params&) -> tune_params {
            tune_params result(8, std::min(max_tile_ofm, 2u), 1, 2, 1, 1, false, EXE_MODE_DEFAULT);

            while (batch % result.tile_b != 0)
                result.tile_b--;

            result.dispatch_bsv = 16;
            while (batch % (result.tile_b * result.dispatch_bsv) != 0)
                result.dispatch_bsv--;

            if (result.tile_b >= 8)
                result.exec_options = EXE_MODE_AGE_BASED;

            return result;
        });
    }

    bool can_use_slm = false;
    if (const auto env_var = std::getenv("SLM")) {
        can_use_slm = convert_to<bool>(env_var);
        std::cout << "Set use SLM\n";
    }

    if (const auto env_var = std::getenv("SLM_FOR_SCALES")) {
        auto val = convert_to<bool>(env_var);
        std::cout << "Slm for scales forced to " << val << "\n";
        can_use_slm |= val;
    }

    auto tuning_res = selector.Default(tune_params(1, 1, 1, 1, 1, 1, false, EXE_MODE_DEFAULT));
    tuning_res.can_use_slm = can_use_slm;

    if (!can_use_slm) {
        return tuning_res;
    }

    std::cout << "Tuning params:" << " tile_b=" << tuning_res.tile_b << " tile_ofm=" << tuning_res.tile_ofm << " tile_ifm=" << tuning_res.tile_ifm << " tile_k=" << tuning_res.tile_k << " dispatch_bsv=" << tuning_res.dispatch_bsv << " dispatch_fsv=" << tuning_res.dispatch_fsv << "\n";

    size_t tile_b_val = tuning_res.tile_b;
    size_t tile_ofm_val = tuning_res.tile_ofm;
    size_t tile_ifm_val = tuning_res.tile_ifm;
    size_t tile_k_val = tuning_res.tile_k;

    bool forced = false;
    if (const auto env_var = std::getenv("TILE_B")) {
        tile_b_val = convert_to<size_t>(env_var);
        forced = true;
    }
    if (const auto env_var = std::getenv("TILE_OFM")) {
        tile_ofm_val = convert_to<size_t>(env_var);
        forced = true;
    }
    if (const auto env_var = std::getenv("TILE_IFM")) {
        tile_ifm_val = convert_to<size_t>(env_var);
        forced = true;
    }
    if (const auto env_var = std::getenv("TILE_K")) {
        tile_k_val = convert_to<size_t>(env_var);
        forced = true;
    }

    if (forced) {
        tuning_res.tile_b = tile_b_val;
        tuning_res.tile_ofm = tile_ofm_val;
        tuning_res.tile_ifm = tile_ifm_val;
        tuning_res.tile_k = tile_k_val;

        std::cout << "Foreced KERNEL PARAMS\n";
        std::cout << "New tuning params:" << " tile_b=" << tuning_res.tile_b << " tile_ofm=" << tuning_res.tile_ofm << " tile_ifm=" << tuning_res.tile_ifm << " tile_k=" << tuning_res.tile_k << " dispatch_bsv=" << tuning_res.dispatch_bsv << " dispatch_fsv=" << tuning_res.dispatch_fsv << "\n";
    }

    return tuning_res;
}

FullyConnected_bf_tiled::DispatchData
FullyConnected_bf_tiled::SetDefault(const fully_connected_params& params, int autoTuneIndex) const {
    auto dispatchData = Parent::SetDefault(params);
    auto tparams = GetAutoTuneParams(params, autoTuneIndex);

    size_t feature_threads = CeilDiv(params.outputs[0].Feature().v, tparams.tile_ofm * simd);
    size_t batch_threads = params.outputs[0].Batch().v;
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        feature_threads = CeilDiv(params.outputs[0].Y().v, tparams.tile_ofm * simd);
        batch_threads = params.outputs[0].Batch().v * params.outputs[0].Feature().v;
    }

    std::cout << "Original batch =  " << params.outputs[0].Batch().v << "\n";
    batch_threads = CeilDiv(batch_threads, tparams.tile_b);

    size_t lws_batches = 8;
    if (const auto env_var = std::getenv("LWS_BATCHES")) {
        auto val = convert_to<size_t>(env_var);
        lws_batches = val;
    }

    std::cout << "Before " << batch_threads << "\n";

    bool use_slm = tparams.can_use_slm;
    dispatchData.use_slm = use_slm;
    if (use_slm) {
        std::cout << "FC tiled: use slm version\n";
        // if (params.is_shape_agnostic && (params.outputs[0].Batch().v % 64 != 0 && params.outputs[0].Batch().v < 512)) {
        //     std::cout << "WARNING !!!\n";
        //     std::cout << "SLM enabled, but use default version!\n";
        //     std::cout << "WARNING !!!\n";
        //     use_slm = false;
        // }
    }


    auto aligned_batch = Align(batch_threads, lws_batches);

    std::cout << "Kernel params: feature_threads=" << feature_threads << " batch_threads=" << batch_threads
              << " (aligned=" << aligned_batch << ") simd=" << simd << " lws_batches=" << lws_batches << "\n";

    if (aligned_batch != batch_threads)
        std::cout << "Batch threads number was aligned: " << batch_threads << " -> " << aligned_batch << "\n";



    dispatchData.gws[0] = use_slm ? feature_threads * simd : feature_threads * batch_threads * simd;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = use_slm ? aligned_batch : 1;

    dispatchData.lws[0] = simd;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = use_slm ? lws_batches : 1;

    dispatchData.tile_m = tparams.tile_b;
    dispatchData.tile_n = tparams.tile_ofm;
    dispatchData.tile_mk = tparams.tile_ifm;
    dispatchData.tile_nk = tparams.tile_k;
    dispatchData.tile_ms = tparams.dispatch_bsv;
    dispatchData.tile_ns = tparams.dispatch_fsv;

    return dispatchData;
}

KernelsPriority FullyConnected_bf_tiled::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    size_t output_b = fc_params.outputs[0].Batch().v;
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx)
        output_b *= fc_params.outputs[0].Feature().v;

    float estimated_time = FORCE_PRIORITY_9;
    if (output_b > 1 && fc_params.inputs[0].GetDType() == Datatype::F32)
        estimated_time = FORCE_PRIORITY_3;
    else if (output_b > 1 && fc_params.inputs[0].GetDType() == Datatype::F16)
        estimated_time = FORCE_PRIORITY_4;

    return estimated_time;
}

JitConstants FullyConnected_bf_tiled::GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    size_t tile_k_ofm = dispatchData.tile_nk * dispatchData.tile_n;
    size_t tile_k_ofm_packed = tile_k_ofm;

    bool use_slm_for_scales = false;
    if (const auto env_var = std::getenv("SLM_FOR_SCALES")) {
        auto val = convert_to<bool>(env_var);
        std::cout << "Slm for scales forced to " << val << "\n";
        use_slm_for_scales = val;
    }

    if (params.weights.GetDType() == WeightsType::UINT4 || params.weights.GetDType() == WeightsType::INT4) {
        tile_k_ofm_packed /= 2;

        jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE", params.weights.GetDType(), tile_k_ofm));
        jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE_PRELOAD", params.weights.GetDType(), 4));
        std::cout << "> JIT: INT4_PACKED_TYPE=1\n";
        const size_t scale_group_size = params.weights.IFM().v / params.decompression_scale.Feature().v;
        if (scale_group_size % simd == 0 && (!dispatchData.use_slm || use_slm_for_scales)) {
            std::cout << "> JIT: DECOMPRESSION_SCALE_POST_OP=1\n";
            jit.AddConstant(MakeJitConstant("DECOMPRESSION_SCALE_POST_OP", 1));
        } else if (scale_group_size % simd == 0) {
            std::cout << "> IGNORE JIT: DECOMPRESSION_SCALE_POST_OP=1 because of SLM\n";
        }
    }

    if (dispatchData.use_slm) {
        if (dispatchData.tile_n != 2) {
            std::cout << "FC error: unsupported OFM size\n";
            OPENVINO_ASSERT(false);
        }

        bool uncompress_before_store = false;
        bool weights_prefetch = false;

        if (const auto env_var = std::getenv("UNCOMP")) {
            auto val = convert_to<bool>(env_var);
            std::cout << "Uncompress before store forced to " << val << "\n";
            uncompress_before_store = val;
        }


        if (const auto env_var = std::getenv("WEIGHTS_PREFETCH")) {
            auto val = convert_to<bool>(env_var);
            std::cout << "Weights prefetch forced to " << val << "\n";
            weights_prefetch = val;
        }

        auto lws_batches = dispatchData.lws[2];

        jit.AddConstant(MakeJitConstant("USE_SLM", 1));
        jit.AddConstant(MakeJitConstant("LWS_BATCHES", lws_batches));
        jit.AddConstant(MakeJitConstant("UNCOMPRESS_BEFORE_STORE", uncompress_before_store));
        jit.AddConstant(MakeJitConstant("USE_WEIGHTS_PREFETCH", weights_prefetch));

        std::cout << "> JIT: USE_SLM=1\n";
        std::cout << "> JIT: LWS_BATCHES=" << lws_batches << "\n";
        std::cout << "> JIT: UNCOMPRESS_BEFORE_STORE=" << uncompress_before_store << "\n";
        std::cout << "> JIT: USE_SLM_FOR_SCALES=" << use_slm_for_scales << "\n";

        if (use_slm_for_scales)
            jit.AddConstant(MakeJitConstant("USE_SLM_FOR_SCALES", use_slm_for_scales));

        auto weights_val_num = simd * dispatchData.tile_n * simd * dispatchData.tile_mk;
        // auto weights_val_size = (params.weights.GetDType() == WeightsType::UINT8 || params.weights.GetDType() == WeightsType::INT8) ? 1 :
        //                         (params.weights.GetDType() == WeightsType::F16) ? 2 : 4;

        auto loads_per_sg = weights_val_num / lws_batches;
        auto weights_load_iters = loads_per_sg / simd;

        if (params.weights.GetDType() == WeightsType::INT4 || params.weights.GetDType() == WeightsType::UINT4) {
            if (weights_load_iters % 4 != 0) {
                std::cout << "FC error: block size for weights loading\n";
                OPENVINO_ASSERT(false);
            }
            // Load by 4 values in each WI per load iteration
            weights_load_iters /= 4;
        }

        if (params.weights.GetDType() == WeightsType::INT8 || params.weights.GetDType() == WeightsType::UINT8 || params.weights.GetDType() == WeightsType::F16) {
            if (weights_load_iters % 2 != 0) {
                std::cout << "FC error: block size for weights loading\n";
                OPENVINO_ASSERT(false);
            }
            // Load by 2 values in each WI per load iteration
            weights_load_iters /= 2;
        }



        jit.AddConstant(MakeJitConstant("WEIGHTS_PRELOAD_ITERS", weights_load_iters));
        std::cout << "> JIT: WEIGHTS_PRELOAD_ITERS=" << weights_load_iters << "\n";

        size_t batch_threads = params.outputs[0].Batch().v;
        if (params.outputs[0].GetLayout() == DataLayout::bfyx)
            batch_threads = params.outputs[0].Batch().v * params.outputs[0].Feature().v;

        batch_threads = CeilDiv(batch_threads, dispatchData.tile_m);

        if (batch_threads != Align(batch_threads, lws_batches)) {
            auto skip_batches_after = batch_threads;
            jit.AddConstant(MakeJitConstant("BATCH_LEFTOVERS", skip_batches_after));
            std::cout << "> JIT: BATCH_LEFTOVERS=" << skip_batches_after << "\n";
        }

        const auto& input = params.inputs[0];
        if (input.LogicalSize()) {
            auto iterations_count = (input.LogicalSize() / input.Batch().v) / (dispatchData.tile_mk * simd);

            std::cout << "Total weights values stored in SLM=" << weights_val_num << ", outer loop iterations count=" << iterations_count << "\n";
        }
    }

    jit.AddConstant(MakeJitConstant("SIMD", simd));
    jit.AddConstant(MakeJitConstant("TILE_B", dispatchData.tile_m));
    jit.AddConstant(MakeJitConstant("TILE_OFM", dispatchData.tile_n));
    jit.AddConstant(MakeJitConstant("TILE_IFM", dispatchData.tile_mk));
    jit.AddConstant(MakeJitConstant("TILE_K", dispatchData.tile_nk));
    jit.AddConstant(MakeJitConstant("TILE_K_OFM", tile_k_ofm));
    jit.AddConstant(MakeJitConstant("TILE_K_OFM_PACKED", tile_k_ofm_packed));
    jit.AddConstant(MakeJitConstant("DISPATCH_BSV", dispatchData.tile_ms));
    jit.AddConstant(MakeJitConstant("DISPATCH_FSV", dispatchData.tile_ns));

    jit.Merge(MakeConstantLoopUnrollJitConstants(dispatchData.tile_m));

    bool realign_fp16_offset = params.inputs[0].GetDType() == Datatype::F16 && params.inputs[0].GetFirstElementOffset() % 2 != 0;
    jit.AddConstant(MakeJitConstant("REALIGN_FP16_OFFSET", realign_fp16_offset));
    if (realign_fp16_offset)
        std::cout << "> JIT: REALIGN_FP16_OFFSET=1\n";

    jit.AddConstant(MakeJitConstant("TILE_K_OFM_PACKED", tile_k_ofm_packed));
    std::cout << "> JIT: TILE_K_OFM_PACKED=" << tile_k_ofm_packed << "\n";

    auto activation_dt = GetActivationType(params);
    auto accumulator_dt = GetAccumulatorType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));

    // for 3d output we are treating spatial as features
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Y().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Y().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", params.inputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("OUTPUT_3D", true));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM)"));
    } else {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Feature().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", params.inputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM)"));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order_scalar = { "(out_b + bi)", "(out_f + sglid)", "0", "0" };
        std::vector<std::string> idx_order_vec = { "(out_b + bi)", "(out_f + fi + sglid)", "0", "0" };
        if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
            idx_order_scalar = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "sglid", "0" };
            idx_order_vec = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "sglid", "0" };
        }

        // Simplify fused ops configuration to prevent mixed layout exception in jitter
        // for common cases with bfyx -> bf layouts and eltwise fusing (such scenarios currently don't work for vectors)
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              idx_order_scalar,
                                              "activated[bi]",
                                              activation_dt,
                                              1 };
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           idx_order_vec,
                                           "activated[bi][fi]",
                                           activation_dt,
                                           1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_scalar, conf_vec }));
    }

    return jit;
}

KernelsData FullyConnected_bf_tiled::GetTunedKernelsDataByIndex(const Params &params,
                                                                const optional_params &options,
                                                                const int autoTuneIndex) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);

    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(auto_tune_params.size())
        && !TuneParamsSelector::VerifyTuneParams(fc_params, auto_tune_params[autoTuneIndex]))
        return {};

    tune_params tparams = GetAutoTuneParams(fc_params, autoTuneIndex);

    WeightsLayout weights_layout = WeightsLayout::os_iyx_osv16;
    if (tparams.tile_ofm * simd == 32)
        weights_layout = WeightsLayout::os_iyx_osv32;
    else if (tparams.tile_ofm * simd == 64)
        weights_layout = WeightsLayout::os_iyx_osv64;

    return GetCommonKernelsData(params,
                                options,
                                fc_params.inputs[0].GetLayout(),
                                weights_layout,
                                tparams.exec_options,
                                autoTuneIndex);
}

KernelsData FullyConnected_bf_tiled::GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const {
    KernelsData res = {};
    for (size_t idx = 0; idx < auto_tune_params.size(); ++idx) {
        KernelsData kds = GetTunedKernelsDataByIndex(params, options, static_cast<int>(idx));

        if (!kds.empty()) {
            res.emplace_back(kds[0]);
        }
    }

    return res;
}

KernelsData FullyConnected_bf_tiled::GetKernelsData(const Params& params, const optional_params& optParams) const {
    KernelsData res = {};
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto tparams = GetAutoTuneParams(fc_params);

    KernelsData kds = GetTunedKernelsDataByIndex(params, optParams, -1);
    if (!kds.empty()) {
        res.emplace_back(kds[0]);
    }

    return res;
}

}  // namespace kernel_selector
