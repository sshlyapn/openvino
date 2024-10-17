// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "ov_ops/dynamic_quantize.hpp"

namespace cldnn {

/// @brief Dynamic Quantize primitive
/// @details Performs dynamic quantization
struct dynamic_quantize : public primitive_base<dynamic_quantize> {
    CLDNN_DECLARE_PRIMITIVE(dynamic_quantize);

    using QuantizationConfig = ov::op::internal::QuantizationConfig;

    dynamic_quantize() : primitive_base("", {}) {}

    /// @brief Constructs dynamic_quantize primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param group_sizes Quantization group size
    /// @param data_type Output data type of quantized
    /// @param output_size Output data size of the primitive
    dynamic_quantize(const primitive_id& id,
           const input_info& input,
           const QuantizationConfig& config,
           const bool combine_scales_and_zp = false,
           const std::vector<uint64_t>& scales_zp_output_order = {})
           : primitive_base(id, {input})
           , combine_scales_and_zp(combine_scales_and_zp)
           , quantization_config(config)
           , scales_zp_output_order(scales_zp_output_order) {}

    bool combine_scales_and_zp = false;
    QuantizationConfig quantization_config;
    std::vector<uint64_t> scales_zp_output_order = {};

    size_t hash() const override {
        size_t seed = primitive::hash();
        // TODO: add more parameters
        seed = hash_range(seed, scales_zp_output_order.begin(), scales_zp_output_order.end());
        seed = hash_range(seed, quantization_config.group_sizes.begin(), quantization_config.group_sizes.end());
        seed = hash_combine(seed, combine_scales_and_zp);
        seed = hash_combine(seed, quantization_config.mode);
        // seed = hash_combine(seed, quantization_config.quantization_dt);
        // seed = hash_combine(seed, quantization_config.scale_dt);
        // seed = hash_combine(seed, quantization_config.zp_dt);

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const dynamic_quantize>(rhs);
        // TODO: add more parameters

        return scales_zp_output_order == rhs_casted.scales_zp_output_order ||
               quantization_config == rhs_casted.quantization_config;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<dynamic_quantize>::save(ob);
        // TODO: add more parameters
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<dynamic_quantize>::load(ib);
    }
};
}  // namespace cldnn
