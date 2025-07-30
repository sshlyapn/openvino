// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "snippets/op/subgraph.hpp"
#include "primitive.hpp"

#include "ocl/ocl_engine.hpp"
#include "jit/gpu_generator.hpp"

namespace cldnn {

/// @brief Subgraph primitive
/// @details Represents snippets configured subgraph
struct subgraph : public primitive_base<subgraph> {
    CLDNN_DECLARE_PRIMITIVE(subgraph);

    subgraph() : primitive_base("", {}) {}

    /// @brief Constructs subgraph primitive
    /// @param id This primitive id
    /// @param inputs Input primitive ids
    /// @param subgraph Original subgraph node
    /// @param eng ocl engine
    subgraph(const primitive_id& id, const std::vector<input_info>& inputs,
             const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph, const cldnn::engine& eng)
        : primitive_base(id, inputs), ov_subgraph(subgraph->clone()) {
        ngen::HW hw;
        switch (eng.get_device()->get_info().arch) {
        case gpu_arch::gen9: hw = ngen::HW::Gen9; break;
        case gpu_arch::gen11: hw = ngen::HW::Gen11; break;
        case gpu_arch::xe_lp: hw = ngen::HW::XeLP; break;
        case gpu_arch::xe_hp: hw = ngen::HW::XeHP; break;
        case gpu_arch::xe_hpg: hw = ngen::HW::XeHPG; break;
        case gpu_arch::xe_hpc: hw = ngen::HW::XeHPC; break;
        case gpu_arch::xe2: hw = ngen::HW::Xe2; break;
        case gpu_arch::xe3: hw = ngen::HW::Xe3; break;
        case gpu_arch::unknown: hw = ngen::HW::Unknown; break;
        default:
            OPENVINO_THROW("Unexpected arch");
        }
        ov_subgraph->set_generator(std::make_shared<ov::intel_gpu::jit::GPUGenerator>(hw));
    }

    std::shared_ptr<ov::snippets::op::Subgraph> ov_subgraph;

    size_t hash() const override {
        size_t seed = primitive::hash();
        // TODO: implement hash
        seed = hash_combine(seed, id);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const subgraph>(rhs);
        // TODO: compare actual parameters
        return id == rhs_casted.id;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<subgraph>::save(ob);
        OPENVINO_THROW("[GPU] Subgraph primitive doesn't support caching");
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<subgraph>::load(ib);
        OPENVINO_THROW("[GPU] Subgraph primitive doesn't support caching");
    }
};
}  // namespace cldnn
