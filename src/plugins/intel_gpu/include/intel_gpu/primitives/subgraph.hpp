// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "snippets/op/subgraph.hpp"
#include "primitive.hpp"

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
    subgraph(const primitive_id& id, const std::vector<input_info>& inputs, const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph)
        : primitive_base(id, inputs), ov_subgraph(subgraph->clone()) {}

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
