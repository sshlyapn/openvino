// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

struct scaled_dot_product_attention : public primitive_base<scaled_dot_product_attention> {
    CLDNN_DECLARE_PRIMITIVE(scaled_dot_product_attention)

    scaled_dot_product_attention() : primitive_base("", {}) {}

    /// @brief Constructs scaled_dot_product_attention primitive.
    /// @param id This primitive id.
    /// @param input Input data primitive id.
    /// @param block_shape Array of block sizes
    /// @param crops_begin Amount to crop from the beginning along each axis of data input
    /// @param crops_end Amount to crop from the ending along each axis of data input
    scaled_dot_product_attention(const primitive_id& id,
                                 const input_info& query,
                                 const input_info& key,
                                 const input_info& value,
                                 const input_info& attention_mask,
                                 const padding& output_padding = padding())
        : primitive_base(id, {query, key, value, attention_mask}, {output_padding}) {}

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scaled_dot_product_attention>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scaled_dot_product_attention>::load(ib);
    }
};
}  // namespace cldnn
