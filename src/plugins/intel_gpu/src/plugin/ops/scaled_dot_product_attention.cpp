// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/scaled_dot_product_attention.hpp"

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"

namespace ov {
namespace intel_gpu {

static void CreateScaledDotProductAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& op) {
    std::cout << "Create SDPA " << op->get_friendly_name() << ", inputs_num=" << op->get_input_size() << "\n";
    validate_inputs_count(op, {/* 3, */ 4, /* 5 */});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs[0],
                                                         inputs[1],
                                                         inputs[2],
                                                         inputs[3]);

    p.add_primitive(*op, sdpa_prim);
}

REGISTER_FACTORY_IMPL(v13, ScaledDotProductAttention);

}  // namespace intel_gpu
}  // namespace ov
