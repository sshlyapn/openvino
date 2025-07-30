// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/subgraph.hpp"

#include "snippets/op/subgraph.hpp"

namespace ov::op::snippets {
    using Subgraph = ov::snippets::op::Subgraph;
}

namespace ov::intel_gpu {

static void CreateSubgraphOp(ProgramBuilder& p, const std::shared_ptr<ov::op::snippets::Subgraph>& op) {
    const auto& inputs = p.GetInputInfo(op);
    const auto& primitive_name = layer_type_name_ID(op);

    auto prim = cldnn::subgraph(primitive_name, inputs, op, p.get_engine());
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(snippets, Subgraph);

}  // namespace ov::intel_gpu
