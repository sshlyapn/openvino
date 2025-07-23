// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/subgraph.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<subgraph> : public typed_program_node_base<subgraph> {
    using parent = typed_program_node_base<subgraph>;

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using subgraph_node = typed_program_node<subgraph>;

template <>
class typed_primitive_inst<subgraph> : public typed_primitive_inst_base<subgraph> {
    using parent = typed_primitive_inst_base<subgraph>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(subgraph_node const& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(subgraph_node const& node, kernel_impl_params const& impl_params);
    static std::string to_string(subgraph_node const& node);

    typed_primitive_inst(network& network, subgraph_node const& node);
};

using subgraph_inst = typed_primitive_inst<subgraph>;

}  // namespace cldnn
