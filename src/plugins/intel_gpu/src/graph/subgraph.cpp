// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/parameter.hpp"
#include "subgraph_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
// GPU_DEFINE_PRIMITIVE_TYPE_ID(subgraph);

template<typename ShapeType>
static std::vector<ShapeType> shape_infer(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph, kernel_impl_params const& impl_params) {
    const auto op = subgraph.get();
    const auto& input_layouts = impl_params.input_layouts;

    OPENVINO_ASSERT(op->get_input_size() == input_layouts.size());

    ov::OutputVector new_inputs;
    for (size_t i = 0; i < op->get_input_size(); ++i) {
        new_inputs.emplace_back(std::make_shared<ov::op::v0::Parameter>(op->get_input_element_type(i),
                                                                        input_layouts[i].get_partial_shape()));
    }
    auto new_subgraph = op->clone_with_new_inputs(new_inputs);
    new_subgraph->validate_and_infer_types();

    std::vector<ShapeType> output_shapes(new_subgraph->get_output_size());
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes[i] = new_subgraph->get_output_partial_shape(i).to_shape();
    }

    return output_shapes;
}

layout subgraph_inst::calc_output_layout(subgraph_node const& node, kernel_impl_params const& impl_params) {
    auto prim = impl_params.typed_desc<subgraph>();

    auto output_shapes = shape_infer<ov::Shape>(prim->ov_subgraph, impl_params);
    auto output_type = prim->output_data_types[0].value();

    OPENVINO_ASSERT(output_shapes.size() == 1);

    return { layout{output_shapes[0], output_type, format::get_default_format(output_shapes[0].size())} };
}

template<typename ShapeType>
std::vector<layout> subgraph_inst::calc_output_layouts(subgraph_node const& /*node*/, const kernel_impl_params& impl_params) {
    auto prim = impl_params.typed_desc<subgraph>();

    // std::vector<ov::PartialShape> input_shapes;
    // input_shapes.reserve(impl_params.input_layouts.size());

    // std::transform(impl_params.input_layouts.begin(), impl_params.input_layouts.end(), std::back_inserter(input_shapes), [](const layout& l) {
    //     return l.get_partial_shape();
    // });

    // auto result = prim->ov_subgraph->shape_infer(input_shapes);
    // auto output_shape = ov::PartialShape(result.dims);

    auto output_shapes = shape_infer<ShapeType>(prim->ov_subgraph, impl_params);

    auto output_layouts = std::vector<layout>{};
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        const auto& output_shape = output_shapes[i];
        const auto& output_dt = prim->output_data_types[i].value();
        const auto& output_format = format::get_default_format(output_shape.size());
        output_layouts.emplace_back(output_shape, output_dt, output_format);
    }

    return output_layouts;
}

template std::vector<layout> subgraph_inst::calc_output_layouts<ov::PartialShape>(const subgraph_node& node, const kernel_impl_params& impl_param);

std::string subgraph_inst::to_string(subgraph_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

subgraph_inst::typed_primitive_inst(network& network, subgraph_node const& node) : parent(network, node) {}

}  // namespace cldnn
