// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "dynamic_quantize_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(dynamic_quantize);

layout dynamic_quantize_inst::calc_output_layout(dynamic_quantize_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();
    auto output_type = data_types::i8;
    auto output_format = input_layout.format;

    return layout(output_type, output_format, input_layout.get_tensor());
}

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::__calc_output_layouts(const layout &act_layout, uint64_t group_size) {
    ov::op::internal::DynamicQuantize op;
    auto output_format = act_layout.format;

    std::vector<ShapeType> input_shapes = {
        act_layout.get<ShapeType>(),
    };

    std::vector<uint64_t> shape_group_size(act_layout.get<ShapeType>().size(), 1);
    shape_group_size[act_layout.get<ShapeType>().size() - 1] = group_size;

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->enable_kv_cache_compression != 1) { // per-token compression
        shape_group_size[act_layout.get<ShapeType>().size() - 2] = group_size;
    }

    // auto print_arr = [&](const std::vector<uint64_t>& vec, size_t max_len, std::string name) {
    //     std::stringstream ss;
    //     for (size_t i = 0; i < std::min(max_len, vec.size()); i++) {
    //         ss << vec[i] << ", ";
    //     }
    //     std::cout << "Array " << name << " (len=" << vec.size() << ") content: " << ss.str() << "\n";
    // };
    // print_arr(shape_group_size, shape_group_size.size(), "shape_group_size");

    auto output_shapes = ov::op::internal::DynamicQuantize::shape_infer(&op, input_shapes, shape_group_size);
    GPU_DEBUG_TRACE_DETAIL << "shape infer dynamic" << output_shapes[0] << " " << output_shapes[1] << "\n";

    return { layout(output_shapes[0], data_types::i8, output_format), layout(output_shapes[1], data_types::f16, output_format) };
}

template std::vector<layout> dynamic_quantize_inst::__calc_output_layouts<ov::PartialShape>(const layout &act_layout, uint64_t group_size);

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::calc_output_layouts(dynamic_quantize_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();
    return __calc_output_layouts<ov::PartialShape>(input_layout, UINT64_MAX /* TODO: handle group_size here */);
}

template std::vector<layout> dynamic_quantize_inst::calc_output_layouts<ov::PartialShape>(dynamic_quantize_node const& node,
                                                                                const kernel_impl_params& impl_param);

std::string dynamic_quantize_inst::to_string(dynamic_quantize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite dynamic_quantize_info;
    dynamic_quantize_info.add("group size", desc->group_size);
    dynamic_quantize_info.add("activation dt", desc->get_output_data_type(0).value_or(data_types::undefined));
    dynamic_quantize_info.add("scale dt", desc->get_output_data_type(1).value_or(data_types::undefined));

    node_info->add("dynamic_quantize info", dynamic_quantize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

dynamic_quantize_inst::typed_primitive_inst(network& network, dynamic_quantize_node const& node) : parent(network, node) {}

}  // namespace cldnn
