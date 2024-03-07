// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "paged_attention_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <string>
#include <sstream>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_attention)


layout paged_attention_inst::calc_output_layout(const paged_attention_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> paged_attention_inst::calc_output_layouts(paged_attention_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<paged_attention>();

    auto out_layout = impl_param.get_input_layout(0);

    return {out_layout};
}

template std::vector<layout> paged_attention_inst::calc_output_layouts<ov::PartialShape>(paged_attention_node const& node, const kernel_impl_params& impl_param);

std::string paged_attention_inst::to_string(const paged_attention_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite custom_gpu_prim_info;
    node_info->add("paged attention primitive info", custom_gpu_prim_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void paged_attention_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    parent::update_shape_info_tensor(params);
    // mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    // auto shape_info_ptr = lock.data();
    // size_t offset = 0;

    // auto query = params.get_input_layout(0);
    // auto key = params.get_input_layout(1);
    // auto value = params.get_input_layout(2);
    // auto key_cache = params.get_input_layout(3);
    // auto value_cache = params.get_input_layout(4);
    // auto slot_mapping = params.get_input_layout(6);

    // // query_shape = [batch_size, seq_len, num_heads * head_size]
    // // key_shape, value_shape = [batch_size, seq_len, num_kv_heads * head_size]
    // // key_cache_shape = [num_blocks, num_kv_heads, head_size/x, block_size, x]
    // // value_cache_shape = [num_blocks, num_kv_heads, head_size, block_size]
    // const auto query_shape = query.get_shape();
    // const auto key_shape = key.get_shape();
    // const auto key_cache_shape = key_cache.get_shape();
    // const auto value_cache_shape = value_cache.get_shape();
    // const size_t batch_size = query_shape[0];
    // const size_t seq_len = query_shape[1];
    // const size_t hidden_size = query_shape[2];
    // const size_t num_kv_heads = value_cache_shape[1];
    // const size_t head_size = value_cache_shape[2];
    // const size_t num_heads = hidden_size / head_size;
    // const size_t block_size = value_cache_shape[3];
    // const size_t x = key_cache_shape[4];
    // const size_t num_tokens = key_shape[0];

    // // Reshape from [batch_size, seq_len, num_heads * head_size] to [batch_size, seq_len, num_heads, head_size]
    // query.set_partial_shape({batch_size, seq_len, num_heads, head_size});
    // key.set_partial_shape({batch_size, seq_len, num_kv_heads, head_size});
    // value.set_partial_shape(key.get_shape());

    // std::vector<std::pair<layout, layout>> input_layouts;
    // for (size_t i = 0; i < _node->get_dependencies().size(); i++) {
    //     const auto& node_in_lay = _node->get_input_layout(i);
    //     const auto& runtime_in_lay = params.input_layouts[i];

    //     input_layouts.emplace_back(runtime_in_lay, node_in_lay);
    // }

    // for (size_t i = 0; i < input_layouts.size(); i++) {
    //     GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
    //     fill_shape_info_data(input_layouts[i].first, input_layouts[i].second, shape_info_ptr, offset);
    // }

    // for (size_t i = 0; i < _node->get_output_layouts().size(); i++) {
    //     GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
    //     const auto& node_out_lay = _node->get_output_layout(i);
    //     const auto& runtime_out_lay = params.output_layouts[i];
    //     fill_shape_info_data(runtime_out_lay, node_out_lay, shape_info_ptr, offset);
    // }
}

paged_attention_inst::typed_primitive_inst(network& network, const paged_attention_node& node)
    : parent(network, node)
    , prefill_network(network::allocate_network(network.get_stream_ptr(),
                                                node.get_primitive()->prefill_stage,
                                                false,
                                                network.is_primary_stream())) { }
}  // namespace cldnn
