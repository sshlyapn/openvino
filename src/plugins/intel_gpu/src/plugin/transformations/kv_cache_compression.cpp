// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"
#include <memory>

#include "intel_gpu/op/kv_cache.hpp"

#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"

#include "ov_ops/dynamic_quantize.hpp"

namespace ov {
namespace intel_gpu {

class KVCacheCompressionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("KVCacheCompressionMatcher", "0");
    KVCacheCompressionMatcher();
};

KVCacheCompressionMatcher::KVCacheCompressionMatcher() {
    using namespace ov::pass::pattern;

    bool first = true;

    int KV_CACHE_COMP = 0;
    if (const auto env_var = std::getenv("KV_CACHE_COMP")) {
        std::istringstream ss(env_var);
        ss >> KV_CACHE_COMP;
    }

    if (KV_CACHE_COMP == 0) {
        if (first) {
            printf("NO_KV_CACHE_COMP\n");
        }
        first = false;
        return;
    } else {
        if (first)
            printf("YES_KV_CACHE_COMP\n");

        first = false;
    }

    auto query = any_input();

    auto k_past = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto k_new_token = any_input();
    auto k_beam_idx = any_input();
    auto key = wrap_type<ov::intel_gpu::op::KVCache>({k_past, k_new_token, k_beam_idx});

    auto v_past = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto v_new_token = any_input();
    auto v_beam_idx = any_input();
    auto value = wrap_type<ov::intel_gpu::op::KVCache>({v_past, v_new_token, v_beam_idx});

    auto input_attn_mask = any_input();
    auto input_scale = any_input();

    auto present = wrap_type<ov::intel_gpu::op::IndirectSDPA>({query, key, value, input_attn_mask, input_scale});

    // k, v, attention_mask, scale
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();

        auto k_new_token_node = pattern_map.at(k_new_token).get_node_shared_ptr();
        auto key_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(key).get_node_shared_ptr());
        auto value_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(value).get_node_shared_ptr());
        auto org_sdpa = std::dynamic_pointer_cast<ov::intel_gpu::op::IndirectSDPA>(pattern_map.at(present).get_node_shared_ptr());

        auto key_past_node = std::dynamic_pointer_cast<ov::intel_gpu::op::ReadValue>(pattern_map.at(k_past).get_node_shared_ptr());
        auto value_past_node = std::dynamic_pointer_cast<ov::intel_gpu::op::ReadValue>(pattern_map.at(v_past).get_node_shared_ptr());

        if (true
            // || org_sdpa->get_friendly_name().find(".h.0.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.1.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.2.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.3.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.4.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.5.") != std::string::npos
            ) {
            std::cout << "pattern matched! " << org_sdpa->get_friendly_name() << std::endl;
            auto rank = key_node->get_input_partial_shape(0).size();
            auto get_shape_group_sizes = [&](const std::vector<int64_t>& transposed_order) {
                std::vector<uint64_t> shape_group_size(rank, 1);
                std::vector<int64_t> order = transposed_order;
                if (transposed_order.size() != rank) {
                    order.resize(rank);
                    std::iota(order.begin(), order.end(), 0);
                }

                shape_group_size[order[rank - 1]] = UINT64_MAX;
                GPU_DEBUG_GET_INSTANCE(debug_config);
                GPU_DEBUG_IF(debug_config->enable_kv_cache_compression != 1) { // per-token compression
                    shape_group_size[order[1]] = UINT64_MAX;
                }

                return shape_group_size;
            };

            auto get_scales_output_order = [&](const std::vector<int64_t>& transposed_order) {
                std::vector<uint64_t> scales_output_order(rank, 1);
                scales_output_order[0] = transposed_order[0];
                scales_output_order[1] = transposed_order[3];
                scales_output_order[2] = transposed_order[2];
                scales_output_order[3] = transposed_order[1];

                return scales_output_order;
            };

            auto key_variable = key_past_node->get_variable();
            key_variable->update_data_type(element::i8);

            auto value_variable = value_past_node->get_variable();
            value_variable->update_data_type(element::i8);

            auto print_arr = [&](const std::vector<uint64_t>& vec, size_t max_len, std::string name) {
                std::stringstream ss;
                for (size_t i = 0; i < std::min(max_len, vec.size()); i++) {
                    ss << vec[i] << ", ";
                }
                std::cout << "Array " << name << " (len=" << vec.size() << ") content: " << ss.str() << "\n";
            };

            auto shape_group_size = get_shape_group_sizes(org_sdpa->get_input1_transpose_order());
            print_arr(shape_group_size, shape_group_size.size(), "shape_group_size");

            auto scales_output_order = get_scales_output_order(org_sdpa->get_input1_transpose_order());
            print_arr(scales_output_order, scales_output_order.size(), "scales_output_order");

            auto replace_read_value_node = [](const std::shared_ptr<Node>& target,
                                              const std::shared_ptr<Node>& replacement) {
                target->output(0).replace(replacement->output(0));

                // replacement->add_node_control_dependents(target);
                // replacement->add_node_control_dependencies(target);
                // target->clear_control_dependents();
            };

            // llama2-7b
            // indirect : 1,
            // gather axis : 0,
            // compressed : 1,
            // concat axis : 2,
            // variable shape : [?,32,?,128],
            // k_order: 0, 1, 2, 3
            // shape_group_size init: 1, 1, 1, 1
            // shape_group_size applied TOKEN: 1, MAX, 1, MAX
            // shape_group_size applied HEAD: 1, 1, 1, MAX
            // sizes TOKEN: [1, 1, 8, 1] + [1, 1, 1, 1] = [1, 1, 9, 1]
            // sizes HEAD: [1, 32, 8, 1] + [1, 32, 1, 1] = [1, 32, 9, 1]
            // GWS TOKEN: BATCH(1) * 1, Y(concat_axis), 1
            // GWS PER HEAD: BATCH(1) * HEADS_NUM(32), Y(concat_axis), 1

            // Scales order: 0, 3, 2, 1


            // qwen
            // indirect : 1,
            // gather axis : 0,
            // compressed : 1,
            // concat axis : 1,
            // variable shape : [?,?,32,128],
            // k_oder: 0, 2, 1, 3
            // shape_group_size init: 1, 1, 1, 1
            // shape_group_size applied TOKEN: 1, 1, MAX, MAX
            // shape_group_size applied HEAD: 1, 1, 1, MAX
            // GWS TOKEN: BATCH * FEATURE, 1, 1
            // GWS HEAD: BATCH * FEATURE, HEADS_NUM, 1

            // Scales order: 0, 3, 1, 2

            if (key_past_node->get_input_size() == 1) {
                auto k_init_dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(key_past_node->get_input_node_shared_ptr(0), shape_group_size, element::f16, scales_output_order);
                auto new_key_past_node = std::make_shared<ov::intel_gpu::op::CompressedReadValue>(k_init_dyn_quan->output(0), k_init_dyn_quan->output(1), key_past_node->get_variable());
                k_init_dyn_quan->set_friendly_name(key_node->get_friendly_name() + "_init_dyn_quan");
                std::cout << "Key outputs: " << key_past_node->get_output_size() << " " << new_key_past_node->get_output_size() << "\n";
                ov::copy_runtime_info(key_past_node, new_key_past_node);

                // TODO: Old ReadValue node is kept in the graph and goes to ShapeOf - this needs to be fixed
                replace_read_value_node(key_past_node, new_key_past_node);
                // ov::replace_node(key_past_node, new_key_past_node);

                key_past_node = new_key_past_node;
            }

            if (value_past_node->get_input_size() == 1) {
                auto v_init_dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(value_past_node->get_input_node_shared_ptr(0), shape_group_size, element::f16, scales_output_order);
                auto new_value_past_node = std::make_shared<ov::intel_gpu::op::CompressedReadValue>(v_init_dyn_quan->output(0), v_init_dyn_quan->output(1), value_past_node->get_variable());

                std::cout << "Value outputs: " << value_past_node->get_output_size() << " " << new_value_past_node->get_output_size() << "\n";

                v_init_dyn_quan->set_friendly_name(value_node->get_friendly_name() + "_init_dyn_quan");
                ov::copy_runtime_info(value_past_node, new_value_past_node);
                replace_read_value_node(value_past_node, new_value_past_node);
                // ov::replace_node(value_past_node, new_value_past_node);

                value_past_node = new_value_past_node;
            }

            auto k_dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(key_node->get_input_node_shared_ptr(1), shape_group_size, element::f16, scales_output_order);
            k_dyn_quan->set_friendly_name("dyn_quan_key");

            // FIXME: need to tell whether it is direct KV cache or indirect kv cache
            auto new_kv_cache_k = std::make_shared<op::KVCache>(key_past_node->output(0),
                                                                k_dyn_quan->output(0),
                                                                key_node->get_input_node_shared_ptr(2),
                                                                key_past_node->output(1),
                                                                k_dyn_quan->output(1),
                                                                key_node->get_variable(),
                                                                key_node->get_concat_axis(),
                                                                key_node->get_gather_axis());

            new_kv_cache_k->set_friendly_name(key_node->get_friendly_name());
            ov::copy_runtime_info(key_node, new_kv_cache_k);

            auto v_dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(value_node->get_input_node_shared_ptr(1), shape_group_size, element::f16, scales_output_order);
            v_dyn_quan->set_friendly_name("dyn_quan_value");
            // FIXME: need to tell whether it is direct KV cache or indirect kv cache
            auto new_kv_cache_v = std::make_shared<op::KVCache>(value_past_node->output(0),
                                                                v_dyn_quan->output(0),
                                                                value_node->get_input_node_shared_ptr(2),
                                                                value_past_node->output(1),
                                                                v_dyn_quan->output(1),
                                                                value_node->get_variable(),
                                                                value_node->get_concat_axis(),
                                                                value_node->get_gather_axis());

            new_kv_cache_v->set_friendly_name(value_node->get_friendly_name());
            ov::copy_runtime_info(value_node, new_kv_cache_v);

            // FIXME: output port from new_kv_cache_k is fixed. compression and indirectness is orthogonal.
            OutputVector sdpa_inputs;
            // QKV -- attention_mask -- input_scale -- key_scale -- beam_idx
            for (size_t i = 0; i < org_sdpa->get_input_size() - 1; i++)
                sdpa_inputs.push_back(org_sdpa->get_input_node_shared_ptr(i));
            sdpa_inputs[1] = new_kv_cache_k->output(0);         // compressed K
            sdpa_inputs[2] = new_kv_cache_v->output(0);         // compressed V
            sdpa_inputs.push_back(new_kv_cache_k->output(2));   // scale for compressed K
            sdpa_inputs.push_back(new_kv_cache_v->output(2));   // scale for compressed V

            auto input0_transpose_order = org_sdpa->get_input0_transpose_order();
            auto input1_transpose_order = org_sdpa->get_input1_transpose_order();
            auto input2_transpose_order = org_sdpa->get_input2_transpose_order();
            auto output_transpose_order = org_sdpa->get_output_transpose_order();

            auto print_arr2 = [&](const std::vector<int64_t>& vec, size_t max_len, std::string name) {
                std::stringstream ss;
                for (size_t i = 0; i < std::min(max_len, vec.size()); i++) {
                    ss << vec[i] << ", ";
                }
                std::cout << "-> Array " << name << " (len=" << vec.size() << ") content: " << ss.str() << "\n";
            };

            print_arr2(input0_transpose_order, input0_transpose_order.size(), "input0_transpose_order");
            print_arr2(input1_transpose_order, input1_transpose_order.size(), "input1_transpose_order");
            print_arr2(input2_transpose_order, input2_transpose_order.size(), "input2_transpose_order");
            print_arr2(output_transpose_order, output_transpose_order.size(), "output_transpose_order");


            auto new_sdpa = std::make_shared<op::IndirectSDPA>(sdpa_inputs,
                                                               new_kv_cache_k->output(1),
                                                               org_sdpa->get_causal(),
                                                               true /* kv_compressed */,
                                                               org_sdpa->get_indirect_axis(),
                                                               input0_transpose_order,
                                                               input1_transpose_order,
                                                               input2_transpose_order,
                                                               output_transpose_order,
                                                               org_sdpa->get_output_type());


            new_kv_cache_k->set_friendly_name(key_node->get_friendly_name());
            ov::copy_runtime_info(key_node, new_kv_cache_k);

            new_kv_cache_v->set_friendly_name(value_node->get_friendly_name());
            ov::copy_runtime_info(value_node, new_kv_cache_v);

            new_sdpa->set_friendly_name(org_sdpa->get_friendly_name());
            ov::copy_runtime_info(org_sdpa, new_sdpa);

            ov::replace_node(org_sdpa, new_sdpa);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(present, "KVCacheCompressionMatcher");
    this->register_matcher(m, callback);

}

bool KVCacheCompression::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool res = pass::GraphRewrite::run_on_model(m);
    std::cout << "KVCacheCompression res=" << res << "\n";



    // TODO: seems it's not needed and copied from kvcache
    if (res) {
        ov::SinkVector sinks = m->get_sinks();
        std::cout << "KVCacheCompression remove sinks " << sinks.size() << "\n";
        for (auto& sink : sinks) {
            if (sink && sink->get_input_node_ptr(0)->get_type_info() == op::KVCache::get_type_info_static()) {
                std::cout << "Remove " << sink->get_friendly_name() << ", kvcache=" << sink->get_input_node_ptr(0)->get_friendly_name() << "\n";
                m->remove_sink(sink);
            }
        }
    }

    return res;
}

KVCacheCompression::KVCacheCompression() {
    add_matcher<ov::intel_gpu::KVCacheCompressionMatcher>();
}

}  // namespace intel_gpu
}  // namespace ov
