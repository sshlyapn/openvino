// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/op/dynamic_quantize.hpp"
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

#include <memory>

namespace ov {
namespace intel_gpu {

namespace {
std::vector<ov::op::util::VariableInfo> get_variable_infos(const ov::op::util::VariableInfo& data_variable_info,
                                                           const ov::op::internal::QuantizationConfig& config,
                                                           const std::vector<uint64_t>& scales_zp_output_order,
                                                           const bool combine_scales_and_zp = false) {
    std::vector<ov::op::util::VariableInfo> infos;

    // add initial data variable info
    infos.push_back(data_variable_info);

    // infer DQ shapes
    ov::intel_gpu::op::DynamicQuantize dq;
    auto dq_shapes =
        ov::intel_gpu::op::DynamicQuantize::shape_infer(&dq, {data_variable_info.data_shape}, config, scales_zp_output_order, combine_scales_and_zp);

    const auto variable_id = data_variable_info.variable_id;
    const auto scale_shape = dq_shapes[1];
    const auto scale_dt = config.scale_dt;

    // add scales variable info
    infos.push_back(ov::op::util::VariableInfo{scale_shape, scale_dt, variable_id});

    if (config.is_asymmetric_quantization() && !combine_scales_and_zp) {
        // add zero points variable info
        const auto zp_dt = config.zp_dt;
        infos.push_back(ov::op::util::VariableInfo{scale_shape, zp_dt, variable_id});
    }

    std::cout << "Generated infos: " << infos.size() << "\n";

    return infos;
}
}

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

    int USE_ZP = 0;
    if (const auto env_var = std::getenv("USE_ZP")) {
        std::istringstream ss(env_var);
        ss >> USE_ZP;
    }

    std::cout << "Set USE_ZP = " << USE_ZP << "\n";

    auto quantization_mode = ov::op::internal::QuantizationConfig::QuantizationMode::Symmetric;
    if (USE_ZP)
        quantization_mode = ov::op::internal::QuantizationConfig::QuantizationMode::Asymmetric;


    bool combine_scales_and_zp = quantization_mode == ov::op::internal::QuantizationConfig::QuantizationMode::Asymmetric;

    int ZP_INPUT = 0;
    if (const auto env_var = std::getenv("ZP_INPUT")) {
        std::istringstream ss(env_var);
        ss >> ZP_INPUT;
    }

    if (ZP_INPUT && combine_scales_and_zp) {
        std::cout << "Use independent ZP INPUT\n";
        combine_scales_and_zp = false;
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

        auto query_node = pattern_map.at(query).get_node_shared_ptr();;

        auto k_new_token_node = pattern_map.at(k_new_token).get_node_shared_ptr();
        auto key_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(key).get_node_shared_ptr());
        auto value_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(value).get_node_shared_ptr());
        auto org_sdpa = std::dynamic_pointer_cast<ov::intel_gpu::op::IndirectSDPA>(pattern_map.at(present).get_node_shared_ptr());

        auto key_past_node = std::dynamic_pointer_cast<ov::intel_gpu::op::ReadValue>(pattern_map.at(k_past).get_node_shared_ptr());
        auto value_past_node = std::dynamic_pointer_cast<ov::intel_gpu::op::ReadValue>(pattern_map.at(v_past).get_node_shared_ptr());

        auto data_rank = key_node->get_input_partial_shape(0).size();
        auto get_shape_group_sizes = [&](const std::vector<int64_t>& transposed_order) {
            std::vector<uint64_t> group_sizes(data_rank, 1);
            std::vector<int64_t> order = transposed_order;
            if (transposed_order.size() != data_rank) {
                order.resize(data_rank);
                std::iota(order.begin(), order.end(), 0);
            }

            group_sizes[order[data_rank - 1]] = UINT64_MAX;
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->enable_kv_cache_compression != 1) { // per-token compression
                group_sizes[order[1]] = UINT64_MAX;
            }

            return group_sizes;
        };

        auto get_scales_output_order = [&](const std::vector<int64_t>& transposed_order) {
            // Reorder scales in static order: [batch, num_heads, seq_len, head_size]
            std::vector<uint64_t> scales_zp_output_order(data_rank);
            scales_zp_output_order[0] = transposed_order[0];
            scales_zp_output_order[1] = transposed_order[1];
            scales_zp_output_order[2] = transposed_order[2];
            scales_zp_output_order[3] = transposed_order[3];

            return scales_zp_output_order;
        };

        auto key_variable = key_past_node->get_variable();
        key_variable->update_data_type(element::i8);

        auto value_variable = value_past_node->get_variable();
        value_variable->update_data_type(element::i8);

        auto print_arr = [&](const std::vector<uint64_t>& vec, size_t max_len) {
            std::stringstream ss;
            for (size_t i = 0; i < std::min(max_len, vec.size()); i++) {
                ss << vec[i] << ", ";
            }

            return ss.str();
        };

        // Rename this
        auto group_sizes = get_shape_group_sizes(org_sdpa->get_input1_transpose_order());
        auto scales_zp_output_order = get_scales_output_order(org_sdpa->get_input1_transpose_order());

        // Use single buffer for scales and zero points

        ov::op::internal::QuantizationConfig config;
        config.mode = quantization_mode;
        config.group_sizes = group_sizes;
        config.quantization_dt = element::i8;
        config.scale_dt = query_node->get_output_element_type(0);

        if (config.is_asymmetric_quantization())
            config.zp_dt = query_node->get_output_element_type(0);


        std::cout << "pattern matched! " << org_sdpa->get_friendly_name() << "; "
                    << "groups: " << print_arr(group_sizes, group_sizes.size()) << "; "
                    << "scales_order: " << print_arr(scales_zp_output_order, scales_zp_output_order.size()) << std::endl;

        auto replace_read_value_node = [](const std::shared_ptr<Node>& target,
                                            const std::shared_ptr<Node>& replacement) {
            target->output(0).replace(replacement->output(0));
        };

        if (key_past_node->get_input_size() == 0) {
            auto variable_infos = get_variable_infos(key_past_node->get_variable()->get_info(), config, scales_zp_output_order, combine_scales_and_zp);
            auto new_key_past_node =
                std::make_shared<ov::intel_gpu::op::ReadValues>(key_past_node->get_variable(),
                                                                variable_infos);

            ov::copy_runtime_info(key_past_node, new_key_past_node);
            replace_read_value_node(key_past_node, new_key_past_node);

            key_past_node = new_key_past_node;
        } else {
            auto variable_infos = get_variable_infos(key_past_node->get_variable()->get_info(), config, scales_zp_output_order, combine_scales_and_zp);
            auto k_init_dyn_quantization =
                std::make_shared<ov::intel_gpu::op::DynamicQuantize>(key_past_node->get_input_node_shared_ptr(0),
                                                                     config,
                                                                     scales_zp_output_order,
                                                                     combine_scales_and_zp);

            OutputVector read_value_initializers = {k_init_dyn_quantization->output(0), k_init_dyn_quantization->output(1)};
            if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
                read_value_initializers.push_back(k_init_dyn_quantization->output(2));

            auto new_key_past_node =
                std::make_shared<ov::intel_gpu::op::ReadValues>(read_value_initializers,
                                                                key_past_node->get_variable(),
                                                                variable_infos);

            k_init_dyn_quantization->set_friendly_name(key_node->get_friendly_name() + "_init_dyn_quan");
            ov::copy_runtime_info(key_past_node, new_key_past_node);
            replace_read_value_node(key_past_node, new_key_past_node);

            key_past_node = new_key_past_node;
        }

        if (value_past_node->get_input_size() == 0) {
            auto variable_infos = get_variable_infos(value_past_node->get_variable()->get_info(), config, scales_zp_output_order, combine_scales_and_zp);
            auto new_value_past_node =
                std::make_shared<ov::intel_gpu::op::ReadValues>(value_past_node->get_variable(),
                                                                variable_infos);

            ov::copy_runtime_info(value_past_node, new_value_past_node);
            replace_read_value_node(value_past_node, new_value_past_node);

            value_past_node = new_value_past_node;
        } else {
            auto variable_infos = get_variable_infos(value_past_node->get_variable()->get_info(), config, scales_zp_output_order, combine_scales_and_zp);
            auto v_init_dyn_quantization =
                std::make_shared<ov::intel_gpu::op::DynamicQuantize>(value_past_node->get_input_node_shared_ptr(0),
                                                                     config,
                                                                     scales_zp_output_order,
                                                                     combine_scales_and_zp);

            OutputVector read_value_initializers = {v_init_dyn_quantization->output(0), v_init_dyn_quantization->output(1)};
            if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
                read_value_initializers.push_back(v_init_dyn_quantization->output(2));


            auto new_value_past_node =
                std::make_shared<ov::intel_gpu::op::ReadValues>(read_value_initializers,
                                                                value_past_node->get_variable(),
                                                                variable_infos);

            v_init_dyn_quantization->set_friendly_name(value_node->get_friendly_name() + "_init_dyn_quan");
            ov::copy_runtime_info(value_past_node, new_value_past_node);
            replace_read_value_node(value_past_node, new_value_past_node);

            value_past_node = new_value_past_node;
        }

        OutputVector key_kv_cache_inputs = { key_past_node->output(0),
                                             key_node->get_input_node_shared_ptr(1),
                                             key_node->get_input_node_shared_ptr(2),
                                             key_past_node->output(1) };

        if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
            key_kv_cache_inputs.push_back(key_past_node->output(2));

        // FIXME: need to tell whether it is direct KV cache or indirect kv cache
        auto new_kv_cache_k = std::make_shared<op::KVCache>(key_kv_cache_inputs,
                                                            key_node->get_variable(),
                                                            key_node->get_concat_axis(),
                                                            key_node->get_gather_axis(),
                                                            combine_scales_and_zp,
                                                            config,
                                                            scales_zp_output_order);

        // new_kv_cache_k->set_asymmetric_quantization(quantization_mode == ov::op::internal::QuantizationConfig::QuantizationMode::Asymmetric);

        new_kv_cache_k->set_friendly_name(key_node->get_friendly_name());
        ov::copy_runtime_info(key_node, new_kv_cache_k);

        // auto v_dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(value_node->get_input_node_shared_ptr(1), group_sizes, element::f16, scales_zp_output_order);
        // v_dyn_quan->set_friendly_name("dyn_quan_value");
        // FIXME: need to tell whether it is direct KV cache or indirect kv cache
        OutputVector value_kv_cache_inputs = { value_past_node->output(0),
                                               value_node->get_input_node_shared_ptr(1),
                                               value_node->get_input_node_shared_ptr(2),
                                               value_past_node->output(1) };

        if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
            value_kv_cache_inputs.push_back(value_past_node->output(2));

        auto new_kv_cache_v = std::make_shared<op::KVCache>(value_kv_cache_inputs,
                                                            value_node->get_variable(),
                                                            value_node->get_concat_axis(),
                                                            value_node->get_gather_axis(),
                                                            combine_scales_and_zp,
                                                            config,
                                                            scales_zp_output_order);

        // new_kv_cache_v->set_asymmetric_quantization(quantization_mode == ov::op::internal::QuantizationConfig::QuantizationMode::Asymmetric);

        new_kv_cache_v->set_friendly_name(value_node->get_friendly_name());
        ov::copy_runtime_info(value_node, new_kv_cache_v);

        // FIXME: output port from new_kv_cache_k is fixed. compression and indirectness is orthogonal.
        OutputVector sdpa_inputs;
        // QKV -- attention_mask -- input_scale -- key_scale -- beam_idx
        for (size_t i = 0; i < org_sdpa->get_input_size() - 1; i++) {
            sdpa_inputs.push_back(org_sdpa->get_input_node_shared_ptr(i));
        }

        sdpa_inputs[1] = new_kv_cache_k->output(0);         // compressed K
        sdpa_inputs[2] = new_kv_cache_v->output(0);         // compressed V

        sdpa_inputs.push_back(new_kv_cache_k->output(2));   // scale for compressed K
        sdpa_inputs.push_back(new_kv_cache_v->output(2));   // scale for compressed V

        if (config.is_asymmetric_quantization() && !combine_scales_and_zp) {
            sdpa_inputs.push_back(new_kv_cache_k->output(3));   // zp for compressed K
            sdpa_inputs.push_back(new_kv_cache_v->output(3));   // zp for compressed V
        }

        auto input0_transpose_order = org_sdpa->get_input0_transpose_order();
        auto input1_transpose_order = org_sdpa->get_input1_transpose_order();
        auto input2_transpose_order = org_sdpa->get_input2_transpose_order();
        auto output_transpose_order = org_sdpa->get_output_transpose_order();

        std::cout << "Indirect axis " << org_sdpa->get_indirect_axis() << "\n";

        auto new_sdpa = std::make_shared<op::IndirectSDPA>(sdpa_inputs,
                                                           new_kv_cache_k->output(1),
                                                           org_sdpa->get_causal(),
                                                           org_sdpa->get_indirect_axis(),
                                                           input0_transpose_order,
                                                           input1_transpose_order,
                                                           input2_transpose_order,
                                                           output_transpose_order,
                                                           config,
                                                           combine_scales_and_zp,
                                                           org_sdpa->get_output_type());

        new_kv_cache_k->set_friendly_name(key_node->get_friendly_name());
        ov::copy_runtime_info(key_node, new_kv_cache_k);

        new_kv_cache_v->set_friendly_name(value_node->get_friendly_name());
        ov::copy_runtime_info(value_node, new_kv_cache_v);

        new_sdpa->set_friendly_name(org_sdpa->get_friendly_name());
        ov::copy_runtime_info(org_sdpa, new_sdpa);

        ov::replace_node(org_sdpa, new_sdpa);
        return true;
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
