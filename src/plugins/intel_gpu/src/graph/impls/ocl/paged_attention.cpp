// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "multi_stage_primitive.hpp"

#include "paged_attention_inst.h"
#include "paged_attention/paged_attention_kernel_selector.hpp"
#include "paged_attention/kv_cache_update_kernel_ref.hpp"

namespace cldnn {
namespace ocl {

struct paged_attention_impl : multi_stage_primitive<paged_attention> {
    using parent = multi_stage_primitive<paged_attention>;
    using parent::parent;
    using kv_cache_update_kernel_selector_t = kernel_selector::kv_cache_update_kernel_selector;
    using kv_cache_update_kernel_params_t = kernel_selector::kv_cache_update_update_params;

    using sdpa_kernel_selector_t = kernel_selector::sdpa_kernel_selector;
    using sdpa_kernel_params_t = kernel_selector::kv_cache_update_kernel_selector;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::paged_attention_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<paged_attention_impl>(*this);
    }

    enum Stage {
        concat,
        sdpa
    };

    cldnn::memory::ptr beam_table_prev = nullptr;
    cldnn::memory::ptr beam_table_new = nullptr;

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            OPENVINO_THROW("[GPU] Unimplemented load func");
            // auto& kernel_selector = kv_cache_update_kernel_selector_t::Instance();
            // auto kernel_impl = kernel_selector.GetImplementation(_kernels_data[concat_stage].kernelName);
            // kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[concat_stage]);
            // if (_kernels_data.size() == 2) {
            //     auto& bt_kernel_selector = sdpa_kernel_selector_t::Instance();
            //     auto bt_kernel_impl = bt_kernel_selector.GetImplementation(_kernels_data[beam_table_stage].kernelName);
            //     bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[beam_table_stage]);
            // }
        }
    }
    void set_arguments_impl(paged_attention_inst& instance) override {}

    kernel_arguments_data get_arguments(const paged_attention_inst& instance, size_t stage) const override {
        kernel_arguments_data args;
        args.shape_info = instance.shape_info_memory_ptr();
        if (stage == Stage::concat) {
            args.inputs = { instance.input_memory_ptr(1),
                            instance.input_memory_ptr(2),
                            instance.input_memory_ptr(6) };
            args.outputs = { instance.input_memory_ptr(3), instance.input_memory_ptr(4) };
        } else if (stage == Stage::sdpa) {
            args.inputs = { beam_table_prev, instance.input_memory_ptr(2) };
            args.outputs = { beam_table_new };
        }

        return args;
    }

    void execute_stage(const std::vector<event::ptr>& events, paged_attention_inst& instance, std::vector<event::ptr>& all_events, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        size_t kernel_offset = 0;
        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto args = get_arguments(instance, stage);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, paged_attention_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        std::vector<event::ptr> res_events;

        execute_stage(events, instance, res_events, Stage::concat);

        return aggregate_events(res_events, stream, res_events.size() > 1);
    }

    static layout get_beam_table_layout(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<paged_attention>();
        auto kv_layout = impl_param.get_input_layout(0);

        // // expected to be normalized already on primitive creation
        // auto concat_axis = primitive->concat_axis;
        // auto gather_axis = primitive->gather_axis;

        // auto kv_shape = kv_layout.get_partial_shape();
        // auto beam_table_shape = ov::PartialShape(std::vector<size_t>(kv_shape.size(), 1));
        // beam_table_shape[gather_axis] = kv_shape[gather_axis];
        // beam_table_shape[concat_axis] = kv_shape[concat_axis];
        return kv_layout;
    }

    static kv_cache_update_kernel_params_t get_kv_cache_update_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        kv_cache_update_kernel_params_t params;
        set_params(impl_param, params);

        auto query = impl_param.get_input_layout(0);
        auto key = impl_param.get_input_layout(1);
        auto value = impl_param.get_input_layout(2);
        auto key_cache = impl_param.get_input_layout(3);
        auto value_cache = impl_param.get_input_layout(4);
        auto slot_mapping = impl_param.get_input_layout(6);

        // query_shape = [batch_size, seq_len, num_heads * head_size]
        // key_shape, value_shape = [batch_size, seq_len, num_kv_heads * head_size]
        // key_cache_shape = [num_blocks, num_kv_heads, head_size/x, block_size, x]
        // value_cache_shape = [num_blocks, num_kv_heads, head_size, block_size]
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

        // Reshape from [batch_size, seq_len, num_heads * head_size] to [batch_size, seq_len, num_heads, head_size]
        // query.set_partial_shape({batch_size, seq_len, num_heads, head_size});
        // key.set_partial_shape({batch_size, seq_len, num_kv_heads, head_size});
        // value.set_partial_shape(key.get_shape());

        params.is_shape_agnostic = is_shape_agnostic;
        params.stage_id = 0;
        params.inputs.resize(3);
        params.outputs.resize(2);
        params.inputs[0] = convert_data_tensor(key);
        params.inputs[1] = convert_data_tensor(value);
        params.inputs[2] = convert_data_tensor(slot_mapping);
        params.outputs[0] = convert_data_tensor(key_cache);
        params.outputs[1] = convert_data_tensor(value_cache);
        params.layerID = impl_param.desc->id;

        // const auto inputs_count = 2;
        // params.inputs.resize(inputs_count);
        // for (size_t i = 0; i < inputs_count; ++i) {
        //     params.inputs[i] = convert_data_tensor(impl_param.input_layouts[i]);
        // }

        // params.axis = convert_axis(axis, impl_param.get_output_layout().get_rank());
        // params.kernelPerInput = true;

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(1)},
            {1, in_offsets_map.at(2)},
            {2, in_offsets_map.at(6)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, in_offsets_map.at(3)},
            {1, in_offsets_map.at(4)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static sdpa_kernel_params_t get_bt_update_kernel_params(const kernel_impl_params& impl_param, bool is_state_set = false) {
        // auto params = get_default_params<kernel_selector::beam_table_update_params>(impl_param, true);

        // auto inputs_count = 2;
        // auto bt_present_layout = impl_param.output_layouts[1];
        // auto bt_shape = extend_shape_to_rank_from_end(bt_present_layout.get_partial_shape(), 1);
        // bt_present_layout.set_partial_shape(bt_shape);
        // layout bt_past_layout = get_beam_table_layout(impl_param);

        // auto beam_idx_l = impl_param.input_layouts[2];
        // beam_idx_l.set_partial_shape(extend_shape_to_rank_from_end(beam_idx_l.get_partial_shape(), 4));

        // params.inputs.resize(inputs_count);
        // params.inputs[0] = convert_data_tensor(bt_past_layout);
        // params.inputs[1] = convert_data_tensor(beam_idx_l);
        // params.outputs[0] = convert_data_tensor(bt_present_layout);
        // params.inputs.resize(inputs_count);
        // params.is_state_set = is_state_set;

        // const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset; // [kv_past, kv_new_token, [beam_idx, beam_table_past]]
        // const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset; // [kv_present, beam_table_present]
        // std::map<size_t, size_t> in_tensor_to_offset_map = {
        //     {0, in_offsets_map.at(3)}, // beam_table_past
        //     {1, in_offsets_map.at(2)}, // beam_idx
        // };
        // std::map<size_t, size_t> out_tensor_to_offset_map = {
        //     {0, out_offsets_map.at(1)}, // beam_table_present
        // };

        // params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return {};
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<paged_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto concat_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        auto& concat_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
        kernels_data.push_back(concat_kernel_selector.get_best_kernel(concat_kernel_params));

        // SDPA
            // auto& concat_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
            // kernels_data.push_back(bt_update_kernel_selector.get_best_kernel(bt_update_kernel_params));
        //
        return cldnn::make_unique<paged_attention_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto paged_attention_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        (_kernels_data[Stage::concat].update_dispatch_data_func)(paged_attention_kernel_params, _kernels_data[Stage::concat]);
        // _kernels_data[concat_stage].kernels[0].skip_execution = impl_param._can_be_optimized || impl_param.get_input_layout(0).count() == 0;
    }
};

namespace detail {

attach_paged_attention_impl::attach_paged_attention_impl() {
    auto types = { data_types::f16, data_types::f32 };
    auto formats = { format::bfyx };
    implementation_map<paged_attention>::add(impl_types::ocl,
                                             shape_types::dynamic_shape,
                                             paged_attention_impl::create,
                                             types,
                                             formats);

    implementation_map<paged_attention>::add(impl_types::ocl,
                                             shape_types::static_shape,
                                             paged_attention_impl::create,
                                             types,
                                             formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::paged_attention_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_attention)
