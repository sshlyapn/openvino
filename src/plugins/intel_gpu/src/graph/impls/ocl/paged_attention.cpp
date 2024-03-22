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
#include "paged_attention/sdpa_kernel_ref.hpp"

namespace cldnn {
namespace ocl {

// static memory::ptr generate_attention_bias(size_t batch_size, size_t seq_len, size_t sliding_window, engine& engine) {
//     ov::Shape attention_mask_shape({batch_size, 1, 1, seq_len, seq_len});
//     layout bias_layout {attention_mask_shape, data_types::f16, format::bfzyx};
//     memory::ptr attention_mask = engine.allocate_memory(bias_layout, false);
//     mem_lock<ov::float16, mem_lock_type::read> attention_mask_lock(attention_mask, engine.get_service_stream());
//     int attention_mask_stride = seq_len * seq_len;

//     ov::float16 negative_inf = -std::numeric_limits<ov::float16>::infinity();

//     for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
//         ov::float16* attention_mask_data = attention_mask_lock.data() + batch_id * attention_mask_stride;
//         size_t left_window = sliding_window, right_window = 1;
//         for (size_t y = 0; y < seq_len; ++y) {
//             for (size_t x = 0; x < seq_len; ++x) {
//                 attention_mask_data[y * seq_len + x] = (x + right_window - 1) > y || (x + left_window - 1) < y ? negative_inf : ov::float16(0);
//             }
//         }
//     }

//     return attention_mask;
// }

struct paged_attention_impl : multi_stage_primitive<paged_attention> {
    using parent = multi_stage_primitive<paged_attention>;
    using parent::parent;
    using kv_cache_update_kernel_selector_t = kernel_selector::kv_cache_update_kernel_selector;
    using kv_cache_update_kernel_params_t = kernel_selector::kv_cache_update_params;

    using sdpa_kernel_selector_t = kernel_selector::sdpa_kernel_selector;
    using sdpa_kernel_params_t = kernel_selector::sdpa_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::paged_attention_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<paged_attention_impl>(*this);
    }

    enum Stage {
        KV_CACHE_UPDATE,
        SDPA
    };

    mutable cldnn::memory::ptr key_cache_mem = nullptr;
    mutable cldnn::memory::ptr value_cache_mem = nullptr;

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
        {
            kernel_arguments_data args;
            args.shape_info = instance.shape_info_memory_ptr();
            if (stage == Stage::KV_CACHE_UPDATE) {
                args.inputs = { instance.input_memory_ptr(1),  /* key */
                                instance.input_memory_ptr(2),  /* value */
                                instance.input_memory_ptr(6)   /* slot_mapping */};
                args.outputs = { instance.input_memory_ptr(3), /* key_cache */
                                instance.input_memory_ptr(4)   /* value_cache */ };
            } else if (stage == Stage::SDPA) {
                // auto& stream = instance.get_network().get_stream();
                auto& service_stream = instance.get_network().get_engine().get_service_stream();
                auto is_prefill_memory = instance.input_memory_ptr(5);
                mem_lock<uint8_t, mem_lock_type::read> is_prefill_memory_lock(is_prefill_memory, service_stream);
                bool is_prefill_stage = is_prefill_memory_lock[0];
                is_prefill_stage = false;

                if (!is_prefill_stage) {
                    args.inputs = { instance.input_memory_ptr(0), /* query */
                                    instance.input_memory_ptr(3), /* key_cache */
                                    instance.input_memory_ptr(4), /* value_cache */
                                    instance.input_memory_ptr(7), /* max_context_len */
                                    instance.input_memory_ptr(8), /* context_lens */
                                    instance.input_memory_ptr(9), /* block_tables */
                                    instance.input_memory_ptr(10) /* scale */ };
                    args.outputs = { instance.output_memory_ptr(0) };
                } else {
                    GPU_DEBUG_TRACE_DETAIL << "Replace arguments for PA\n";
                    args.inputs = { instance.input_memory_ptr(0), /* query */
                                    instance.input_memory_ptr(3), /* key_cache */
                                    instance.input_memory_ptr(4), /* value_cache */
                                    instance.input_memory_ptr(7), /* max_context_len */
                                    instance.context_lens_mem,    /* context_lens */
                                    instance.blocks_mem,          /* block_tables */
                                    instance.input_memory_ptr(10) /* scale */ };
                    args.outputs = { instance.output_memory_ptr(0) };
                }
            }

            return args;
        }

        // WA due to lack of proper handling of key and value cache buffers. Keep them in impl for test purpose.
        if (value_cache_mem == nullptr) {
            const auto key_cache_layout = instance.get_impl_params()->get_input_layout(3);
            const auto value_cache_layout = instance.get_impl_params()->get_input_layout(4);
            key_cache_mem = instance.get_network().get_engine().allocate_memory(key_cache_layout, cldnn::allocation_type::usm_device, false);
            value_cache_mem = instance.get_network().get_engine().allocate_memory(value_cache_layout, cldnn::allocation_type::usm_device, false);
        }

        kernel_arguments_data args;
        args.shape_info = instance.shape_info_memory_ptr();
        if (stage == Stage::KV_CACHE_UPDATE) {
            args.inputs = { instance.input_memory_ptr(1),  /* key */
                            instance.input_memory_ptr(2),  /* value */
                            instance.input_memory_ptr(6)   /* slot_mapping */};
            args.outputs = { key_cache_mem,                /* key_cache */
                             value_cache_mem               /* value_cache */ };
        } else if (stage == Stage::SDPA) {
            args.inputs = { instance.input_memory_ptr(0), /* query */
                            key_cache_mem,                /* key_cache */
                            value_cache_mem,              /* value_cache */
                            instance.input_memory_ptr(7), /* max_context_len */
                            instance.input_memory_ptr(8), /* context_lens */
                            instance.input_memory_ptr(9), /* block_tables */
                            instance.input_memory_ptr(10) /* scale */ };
            args.outputs = { instance.output_memory_ptr(0) };
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


        if (instance.get_network().get_config().get_property(ov::enable_profiling)) {
            auto final_event = stream.group_events(all_events);
            if (final_event != nullptr) {
                stream.wait_for_events({final_event});
                auto profiling_info = final_event->get_profiling_info();
                for (const auto &interval : profiling_info) {
                    if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                        auto time_res0 = std::chrono::duration_cast<std::chrono::microseconds>(interval.value->value()).count();
                        GPU_DEBUG_INFO << "PagedAttention " << stage << " stage time: " << time_res0 << " mcs\n";
                    }
                }
            }
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, paged_attention_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        // auto& service_stream = instance.get_network().get_engine().get_service_stream();
        std::vector<event::ptr> res_events;

        // auto is_prefill_memory = instance.input_memory_ptr(5);
        // mem_lock<uint8_t, mem_lock_type::read> is_prefill_memory_lock(is_prefill_memory, service_stream);
        // bool is_prefill_stage = is_prefill_memory_lock[0];

        // GPU_DEBUG_TRACE_DETAIL << instance.id() << " stage is " << (is_prefill_stage ? "prefill" : "tokens generating") << "\n";

        execute_stage(events, instance, res_events, Stage::KV_CACHE_UPDATE);

        if (false) {
            // auto sliding_window_memory = instance.input_memory_ptr(12);
            // auto new_sliding_window_layout = layout{{1}, sliding_window_memory->get_layout().data_type, format::bfyx};
            // auto reshaped_sliding_window_mem = instance.get_network().get_engine().reinterpret_buffer(*sliding_window_memory, new_sliding_window_layout);
            // mem_lock<int32_t, mem_lock_type::read> sliding_window_memory_lock(reshaped_sliding_window_mem, service_stream);
            // int32_t sliding_window = sliding_window_memory_lock[0];
            // if (sliding_window == 0) {
            //     sliding_window = std::numeric_limits<std::int32_t>::max();
            // }

            // const auto query_layout = instance.get_impl_params()->get_input_layout(0);
            // const auto query_shape = query_layout.get_shape();
            // const auto key_cache_layout = instance.get_impl_params()->get_input_layout(3);
            // const auto value_cache_layout = instance.get_impl_params()->get_input_layout(4);
            // const auto key_cache_shape = key_cache_layout.get_shape();
            // const auto value_cache_shape = value_cache_layout.get_shape();

            // const int64_t batch_size = query_shape[0];
            // const int64_t seq_len = query_shape[1];
            // const int64_t hidden_size = query_shape[2];
            // const int64_t kv_heads_num = value_cache_shape[1];
            // const int64_t head_size = value_cache_shape[2];
            // const int64_t heads_num = hidden_size / head_size;
            // const int64_t num_queries_per_kv = heads_num / kv_heads_num;

            // std::cout << "Prefill stage: batch_size=" << batch_size << " seq_len=" << seq_len << " hidden_size=" << hidden_size
            //           << " kv_heads_num=" << kv_heads_num << " heads_num=" << heads_num << " head_size=" << head_size
            //           << " q=" << query_layout.to_short_string() << " k_cache=" << key_cache_layout.to_short_string()
            //           << " v_cache=" << value_cache_layout.to_short_string() << "\n";

            // auto attention_bias = generate_attention_bias(batch_size, seq_len, sliding_window, instance.get_network().get_engine());

            // auto query_mem = instance.input_memory_ptr(0);
            // auto query_layout_new = layout{{batch_size, seq_len, kv_heads_num, num_queries_per_kv, head_size}, query_mem->get_layout().data_type, format::bfzyx};
            // auto reshaped_query_mem = instance.get_network().get_engine().reinterpret_buffer(*query_mem, query_layout_new);

            // auto key_mem = instance.input_memory_ptr(1);
            // auto key_layout_new = layout{{batch_size, seq_len, kv_heads_num, 1, head_size}, key_mem->get_layout().data_type, format::bfzyx};
            // auto reshaped_key_mem = instance.get_network().get_engine().reinterpret_buffer(*key_mem, key_layout_new);

            // auto value_mem = instance.input_memory_ptr(2);
            // auto value_layout_new = layout{{batch_size, seq_len, kv_heads_num, 1, head_size}, value_mem->get_layout().data_type, format::bfzyx};
            // auto reshaped_value_mem = instance.get_network().get_engine().reinterpret_buffer(*value_mem, value_layout_new);

            // auto scale_mem = instance.input_memory_ptr(10);
            // auto scale_layout_new = layout{{1}, scale_mem->get_layout().data_type, format::bfyx};
            // auto reshaped_scale_mem = instance.get_network().get_engine().reinterpret_buffer(*scale_mem, scale_layout_new);

            // auto prefill_network = instance.prefill_network;
            // prefill_network->set_input_data("parameter:query", reshaped_query_mem);
            // prefill_network->set_input_data("parameter:key", reshaped_key_mem);
            // prefill_network->set_input_data("parameter:value", reshaped_value_mem);
            // prefill_network->set_input_data("parameter:mask", attention_bias);
            // prefill_network->set_input_data("parameter:scale", reshaped_scale_mem);

            // auto results = prefill_network->execute(events);

            // OPENVINO_ASSERT(results.size() == 1, "[GPU] Unexpected number of outputs of PagedAttention operation");

            // auto output_mem = results.begin()->second.get_memory();
            // auto output_layout_new = layout{query_shape, output_mem->get_layout().data_type, format::bfyx};
            // auto reshaped_output_mem = instance.get_network().get_engine().reinterpret_buffer(*output_mem, output_layout_new);

            // instance.set_output_memory(reshaped_output_mem);
            // instance.set_mem_changed(true);

            // return results.begin()->second.get_event();
        } else {
            // Add key/value cache update to dependencies
            auto all_events = events;
            for (auto& ev : res_events)
                all_events.push_back(ev);

            execute_stage(all_events, instance, res_events, Stage::SDPA);

            return aggregate_events(res_events, stream, res_events.size() > 1);
        }
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param) {
        kernel_selector::sdpa_configuration config;

        const auto query_layout = impl_param.get_input_layout(0);
        const auto key_cache_layout = impl_param.get_input_layout(3);
        const auto value_cache_layout = impl_param.get_input_layout(4);

        if (query_layout.is_static() && key_cache_layout.is_static() && value_cache_layout.is_static()) {
            // query_shape = [batch_size, seq_len, heads_num * head_size]
            const auto query_shape = query_layout.get_shape();
            // key_cache_shape = [num_blocks, kv_heads_num, head_size / x_size, block_size, x_size]
            const auto key_cache_shape = key_cache_layout.get_shape();
            // value_cache_shape = [num_blocks, kv_heads_num, head_size, block_size]
            const auto value_cache_shape = value_cache_layout.get_shape();

            const size_t hidden_size = query_shape[2];
            const size_t kv_heads_num = value_cache_shape[1];
            const size_t head_size = value_cache_shape[2];
            const size_t heads_num = hidden_size / head_size;
            const size_t block_size = value_cache_shape[3];
            const size_t x_size = key_cache_shape[4];

            const size_t simd_size = 16;
            OPENVINO_ASSERT(head_size % simd_size == 0, "[GPU] Head size is expected to be divisible by 16");

            config.head_size = head_size;
            config.heads_num = heads_num;
            config.kv_heads_num = kv_heads_num;
            config.block_size = block_size;
            config.x_size = x_size;
            config.max_context_len = 1;
        }

        return config;
    }

    static kv_cache_update_kernel_params_t get_kv_cache_update_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic = false) {
        kv_cache_update_kernel_params_t params;
        set_params(impl_param, params);

        auto query = impl_param.get_input_layout(0);
        auto key = impl_param.get_input_layout(1);
        auto value = impl_param.get_input_layout(2);
        auto key_cache = impl_param.get_input_layout(3);
        auto value_cache = impl_param.get_input_layout(4);
        auto slot_mapping = impl_param.get_input_layout(6);

        params.is_shape_agnostic = is_dynamic;
        params.stage_id = 0;
        params.inputs.resize(3);
        params.outputs.resize(2);
        params.inputs[0] = convert_data_tensor(key);
        params.inputs[1] = convert_data_tensor(value);
        params.inputs[2] = convert_data_tensor(slot_mapping);
        params.outputs[0] = convert_data_tensor(key_cache);
        params.outputs[1] = convert_data_tensor(value_cache);
        params.layerID = impl_param.desc->id;

        params.configuration = get_sdpa_configuration(impl_param);

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

    static sdpa_kernel_params_t get_sdpa_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic = false) {
        auto params = get_default_params<kernel_selector::sdpa_params>(impl_param, is_dynamic);

        const auto inputs_count = 7;
        const auto query_layout = impl_param.get_input_layout(0);
        const auto key_cache_layout = impl_param.get_input_layout(3);
        const auto value_cache_layout = impl_param.get_input_layout(4);
        const auto max_context_len_layout = impl_param.get_input_layout(7);
        const auto context_lens_layout = impl_param.get_input_layout(8);
        const auto block_tables_layout = impl_param.get_input_layout(9);
        const auto scale_layout = impl_param.get_input_layout(10);

        params.inputs.resize(inputs_count);
        params.inputs[1] = convert_data_tensor(key_cache_layout);
        params.inputs[2] = convert_data_tensor(value_cache_layout);
        params.inputs[3] = convert_data_tensor(max_context_len_layout);
        params.inputs[4] = convert_data_tensor(context_lens_layout);
        params.inputs[5] = convert_data_tensor(block_tables_layout);
        params.inputs[6] = convert_data_tensor(scale_layout);

        params.configuration = get_sdpa_configuration(impl_param);
        GPU_DEBUG_TRACE_DETAIL << "Number of constant_mem " << impl_param.memory_deps.size() << ", dynamic=" << is_dynamic << "\n";
        if (!is_dynamic) {
            auto& constant_mem = impl_param.memory_deps;


            const auto max_context_len_mem = constant_mem.at(7);
            mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len_mem, impl_param.get_stream());
            GPU_DEBUG_TRACE_DETAIL << "max_context_len_mem_lock=" << max_context_len_mem_lock[0] << "\n";

            const auto is_prompt_stage_mem = constant_mem.at(5);
            mem_lock<uint8_t, mem_lock_type::read> is_prompt_stage_mem_lock(is_prompt_stage_mem, impl_param.get_stream());
            bool is_prompt_stage = is_prompt_stage_mem_lock[0];

            if (is_prompt_stage) {
                // Use number of slots for KV cache as a maximum context length for the first iteration
                auto slot_mapping = impl_param.get_input_layout(6);
                params.configuration.max_context_len = slot_mapping.get_shape()[1];
            } else {
                const auto max_context_len_mem = constant_mem.at(7);
                mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len_mem, impl_param.get_stream());
                params.configuration.max_context_len = max_context_len_mem_lock[0];
            }
        }

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(3)},
            {2, in_offsets_map.at(4)},
            {3, in_offsets_map.at(7)},
            {4, in_offsets_map.at(8)},
            {5, in_offsets_map.at(9)},
            {6, in_offsets_map.at(10)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<paged_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        auto& kv_cache_update_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
        kernels_data.push_back(kv_cache_update_kernel_selector.get_best_kernel(kv_cache_update_kernel_params));

        auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, impl_param.is_dynamic());
        auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(sdpa_kernel_selector.get_best_kernel(sdpa_kernel_params));

        return cldnn::make_unique<paged_attention_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        (_kernels_data[Stage::KV_CACHE_UPDATE].update_dispatch_data_func)(kv_cache_update_kernel_params, _kernels_data[Stage::KV_CACHE_UPDATE]);

        auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, impl_param.is_dynamic());
        (_kernels_data[Stage::SDPA].update_dispatch_data_func)(sdpa_kernel_params, _kernels_data[Stage::SDPA]);
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

// const ov::Output<ov::Node>& query,
// const ov::Output<ov::Node>& key,
// const ov::Output<ov::Node>& value,
// const ov::Output<ov::Node>& key_cache,
// const ov::Output<ov::Node>& value_cache,
// // start of arguments from InputMetadata
// const ov::Output<ov::Node>& is_prompt,
// const ov::Output<ov::Node>& slot_mapping,
// //    const ov::Output<ov::Node>& prompt_lens,
// //    const ov::Output<ov::Node>& max_seq_len,
// //    const ov::Output<ov::Node>& start_loc,
// const ov::Output<ov::Node>& max_context_len, // 7
// const ov::Output<ov::Node>& context_lens, // 8
// const ov::Output<ov::Node>& block_tables, // 9
// //    const ov::Output<ov::Node>& use_cuda_graph,
// //    const ov::Output<ov::Node>& attn_bias
// // end of arguments from InputMetadata
// const ov::Output<ov::Node>& scale,
// const ov::Output<ov::Node>& alibi_slopes,
// const ov::Output<ov::Node>& sliding_window

// outputs[0],
// query, key_cache, value_cache,
// kv_heads_num, scale,
// block_tables, context_lens,
// block_size, max_context_len
