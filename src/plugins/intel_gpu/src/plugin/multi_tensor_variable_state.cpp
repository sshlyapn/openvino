// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

MultiTensorState::MultiTensorState(const std::vector<VariableStateInfo>& infos,
                                   std::shared_ptr<RemoteContextImpl> context,
                                   ShapePredictor::Ptr shape_predictor) : ov::intel_gpu::GPUVariableState(infos[0].m_id, context) {
    for (auto& info : infos) {
        m_states.push_back(std::make_shared<VariableState>(info, context, shape_predictor));
    }
}

VariableStateIndirectKVCache::VariableStateIndirectKVCache(const VariableStateInfo& info,
                                                           RemoteContextImpl::Ptr context,
                                                           std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                                                           size_t beam_idx,
                                                           size_t concat_idx)
    : MultiTensorState { {info}, context, shape_predictor}
    , m_beam_idx(beam_idx)
    , m_concat_idx(concat_idx) {
    cldnn::layout beam_table_layout(get_beam_table_shape(info.m_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    VariableStateInfo beam_table_state_info(info.m_id + "/beam_table", beam_table_layout);
    m_states.push_back(std::make_shared<VariableState>(beam_table_state_info, context, shape_predictor));
    OPENVINO_ASSERT(m_states.size() == 2, "[GPU] VariableStateIndirectKVCache expects 2 internal states to be initialized");
}

void VariableStateIndirectKVCache::reset() {
    for (auto& state : m_states) {
        state->reset();
    }
    m_is_set = false;
}

bool VariableStateIndirectKVCache::is_set() const {
    return m_is_set;
}

cldnn::memory::ptr VariableStateIndirectKVCache::get_memory() const {
    return m_states[0]->get_memory();
}

const cldnn::layout& VariableStateIndirectKVCache::get_layout() const {
    return m_states[0]->get_layout();
}

void VariableStateIndirectKVCache::set_state(const ov::SoPtr<ov::ITensor>& state) {
    OPENVINO_ASSERT(m_states.size() == 2, "[GPU] Corrupted VariableStateIndirectKVCache. Expected 2 internal states. Got: ", m_states.size());
    auto kv_cache_state = m_states[0];
    m_states[0]->set_state(state); // user can set only KV cache itself
    ov::Tensor default_beam_table(ov::element::i32, get_beam_table_shape(state->get_shape()).to_shape());
    m_states[1]->set_state(ov::get_tensor_impl(default_beam_table));
    m_states[1]->set();
}

static void rearrange_cache(cldnn::memory::ptr kv_in_mem, cldnn::memory::ptr bt_mem, cldnn::memory::ptr kv_out_mem, cldnn::stream& stream) {
    auto kv_shape = kv_in_mem->get_layout().get_shape();
    cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> kv_in_ptr(kv_in_mem, stream);
    cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> bt_in_ptr(bt_mem, stream);
    cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::write> kv_out_ptr(kv_out_mem, stream);

    std::cerr << kv_in_mem->get_layout().to_short_string() << std::endl;
    std::cerr << bt_mem->get_layout().to_short_string() << std::endl;
    std::cerr << kv_out_mem->get_layout().to_short_string() << std::endl;
    for (size_t b = 0; b < kv_shape[0]; b++) {
        for (size_t f = 0; f < kv_shape[1]; f++) {
            for (size_t y = 0; y < kv_shape[2]; y++) {
                for (size_t x = 0; x < kv_shape[3]; x++) {
                    size_t b_kv = bt_in_ptr[b* kv_shape[2] + y];

                    auto in_idx = std::vector<int>{static_cast<int>(b_kv), static_cast<int>(f), static_cast<int>(y), static_cast<int>(x)};
                    auto out_idx = std::vector<int>{static_cast<int>(b), static_cast<int>(f), static_cast<int>(y), static_cast<int>(x)};

                    cldnn::tensor in(cldnn::format::bfyx, in_idx, 0);
                    cldnn::tensor out(cldnn::format::bfyx, out_idx, 0);

                    size_t out_offset = kv_out_mem->get_layout().get_linear_offset(out);
                    size_t in_offset = kv_in_mem->get_layout().get_linear_offset(in);

                    kv_out_ptr[out_offset] = kv_in_ptr[in_offset];
                }
            }
        }
    }
}

ov::SoPtr<ov::ITensor> VariableStateIndirectKVCache::get_state() const {
    auto kv_layout = m_states[0]->get_layout();
    auto kv_mem = m_states[0]->get_memory();
    auto bt_mem = m_states[1]->get_memory();
    auto tensor = m_context->create_host_tensor(m_states[0]->get_user_specified_type(), kv_layout.get_shape());

    auto& engine = m_context->get_engine();
    auto tmp_mem = engine.allocate_memory(kv_layout, engine.get_lockable_preferred_memory_allocation_type(), false);

    rearrange_cache(kv_mem, bt_mem, tmp_mem, m_context->get_engine().get_service_stream());

    convert_and_copy(tmp_mem, tensor._ptr.get(), m_context->get_engine().get_service_stream());

    return tensor;
}

void VariableStateIndirectKVCache::set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) {
    m_states[0]->set_memory(new_mem, actual_layout);

    cldnn::layout beam_table_layout(get_beam_table_shape(actual_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    std::cerr << "set_memory: update beam table to" << beam_table_layout.to_short_string() << std::endl;
    auto prev_table = m_states[1]->get_memory();
    m_states[1]->set_layout(beam_table_layout);
    auto curr_table = m_states[1]->get_memory();

    if (prev_table && curr_table && !prev_table->get_engine()->is_the_same_buffer(*prev_table, *curr_table)) {
        curr_table->copy_from(m_context->get_engine().get_service_stream(), *prev_table, true);
    }
}

void VariableStateIndirectKVCache::set_layout(const cldnn::layout& new_layout) {
    m_states[0]->set_layout(new_layout);
    cldnn::layout beam_table_layout(get_beam_table_shape(new_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    std::cerr << "set_layout: update beam table to" << beam_table_layout.to_short_string() << std::endl;
    auto prev_table = m_states[1]->get_memory();
    m_states[1]->set_layout(beam_table_layout);
    auto curr_table = m_states[1]->get_memory();

    if (prev_table && curr_table && !prev_table->get_engine()->is_the_same_buffer(*prev_table, *curr_table)) {
        curr_table->copy_from(m_context->get_engine().get_service_stream(), *prev_table, true);
    }
}

size_t VariableStateIndirectKVCache::get_actual_mem_size() const {
    return m_states[0]->get_actual_mem_size();
}

cldnn::memory::ptr VariableStateIndirectKVCache::get_beam_table_mem() const {
    return m_states[1]->get_memory();
}

ov::PartialShape VariableStateIndirectKVCache::get_beam_table_shape(const ov::PartialShape& kv_cache_shape) {
    return ov::PartialShape{kv_cache_shape[m_beam_idx], kv_cache_shape[m_concat_idx]};
}

VariableState::Ptr VariableStateIndirectKVCache::get_kv_cache_state() const {
    return m_states[0];
}

VariableState::Ptr VariableStateIndirectKVCache::get_beam_table_state() const {
    return m_states[1];
}

}  // namespace intel_gpu
}  // namespace ov
