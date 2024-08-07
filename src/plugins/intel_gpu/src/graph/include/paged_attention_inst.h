// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/paged_attention.hpp"
#include "primitive_inst.h"

namespace cldnn {

bool is_prefill_stage(const kernel_impl_params& impl_param);

template <>
struct typed_program_node<paged_attention> : public typed_program_node_base<paged_attention> {
private:
    using parent = typed_program_node_base<paged_attention>;

public:
    using parent::parent;

    program_node& query_input() const { return get_dependency(0); }
    program_node& key_input() const { return get_dependency(1); }
    program_node& value_input() const { return get_dependency(2); }
    program_node& key_cache() const { return get_dependency(3); }
    program_node& value_cache() const { return get_dependency(4); }

    // std::vector<size_t> get_shape_infer_dependencies() const override { return { 5 /* is_prompt */, 7 /* max_context_len */ }; }
};

using paged_attention_node = typed_program_node<paged_attention>;

template<>
class typed_primitive_inst<paged_attention> : public typed_primitive_inst_base<paged_attention> {
    using parent = typed_primitive_inst_base<paged_attention>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(paged_attention_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const paged_attention_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const paged_attention_node& node);

    typed_primitive_inst(network& network, const paged_attention_node& desc);
    typed_primitive_inst(network& network) : parent(network) {}

    memory::ptr query_memory_ptr() const { return input_memory_ptr(0); }
    memory::ptr key_memory_ptr() const { return input_memory_ptr(1); }
    memory::ptr value_memory_ptr() const { return input_memory_ptr(2); }
    memory::ptr key_cache_memory_ptr() const { return input_memory_ptr(3); }
    memory::ptr value_cache_memory_ptr() const { return input_memory_ptr(4); }
    memory::ptr past_lens_memory_ptr() const { return input_memory_ptr(5); }
    memory::ptr subsequence_begins_memory_ptr() const { return input_memory_ptr(6); }
    memory::ptr block_indices_memory_ptr() const { return input_memory_ptr(7); }
    memory::ptr block_indices_begins_memory_ptr() const { return input_memory_ptr(8); }

    mutable cldnn::memory::ptr blocks_mem = nullptr;
    mutable cldnn::memory::ptr context_lens_mem = nullptr;

    std::shared_ptr<network> prefill_network;

protected:
    void on_execute() override;

    void update_shape_info_tensor(const kernel_impl_params& params) override;
};

using paged_attention_inst = typed_primitive_inst<paged_attention>;

} // namespace cldnn
