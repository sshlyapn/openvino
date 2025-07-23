// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.hpp"

#include "primitive_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"

#include <vector>

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::jit {

class SubgraphImpl : public primitive_impl {
    using primitive_impl::primitive_impl;

public:
    explicit SubgraphImpl(const program_node& /*node*/, const kernel_impl_params& /*impl_params*/)
        : primitive_impl("jit::subgraph") { }

    SubgraphImpl() : primitive_impl() {}

    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::jit::SubgraphImpl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<SubgraphImpl>(*this);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void set_arguments(primitive_inst& /*instance*/) override {}
    void set_arguments(primitive_inst& /*instance*/, kernel_arguments_data& /*args*/) override {}
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override { return {}; }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        return stream.aggregate_events(events);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override { }
};

std::unique_ptr<primitive_impl> Subgraph::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<subgraph>());
    return std::make_unique<SubgraphImpl>(node, params);
}

}  // namespace ov::intel_gpu::jit

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::jit::SubgraphImpl)
