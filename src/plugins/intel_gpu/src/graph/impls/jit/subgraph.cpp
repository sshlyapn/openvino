// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu/intel/jit/generator.hpp"

#include "primitive_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"
#include "subgraph.hpp"

#include "runtime/ocl/ocl_engine.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "plugin/transformations/snippets/lowered/set_single_kernel_work_amount.hpp"

#include "snippets/lowered/pass/optimize_domain.hpp"
#include "snippets/utils/utils.hpp"
#include "gpu_generator.hpp"

#include <vector>
namespace ov::intel_gpu::jit {

using namespace dnnl::impl::gpu::intel::jit;
using namespace ngen;
class SubgraphImpl : public primitive_impl {
    using primitive_impl::primitive_impl;

    using DataFlowPasses = std::vector<ov::snippets::pass::Manager::PositionedPassBase>;
    using ControlFlowPasses = std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered>;

    std::shared_ptr<ov::snippets::op::Subgraph> m_subgraph {nullptr};

public:
    explicit SubgraphImpl(const program_node& node, const kernel_impl_params& impl_params)
        : primitive_impl("jit::subgraph"), m_subgraph(node.as<subgraph>().get_primitive()->ov_subgraph->clone())  {
            m_subgraph->set_generator(
                std::make_shared<ov::intel_gpu::jit::GPUGenerator>(ngenHW2pluginHW(impl_params.get_device_info().arch)));

            const auto in_blocked_shapes = getSnippetsBlockedShapes(impl_params);
            const auto precisions = getIOPrecisions(impl_params);
            m_subgraph->data_flow_transformations(in_blocked_shapes, precisions.first, precisions.second);

            const auto control_flow_config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
            control_flow_config->disable<ov::snippets::lowered::pass::OptimizeDomain>();
            m_subgraph->set_tile_rank(1UL);

            m_subgraph->control_flow_transformations(0,   // unused
                                                     256, // unused
                                                     std::make_shared<ov::snippets::IShapeInferSnippetsFactory>(),
                                                     control_flow_config,
                                                     getControlFlowPasses());
        }
    
    ControlFlowPasses getControlFlowPasses() const {
        using PassPosition = ov::snippets::pass::PassPosition;
        using Place = PassPosition::Place;

        ControlFlowPasses backend_passes;
#define SNIPPETS_REGISTER_PASS_ABSOLUTE(PASS_PLACE, PASS, ...)             \
        backend_passes.emplace_back(PassPosition(PASS_PLACE), std::make_shared<PASS>(__VA_ARGS__))


        SNIPPETS_REGISTER_PASS_ABSOLUTE(Place::PipelineStart,
                                        ov::intel_gpu::pass::SetSingleKernelWorkAmount);
#undef SNIPPETS_REGISTER_PASS_ABSOLUTE
        return backend_passes;
    }

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

private:
    static ngen::HW ngenHW2pluginHW(gpu_arch arch) {
        switch (arch) {
        case gpu_arch::gen9: return ngen::HW::Gen9;
        case gpu_arch::gen11: return ngen::HW::Gen11;
        case gpu_arch::xe_lp: return ngen::HW::XeLP;
        case gpu_arch::xe_hp: return ngen::HW::XeHP;
        case gpu_arch::xe_hpg: return ngen::HW::XeHPG;
        case gpu_arch::xe_hpc: return ngen::HW::XeHPC;
        case gpu_arch::xe2: return ngen::HW::Xe2;
        case gpu_arch::xe3: return ngen::HW::Xe3;
        case gpu_arch::unknown: return ngen::HW::Unknown;
        default:
            OPENVINO_THROW("Unexpected arch");
        }
    }

    static ov::snippets::op::Subgraph::BlockedShapeVector getSnippetsBlockedShapes(const kernel_impl_params& impl_params) {
        ov::snippets::op::Subgraph::BlockedShapeVector in_blocked_shapes(impl_params.input_layouts.size());
        for (size_t i = 0; i < in_blocked_shapes.size(); i++) {
            // support only planar shapes
            const auto blocked_dims = ov::snippets::utils::pshape_to_vdims(impl_params.input_layouts[i].get_partial_shape());
            const auto blocked_layout = ov::snippets::utils::get_planar_layout(blocked_dims.size());
            in_blocked_shapes[i] = {blocked_dims, blocked_layout};
        }
        return in_blocked_shapes;
    }

    static std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> getIOPrecisions(const kernel_impl_params& impl_params) {
        std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> prc;
        prc.first.reserve(impl_params.input_layouts.size());
        prc.second.reserve(impl_params.output_layouts.size());
        for (const auto& in : impl_params.input_layouts) {
            prc.first.push_back(in.data_type);
        }
        for (const auto& out : impl_params.output_layouts) {
            prc.second.push_back(out.data_type);
        }
        return prc;
    }
};

std::unique_ptr<primitive_impl> Subgraph::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<subgraph>());
    return std::make_unique<SubgraphImpl>(node, params);
}

}  // namespace ov::intel_gpu::jit

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::jit::SubgraphImpl)
