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

#include "snippets/lowered/pass/optimize_domain.hpp"
#include "snippets/utils/utils.hpp"
#include "gpu_generator.hpp"

#include <vector>
namespace ov::intel_gpu::jit {

using namespace dnnl::impl::gpu::intel::jit;
using namespace ngen;

template <HW hw>
class VectorScaleKernelGenerator : public OpenCLCodeGenerator<hw>
{
protected:
    NGEN_FORWARD_OPENCL(hw);

public:
    VectorScaleKernelGenerator() : OpenCLCodeGenerator<hw>()
    {
        // Define kernel interface for OpenCL.
        newArgument("buffer", ExternalArgumentType::GlobalPtr);
        newArgument("alpha", DataType::f);
        requireLocalID(1);
        requireLocalSize();
        requireSIMD((GRF::bytes(hw) == 64) ? 16 : 8);
        externalName("vector_scale");

        finalizeInterface();

        auto bufferSurface = Surface(getArgumentSurfaceIfExists("buffer"));     // Surface # for buffer.
        auto bufferPtr = getArgument("buffer");                                 // A64 pointer for buffer.
        auto alpha = getArgument("alpha");

        auto localSize = getLocalSize(0).uw();
        auto localID = getLocalID(0);               // Vector of local IDs.
        auto groupID = r0.ud(1);                    // Thread group (a.k.a. workgroup) IDs are in r0.ud(1) (X) r0.ud(6) (Y) r0.ud(7) (Z)

        // Local variables.
        auto globalID = r12.ud(0);
        auto header = r13;
        auto data = r14;
        auto temp = r15;

        // Decide on load/store messages.
        bool useLSC = (hw >= HW::XeHPC);

        // All instructions use W (NoMask) by default.
        setDefaultNoMask();

        // Enable automatic SWSB for Gen12.
        setDefaultAutoSWSB();

        // Prologue for ATS+.
        prologue();

        // Enable IEEE denormals.
        or_(1 | Switch, cr0[0], cr0[0], 0x4C0);

        // Calculate global ID = (group ID) * (local size) + (local ID for lane 0).
        mul(1, globalID, groupID, localSize);
        add(1, globalID, globalID, localID[0]);

        // Do 32 byte (2 OWord) block read at offset (global ID) * sizeof(float).
        if (!useLSC) {
            shr<uint32_t>(1, header[2], globalID, 2);
            load(8, data, block_oword(2), bufferSurface, header);
        } else {
            shl(1, globalID, globalID, 2);
            addc(1, header.ud(0), bufferPtr.ud(0), globalID);
            mov(1, temp.ud(0), acc0.ud(0));
            add(1, header.ud(1), bufferPtr.ud(1), temp.ud(0));
            load(1, data, D32 | V8T, A64, header);
        }

        // Scale data.
        mul<float>(8, data, data, alpha);

        // Store updated data.
        if (!useLSC)
            store(8, block_oword(2), bufferSurface, header, data);
        else
            store(1, D32 | V8T, A64, header, data);

        // End thread. Must move r0 to one of r112-r127, then call threadend.
        mov<uint32_t>(8, r127, r0);
        threadend(r127);
    }
};


class SubgraphImpl : public primitive_impl {
    using primitive_impl::primitive_impl;

    std::shared_ptr<ov::snippets::op::Subgraph> m_subgraph {nullptr};

public:
    explicit SubgraphImpl(const program_node& node, const kernel_impl_params& impl_params)
        : primitive_impl("jit::subgraph"), m_subgraph(node.as<subgraph>().get_primitive()->ov_subgraph->clone())  {
            m_subgraph->set_generator(
                std::make_shared<ov::intel_gpu::jit::GPUGenerator>(ngenHW2pluginHW(impl_params.get_device_info().arch)));

            const auto in_blocked_shapes = getSnippetsBlockedShapes(impl_params);
            const auto precisions = getIOPrecisions(impl_params);
            m_subgraph->data_flow_transformations(in_blocked_shapes, precisions.first, precisions.second, {});

            const auto control_flow_config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
            control_flow_config->disable<ov::snippets::lowered::pass::OptimizeDomain>();
            m_subgraph->set_tile_rank(1UL);

            m_subgraph->control_flow_transformations(0,   // unused
                                                     256, // unused
                                                     std::make_shared<ov::snippets::IShapeInferSnippetsFactory>(),
                                                     control_flow_config);
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
