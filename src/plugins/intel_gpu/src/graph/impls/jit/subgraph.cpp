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

public:
    explicit SubgraphImpl(const program_node& /*node*/, const kernel_impl_params& impl_params)
        : primitive_impl("jit::subgraph") {
            const auto& engine = downcast<ocl::ocl_engine>(impl_params.get_program().get_engine());

            HW hw = VectorScaleKernelGenerator<HW::Unknown>::detectHW(engine.get_cl_context().get(), engine.get_cl_device().get());
            const char *gpuString = "unknown";

            switch (hw) {
                case HW::Gen9:    gpuString = "Gen9"; break;
                case HW::Gen11:   gpuString = "Gen11"; break;
                case HW::Gen12LP: gpuString = "Gen12LP"; break;
                case HW::XeHP:    gpuString = "XeHP"; break;
                case HW::XeHPG:   gpuString = "XeHPG"; break;
                case HW::XeHPC:   gpuString = "XeHPC"; break;
                case HW::Xe2:     gpuString = "Xe2"; break;
                case HW::Xe3:     gpuString = "Xe3"; break;
                default:          OPENVINO_THROW("[GPU] Unsupported architecture");
            }

            std::cout << "GPU arch: " << gpuString << "\n";

            // Create appropriate kernel generator object for the detected HW, and get a cl_kernel.
            // switch (hw) {
            //     case HW::Gen9:    VectorScaleKernelGenerator<HW::Gen9>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     case HW::Gen11:   VectorScaleKernelGenerator<HW::Gen11>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     case HW::Gen12LP: VectorScaleKernelGenerator<HW::Gen12LP>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     case HW::XeHP:    VectorScaleKernelGenerator<HW::XeHP>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     case HW::XeHPG:   VectorScaleKernelGenerator<HW::XeHPG>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     case HW::XeHPC:   VectorScaleKernelGenerator<HW::XeHPC>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     case HW::Xe2:     VectorScaleKernelGenerator<HW::Xe2>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     case HW::Xe3:     VectorScaleKernelGenerator<HW::Xe3>().getKernel(engine.get_cl_context().get(), engine.get_cl_device().get());
            //     default:          OPENVINO_THROW("[GPU] Unsupported architecture");;
            // }

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
};

std::unique_ptr<primitive_impl> Subgraph::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<subgraph>());
    return std::make_unique<SubgraphImpl>(node, params);
}

}  // namespace ov::intel_gpu::jit

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::jit::SubgraphImpl)
