// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitter.hpp"

#include "snippets/op/kernel.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::intel_gpu::jit {

using namespace ngen;

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
jit_kernel_emitter<hw>::jit_kernel_emitter(jit_snippet_t<hw>* host,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter<hw>(host, ov::element::dynamic) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OPENVINO_ASSERT(kernel != nullptr, "invoked with invalid op argument");
    OPENVINO_ASSERT(!kernel->region->empty(), "invoked with empty body");
    body = kernel->region;

    const auto& parameters = body->get_parameters();
    const auto& results = body->get_results();
    std::vector<snippets::Reg> data_ptr_regs;
    for (const auto& param : parameters) {
        const auto& reg = param->get_output_port_descriptor(0)->get_reg();
        if (!reg.is_address()) {
            data_ptr_regs.push_back(reg);
        }
    }
    num_inputs = data_ptr_regs.size();
    for (const auto& result : results) {
        data_ptr_regs.push_back(result->get_input_port_descriptor(0)->get_reg());
    }
    num_outputs = data_ptr_regs.size() - num_inputs;
};

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
void jit_kernel_emitter<hw>::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OPENVINO_ASSERT(out.empty() && out.empty(), "Unexpected number of input/output arguments");
}

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
void jit_kernel_emitter<hw>::emit_code_impl(const std::vector<size_t>& in,
                                            const std::vector<size_t>& out,
                                            const std::vector<size_t>& pool_vec_idxs,
                                            const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    this->aux_vec_idxs = pool_vec_idxs;
    this->aux_gpr_idxs = pool_gpr_idxs;
    emit_impl(in, out);
}

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
void jit_kernel_emitter<hw>::emit_impl(const std::vector<size_t>& in,
                                      [[maybe_unused]] const std::vector<size_t>& out) const {
    // Define kernel interface for OpenCL.
    for (size_t i = 0; i < num_inputs; ++i) {
        this->m_h->newArgument("src" + std::to_string(i), ExternalArgumentType::GlobalPtr);
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        this->m_h->newArgument("dst" + std::to_string(i), ExternalArgumentType::GlobalPtr);
    }

    this->m_h->requireLocalID(1);
    this->m_h->requireLocalSize();

    this->m_h->finalizeInterface();

    auto src0_ptr = this->m_h->getArgument("src0");
    auto src1_ptr = this->m_h->getArgument("src1");
    auto dst_ptr = this->m_h->getArgument("dst0");

    auto local_size = this->m_h->getLocalSize(0).uw();
    auto local_id = this->m_h->getLocalID(0);               // Vector of local IDs.
    auto group_id = this->m_h->r0.ud(1);                    // Thread group (a.k.a. workgroup) IDs are in r0.ud(1) (X) r0.ud(6) (Y) r0.ud(7) (Z)
 
    // Local variables.
    auto global_id = this->m_h->r12.ud(0);
    auto header = this->m_h->r13;
    auto temp = this->m_h->r11;

    auto reg_src0 = this->m_h->r14;
    auto reg_src1 = this->m_h->r15;

    // All instructions use W (NoMask) by default.
    this->m_h->setDefaultNoMask();

    // Enable automatic SWSB for Gen12.
    this->m_h->setDefaultAutoSWSB();

    // Prologue for ATS+.
    this->m_h->prologue();

    // Enable IEEE denormals.
    this->m_h->or_(1 | this->m_h->Switch, this->m_h->cr0[0], this->m_h->cr0[0], 0x4C0);

    // Calculate global ID = (group ID) * (local size) + (local ID for lane 0).
    this->m_h->mul(1, global_id, group_id, local_size);
    this->m_h->add(1, global_id, global_id, local_id[0]);

    this->m_h->shl(1, global_id, global_id, 2);
    {
        this->m_h->addc(1, header.ud(0), src0_ptr.ud(0), global_id);
        this->m_h->mov(1, temp.ud(0), this->m_h->acc0.ud(0));
        this->m_h->add(1, header.ud(1), src0_ptr.ud(1), temp.ud(0));
        this->m_h->load(1, reg_src0, this->m_h->D32 | this->m_h->V8T, this->m_h->A64, header);
    }
    {
        this->m_h->addc(1, header.ud(0), src1_ptr.ud(0), global_id);
        this->m_h->mov(1, temp.ud(0), this->m_h->acc0.ud(0));
        this->m_h->add(1, header.ud(1), src1_ptr.ud(1), temp.ud(0));
        this->m_h->load(1, reg_src1, this->m_h->D32 | this->m_h->V8T, this->m_h->A64, header);
    }

    this->m_h->template add<float>(8, reg_src0, reg_src0, reg_src1);

    {
        this->m_h->addc(1, header.ud(0), dst_ptr.ud(0), global_id);
        this->m_h->mov(1, temp.ud(0), this->m_h->acc0.ud(0));
        this->m_h->add(1, header.ud(1), dst_ptr.ud(1), temp.ud(0));
        this->m_h->store(1, this->m_h->D32 | this->m_h->V8T, this->m_h->A64, header, reg_src0);
    }

    this->m_h->epilogue();
}

TEMPLATE_INSTANCES(jit_kernel_emitter)

}  // namespace ov::intel_gpu::jit
