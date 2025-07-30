// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_generator.hpp"

#include "snippets/runtime_configurator.hpp"


using namespace dnnl::impl::gpu::intel::jit;

namespace ov::intel_gpu::jit {

#define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                      \
        {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
             return std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__);                      \
         },                                                                                           \
         [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
             return e_type::get_supported_precisions(n);                                              \
         }}

GPUTargetMachine::GPUTargetMachine(dnnl::impl::gpu::intel::jit::gpu_gen_t hw)
    : TargetMachine(std::make_shared<ov::snippets::RuntimeConfigurator>(std::make_shared<ov::snippets::RuntimeConfig>())),
      m_hw(hw) {
    // init generator by hw
    switch (hw) {
    case ngen::HW::Gen9:    m_h = std::make_unique<jit_snippet_t<ngen::HW::Gen9>>(); break;
    case ngen::HW::Gen11:   m_h = std::make_unique<jit_snippet_t<ngen::HW::Gen11>>(); break;
    case ngen::HW::Gen12LP: m_h = std::make_unique<jit_snippet_t<ngen::HW::Gen12LP>>(); break;
    case ngen::HW::XeHP:    m_h = std::make_unique<jit_snippet_t<ngen::HW::XeHP>>(); break;
    case ngen::HW::XeHPG:   m_h = std::make_unique<jit_snippet_t<ngen::HW::XeHPG>>(); break;
    case ngen::HW::XeHPC:   m_h = std::make_unique<jit_snippet_t<ngen::HW::XeHPC>>(); break;
    case ngen::HW::Xe2:     m_h = std::make_unique<jit_snippet_t<ngen::HW::Xe2>>(); break;
    case ngen::HW::Xe3:     m_h = std::make_unique<jit_snippet_t<ngen::HW::Xe3>>(); break;
    default:
        OPENVINO_THROW("Unknown GPU hardware!");
    }
    OPENVINO_ASSERT(m_h, "Unitialized generator");

    // data movement
    //jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    //jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
}

std::shared_ptr<snippets::TargetMachine> GPUTargetMachine::clone() const {
    const auto cloned = std::make_shared<GPUTargetMachine>(m_hw);
    cloned->configurator = std::make_shared<ov::snippets::RuntimeConfigurator>(*configurator);
    return cloned;
}

size_t GPUTargetMachine::get_lanes() const {
    assert(m_h);
    return m_h->getSIMD();
}

std::vector<snippets::Reg> GPUTargetMachine::get_abi_arg_regs() const {
    OPENVINO_THROW("Unimplemented!");
    return {};
}

std::vector<snippets::Reg> GPUTargetMachine::get_gp_reg_pool() const {
    OPENVINO_THROW("Unimplemented!");
    return {};
}

std::vector<snippets::Reg> GPUTargetMachine::get_vec_reg_pool() const {
    OPENVINO_THROW("Unimplemented!");
    return {};
}

dnnl::impl::gpu::intel::jit::gpu_gen_t GPUTargetMachine::get_hw() const {
    return m_hw;
}


snippets::CompiledSnippetPtr GPUTargetMachine::get_snippet() {
    // OPENVINO_ASSERT(h->create_kernel() == dnnl::impl::status::success, "Failed to create jit_kernel in get_snippet()");
    // const auto& result =
    //     std::make_shared<CompiledSnippetGPU>(std::unique_ptr<dnnl::impl::cpu::x64::jit_generator_t>(h.release()));
    // // Note that we reset all the generated code, since it was copied into CompiledSnippetGPU
    // h = std::make_unique<jit_snippet>();
    // return result;
    OPENVINO_THROW("Unimplemented!");
    return nullptr;
}

CompiledSnippetGPU::CompiledSnippetGPU(std::unique_ptr<jit_snippet_base_t> h)
    : h_compiled(std::move(h)) {
    //OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was nopt compiled");
}

const uint8_t* CompiledSnippetGPU::get_code() const {
    //return h_compiled->jit_ker();
    OPENVINO_THROW("Unimplemented!");
    return nullptr;
}

size_t CompiledSnippetGPU::get_code_size() const {
    OPENVINO_THROW("Unimplemented!");
}

bool CompiledSnippetGPU::empty() const {
    return get_code_size() == 0;
}

GPUGenerator::GPUGenerator(dnnl::impl::gpu::intel::jit::gpu_gen_t hw)
    : Generator(std::make_shared<GPUTargetMachine>(hw)) {}
GPUGenerator::GPUGenerator(const std::shared_ptr<GPUTargetMachine>& target) : Generator(target) {}

std::shared_ptr<snippets::Generator> GPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<GPUTargetMachine>(target->clone());
    OPENVINO_ASSERT(cpu_target_machine,
                    "Failed to clone GPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<GPUGenerator>(cpu_target_machine);
}

}  // namespace ov::intel_gpu::jit
