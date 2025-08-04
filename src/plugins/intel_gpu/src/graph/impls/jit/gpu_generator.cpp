// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_generator.hpp"

#include "snippets/runtime_configurator.hpp"
#include "emitters/jit_eltwise_emitters.hpp"
#include "emitters/jit_snippets_emitters.hpp"

#include "openvino/op/add.hpp"


using namespace dnnl::impl::gpu::intel::jit;

namespace ov::intel_gpu::jit {

#define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                          \
        {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
             return std::make_shared<e_type<hw>>(m_h.get(), ##__VA_ARGS__);                           \
         },                                                                                           \
         [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
             return e_type<hw>::get_supported_precisions(n);                                          \
         }}

template <ngen::HW hw>
GPUTargetMachine<hw>::GPUTargetMachine()
    : TargetMachine(std::make_shared<ov::snippets::RuntimeConfigurator>(std::make_shared<ov::snippets::RuntimeConfig>())),
      m_h(std::make_unique<jit_snippet_t<hw>>()) {
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[op::v1::Add::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_add_emitter);
}

template <ngen::HW hw>
std::shared_ptr<snippets::TargetMachine> GPUTargetMachine<hw>::clone() const {
    const auto cloned = std::make_shared<GPUTargetMachine<hw>>();
    cloned->configurator = std::make_shared<ov::snippets::RuntimeConfigurator>(*configurator);
    return cloned;
}

template <ngen::HW hw>
size_t GPUTargetMachine<hw>::get_lanes() const {
    assert(m_h);
    return m_h->getSIMD();
}

template <ngen::HW hw>
std::vector<snippets::Reg> GPUTargetMachine<hw>::get_abi_arg_regs() const {
    OPENVINO_THROW("Unimplemented!");
    return {};
}

template <ngen::HW hw>
std::vector<snippets::Reg> GPUTargetMachine<hw>::get_gp_reg_pool() const {
    OPENVINO_THROW("Unimplemented!");
    return {};
}

template <ngen::HW hw>
std::vector<snippets::Reg> GPUTargetMachine<hw>::get_vec_reg_pool() const {
    OPENVINO_THROW("Unimplemented!");
    return {};
}

template <ngen::HW hw>
ngen::HW GPUTargetMachine<hw>::get_hw() const {
    return hw;
}

template <ngen::HW hw>
snippets::CompiledSnippetPtr GPUTargetMachine<hw>::get_snippet() {
    // OPENVINO_ASSERT(h->create_kernel() == dnnl::impl::status::success, "Failed to create jit_kernel in get_snippet()");
    // const auto& result =
    //     std::make_shared<CompiledSnippetGPU>(std::unique_ptr<dnnl::impl::cpu::x64::jit_generator_t>(h.release()));
    // // Note that we reset all the generated code, since it was copied into CompiledSnippetGPU
    // h = std::make_unique<jit_snippet>();
    // return result;
    OPENVINO_THROW("Unimplemented!");
    return nullptr;
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

GPUGenerator::GPUGenerator(ngen::HW hw)
    : Generator(create_target_machine(hw)) {}

template <ngen::HW hw>
GPUGenerator::GPUGenerator(const std::shared_ptr<GPUTargetMachine<hw>>& target) : Generator(target) {}

std::shared_ptr<snippets::Generator> GPUGenerator::clone() const {
    //const auto hw = target->get_hw();
    //return std::make_shared<GPUGenerator>(target->clone());
    OPENVINO_THROW("Unimplemented!");
}

ov::snippets::RegType GPUGenerator::get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const {
    return ov::snippets::RegType::undefined;
}

std::shared_ptr<ov::snippets::TargetMachine> GPUGenerator::create_target_machine(ngen::HW hw) {
    switch (hw) {
    case ngen::HW::Gen9:    return std::make_unique<GPUTargetMachine<ngen::HW::Gen9>>();
    case ngen::HW::Gen11:   return std::make_unique<GPUTargetMachine<ngen::HW::Gen11>>();
    case ngen::HW::Gen12LP: return std::make_unique<GPUTargetMachine<ngen::HW::Gen12LP>>();
    case ngen::HW::XeHP:    return std::make_unique<GPUTargetMachine<ngen::HW::XeHP>>();
    case ngen::HW::XeHPG:   return std::make_unique<GPUTargetMachine<ngen::HW::XeHPG>>();
    case ngen::HW::XeHPC:   return std::make_unique<GPUTargetMachine<ngen::HW::XeHPC>>();
    case ngen::HW::Xe2:     return std::make_unique<GPUTargetMachine<ngen::HW::Xe2>>();
    case ngen::HW::Xe3:     return std::make_unique<GPUTargetMachine<ngen::HW::Xe3>>();
    default:
        OPENVINO_THROW("Unknown GPU hardware!");
    }
}

}  // namespace ov::intel_gpu::jit
