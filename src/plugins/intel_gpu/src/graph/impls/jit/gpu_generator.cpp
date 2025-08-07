// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_generator.hpp"

#include "snippets/lowered/port_connector.hpp"
#include "snippets/lowered/reg_manager.hpp"
#include "snippets/runtime_configurator.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/store.hpp"
#include "emitters/jit_eltwise_emitters.hpp"
#include "emitters/jit_kernel_emitter.hpp"
#include "emitters/jit_snippets_emitters.hpp"

#include "openvino/op/add.hpp"



using namespace dnnl::impl::gpu::intel::jit;

namespace ov::intel_gpu::jit {

static ngen::HW pluginHW2ngen(cldnn::gpu_arch arch) {
    switch (arch) {
    case cldnn::gpu_arch::gen9: return ngen::HW::Gen9;
    case cldnn::gpu_arch::gen11: return ngen::HW::Gen11;
    case cldnn::gpu_arch::xe_lp: return ngen::HW::XeLP;
    case cldnn::gpu_arch::xe_hp: return ngen::HW::XeHP;
    case cldnn::gpu_arch::xe_hpg: return ngen::HW::XeHPG;
    case cldnn::gpu_arch::xe_hpc: return ngen::HW::XeHPC;
    case cldnn::gpu_arch::xe2: return ngen::HW::Xe2;
    case cldnn::gpu_arch::xe3: return ngen::HW::Xe3;
    case cldnn::gpu_arch::unknown: return ngen::HW::Unknown;
    default:
        OPENVINO_THROW("[GPU] Unexpected GPU arch");
    }
}

#define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                          \
        {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
             return std::make_shared<e_type<hw>>(m_h.get(), expr, ##__VA_ARGS__);                     \
         },                                                                                           \
         [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
             return e_type<hw>::get_supported_precisions(n);                                          \
         }}

template <ngen::HW hw>
GPUTargetMachine<hw>::GPUTargetMachine(cldnn::engine& engine)
    : TargetMachine(std::make_shared<ov::snippets::RuntimeConfigurator>(std::make_shared<ov::snippets::RuntimeConfig>())),
      m_h(std::make_unique<jit_snippet_t<hw>>()),
      engine(engine) {
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);

    jitters[ov::snippets::op::KernelStatic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_kernel_emitter);
    jitters[ov::snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[ov::snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);

    jitters[op::v1::Add::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_add_emitter);
}

template <ngen::HW hw>
std::shared_ptr<snippets::TargetMachine> GPUTargetMachine<hw>::clone() const {
    const auto cloned = std::make_shared<GPUTargetMachine<hw>>(engine);
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
    // OPENVINO_THROW("Unimplemented!");
    // TODO: REWRIE THIS PART, THIS IS TEMPORARY SOLUTION
    std::vector<snippets::Reg> regs(10);
    for (size_t i = 0; i < regs.size(); ++i) {
        regs[i] = {ov::snippets::RegType::vec, 10 + i};
    }
    return regs;
}

template <ngen::HW hw>
std::vector<snippets::Reg> GPUTargetMachine<hw>::get_gp_reg_pool() const {
    // OPENVINO_THROW("Unimplemented!");
    // TODO: REWRIE THIS PART, THIS IS TEMPORARY SOLUTION
    std::vector<snippets::Reg> regs(30);
    for (size_t i = 0; i < regs.size(); ++i) {
        regs[i] = {ov::snippets::RegType::vec, 20 + i};
    }
    return regs;
}

template <ngen::HW hw>
std::vector<snippets::Reg> GPUTargetMachine<hw>::get_vec_reg_pool() const {
    // OPENVINO_THROW("Unimplemented!");
    // TODO: REWRIE THIS PART, THIS IS TEMPORARY SOLUTION
    std::vector<snippets::Reg> regs(30);
    for (size_t i = 0; i < regs.size(); ++i) {
        regs[i] = {ov::snippets::RegType::vec, 50 + i};
    }
    return regs;
}

template <ngen::HW hw>
ngen::HW GPUTargetMachine<hw>::get_hw() const {
    return hw;
}

template <ngen::HW hw>
snippets::CompiledSnippetPtr GPUTargetMachine<hw>::get_snippet() {
    auto compiled_snippets = std::make_shared<CompiledSnippetGPU>();

    const auto& ocl_engine = cldnn::downcast<cldnn::ocl::ocl_engine>(engine);
    const auto& ocl_device = cldnn::downcast<cldnn::ocl::ocl_device>(*engine.get_device());

    auto cl_kernel = cl::Kernel(m_h->getKernel(ocl_engine.get_cl_context().get(), ocl_device.get_device().get()));

    compiled_snippets->kernel = std::make_shared<cldnn::ocl::ocl_kernel>(cldnn::ocl::ocl_kernel_type(cl_kernel, ocl_device.get_usm_helper()),
                                                                         cl_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>());

    return compiled_snippets;
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

GPUGenerator::GPUGenerator(cldnn::engine& engine)
    : Generator(create_target_machine(engine)) {}

GPUGenerator::GPUGenerator(const std::shared_ptr<ov::snippets::TargetMachine>& target)
    : Generator(target) {
    OPENVINO_ASSERT(typeid(*target) == typeid(GPUTargetMachine<ngen::HW::Gen9>) ||
                    typeid(*target) == typeid(GPUTargetMachine<ngen::HW::Gen11>) ||
                    typeid(*target) == typeid(GPUTargetMachine<ngen::HW::Gen12LP>) ||
                    typeid(*target) == typeid(GPUTargetMachine<ngen::HW::XeHP>) ||
                    typeid(*target) == typeid(GPUTargetMachine<ngen::HW::XeHPG>) ||
                    typeid(*target) == typeid(GPUTargetMachine<ngen::HW::XeHPC>) ||
                    typeid(*target) == typeid(GPUTargetMachine<ngen::HW::Xe2>) ||
                    typeid(*target) == typeid(GPUTargetMachine<ngen::HW::Xe3>));
}

std::shared_ptr<snippets::Generator> GPUGenerator::clone() const {
    return std::shared_ptr<GPUGenerator>(new GPUGenerator(target->clone()));
}

ov::snippets::RegType GPUGenerator::get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const {
    return ov::snippets::RegType::undefined;
}

std::shared_ptr<ov::snippets::TargetMachine> GPUGenerator::create_target_machine(cldnn::engine& engine) {
    auto hw = pluginHW2ngen(engine.get_device_info().arch);
    switch (hw) {
    case ngen::HW::Gen9:    return std::make_unique<GPUTargetMachine<ngen::HW::Gen9>>(engine);
    case ngen::HW::Gen11:   return std::make_unique<GPUTargetMachine<ngen::HW::Gen11>>(engine);
    case ngen::HW::Gen12LP: return std::make_unique<GPUTargetMachine<ngen::HW::Gen12LP>>(engine);
    case ngen::HW::XeHP:    return std::make_unique<GPUTargetMachine<ngen::HW::XeHP>>(engine);
    case ngen::HW::XeHPG:   return std::make_unique<GPUTargetMachine<ngen::HW::XeHPG>>(engine);
    case ngen::HW::XeHPC:   return std::make_unique<GPUTargetMachine<ngen::HW::XeHPC>>(engine);
    case ngen::HW::Xe2:     return std::make_unique<GPUTargetMachine<ngen::HW::Xe2>>(engine);
    case ngen::HW::Xe3:     return std::make_unique<GPUTargetMachine<ngen::HW::Xe3>>(engine);
    default:
        OPENVINO_THROW("Unknown GPU hardware!");
    }
}

}  // namespace ov::intel_gpu::jit
