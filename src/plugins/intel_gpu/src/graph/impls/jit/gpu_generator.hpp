// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "jit_generator.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "snippets/emitter.hpp"
#include "snippets/generator.hpp"
#include "snippets/target_machine.hpp"

#include "runtime/ocl/ocl_device.hpp"
#include "runtime/ocl/ocl_kernel.hpp"
#include "common_utils/kernel_generator_base.hpp"

#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/utils.hpp"

namespace ov::intel_gpu::jit {

class CompiledSnippetGPU : public snippets::CompiledSnippet {
public:
    [[nodiscard]] const uint8_t* get_code() const override;
    [[nodiscard]] size_t get_code_size() const override;
    [[nodiscard]] bool empty() const override;
    explicit CompiledSnippetGPU() = default;

    std::shared_ptr<cldnn::ocl::ocl_kernel> kernel{nullptr};
    ov::intel_gpu::KernelData kernels_data{};
};

template <ngen::HW hw>
class GPUTargetMachine : public ov::snippets::TargetMachine {
public:
    explicit GPUTargetMachine(cldnn::engine& engine);

    [[nodiscard]] bool is_supported() const override { return true; }
    [[nodiscard]] std::shared_ptr<snippets::TargetMachine> clone() const override;

    [[nodiscard]] size_t get_lanes() const override;

    [[nodiscard]] std::vector<snippets::Reg> get_abi_arg_regs() const override;
    [[nodiscard]] std::vector<snippets::Reg> get_gp_reg_pool() const override;
    [[nodiscard]] std::vector<snippets::Reg> get_vec_reg_pool() const override;

    [[nodiscard]] dnnl::impl::gpu::intel::jit::gpu_gen_t get_hw() const;

    snippets::CompiledSnippetPtr get_snippet() override;

private:
    std::unique_ptr<jit_snippet_t<hw>> m_h;
    cldnn::engine& engine;
};

class GPUGenerator : public ov::snippets::Generator {
public:
    GPUGenerator(cldnn::engine& engine);
    std::shared_ptr<Generator> clone() const override;

    ov::snippets::RegType get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const override;

private:
    GPUGenerator(const std::shared_ptr<ov::snippets::TargetMachine>& target);

    static std::shared_ptr<ov::snippets::TargetMachine> create_target_machine(cldnn::engine& engine);
};

}  // namespace ov::intel_gpu::jit
