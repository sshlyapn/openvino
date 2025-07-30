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

namespace ov::intel_gpu::jit {

class CompiledSnippetGPU : public snippets::CompiledSnippet {
    const std::unique_ptr<const jit_snippet_base_t> h_compiled;

public:
    [[nodiscard]] const uint8_t* get_code() const override;
    [[nodiscard]] size_t get_code_size() const override;
    [[nodiscard]] bool empty() const override;
    explicit CompiledSnippetGPU(std::unique_ptr<jit_snippet_base_t> h);
};

class GPUTargetMachine : public ov::snippets::TargetMachine {
public:
    explicit GPUTargetMachine(dnnl::impl::gpu::intel::jit::gpu_gen_t hw);

    [[nodiscard]] bool is_supported() const override { return true; }
    [[nodiscard]] std::shared_ptr<snippets::TargetMachine> clone() const override;

    [[nodiscard]] size_t get_lanes() const override;

    [[nodiscard]] std::vector<snippets::Reg> get_abi_arg_regs() const override;
    [[nodiscard]] std::vector<snippets::Reg> get_gp_reg_pool() const override;
    [[nodiscard]] std::vector<snippets::Reg> get_vec_reg_pool() const override;

    [[nodiscard]] dnnl::impl::gpu::intel::jit::gpu_gen_t get_hw() const;

    snippets::CompiledSnippetPtr get_snippet() override;

private:
    dnnl::impl::gpu::intel::jit::gpu_gen_t m_hw;
    std::unique_ptr<jit_snippet_base_t> m_h;
};

class GPUGenerator : public ov::snippets::Generator {
public:
    GPUGenerator(dnnl::impl::gpu::intel::jit::gpu_gen_t hw);
    GPUGenerator(const std::shared_ptr<GPUTargetMachine>& target);
    std::shared_ptr<Generator> clone() const override;
};

}  // namespace ov::intel_gpu::jit
