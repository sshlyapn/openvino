// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

#include "snippets/lowered/expression.hpp"

namespace ov::intel_gpu::jit {

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
class jit_kernel_emitter : public jit_emitter<hw> {
public:
    jit_kernel_emitter(jit_snippet_t<hw>* host,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    static std::set<std::vector<ov::element::Type>> get_supported_precisions(
        [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
        return {};
    }

    size_t get_inputs_count() const override { return 0; };

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    std::vector<size_t> data_ptr_regs_idx;
    size_t num_inputs = 0;
    size_t num_outputs = 0;

    std::shared_ptr<snippets::lowered::LinearIR> body;
};

}  // namespace ov::intel_gpu::jit
