// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

#include "snippets/lowered/expression.hpp"


namespace ov::intel_gpu::jit {

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
class jit_add_emitter : public jit_emitter<hw> {
public:
    jit_add_emitter(jit_snippet_t<hw>* host,
                    [[maybe_unused]] const ov::snippets::lowered::ExpressionPtr& expr,
                    ov::element::Type exec_prc = ov::element::f32) : jit_emitter<hw>(host, exec_prc) {};

    static std::set<std::vector<ov::element::Type>> get_supported_precisions(
        [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
        return {{element::f32, element::f32}, {element::f16, element::f16}};
    }

    size_t get_inputs_count() const override { return 2; };

protected:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override { 
        OPENVINO_THROW("Unimplemented");
    }
};

}  // namespace ov::intel_gpu::jit
