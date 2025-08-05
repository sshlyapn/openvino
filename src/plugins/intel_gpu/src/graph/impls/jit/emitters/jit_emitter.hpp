// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/emitter.hpp"

#include "graph/impls/jit/jit_generator.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_gpu::jit {

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
class jit_emitter : public ov::snippets::Emitter {
public:
    jit_emitter(dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>* host,
                ov::element::Type exec_prc = ov::element::f32) :
        m_h(host),
        m_exec_prc(exec_prc) {}

    /**
     * @brief Returns supported precisions.
     * Precisions are ordered, the first bigger bitness precision with the same type will be selected.
     * Empty collection means the emitter supports any input precisions.
     */
    static std::set<std::vector<ov::element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr) {
        return {};
    }

protected:
    void emit_code_impl(const std::vector<size_t>& in,
                        const std::vector<size_t>& out,
                        const std::vector<size_t>& pool,
                        const std::vector<size_t>& gpr) const override {
        OPENVINO_THROW("Unimplemented");
    }

    dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>* m_h;
    ov::element::Type m_exec_prc;
};

}  // namespace ov::intel_gpu::jit
