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
    jit_emitter(jit_snippet_t<hw>* host, ov::element::Type exec_prc = ov::element::f32) :
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

    virtual size_t get_inputs_count() const = 0;
    virtual size_t get_aux_vecs_count() const { return 0; }
    virtual size_t get_aux_gprs_count() const { return 0; }

protected:
    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_aux_vec_idxs,
                        const std::vector<size_t>& pool_aux_gpr_idxs) const override {
        emitter_preamble(in_idxs, out_idxs, pool_aux_vec_idxs, pool_aux_gpr_idxs);

        emit_impl(in_idxs, out_idxs);

        emitter_postamble();
    }

    virtual void emitter_preamble(const std::vector<size_t>& in_idxs,
                                  const std::vector<size_t>& out_idxs,
                                  const std::vector<size_t>& pool_aux_vec_idxs,
                                  const std::vector<size_t>& pool_aux_gpr_idxs) const {
        aux_vec_idxs = pool_aux_vec_idxs;
        aux_gpr_idxs = pool_aux_gpr_idxs;
        OPENVINO_ASSERT(aux_vec_idxs.size() >= get_aux_vecs_count(), "Not enough aux vec regs");
        OPENVINO_ASSERT(aux_gpr_idxs.size() >= get_aux_gprs_count(), "Not enough aux gpr regs");
    }

    virtual void emitter_postamble() const {}

    virtual void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const = 0;

    jit_snippet_t<hw>* m_h;
    ov::element::Type m_exec_prc;

    mutable std::vector<size_t> aux_vec_idxs;
    mutable std::vector<size_t> aux_gpr_idxs;
};

#define TEMPLATE_INSTANCE(emitter, hw) \
    template class emitter<hw>;

#define TEMPLATE_INSTANCES(emitter) \
    TEMPLATE_INSTANCE(emitter, ngen::HW::Gen9)    \
    TEMPLATE_INSTANCE(emitter, ngen::HW::Gen11)   \
    TEMPLATE_INSTANCE(emitter, ngen::HW::Gen12LP) \
    TEMPLATE_INSTANCE(emitter, ngen::HW::XeHP)    \
    TEMPLATE_INSTANCE(emitter, ngen::HW::XeHPG)   \
    TEMPLATE_INSTANCE(emitter, ngen::HW::XeHPC)   \
    TEMPLATE_INSTANCE(emitter, ngen::HW::Xe2)     \
    TEMPLATE_INSTANCE(emitter, ngen::HW::Xe3)
 

}  // namespace ov::intel_gpu::jit
