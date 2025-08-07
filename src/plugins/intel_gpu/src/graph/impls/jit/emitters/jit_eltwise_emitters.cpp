// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"


namespace ov::intel_gpu::jit {


template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
void jit_add_emitter<hw>::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    switch (this->m_exec_prc) {
    case ov::element::Type_t::f16:
        this->m_h->template add<ngen::half>(this->m_h->getSIMD(), ngen::GRF(out_idxs[0]), ngen::GRF(in_idxs[0]), ngen::GRF(in_idxs[1]));
        break;
    case ov::element::Type_t::f32:
        this->m_h->template add<float>(this->m_h->getSIMD(), ngen::GRF(out_idxs[0]), ngen::GRF(in_idxs[0]), ngen::GRF(in_idxs[1]));
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported add emitter data type:", this->m_exec_prc);
        break;
    }
}

template class jit_add_emitter<ngen::HW::Gen9>;
template class jit_add_emitter<ngen::HW::Gen11>;
template class jit_add_emitter<ngen::HW::Gen12LP>;
template class jit_add_emitter<ngen::HW::XeHP>;
template class jit_add_emitter<ngen::HW::XeHPG>;
template class jit_add_emitter<ngen::HW::XeHPC>;
template class jit_add_emitter<ngen::HW::Xe2>;
template class jit_add_emitter<ngen::HW::Xe3>;

}  // namespace ov::intel_gpu::jit
