// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"


namespace ov::intel_gpu::jit {

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
class jit_nop_emitter : public jit_emitter<hw> {
public:
    jit_nop_emitter(dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>* host,
                    ov::element::Type exec_prc = ov::element::f32) : jit_emitter<hw>(host, exec_prc) {};

    static std::set<std::vector<ov::element::Type>> get_supported_precisions(
        [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
        return {};
    }
};

}  // namespace ov::intel_gpu::jit
