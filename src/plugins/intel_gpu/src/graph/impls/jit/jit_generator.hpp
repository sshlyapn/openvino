// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/intel/jit/generator.hpp"


namespace ov::intel_gpu::jit {

struct jit_snippet_base_t {
    virtual ~jit_snippet_base_t() = default;
    virtual const char *kernel_name() const = 0;

    virtual int getSIMD() const = 0;
    virtual int getGRFCount() const = 0;
};

template <dnnl::impl::gpu::intel::jit::gpu_gen_t hw>
class jit_snippet_t : public dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>, public jit_snippet_base_t {
public:
    jit_snippet_t()
        : dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>(0, {GENERATOR_NAME, GENERATOR_LINE, false}) {};

    const char *kernel_name() const override {
        return dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>::getExternalName().c_str();
    }
    int getSIMD() const override {
        return dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>::getSIMD();
    };
    int getGRFCount() const override {
        return dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>::getGRFCount();
    }
};

}  // namespace ov::intel_gpu::jit
