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

template <ngen::HW hw>
class jit_snippet_t : public ngen::OpenCLCodeGenerator<hw> {
public:
    jit_snippet_t()
        : ngen::OpenCLCodeGenerator<hw>(0, {GENERATOR_NAME, GENERATOR_LINE, false}) {};

    NGEN_FORWARD_OPENCL(hw);
};

}  // namespace ov::intel_gpu::jit
