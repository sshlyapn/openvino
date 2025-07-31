// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"


namespace ov::intel_gpu::jit {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter() = default;

    static std::set<std::vector<ov::element::Type>> get_supported_precisions(
        [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
        return {{element::f32, element::f32}, {element::f16, element::f16}};
    }
};

}  // namespace ov::intel_gpu::jit
