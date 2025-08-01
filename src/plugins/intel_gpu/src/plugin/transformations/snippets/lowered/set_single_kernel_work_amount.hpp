// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/shape_types.hpp"

namespace ov::intel_gpu::pass {

/**
 * @interface SetSingleKernelWorkAmount
 * @brief TODO
 * @ingroup snippets
 */

class SetSingleKernelWorkAmount : public ov::snippets::lowered::pass::Pass {
public:
    OPENVINO_RTTI("SetSingleKernelWorkAmount", "", Pass)
    explicit SetSingleKernelWorkAmount() = default;
    bool run(ov::snippets::lowered::LinearIR& linear_ir) override;
};

}  // namespace ov::intel_gpu::pass
