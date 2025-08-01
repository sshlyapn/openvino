// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_single_kernel_work_amount.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_gpu::pass {

bool SetSingleKernelWorkAmount::run(snippets::lowered::LinearIR& linear_ir) {
    // GPU Plugin requires 1D tile
    linear_ir.set_loop_depth(1);

    const auto& config = linear_ir.get_config();
    if (linear_ir.empty()) {
        return false;
    }

    if (!config.m_enable_domain_optimization) {
        // Unsupported
        return false;
    }

    if (linear_ir.is_dynamic()) {
        // [134873] In dynamic case we need to implement own shape inference in runtime configurator
        return false;
    }

    auto master_shape = linear_ir.get_master_shape();
    if (master_shape.back() == 1) {
        // Already single work amount
        return false;
    }

    auto CollapseDims = [](ov::snippets::VectorDims& dims) {
        OPENVINO_ASSERT(dims.size() >= 2, "CollapseDims can't process shape with less than two dims");
        const auto full_wa_idx = dims.size() - 2;
        dims[full_wa_idx] *= dims[dims.size() - 1];
        dims[dims.size() - 1] = 1;
        for (size_t i = 0; i < full_wa_idx; i++) {
            dims[full_wa_idx] *= dims[i];
            dims[i] = 1;
        }
    };

    const auto& params = linear_ir.get_parameters();
    std::vector<ov::snippets::VectorDims> input_shapes;
    for (const auto& param : params) {
        const auto& desc = param->get_output_port_descriptor(0);
        OPENVINO_ASSERT(ov::snippets::utils::is_planar_layout(desc->get_layout()),
                        "SetSingleKernelWorkAmount supports only planar layout on inputs");
        auto shape = desc->get_shape();
        OPENVINO_ASSERT(std::none_of(shape.begin(),
                                     shape.end(),
                                     [](size_t d) {
                                         return ov::snippets::utils::is_dynamic_value(d);
                                     }),
                        "SetSingleKernelWorkAmount pass does not support dynamic shapes");
        OPENVINO_ASSERT(shape == params.front()->get_output_port_descriptor(0)->get_shape(),
                        "SetSingleKernelWorkAmount pass supports only similar shapes on input");
        CollapseDims(shape);
        input_shapes.emplace_back(shape);
    }

    std::vector<ov::snippets::VectorDimsRef> infer_shapes;
    infer_shapes.reserve(input_shapes.size());
    for (const auto& is : input_shapes) {
        infer_shapes.emplace_back(is);
    }
    // Need to propagate updated shapes through LIR
    linear_ir.shape_infer(infer_shapes);

    return true;
}

}  // namespace ov::intel_gpu::pass
