// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "intel_gpu/op/dynamic_quantize.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/dynamic_quantize.hpp"


namespace ov {
namespace op {
namespace internal {
using DynamicQuantizeExtended = ov::intel_gpu::op::DynamicQuantize;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateDynamicQuantizeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::DynamicQuantize>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto prim = cldnn::dynamic_quantize(primitive_name,
                                        inputs[0],
                                        op->get_quantization_config());

    p.add_primitive(*op, prim);
}

static void CreateDynamicQuantizeExtendedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::DynamicQuantizeExtended>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto group_sizes = op->get_group_sizes();

    auto prim = cldnn::dynamic_quantize(primitive_name,
                                        inputs[0],
                                        op->get_quantization_config(),
                                        op->get_combine_scales_and_zp(),
                                        op->get_scales_zp_output_order());

    prim.num_outputs = op->get_output_size();

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, DynamicQuantize);
REGISTER_FACTORY_IMPL(internal, DynamicQuantizeExtended);

}  // namespace intel_gpu
}  // namespace ov
