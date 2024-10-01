// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/dynamic_quantize.hpp"

namespace ov {
namespace intel_gpu {

static void CreateDynamicQuantizeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::DynamicQuantize>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto group_sizes = op->get_group_sizes();

    const auto& users = op->get_users();
    if (users.size() >= 1) {
        // std::cout << "KV cache user of dynamic quantization " << users[0]->get_friendly_name() << "\n";
    } else {
        for (size_t i = 0; i < group_sizes.size() - 1; i++)
            OPENVINO_ASSERT(group_sizes[i] == 1, "Not supported group size at ", i, ": ", group_sizes[i]);

        OPENVINO_ASSERT(group_sizes.back() == UINT64_MAX, "Not supported group size: ", group_sizes.back());
    }

    GPU_DEBUG_TRACE_DETAIL << "Create DQ " << primitive_name << " with scales " << op->get_scales_output_order().size() << " number\n";

    auto prim = cldnn::dynamic_quantize(primitive_name,
                                        inputs[0],
                                        group_sizes,
                                        op->get_scales_output_order(),
                                        get_output_data_types(op));
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, DynamicQuantize);

}  // namespace intel_gpu
}  // namespace ov
