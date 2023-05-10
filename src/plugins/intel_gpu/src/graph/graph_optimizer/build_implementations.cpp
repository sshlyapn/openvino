// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "reshape_inst.h"
#include "shape_of_inst.h"
#include "eltwise_inst.h"
#include "strided_slice_inst.h"
#include "select_inst.h"

#include "intel_gpu/runtime/itt.hpp"

using namespace cldnn;

void build_implementations::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::build_implementations");
    if (p.get_config().get_property(ov::intel_gpu::partial_build_program)) {
        return;
    }

    auto& cache = p.get_kernels_cache();
    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            cache.add_kernels_source(*params, impl->get_kernels_source());
        }
    }
    cache.build_all();
    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            impl->init_kernels(cache, *params);
            impl->reset_kernels_source();
        }
    }
    cache.reset();

    std::function<bool(program_node& node, size_t iter, size_t max_depth, bool print)>
    look_for_shape_of_subgraph = [&](program_node& node, size_t iter, size_t max_depth, bool print) -> bool {
        bool result = false;
        if (iter == max_depth)
            return false;
        for (auto& user : node.get_users()) {
            if (user->is_type<reshape>() &&
                node.get_users().size() == 1 &&
                node.get_output_layout(false).get_partial_shape().size() == 1 &&
                user->get_dependency_index(node) == 1) {
                std::cout << "Marking " << node.id() << " as user of reshape" << std::endl;
                node.in_shape_of_subgraph = true;
                return true;
            } else if (user->is_type<eltwise>() &&
                       node.get_users().size() == 1 &&
                       user->get_dependency_index(node) == 1) {
                std::cout << "Marking " << node.id() << " as user of eltwise with shape: " << node.get_output_layout(false).get_partial_shape() << std::endl;
                node.in_shape_of_subgraph = true;
                return true;
            } else if (user->is_type<select>() &&
                       node.get_users().size() == 1) {
                std::cout << "Marking " << node.id() << " as user of select with shape: " << node.get_output_layout(false).get_partial_shape() << std::endl;
                node.in_shape_of_subgraph = true;
                return true;
            } else {
                if (look_for_shape_of_subgraph(*user, iter + 1, max_depth, print)) {
                    std::cout << "Marking " << node.id() << "" << std::endl;
                    node.in_shape_of_subgraph = true;
                    result |= true;
                } else {
                    if (print)
                        std::cout << "Can't mark " << node.id() << "" << std::endl;
                    result |= false;
                }
            }
        }
        return result;
    };

    if (p.get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
        for (auto& node : p.get_processing_order()) {
            if (node->is_type<shape_of>()) {
                std::cout << "Checking " << node->id() << std::endl;
                auto result = look_for_shape_of_subgraph(*node, 0, 10, node->id() == "shapeof:/model/transformer/h.0/attn/Shape_7");
                if (result == false) {
                    node->in_shape_of_subgraph = true;
                    std::cout << "Marking anyway " << node->id() << "" << std::endl;
                }
                std::cout << "Finished === " << node->id() << " result " << result << std::endl;
            }
        }
    }
}
