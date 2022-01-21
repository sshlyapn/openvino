// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "intel_gpu/runtime/error_handler.hpp"
#include "pass_manager.h"
#include "program_helpers.h"
#include "strided_slice_inst.h"
#include "reshape_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include <vector>
#include <memory>

using namespace cldnn;

void prepare_primitive_fusing_through::run(program& p) {
    auto try_fuse_through = [&](program_node& node) -> std::vector<program_node*> {
        // This function tries to fuse peer_node to first non reorder or reshape previous primitive.
        // It returns chain of primitives (reshapes and reorders) including potential fused_node (e.g. Conv, FC, etc)
        // at the end of vector.
        // There are some limitations:
        // 1. As for now we can fuse only through chain of reorders and reshapes primitives
        // 2. There are three types of supported primitives:
        //    - Quantize with per-tensor parameters
        //    - Eltwise with single scale value in constant buffer
        //    - Activation w/o additional buffers
        // 3. Paddings are not allowed for fused node
        auto can_raise_up_through = [](program_node* node) {
            if (!node->is_type<reshape>() && !node->is_type<reorder>())
                return false;

            if (node->get_dependencies().empty())
                return false;

            if (node->get_users().size() > 1)
                return false;

            if (node->is_type<reorder>() &&
                node->get_output_layout().data_type != node->get_dependency(0).get_output_layout().data_type)
                return false;

            return true;
        };

        std::vector<program_node*> pass_through;
        program_node* fuse_through = &node;
        pass_through.push_back(fuse_through);

        bool can_raise_up = can_raise_up_through(fuse_through);
        while (can_raise_up) {
            fuse_through = &fuse_through->get_dependency(0);
            can_raise_up = can_raise_up_through(fuse_through);
            pass_through.push_back(fuse_through);
        }

        return pass_through;
    };

    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        cldnn::program_node* input_node;

        if (node->is_type<activation>()) {
            if (node->get_dependencies().size() > 1)
                continue;

            input_node = &node->get_dependency(0);
        } else if (node->is_type<quantize>()) {
            auto& quantize_node = node->as<quantize>();
            bool per_tensor_values = quantize_node.get_scale_shift_opt() &&
                                     quantize_node.get_per_tensor_input_scale() &&
                                     quantize_node.get_per_tensor_input_shift() &&
                                     quantize_node.get_per_tensor_input_range() &&
                                     quantize_node.get_per_tensor_output_scale() &&
                                     quantize_node.get_per_tensor_output_shift() &&
                                     quantize_node.get_per_tensor_output_range();

            if (!per_tensor_values)
                continue;

            input_node = &node->get_dependency(0);
        } else if (node->is_type<eltwise>()) {
            if (node->get_dependencies().size() !=2)
                continue;

            size_t second_input_idx = 0;
            bool has_constant_input = false;
            for (size_t i = 0; i < node->get_dependencies().size(); i++) {
                auto& dep = node->get_dependency(i);
                if (dep.is_constant() && dep.get_output_layout().size == cldnn::tensor(1)) {
                    second_input_idx = i ^ 1;
                    has_constant_input = true;
                    break;
                }
            }

            if (!has_constant_input)
                continue;

            input_node = &node->get_dependency(second_input_idx);
        } else {
            continue;
        }

        auto fuse_through_order = try_fuse_through(*input_node);
        bool use_fuse_through_order = fuse_through_order.size() > 1;
        bool fused_node_has_multiple_users = fuse_through_order.back()->get_users().size() > 1;

                if (!use_fuse_through_order || fused_node_has_multiple_users)
            continue;

        if (static_cast<bool>(node->get_output_layout().data_padding))
            continue;

        printf("Fuse through size %lu for %s:\n", fuse_through_order.size(), node->id().c_str());
        for (auto& nn : fuse_through_order) {
            printf("-> %s\n", nn->id().c_str());
        }

        auto new_prev = fuse_through_order[fuse_through_order.size() - 1];
        auto new_next = fuse_through_order[fuse_through_order.size() - 2];

        std::vector<cldnn::program_node*> dependencies;
        for (auto dep : node->get_dependencies()) {
            if (dep == input_node)
                continue;
            dependencies.push_back(dep);
        }

        for (auto dep : dependencies)
            p.remove_connection(*dep, *node);

        p.move_node(*node, *new_prev, *new_next);

        for (auto dep : dependencies)
            p.add_connection(*dep, *node);

        auto target_dt = node->get_output_layout().data_type;
        auto node_itr = std::next(fuse_through_order.rbegin());
        while (node_itr != fuse_through_order.rend()) {
            auto itermediate_node = *node_itr++;

            if (itermediate_node->get_output_layout().data_type == target_dt)
                continue;

            if (itermediate_node->is_type<reorder>()) {
                // We can't modify reorder's output type directly, so replace old node with new one
                auto reorder_layout = itermediate_node->get_output_layout();
                reorder_layout.data_type = target_dt;
                auto r_prim = std::make_shared<reorder>(itermediate_node->id() + "_reorder_to_req_dt", itermediate_node->id(), reorder_layout);
                p.add_intermediate(r_prim, *itermediate_node->get_users().front(), 0);

                p.add_optimized_primitive_info(itermediate_node->id());
                p.extract_and_remove(*itermediate_node);
            } else {
                itermediate_node->recalc_output_layout(true);
            }
        }
    }
}
