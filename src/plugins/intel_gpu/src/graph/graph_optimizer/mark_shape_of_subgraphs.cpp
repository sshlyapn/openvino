// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "shape_of_inst.h"
#include "reshape_inst.h"
#include "gather_elements_inst.h"
#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

void mark_shape_of_subgraphs::run(program& p) {
    auto mark_node = [&](program_node& node, program_node& parent_shape_of) {
        node.in_shape_of_subgraph = true;
        node.dependant_shape_of_nodes.insert(&parent_shape_of);
        for (auto dep : node.get_dependencies()) {
            if (dep.first->in_shape_of_subgraph) {
                for (auto shape_of : dep.first->dependant_shape_of_nodes) {
                    node.dependant_shape_of_nodes.insert(shape_of);
                }
            }
        }
        if (update_impls)
            // if (usr->type()->does_possible_implementation_exist(*usr)) {
            if (!node.is_type<reshape>())
                node.set_preferred_impl_type(impl_types::cpu);
    };

    auto can_mark_node = [&](program_node& node) {
        if (node.is_type<reshape>())
            return true;

        impl_types prev_impl = node.get_preferred_impl_type();
        node.set_preferred_impl_type(impl_types::cpu);

        bool cpu_impl_found = (!node.is_dynamic() && node.type()->does_possible_implementation_exist(node)) ||
                              (node.is_dynamic() && node.type()->does_dynamic_implementation_exist(node));

        std::cout << "CPU impl for " << node.id() << "=" << cpu_impl_found << "\n";

        node.set_preferred_impl_type(prev_impl);
        if (cpu_impl_found)
            return true;

        return false;
    };

    std::function<void(program_node& node, program_node& parent_shape_of, size_t iter, size_t max_depth, bool print)>
    look_for_shape_of_subgraph = [&](program_node& node, program_node& parent_shape_of, size_t iter, size_t max_depth, bool print) {
        std::cout << "--> Call for " << node.id() << " [" << iter << "]" << "\n";

        bool shape_of_node = node.is_type<shape_of>();

        if (shape_of_node) {
            mark_node(node, parent_shape_of);
        }

        // TODO: call can_include_to_shape_of_subgraph (node) -> bool, 1) has cpu impl || type<reshape>

        // Check if all dependencise are constant or in shape_of subgraphs
        bool can_traverse_further = true;
        std::stringstream ss;
        for (auto& dependency : node.get_dependencies()) {
            if (!dependency.first->in_shape_of_subgraph && !dependency.first->is_constant()) {
                can_traverse_further = false;
                break;
            } else {
                ss << dependency.first->id() << " (" << dependency.first->in_shape_of_subgraph << ", " << dependency.first->is_constant() << "), ";
            }
        }

        bool in_shape_infer_dep = false;

        for (auto& user : node.get_users()) {
            auto shape_infer_deps = user->get_shape_infer_dependencies();
            auto dep_idx = user->get_dependency_index(node);
            in_shape_infer_dep = std::find(shape_infer_deps.begin(), shape_infer_deps.end(), dep_idx) != shape_infer_deps.end();

            auto print_shape_infer_deps = [](std::vector<size_t> vec) {
                std::stringstream ss;
                for (size_t i = 0; i < vec.size(); ++i) {
                    ss << vec[i];
                    if (i < vec.size() - 1)
                        ss << ", ";
                }
                return ss.str();
            };

            if (in_shape_infer_dep) {
                std::cout << "----> Shape infer dependencies=" << " {" << print_shape_infer_deps(shape_infer_deps) << "}, dep_idx=" << dep_idx << " [" << iter << "]" << "\n";
                break;
            }
        }

        if (!can_traverse_further && !in_shape_infer_dep && !shape_of_node)
            return;

        std::cout << "NEW! Can traverse futher: {" << ss.str() << "}\n";

        if (can_mark_node(node))
            mark_node(node, parent_shape_of);
        else
            return;

        if (can_traverse_further)
            std::cout << "Marking " << node.id() << " because we can traverse futher\n";
        if (in_shape_infer_dep)
            std::cout << "Marking " << node.id() << " because it is in shape infer dependencies\n";

        for (auto& user : node.get_users())
            look_for_shape_of_subgraph(*user, parent_shape_of, iter + 1, max_depth, print);
    };

    if (p.get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
        for (auto& node : p.get_processing_order()) {
            if (node->is_type<shape_of>()) {
                std::cout << "====== Checking " << node->id() << " ======\n";
                look_for_shape_of_subgraph(*node, *node, 0, 10, node->id() == "shapeof:/model/transformer/h.0/attn/Shape_7");
                std::cout << "====== Finished " << node->id() << " result " << " ======" << std::endl;
            }
        }
    }
}
