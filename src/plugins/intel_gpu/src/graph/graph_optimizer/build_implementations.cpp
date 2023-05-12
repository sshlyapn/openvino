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
        if (iter == max_depth)
            return false;

        std::cout << "--> Call for " << node.id() << " [" << iter << "]" << "\n";

        bool result = false;
        for (auto& user : node.get_users()) {
            std::cout << "----> Check for " << user->id() << " [" << iter << "]" << "\n";
            if (user->is_type<shape_of>())
                continue;

            if (user->in_shape_of_subgraph) {
                node.in_shape_of_subgraph = true;
                std::cout << "Marking " << node.id() << " because it's user already in shape_of subgraph (" << user->id() << ")\n";
                result = true;
                break;
            }

            auto shape_infer_deps = user->get_shape_infer_dependencies();
            auto dep_idx = user->get_dependency_index(node);
            bool in_shape_infer_dep = std::find(shape_infer_deps.begin(), shape_infer_deps.end(), dep_idx) != shape_infer_deps.end();

            auto print_shape_infer_deps = [](std::vector<size_t> vec) {
                std::stringstream ss;
                for (size_t i = 0; i < vec.size(); ++i) {
                    ss << vec[i];
                    if (i < vec.size() - 1)
                        ss << ", ";
                }
                return ss.str();
            };

            bool eltwise_branch = !in_shape_infer_dep;
            if (eltwise_branch) {
                eltwise_branch &= user->is_type<eltwise>();
            }

            std::cout << "----> Shape infer dependencies=" << " {" << print_shape_infer_deps(shape_infer_deps) << "}, dep_idx=" << dep_idx << " [" << iter << "]" << "\n";
            if (in_shape_infer_dep) {
                node.in_shape_of_subgraph = true;
                result = true;
                std::cout << "Marking " << node.id() << " because it is a shape infer dependency of " << user->id() << " {" << print_shape_infer_deps(shape_infer_deps) << "}, dep_idx=" << dep_idx << "\n";

                // Check if all infer_dependencise in shape_of subgraphs
                bool can_traverse_futher = true;
                // for (auto dep_idx : shape_infer_deps) {
                //     const auto& dependency = user->get_dependency(dep_idx);
                //     if (!dependency.in_shape_of_subgraph && !dependency.is_constant()) {
                //         can_traverse_futher = false;
                //         break;
                //     }
                // }
                for (auto& dependency : user->get_dependencies()) {
                    // auto dep_idx = user->get_dependency_index(*dependency.first);
                    if (!dependency.first->in_shape_of_subgraph && !dependency.first->is_constant()) {
                        can_traverse_futher = false;
                        break;
                    }
                }

                if (can_traverse_futher) {
                    std::cout << "NEW! Can traverse futher\n";
                    user->in_shape_of_subgraph = true;
                    std::cout << "Marking " << user->id() << " because we can traverse futher\n";

                    bool res = look_for_shape_of_subgraph(*user, iter + 1, max_depth, print);
                    if (res) {
                        std::cout << "NEW! And got it! For " << user->id() << "\n";
                    }
                }
            } else {
                if (eltwise_branch) {
                    std::cout << "*    Marking " << node.id() << " since it leads to eltwise\n";
                    node.in_shape_of_subgraph = true;
                    result = true;
                }
                bool in_shape_of_subgraph = look_for_shape_of_subgraph(*user, iter + 1, max_depth, print);
                result |= in_shape_of_subgraph;
                if (in_shape_of_subgraph) {
                    std::cout << "Marking " << node.id() << " because it's user in shape_of subgraph (" << user->id() << ")\n";
                    node.in_shape_of_subgraph = true;
                }
            }
        }
        return result;
    };

    if (p.get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
        for (auto& node : p.get_processing_order()) {
            if (node->is_type<shape_of>()) {
                std::cout << "====== Checking " << node->id() << " ======\n";
                auto result = look_for_shape_of_subgraph(*node, 0, 10, node->id() == "shapeof:/model/transformer/h.0/attn/Shape_7");
                if (result == false) {
                    node->in_shape_of_subgraph = true;
                    std::cout << "Marking anyway " << node->id() << "" << std::endl;
                }
                std::cout << "====== Finished " << node->id() << " result " << result << " ======" << std::endl;
            }
        }
    }
}
