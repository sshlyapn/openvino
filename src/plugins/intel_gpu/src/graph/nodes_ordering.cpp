// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/program.hpp"
#include "program_node.h"
#include <vector>
#include <map>
#include <algorithm>

namespace cldnn {
// helper method for calc_processing order
void program::nodes_ordering::calc_processing_order_visit(program_node* node) {
    if (node->is_marked())
        return;
    for (auto user : node->users) {
        calc_processing_order_visit(user);
    }
    node->mark();
    _processing_order.push_front(node);
    processing_order_iterators[node] = _processing_order.begin();
    return;
}

// DFS to sort nodes topologically
// any topological sort of nodes is required for further optimizations
void program::nodes_ordering::calc_processing_order(program& p) {
    _processing_order.clear();
    for (auto input : p.get_inputs()) {
        calc_processing_order_visit(input);
    }
    for (auto& node : _processing_order) {
        node->unmark();
    }
    return;
}

/*
    recalculate processing_order
    algorithm based on: CLRS 24.5 (critical path in DAG)
    modifications: adjust for multiple inputs
    input: any topological order in processing order
    output: BFS topological order.
    */
void program::nodes_ordering::calculate_BFS_processing_order() {
// GPU_DEBUG_DEFINE_MEM_LOGGER("calculate_BFS_processing_order_alap");
//         std::map<program_node*, int> distances;
//         for (auto itr : _processing_order) {
//             distances[itr] = -1;
//         }
//         int max_distance = 0;
//         for (auto itr : _processing_order) {
//             // Init
//             if (distances[itr] == -1) {  // this must be an input
//                 distances[itr] = 0;      // initialize input
//             }
//             // RELAX
//             for (auto& user : itr->get_users()) {
//                 distances[user] = std::max(distances[user], distances[itr] + 1);
//                 max_distance = std::max(max_distance, distances[user]);
//             }
//         }

//         // Relax distances more to be ALAP scheduling
//         for (auto itr = _processing_order.rbegin(); itr != _processing_order.rend(); ++itr) {
//             const auto& node_ptr = *itr;
//             if (!node_ptr->get_users().size()) {
//                 // leaf node
//                 distances[node_ptr] = max_distance;
//                 continue;
//             }
//             int32_t min_of_users = INT32_MAX;
//             for (auto & usr : (*itr)->get_users()) {
//                 min_of_users = std::min(min_of_users, distances[usr]);
//             }
//             distances[node_ptr] = min_of_users - 1;
//         }

//         // bucket sort nodes based on their max distance from input
//         std::vector<std::vector<program_node*>> dist_lists;
//         dist_lists.resize(max_distance + 1);
//         for (auto itr : _processing_order) {
//             dist_lists[distances[itr]].push_back(itr);
//         }


//         // replace the old processing order by the new one, still topological.
//         _processing_order.clear();
//         for (auto& dist : dist_lists) {
//             for (auto& node : dist) {
//                 _processing_order.push_back(node);
//                 processing_order_iterators[node] = _processing_order.end();
//                 processing_order_iterators[node]--;
//             }
//         }
//         return;


    GPU_DEBUG_DEFINE_MEM_LOGGER("calculate_BFS_processing_order");
    std::map<program_node*, int> distances;
    for (auto itr : _processing_order) {
        distances[itr] = -1;
    }
    int max_distance = 0;
    for (auto itr : _processing_order) {
        // Init
        if (distances[itr] == -1) {  // this must be an input
            distances[itr] = 0;      // initialize input
        }
        // RELAX
        for (auto& user : itr->get_users()) {
            distances[user] = std::max(distances[user], distances[itr] + 1);
            max_distance = std::max(max_distance, distances[user]);
        }
    }

    auto is_cpu_impl = [&](program_node* node) {
        return node->in_shape_of_subgraph;
    };

    auto has_cpu_impl_usr = [&](program_node* node) {
        for (auto usr : node->get_users()) {
            if (is_cpu_impl(usr))
                return true;
        }
        return false;
    };

    // bucket sort nodes based on their max distance from input
    std::vector<std::vector<program_node*>> dist_lists;
    dist_lists.resize(max_distance + 1);
    for (auto itr : _processing_order) {
        dist_lists[distances[itr]].push_back(itr);
    }

    GPU_DEBUG_IF_ENV_VAR(recalculate_po, "UPDATE_PO");

    if (recalculate_po) {
        for (auto& dist : dist_lists) {
            std::vector<program_node*> tmp = dist;

            std::vector<program_node*> has_cpu_impl_usr_nodes;
            std::vector<program_node*> cpu_impls_nodes;
            for (auto node : dist) {
                if (has_cpu_impl_usr(node) && !is_cpu_impl(node))
                    has_cpu_impl_usr_nodes.push_back(node);
                else if (is_cpu_impl(node))
                    cpu_impls_nodes.push_back(node);
            }

            for (auto node : has_cpu_impl_usr_nodes) {
                auto it = std::find(dist.rbegin(), dist.rend(), node);
                std::rotate(it, it + 1, dist.rend());
            }

            for (auto node : cpu_impls_nodes) {
                auto it = std::find(dist.begin(), dist.end(), node);
                std::rotate(it, it + 1, dist.end());
            }
        }
    }

    // replace the old processing order by the new one, still topological.
    _processing_order.clear();
    for (auto& dist : dist_lists) {
        for (auto& node : dist) {
            _processing_order.push_back(node);
            processing_order_iterators[node] = _processing_order.end();
            processing_order_iterators[node]--;
        }
    }
    return;
}

// verifies if a given node will be processed before all its dependent nodes
bool program::nodes_ordering::is_correct(program_node* node) {
    for (auto& dep : node->get_dependencies()) {
        if (get_processing_number(node) < get_processing_number(dep.first)) {
            return false;
        }
    }
    return true;
}
}  // namespace cldnn
