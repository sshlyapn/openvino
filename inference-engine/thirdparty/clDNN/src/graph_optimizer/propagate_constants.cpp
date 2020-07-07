/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "engine_impl.h"
#include "program_impl.h"
#include "network_impl.h"
#include "data_inst.h"
#include <vector>
#include <list>
#include <memory>
#include <utility>

using namespace cldnn;

// ToDo remove friendship relation from  program_node and program_impl
void propagate_constants::run(program_impl& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_constant())
            handle_constant(p, *node);
    }

    auto&& to_replace = calculate(p, p.get_engine(), p.get_options());

    // remove all nodes which are no longer relevant, i.e. nodes which:
    // 1. are constants, and
    // 2. do not have non-const user (so their data are not used during inference), and
    // 3. are not marked as outputs.
    // in case if node has either non-const user or is marked as output, it should be replace with cldnn::data rather
    // than removed (see next loop)
    auto proc_itr = p.get_processing_order().begin();
    while (proc_itr != p.get_processing_order().end()) {
        auto& node = (*proc_itr++);
        if (!node->is_constant())
            continue;
        if (has_non_const_user(*node) || (node->is_output() && !node->is_type<data>()))
            continue;

        auto& users = node->users;
        auto& deps = node->dependencies;

        for (size_t idx = 0; idx < deps.size(); idx++) {
            deps.at(idx)->users.remove(node);
        }
        deps.clear();

        for (auto& usr : users) {
            auto& usr_deps = usr->dependencies;
            usr_deps.erase(std::remove(usr_deps.begin(), usr_deps.end(), node), usr_deps.end());
        }
        users.clear();

        if (!node->is_output()) {
            auto rem = p.remove_if_dangling(*node);
            assert(rem &&
                   "Non-output constant node which has only constant users should have been removed during constants "
                   "propagation pass");
            (void)rem;
        }
    }

    // replace all constant nodes which are relevant for inference (either used by non-const user or marked as output)
    // with recomputed cldnn::data
    for (auto& cout : to_replace) {
        auto& id_to_replace = cout.first;
        auto mem_impl = cout.second;

        memory api_memory = memory(mem_impl.detach());

        auto const_data =
            std::make_shared<data>("_cldnn_const_prop_" + id_to_replace, api_memory /* <<< REMOVE ME WHEN POSSIBLE */);
        auto& new_node = p.get_or_create(const_data);
        auto& curr_node = p.get_node(id_to_replace);

        // Remove dependencies
        auto curr_node_deps = curr_node.get_dependencies();
        for (auto& dep : curr_node_deps) {
            auto dep_users = dep->get_users();
            for (auto& dep_user : dep_users) {
                if (dep_user == &curr_node)
                    p.remove_connection(*dep, curr_node);
            }
        }

        curr_node.dependencies.clear();
        // remove all constant users (as they will be either removed or replaced by cldnn::data which does not have any
        // dependencies)
        curr_node.users.erase(std::remove_if(curr_node.users.begin(),
                                             curr_node.users.end(),
                                             [](program_node* node) { return node->is_constant(); }),
                              curr_node.users.end());
        p.replace(curr_node, new_node);
    }
}

bool propagate_constants::has_non_const_user(program_node& node) const {
    if (!node.is_constant())
        return true;
    for (auto& user : node.get_users()) {
        if (!user->is_constant())
            return true;
    }
    return false;
}

inline size_t get_os_is_yx_isv16_osv16_index(size_t o, size_t i, size_t y, size_t x, size_t i_size, size_t y_size, size_t x_size) {
    const size_t idx = o%16 + (o / 16)*i_size*x_size*y_size*16 +
                       16 *(i+ x*i_size + y*i_size*x_size);
    return idx;
}

std::list<std::pair<primitive_id, memory_impl::ptr>> propagate_constants::calculate(program_impl& prog, engine_impl& engine, build_options bo) {
    if (!has_non_trivial_constants)
        return {};
/*
build_time took 22120.3 ms
input_time took 1.03303 ms
exec_time took 159.505 ms
output_time took 0.018459 ms

build_time took 8868.78 ms
input_time took 0.406849 ms
exec_time took 32.8557 ms
output_time took 0.018 ms

build_time took 8848.86 ms
input_time took 0.36911 ms
exec_time took 33.0224 ms
output_time took 0.019228 ms

Kernel execution took 228.331 ms
Kernel execution took 1.93725 ms
Kernel execution took 216.723 ms
Kernel execution took 493.409 ms
Kernel execution took 217.171 ms
Kernel execution took 216.534 ms
Kernel execution took 514.311 ms
Kernel execution took 220.925 ms
Kernel execution took 223.924 ms
Kernel execution took 493.691 ms
Kernel execution took 110.755 ms
Kernel execution took 437.372 ms
Kernel execution took 54.5999 ms
Kernel execution took 131.777 ms
Kernel execution took 59.1095 ms
Kernel execution took 56.7519 ms
Kernel execution took 123.873 ms
Kernel execution took 56.7898 ms
Kernel execution took 54.8657 ms
Kernel execution took 134.968 ms
Kernel execution took 57.0889 ms
Kernel execution took 54.7504 ms
Kernel execution took 132.178 ms
Kernel execution took 58.0358 ms
Kernel execution took 57.5134 ms
Kernel execution took 131.258 ms
Kernel execution took 57.5604 ms
Kernel execution took 57.6085 ms
Kernel execution took 129.47 ms
Kernel execution took 28.4483 ms
Kernel execution took 109.309 ms
Kernel execution took 14.0315 ms
Kernel execution took 31.9061 ms
Kernel execution took 15.0036 ms
Kernel execution took 15.1069 ms
Kernel execution took 33.6183 ms
Kernel execution took 13.754 ms
Kernel execution took 14.497 ms
Kernel execution took 33.2015 ms
Kernel execution took 14.5165 ms
Kernel execution took 14.3987 ms
Kernel execution took 33.5299 ms
Kernel execution took 7.26093 ms
Kernel execution took 27.2417 ms
Kernel execution took 3.50885 ms
Kernel execution took 8.16403 ms
Kernel execution took 3.54645 ms
Kernel execution took 3.57583 ms
Kernel execution took 8.13936 ms
Kernel execution took 3.57489 ms
Kernel execution took 3.55556 ms
Kernel execution took 8.40556 ms
Kernel execution took 0.88692 ms
Kernel execution took 3.3837 ms
*/


    std::chrono::high_resolution_clock::time_point tt = std::chrono::high_resolution_clock::now();
    std::list<std::pair<primitive_id, memory_impl::ptr>> ret;
    // for (auto it = nodes.begin(); it != nodes.end();) {
    //     // if ((*it)->get_output_layout().format == cldnn::format::os_is_yx_isv16_osv16) {
    //     //     printf("os_is_yx_isv16_osv16 for %s\n", (*it)->id().c_str());
    //     // } else {
    //     //     it++;
    //     //     printf("%d for %s\n", (int)(*it)->get_output_layout().format,  (*it)->id().c_str());
    //     // }
    //     if ((*it)->get_output_layout().format == cldnn::format::os_is_yx_isv16_osv16 && 
    //         (*it)->get_dependencies().size() != 0 &&
    //         (*it)->get_dependency(0).is_type<data>()) {
    //         printf("Dependency is a data!\n");
    //         auto &original_weights = (*it)->get_dependency(0).template as<data>();
    //         auto &mem = original_weights.get_attached_memory();
    //         auto data = static_cast<float*>(mem.lock());
    //         printf("Dependency is a data! %f\n", data[0]);
    //         size_t ofm_size = mem.get_layout().size.batch[0];
    //         size_t ifm_size = mem.get_layout().size.feature[0];
    //         size_t x_size = mem.get_layout().size.spatial[0];
    //         size_t y_size = mem.get_layout().size.spatial[1];

    //         memory_impl::ptr data_to_allocate = engine.allocate_memory((*it)->get_output_layout(), 0);
    //         auto out = static_cast<float*>(data_to_allocate->lock());

    //         size_t oiyx_idx = 0;
    //         for (size_t ofm = 0; ofm < ofm_size; ofm++) {
    //             for (size_t ifm = 0; ifm < ifm_size; ifm++) {
    //                 for (size_t y = 0; y < y_size; y++) {
    //                     for (size_t x = 0; x < x_size; x++) {
    //                         float w = data[oiyx_idx];
    //                         out[get_os_is_yx_isv16_osv16_index(ofm, ifm, y, x, ifm_size, y_size, x_size)] = w;
    //                     }
    //                 }
    //             }
    //         }
    //         mem.unlock();
    //         data_to_allocate->unlock();
    //         ret.push_back({(*it)->id(), data_to_allocate});
    //         it++;
    //     } else {
    //         it++;
    //     }
    // }

    printf("Here\n");

    bo.set_option(build_option::optimize_data(false));
    bo.set_option(build_option::outputs(const_outputs));
    std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
    
    network_impl::ptr net = engine.build_network(nodes, bo, true);
    
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    
    for (auto& cin : const_inputs) net->set_input_data(cin->id(), cin->get_attached_memory());

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    net->execute({});
    net->reset_execution(true);  // wait for computations to complete
    auto outputs = net->get_outputs();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    for (auto& out : outputs) {
        // bool check = false;
        // for (auto& r : ret)
        //     if (r.first == out->id()) {
        //         check = true;
        //         continue;
        //     }
        // if (check)
        //     continue;
        ret.push_back({out->id(), (memory_impl::ptr) &out->output_memory()});
    }

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> online_time = t - tt;
    std::chrono::duration<double, std::milli> build_time = t0 - t;
    std::chrono::duration<double, std::milli> input_time = t1 - t0;
    std::chrono::duration<double, std::milli> exec_time = t2 - t1;
    std::chrono::duration<double, std::milli> output_time = t3 - t2;

    std::cout << "online_time took " << online_time.count() << " ms\n";
    std::cout << "build_time took " << build_time.count() << " ms\n";
    std::cout << "input_time took " << input_time.count() << " ms\n";
    std::cout << "exec_time took " << exec_time.count() << " ms\n";
    std::cout << "output_time took " << output_time.count() << " ms\n";

    return ret;
}

void propagate_constants::handle_constant(program_impl& prog, program_node& node) {
    // printf("Handle constant: %s %d - %d\n", node.id().c_str(), (bool)node.is_type<data>(), (bool)node.is_type<generic_layer>());
    if (!node.is_type<data>()) {
        add_constant(prog, node);
        if (has_non_const_user(node))
            const_outputs.push_back(node.id());
    }
}

void propagate_constants::add_constant(program_impl& prog, program_node& node) {
    if (node.is_type<data>())
        return;
    if (node.get_output_layout().format == cldnn::format::os_is_yx_isv16_osv16 && 
            node.get_dependencies().size() != 0 &&
            node.get_dependency(0).is_type<data>()) {
        online_nodes.insert(prog.get_node_ptr(node.get_primitive()->id));
        printf("Online\n");
    }
    nodes.insert(prog.get_node_ptr(node.get_primitive()->id));
    has_non_trivial_constants = true;

    // if a node is either an endpoint or an output, always add it as an output
    if (node.is_endpoint() || node.is_output())
        const_outputs.push_back(node.id());

    // if a non-tirivial constant has a trivial input, add this input as an input for our network
    add_deps_to_tpl(prog, node.get_dependencies());
}

void propagate_constants::add_deps_to_tpl(program_impl& prog, const std::vector<program_node*>& deps) {
    /*
    Nodes can share dependencies, if we already have dep in tpl, don't add it again.
    example:
    C   <--- shared dep
    / \
    /   \
    A     B
    */
    for (auto& dep : deps) {
        if (dep->is_type<data>()) {
            auto dep_ptr = prog.get_node_ptr(dep->get_primitive()->id);
            if (nodes.find(dep_ptr) == nodes.end()) {
                nodes.insert(prog.get_node_ptr(dep->get_primitive()->id));
                const_inputs.push_back(&dep->as<data>());
            }
        }
    }
}
