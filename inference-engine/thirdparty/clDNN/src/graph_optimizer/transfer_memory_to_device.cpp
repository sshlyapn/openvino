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

#include <algorithm>

#include "pass_manager.h"
#include "program_node.h"
#include "mutable_data_inst.h"
#include "concatenation_inst.h"
#include "scale_inst.h"
#include "tensor_type.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include "gpu/memory_gpu.h"

void transfer_memory_to_device_pass::run(program_impl& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<data>() && !node->need_lockable_memory()) {
            auto& data_node = node->as<data>();
            auto& mem = data_node.get_attached_memory();
            auto alloc_type = mem.get_allocation_type();

            if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
                // Allocate and transfer memory
                auto device_mem = mem.get_engine()->allocate_memory(mem.get_layout(),
                                                                    allocation_type::usm_device,
                                                                    mem.get_net_id());
                dynamic_cast<gpu::gpu_usm&>(*device_mem).copy_from_other(dynamic_cast<gpu::gpu_usm&>(mem));
                data_node.attach_memory(*device_mem);
                const_cast<memory&>(data_node.get_primitive()->mem).reset();
            }
        }
    }
}
