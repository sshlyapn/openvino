// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "fully_connected_inst.h"

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
            if (n->is_type<fully_connected>() && n->is_dynamic()) {
                if (n->as<fully_connected>().weights().get_output_layout().data_type == data_types::i4 ||
                    n->as<fully_connected>().weights().get_output_layout().data_type == data_types::u4) {
                    auto& impls_cache = p.get_implementations_cache();
                    auto input_layout = n->get_input_layout();
                    auto output_layout = n->get_output_layout();
                    auto new_input_shape = input_layout.get_partial_shape();
                    new_input_shape[0] = 1;
                    if (new_input_shape.size() > 2)
                        new_input_shape[1] = 1;
                    auto new_output_shape = output_layout.get_partial_shape();
                    new_output_shape[0] = 1;
                    if (new_output_shape.size() > 2)
                        new_output_shape[1] = 1;

                    GPU_DEBUG_TRACE_DETAIL << n->id() << ": generate static kernel. Input: " << input_layout.to_short_string() << " to " << new_input_shape << std::endl
                                        << "Output: " << output_layout.to_short_string() << " to " << new_output_shape << std::endl;

                    input_layout.set_partial_shape(new_input_shape);
                    output_layout.set_partial_shape(new_output_shape);

                    OPENVINO_ASSERT(!new_input_shape.is_dynamic(), "Unexpected dynamic input shape", input_layout.to_short_string());
                    OPENVINO_ASSERT(!new_output_shape.is_dynamic(), "Unexpected dynamic output shape", output_layout.to_short_string());

                    auto static_kernel_params = *params;

                    static_kernel_params.input_layouts[0] = input_layout;
                    static_kernel_params.output_layouts[0] = output_layout;

                    if (!impls_cache.has(static_kernel_params)) {
                        auto impl = n->type()->choose_impl(*n, static_kernel_params);
                        if (impl->get_kernels_source().size() > 0) {
                            auto kernels = p.get_kernels_cache().compile(static_kernel_params, impl->get_kernels_source());
                            impl->set_kernels(kernels);
                        }
                        impls_cache.add(static_kernel_params, impl->clone());
                    }
                }
            }
            impl->init_kernels(cache, *params);
            impl->reset_kernels_source();
        }
    }
    cache.reset();
}
