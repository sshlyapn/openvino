// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "shape_of_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/shape_of.hpp"

namespace cldnn {
namespace cpu {

struct shape_of_impl : public typed_primitive_impl<shape_of> {
    using parent = typed_primitive_impl<shape_of>;
    using parent::parent;

    std::string variable_id;
    bool calculated = false;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<shape_of_impl>(*this);
    }

    shape_of_impl() : parent("shape_of_cpu_impl") {}

    explicit shape_of_impl(const shape_of_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<shape_of>());
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, shape_of_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "shape_of::execute_impl");
        auto& stream = instance.get_network().get_stream();


        auto ev = stream.create_user_event(false);

        // GPU_DEBUG_IF_ENV_VAR(execute_once, "EXECUTE_ONCE");

        // if (execute_once)

        // {
        //     OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "shape_of_cpu::wait_for_events");
        //     for (auto e : events) {
        //         e->wait();
        //     }
        // }


        // if (!calculated) {
            auto output_mem_ptr = instance.output_memory_ptr();

            cldnn::mem_lock<int32_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);

            auto shape = instance.get_input_layout().get_shape();
            for (size_t i = 0; i < shape.size(); i++)
                output_lock[i] = shape[i];

            calculated = true;
        // }

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const shape_of_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<shape_of_impl>();
    }
};


namespace detail {

attach_shape_of_impl::attach_shape_of_impl() {
    std::cout << "ShapeOf attach: CPU impl\n";
    implementation_map<shape_of>::add(impl_types::cpu, shape_of_impl::create, {});

    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<shape_of>::add(impl_types::cpu,
                                      shape_types::dynamic_shape,
                                      shape_of_impl::create,
                                      dyn_types,
                                      dyn_formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::shape_of_impl)
