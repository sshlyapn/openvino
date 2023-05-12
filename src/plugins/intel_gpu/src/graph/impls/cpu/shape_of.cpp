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
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        std::cout << "Cpu impl Concat: " << instance.id() << " axis=" << instance.get_typed_desc<concatenation>()->axis << "\n";

        // auto input_mem_ptr = instance.dep_memory_ptr(0);
        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<int32_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);

        auto shape = instance.get_input_layout().get_shape();
        for (size_t i = 0; i < shape.size(); i++)
            output_lock[i] = shape[i];

        {
            // auto input_mem_ptr = instance.dep_memory_ptr(0);
            // auto output_mem_ptr = instance.output_memory_ptr();

            // cldnn::mem_lock<uint8_t, mem_lock_type::read> input_lock(input_mem_ptr, stream);
            // cldnn::mem_lock<uint8_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);

            // auto input = make_host_tensor(input_mem_ptr->get_layout(), input_lock.data());
            // auto output = make_host_tensor(output_mem_ptr->get_layout(), output_lock.data());

            // ov::op::v0::ShapeOf op;

            // op.evaluate({output}, {input});
        }

        return stream.create_user_event(true);
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
