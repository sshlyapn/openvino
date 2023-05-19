// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "activation_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/power.hpp"


namespace cldnn {
namespace cpu {

struct activation_impl : public typed_primitive_impl<activation> {
    using parent = typed_primitive_impl<activation>;
    using parent::parent;

    activation_func activation_function;
    activation_additional_params params;

    ov::TensorVector input_host_tensors;
    ov::TensorVector output_host_tensors;

    std::shared_ptr<ov::op::util::BinaryElementwiseArithmetic> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<activation_impl>(*this);
    }

    activation_impl() : parent("activation_cpu_impl") {}

    explicit activation_impl(const activation_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<activation>());
        const auto& node = arg.as<activation>();
        activation_function = node.get_primitive()->activation_function;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << make_data(&activation_function, sizeof(activation_func));
        ob << make_data(&params, sizeof(activation_additional_params));
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> make_data(&activation_function, sizeof(activation_func));
        ib >> make_data(&params, sizeof(activation_additional_params));
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, activation_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "activation::execute_impl");
        auto& stream = instance.get_network().get_stream();

        // std::cout << "Cpu impl activation: " << instance.id() << "\n";

        for (auto e : events) {
            e->wait();
        }
        auto ev = stream.create_user_event(false);

        bool need_tensor_creation = input_host_tensors.empty();


        if (need_tensor_creation) {

            if (activation_function == activation_func::pow) {
                op = std::make_shared<ov::op::v1::Power>();
            } else {
                OPENVINO_THROW("[GPU] Unsupported activation\n");
            }

            std::vector<memory::ptr> input_mem_ptrs;
            for (size_t i = 0; i < instance.dependencies().size(); i++)
                input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

            // ToDo: consider to re-implement lock in more exception-safetest way
            for (size_t i = 0; i < input_mem_ptrs.size(); i++)
                input_host_tensors.push_back(make_tensor(input_mem_ptrs[i]->get_layout(), input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

            if (activation_function == activation_func::pow) {
                input_host_tensors.push_back(ov::Tensor(ov::element::Type_t::f32, {}, &params.a));
            } else {
                OPENVINO_THROW("[GPU] Unsupported activation\n");
            }

            auto output_mem_ptr = instance.output_memory_ptr();

            cldnn::mem_lock<int32_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);
            output_host_tensors.push_back(make_tensor(output_mem_ptr->get_layout(), output_lock.data()));
        }

        op->evaluate(output_host_tensors, input_host_tensors);

        if (need_tensor_creation) {
            for (size_t i = 0; i < instance.dependencies().size(); i++)
                instance.dep_memory_ptr(i)->unlock(stream);
        }

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const activation_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<activation_impl>();
    }
};


namespace detail {

attach_activation_impl::attach_activation_impl() {
    std::cout << "Activation attach: CPU impl\n";
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32,
        data_types::i64,
    };

    implementation_map<activation>::add(impl_types::cpu, shape_types::static_shape, activation_impl::create, types, formats);
    implementation_map<activation>::add(impl_types::cpu, shape_types::dynamic_shape, activation_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::activation_impl)
