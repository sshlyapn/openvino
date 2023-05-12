// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "gather_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/gather.hpp"

namespace cldnn {
namespace cpu {

struct gather_impl : public typed_primitive_impl<gather> {
    using parent = typed_primitive_impl<gather>;
    using parent::parent;

    int64_t axis;
    int64_t batch_dims;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_impl>(*this);
    }

    gather_impl() : parent("gather_cpu_impl") {}

    // ADD Impl name here
    explicit gather_impl(const gather_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<gather>());
        const auto& node = arg.as<gather>();
        axis = node.get_primitive()->axis;
        batch_dims = node.get_primitive()->batch_dim;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << axis;
        ob << batch_dims;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> axis;
        ib >> batch_dims;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, gather_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        std::cout << "Cpu impl Gather: " << instance.id() << "\n";

        for (auto e : events) {
            e->wait();
        }

        ov::op::v8::Gather op;
        op.set_batch_dims(batch_dims);

        auto axis_tensor = std::make_shared<ngraph::runtime::HostTensor>(ov::element::i64, ov::Shape{1}, static_cast<void*>(&axis));

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        auto output_mem_ptr = instance.output_memory_ptr();

        std::cout << "Gather input types and sizes:\n";
        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            std::cout << "-> " << i << ": " << input_mem_ptrs[i]->get_allocation_type() << " " << input_mem_ptrs[i]->get_layout().to_short_string() << std::endl;

        std::cout << "Gather output types and sizes\n";
        std::cout << "-> " << output_mem_ptr->get_allocation_type() << " " << output_mem_ptr->get_layout().to_short_string() << std::endl;

        cldnn::mem_lock<int32_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);

        ov::HostTensorVector input_host_tensors;
        // ToDo: consider to re-implement lock in more exception-safetest way
        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_host_tensor(input_mem_ptrs[i]->get_layout(), input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        input_host_tensors.push_back(axis_tensor);

        for (size_t i = 0; i < input_host_tensors.size(); i++) {
            std::cout << "Tensor " << i << ": " << input_host_tensors[i]->get_element_type() << " " << input_host_tensors[i]->get_partial_shape() << " " << input_host_tensors[i]->get_data_ptr() << std::endl;
            // for (size_t j = 0; j < (input_host_tensors[i]->get_element_count()); j += 2)
            //     std::cout << j << "." << (int)(static_cast<char*>(input_host_tensors[i]->get_data_ptr())[j]) << " "
            //                           << (int)(static_cast<char*>(input_host_tensors[i]->get_data_ptr())[j + 1]) << std::endl;
        }

        auto output_host_tensor = make_host_tensor(output_mem_ptr->get_layout(), output_lock.data());

        op.evaluate({output_host_tensor}, input_host_tensors);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const gather_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<gather_impl>();
    }
};


namespace detail {

attach_gather_impl::attach_gather_impl() {
    std::cout << "Attach CPU impl for Gather\n";
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

    implementation_map<gather>::add(impl_types::cpu, shape_types::static_shape, gather_impl::create, types, formats);
    implementation_map<gather>::add(impl_types::cpu, shape_types::dynamic_shape, gather_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::gather_impl)
