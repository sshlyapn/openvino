// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "concatenation_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/concat.hpp"

namespace cldnn {
namespace cpu {

struct concatenation_impl : public typed_primitive_impl<concatenation> {
    using parent = typed_primitive_impl<concatenation>;
    using parent::parent;

    ov::HostTensorVector input_host_tensors_cache;
    ov::HostTensorVector output_host_tensors_cache;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<concatenation_impl>(*this);
    }

    concatenation_impl() : parent("concatenation_cpu_impl") {}

    explicit concatenation_impl(const concatenation_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<concatenation>());
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, concatenation_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "concat::execute_impl");
        auto& stream = instance.get_network().get_stream();

        {
            OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "concat_cpu::wait_for_events");
            for (auto e : events) {
                e->wait();
            }
        }
        auto ev = stream.create_user_event(false);

        bool reallocate_tensors = input_host_tensors_cache.empty() || instance.get_network().reallocate_tensors;
        ov::HostTensorVector input_host_tensors;
        ov::HostTensorVector output_host_tensors;

        for (auto in_layout : instance.get_impl_params()->input_layouts)
            OPENVINO_ASSERT(in_layout.data_type == instance.get_impl_params()->get_output_layout().data_type, "[GPU] Unsupported mixed formats");

        if (reallocate_tensors) {
            std::vector<memory::ptr> input_mem_ptrs;
            for (size_t i = 0; i < instance.dependencies().size(); i++)
                input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

            auto output_mem_ptr = instance.output_memory_ptr();

            cldnn::mem_lock<int32_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);

            for (size_t i = 0; i < input_mem_ptrs.size(); i++)
                input_host_tensors.push_back(make_host_tensor(input_mem_ptrs[i]->get_layout(), input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

            output_host_tensors.push_back(make_host_tensor(output_mem_ptr->get_layout(), output_lock.data()));

            if (!instance.get_network().reallocate_tensors) {
                input_host_tensors_cache = input_host_tensors;
                output_host_tensors_cache = output_host_tensors;
            }
        } else {
            input_host_tensors = input_host_tensors_cache;
            output_host_tensors = output_host_tensors_cache;
        }

        ov::op::v0::Concat op;
        op.set_axis(instance.get_typed_desc<concatenation>()->axis);

        op.evaluate(output_host_tensors, input_host_tensors);

        if (reallocate_tensors) {
            for (size_t i = 0; i < instance.dependencies().size(); i++)
                instance.dep_memory_ptr(i)->unlock(stream);
        }

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const concatenation_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<concatenation_impl>();
    }
};


namespace detail {

attach_concatenation_impl::attach_concatenation_impl() {
    std::cout << "Concat attach: CPU impl\n";
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

    implementation_map<concatenation>::add(impl_types::cpu, shape_types::static_shape, concatenation_impl::create, types, formats);
    implementation_map<concatenation>::add(impl_types::cpu, shape_types::dynamic_shape, concatenation_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::concatenation_impl)
