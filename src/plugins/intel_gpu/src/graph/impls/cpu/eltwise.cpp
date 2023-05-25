// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "eltwise_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/maximum.hpp"

namespace cldnn {
namespace cpu {

struct eltwise_impl : public typed_primitive_impl<eltwise> {
    using parent = typed_primitive_impl<eltwise>;
    using parent::parent;

    eltwise_mode mode;

    std::shared_ptr<ov::op::util::BinaryElementwiseArithmetic> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<eltwise_impl>(*this);
    }

    eltwise_impl() : parent("eltwise_cpu_impl") {}

    explicit eltwise_impl(const eltwise_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<eltwise>());
        const auto& node = arg.as<eltwise>();
        mode = node.get_primitive()->mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << make_data(&mode, sizeof(eltwise_mode));
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> make_data(&mode, sizeof(eltwise_mode));
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, eltwise_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "eltwise::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        if (!op) {
            switch (mode) {
            case eltwise_mode::sum:
                op = std::make_shared<ov::op::v1::Add>();
                break;
            case eltwise_mode::prod:
                op = std::make_shared<ov::op::v1::Multiply>();
                break;
            case eltwise_mode::max:
                op = std::make_shared<ov::op::v1::Maximum>();
                break;
            default:
                OPENVINO_THROW("[GPU] Couldn't create eltwise operation: unsupported eltwise operation (", static_cast<size_t>(mode), ")");
            }

            OPENVINO_ASSERT(op->has_evaluate(), "[GPU] Couldn't find evaluate() function for eltwise ",
                                                "primitive with id ", instance.id());
        }

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(input_mem_ptrs[i]->get_layout(), input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        output_host_tensors.push_back(make_tensor(output_mem_ptr->get_layout(), output_lock.data()));

        op->evaluate(output_host_tensors, input_host_tensors);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const eltwise_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<eltwise_impl>();
    }
};


namespace detail {

attach_eltwise_impl::attach_eltwise_impl() {
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

    implementation_map<eltwise>::add(impl_types::cpu, shape_types::static_shape, eltwise_impl::create, types, formats);
    implementation_map<eltwise>::add(impl_types::cpu, shape_types::dynamic_shape, eltwise_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::eltwise_impl)
