// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "strided_slice_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/strided_slice.hpp"

namespace cldnn {
namespace cpu {

struct strided_slice_impl : public typed_primitive_impl<strided_slice> {
    using parent = typed_primitive_impl<strided_slice>;
    using parent::parent;

    std::vector<int64_t> begin_data;
    std::vector<int64_t> end_data;
    std::vector<int64_t> strides_data;

    std::vector<int64_t> begin_mask;
    std::vector<int64_t> end_mask;
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_mask;

    std::shared_ptr<ov::op::v1::StridedSlice> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<strided_slice_impl>(*this);
    }

    strided_slice_impl() : parent("strided_slice_cpu_impl") {}

    explicit strided_slice_impl(const strided_slice_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<strided_slice>());
        const auto& node = arg.as<strided_slice>();
        begin_data = node.get_primitive()->begin;
        end_data = node.get_primitive()->end;
        strides_data = node.get_primitive()->strides;

        begin_mask = node.get_primitive()->begin_mask;
        end_mask = node.get_primitive()->end_mask;
        new_axis_mask = node.get_primitive()->new_axis_mask;
        shrink_axis_mask = node.get_primitive()->shrink_axis_mask;
        ellipsis_mask = node.get_primitive()->ellipsis_mask;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << begin_data;
        ob << end_data;
        ob << strides_data;

        ob << begin_mask;
        ob << end_mask;
        ob << new_axis_mask;
        ob << shrink_axis_mask;
        ob << ellipsis_mask;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> begin_data;
        ib >> end_data;
        ib >> strides_data;

        ib >> begin_mask;
        ib >> end_mask;
        ib >> new_axis_mask;
        ib >> shrink_axis_mask;
        ib >> ellipsis_mask;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, strided_slice_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "strided_slice::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        ov::Tensor begin_host_tensor;
        ov::Tensor end_host_tensor;
        ov::Tensor strides_host_tensor;

        if (begin_data.empty()) {
            auto begin_mem = instance.dep_memory_ptr(1);
            begin_host_tensor = make_tensor(begin_mem->get_layout(), begin_mem->lock(stream, mem_lock_type::read));
        } else {
            ov::Shape begin_shape = ov::Shape{ begin_data.size() };
            begin_host_tensor = make_tensor({ begin_shape, data_types::i64, format::bfyx }, static_cast<void*>(begin_data.data()));
        }

        if (end_data.empty()) {
            auto end_mem = instance.dep_memory_ptr(2);
            end_host_tensor = make_tensor(end_mem->get_layout(), end_mem->lock(stream, mem_lock_type::read));
        } else {
            ov::Shape end_shape = ov::Shape{ end_data.size() };
            end_host_tensor = make_tensor({ end_shape, data_types::i64, format::bfyx }, static_cast<void*>(end_data.data()));
        }

        if (strides_data.empty()) {
            auto strides_mem = instance.dep_memory_ptr(3);
            strides_host_tensor = make_tensor(strides_mem->get_layout(), strides_mem->lock(stream, mem_lock_type::read));
        } else {
            ov::Shape strides_shape = ov::Shape{ strides_data.size() };
            strides_host_tensor = make_tensor({ strides_shape, data_types::i64, format::bfyx }, static_cast<void*>(strides_data.data()));
        }

        auto input_mem_ptr = instance.dep_memory_ptr(0);
        auto output_mem_ptr = instance.output_memory_ptr();

        input_host_tensors.push_back(make_tensor(input_mem_ptr->get_layout(), input_mem_ptr->lock(stream, mem_lock_type::read)));
        input_host_tensors.push_back(begin_host_tensor);
        input_host_tensors.push_back(end_host_tensor);
        input_host_tensors.push_back(strides_host_tensor);
        output_host_tensors.push_back(make_tensor(output_mem_ptr->get_layout(), output_mem_ptr->lock(stream, mem_lock_type::write)));

        if (!op) {
            op = std::make_shared<ov::op::v1::StridedSlice>();

            op->set_begin_mask(begin_mask);
            op->set_end_mask(end_mask);
            op->set_new_axis_mask(new_axis_mask);
            op->set_shrink_axis_mask(shrink_axis_mask);
            op->set_ellipsis_mask_mask(ellipsis_mask);

            OPENVINO_ASSERT(op->has_evaluate(), "[GPU] Couldn't find evaluate() function for strided_slice ",
                                                "primitive with id ", instance.id());
        }

        op->evaluate(output_host_tensors, input_host_tensors);

        if (begin_data.empty()) {
            auto begin_mem = instance.dep_memory_ptr(1);
            begin_mem->unlock(stream);
        }

        if (end_data.empty()) {
            auto end_mem = instance.dep_memory_ptr(2);
            end_mem->unlock(stream);
        }

        if (strides_data.empty()) {
            auto strides_mem = instance.dep_memory_ptr(3);
            strides_mem->unlock(stream);
        }

        input_mem_ptr->unlock(stream);
        output_mem_ptr->unlock(stream);

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const strided_slice_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<strided_slice_impl>();
    }
};


namespace detail {

attach_strided_slice_impl::attach_strided_slice_impl() {
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

    implementation_map<strided_slice>::add(impl_types::cpu, shape_types::static_shape, strided_slice_impl::create, types, formats);
    implementation_map<strided_slice>::add(impl_types::cpu, shape_types::dynamic_shape, strided_slice_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::strided_slice_impl)
