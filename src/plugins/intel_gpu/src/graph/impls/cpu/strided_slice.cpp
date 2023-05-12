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
        auto& stream = instance.get_network().get_stream();

        std::cout << "Cpu impl strided_slice: " << instance.id() << "\n";

        for (auto e : events) {
            e->wait();
        }

        auto& constant_mem = instance.get_impl_params()->memory_deps;
        std::cout << "Constan mem size=" << constant_mem.size() << std::endl;

        bool use_runtime_inputs = (begin_data.empty() && !constant_mem.count(1))
                                || (end_data.empty() && !constant_mem.count(2))
                                || (strides_data.empty() && !constant_mem.count(3));

        if (use_runtime_inputs && instance.dependencies().size() != 4) {
            OPENVINO_THROW("[GPU] Unexpected configuration of Strided Slice");
        }

        ov::op::v1::StridedSlice op;
        ov::Shape begin_shape = begin_data.empty() ? instance.get_impl_params()->get_input_layout(1).get_shape() : ov::Shape{ begin_data.size() };
        ov::Shape end_shape = end_data.empty() ? instance.get_impl_params()->get_input_layout(2).get_shape() : ov::Shape{ end_data.size() };
        ov::Shape strides_shape = strides_data.empty() ? instance.get_impl_params()->get_input_layout(3).get_shape() : ov::Shape{ strides_data.size() };

        op.set_begin_mask(begin_mask);
        op.set_end_mask(end_mask);
        op.set_new_axis_mask(new_axis_mask);
        op.set_shrink_axis_mask(shrink_axis_mask);
        op.set_ellipsis_mask_mask(ellipsis_mask);

        auto print_arr = [&](std::vector<int64_t> vec, std::string name) {
            std::cout << name << ": ";
            for (size_t i = 0; i < vec.size(); i++) {
                std::cout << vec[i] << ", ";
            }
            std::cout << "\n";
        };

        print_arr(begin_data, "begin_data");
        print_arr(end_data, "end_data");
        print_arr(strides_data, "strides_data");
        print_arr(begin_mask, "begin_mask");
        print_arr(end_mask, "end_mask");
        print_arr(new_axis_mask, "new_axis_mask");
        print_arr(shrink_axis_mask, "shrink_axis_mask");
        print_arr(ellipsis_mask, "ellipsis_mask");

        std::shared_ptr<ngraph::runtime::HostTensor> begin_host_tensor;
        std::shared_ptr<ngraph::runtime::HostTensor> end_host_tensor;
        std::shared_ptr<ngraph::runtime::HostTensor> strides_host_tensor;

        if (begin_data.empty()) {
            auto begin_mem = use_runtime_inputs ? instance.dep_memory_ptr(1) : constant_mem.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(begin_mem, stream);
            begin_host_tensor = make_host_tensor(begin_mem->get_layout(), begin_mem->lock(stream, mem_lock_type::read));
            std::cout << "Begin from const: " << begin_mem->get_layout().to_short_string() << "\n";
            for (size_t i = 0; i < begin_host_tensor->get_element_count(); i++) {
                std::cout << static_cast<int64_t*>(begin_host_tensor->get_data_ptr())[i] << ", ";
            }
            if (begin_host_tensor->get_element_count())
                std::cout << "\n";
        } else {
            std::cout << "Begin from dep\n";
            begin_host_tensor = make_host_tensor({ begin_shape, data_types::i64, format::bfyx }, static_cast<void*>(begin_data.data()));
        }

        if (end_data.empty()) {
            auto end_mem = use_runtime_inputs ? instance.dep_memory_ptr(2) : constant_mem.at(2);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(end_mem, stream);
            end_host_tensor = make_host_tensor(end_mem->get_layout(), end_mem->lock(stream, mem_lock_type::read));
            std::cout << "End from const: " << end_mem->get_layout().to_short_string() << "\n";
            for (size_t i = 0; i < end_host_tensor->get_element_count(); i++) {
                std::cout << static_cast<int64_t*>(end_host_tensor->get_data_ptr())[i] << ", ";
            }
            if (end_host_tensor->get_element_count())
                std::cout << "\n";
        } else {
            std::cout << "End from dep\n";
            end_host_tensor = make_host_tensor({ end_shape, data_types::i64, format::bfyx }, static_cast<void*>(end_data.data()));
        }

        if (strides_data.empty()) {
            auto strides_mem = use_runtime_inputs ? instance.dep_memory_ptr(3) : constant_mem.at(3);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(strides_mem, stream);
            strides_host_tensor = make_host_tensor(strides_mem->get_layout(), strides_mem->lock(stream, mem_lock_type::read));
            std::cout << "Strides from const: " << strides_mem->get_layout().to_short_string() << "\n";
            for (size_t i = 0; i < strides_host_tensor->get_element_count(); i++) {
                std::cout << static_cast<int64_t*>(strides_host_tensor->get_data_ptr())[i] << ", ";
            }
            if (strides_host_tensor->get_element_count())
                std::cout << "\n";
        } else {
            std::cout << "Strides from dep\n";
            strides_host_tensor = make_host_tensor({ strides_shape, data_types::i64, format::bfyx }, static_cast<void*>(strides_data.data()));
        }

        memory::ptr input_mem_ptr = instance.dep_memory_ptr(0);
        auto output_mem_ptr = instance.output_memory_ptr();

        auto input_host_tensor = make_host_tensor(input_mem_ptr->get_layout(), input_mem_ptr->lock(stream, mem_lock_type::read));
        auto output_host_tensor = make_host_tensor(output_mem_ptr->get_layout(), output_mem_ptr->lock(stream, mem_lock_type::write));

        op.evaluate({output_host_tensor}, {input_host_tensor, begin_host_tensor, end_host_tensor, strides_host_tensor});

        if (begin_data.empty()) {
            auto begin_mem = use_runtime_inputs ? instance.dep_memory_ptr(1) : constant_mem.at(1);
            begin_mem->unlock(stream);
        }

        if (end_data.empty()) {
            auto end_mem = use_runtime_inputs ? instance.dep_memory_ptr(2) : constant_mem.at(2);
            end_mem->unlock(stream);
        }

        if (strides_data.empty()) {
            auto strides_mem = use_runtime_inputs ? instance.dep_memory_ptr(3) : constant_mem.at(3);
            strides_mem->unlock(stream);
        }

        input_mem_ptr->unlock(stream);
        output_mem_ptr->unlock(stream);

        std::cout << "StridedSlice input types and sizes:\n";
        std::cout << "-> " << input_mem_ptr->get_allocation_type() << " " << input_mem_ptr->get_layout().to_short_string() << std::endl;

        std::cout << "StridedSlice output types and sizes\n";
        std::cout << "-> " << output_mem_ptr->get_allocation_type() << " " << output_mem_ptr->get_layout().to_short_string() << std::endl;

        return stream.create_user_event(true);
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
    std::cout << "StridedSlice attach: CPU impl\n";
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
