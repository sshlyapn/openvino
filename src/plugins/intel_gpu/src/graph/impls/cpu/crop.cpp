// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "crop_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace cpu {

struct crop_impl : public typed_primitive_impl<crop> {
    using parent = typed_primitive_impl<crop>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<crop_impl>(*this);
    }

    crop_impl() : parent("crop_cpu_impl") {}

    explicit crop_impl(const crop_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<crop>());
    }

    template <class T>
    void calculate_crop(const cldnn::kernel_impl_params* params, cldnn::memory::ptr input_mem_ptr, cldnn::memory::ptr output_mem_ptr, cldnn::stream &stream) {
        auto input_layout = params->input_layouts[0];
        auto output_layout = params->output_layouts[0];
        auto input_offset = params->input_offsets[0];

        cldnn::mem_lock<T, mem_lock_type::read> input_lock(input_mem_ptr, stream);
        cldnn::mem_lock<T, mem_lock_type::write> output_lock(output_mem_ptr, stream);

        auto size_out = output_layout.get_tensor();

        size_t out_idx = 0;
        for (cldnn::tensor::value_type b = input_offset.batch[0]; b < input_offset.batch[0] + size_out.batch[0]; ++b) {
            for (cldnn::tensor::value_type f = input_offset.feature[0]; f < input_offset.feature[0] + size_out.feature[0]; ++f) {
                for (cldnn::tensor::value_type w = input_offset.spatial[3]; w < input_offset.spatial[3] + size_out.spatial[3]; ++w) {
                    for (cldnn::tensor::value_type z = input_offset.spatial[2]; z < input_offset.spatial[2] + size_out.spatial[2]; ++z) {
                        for (cldnn::tensor::value_type y = input_offset.spatial[1]; y < input_offset.spatial[1] + size_out.spatial[1]; ++y) {
                            cldnn::tensor input_t(cldnn::group(0), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(input_offset.spatial[0], y, z, w));
                            size_t input_idx = input_layout.get_linear_offset(input_t);
                            for (cldnn::tensor::value_type x = input_offset.spatial[0]; x < input_offset.spatial[0] + size_out.spatial[0]; ++x) {
                                output_lock[out_idx++] = input_lock[input_idx++];
                            }
                        }
                    }
                }
            }
        }
    };

    event::ptr execute_impl(const std::vector<event::ptr>& events, crop_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "crop::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);

        auto params = instance.get_impl_params();

        auto input_mem_ptr = instance.input_memory_ptr();
        auto output_mem_ptr = instance.output_memory_ptr();

        switch (params->input_layouts[0].data_type) {
        case data_types::f32:
            calculate_crop<float>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::f16:
            calculate_crop<half_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::i64:
            calculate_crop<int64_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::i32:
            calculate_crop<int32_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::u8:
            calculate_crop<uint8_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::i8:
            calculate_crop<int8_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        default:
            OPENVINO_THROW("[GPU] Couldn't execute crop operation: unsupported input data type");
        }

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const crop_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<crop_impl>();
    }
};


namespace detail {

attach_crop_impl::attach_crop_impl() {
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

    implementation_map<crop>::add(impl_types::cpu, shape_types::static_shape, crop_impl::create, types, formats);
    implementation_map<crop>::add(impl_types::cpu, shape_types::dynamic_shape, crop_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::crop_impl)
