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

    // tensor input_offset;

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
        // const auto& node = arg.as<crop>();
        // input_offset = node.get_primitive()->inpu;
    }

    // void save(BinaryOutputBuffer& ob) const override {
    //     ob << input_offset;
    // }

    // void load(BinaryInputBuffer& ib) override {
    //     ib >> input_offset;
    // }

    event::ptr execute_impl(const std::vector<event::ptr>& events, crop_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "crop::execute_impl");
        auto& stream = instance.get_network().get_stream();

        // std::cout << "Cpu impl crop: " << instance.id() << "\n";

        {
            OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "crop_cpu::wait_for_events");
            for (auto e : events) {
                e->wait();
            }
        }
        auto ev = stream.create_user_event(false);

        auto params = instance.get_impl_params();
        // std::cout << "Inputs:\n";
        // for (size_t i = 0; i < params->input_layouts.size(); i++)
        //     std::cout << "- " << params->input_layouts[i].to_short_string() << std::endl;

        // std::cout << "Input offsets:\n";
        // for (size_t i = 0; i < params->input_offsets.size(); i++)
        //     std::cout << "- " << params->input_offsets[i].to_string() << std::endl;

        // std::cout << "Outputs:\n";
        // for (size_t i = 0; i < params->output_layouts.size(); i++)
        //     std::cout << "- " << params->output_layouts[i].to_short_string() << std::endl;

        {
            auto input_layout = params->input_layouts[0];
            auto output_layout = params->output_layouts[0];
            auto input_offset = params->input_offsets[0];

            auto input_mem_ptr = instance.input_memory_ptr();
            auto output_mem_ptr = instance.output_memory_ptr();

            OPENVINO_ASSERT(input_layout.data_type == data_types::i32 && output_layout.data_type == data_types::i32, "[GPU] Crop error");

            cldnn::mem_lock<int32_t, mem_lock_type::read> input_lock(input_mem_ptr, stream);
            cldnn::mem_lock<int32_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);

            auto size_out = output_layout.get_tensor();

            {
                OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "crop_cpu::caclulation");
                size_t out_idx = 0;
                for (cldnn::tensor::value_type b = input_offset.batch[0]; b < input_offset.batch[0] + size_out.batch[0]; ++b) {
                    for (cldnn::tensor::value_type f = input_offset.feature[0]; f < input_offset.feature[0] + size_out.feature[0]; ++f) {
                        for (cldnn::tensor::value_type w = input_offset.spatial[3]; w < input_offset.spatial[3] + size_out.spatial[3]; ++w) {
                            for (cldnn::tensor::value_type z = input_offset.spatial[2]; z < input_offset.spatial[2] + size_out.spatial[2]; ++z) {
                                for (cldnn::tensor::value_type y = input_offset.spatial[1]; y < input_offset.spatial[1] + size_out.spatial[1]; ++y) {
                                    for (cldnn::tensor::value_type x = input_offset.spatial[0]; x < input_offset.spatial[0] + size_out.spatial[0]; ++x) {
                                        cldnn::tensor input_t(cldnn::group(0), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(x, y, z, w));
                                        size_t input_it = input_layout.get_linear_offset(input_t);

                                        output_lock[out_idx++] = input_lock[input_it];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // std::cout << "Input data:\n";
            // for (size_t i = 0; i < input_lock.size(); i++)
            //     std::cout << i << ". " << input_lock[i] << std::endl;

            // std::cout << "Output data:\n";
            // for (size_t i = 0; i < output_lock.size(); i++)
            //     std::cout << i << ". " << output_lock[i] << std::endl;
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
    std::cout << "crop attach: CPU impl\n";
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
