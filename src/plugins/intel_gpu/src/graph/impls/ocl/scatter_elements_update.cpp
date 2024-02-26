// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "scatter_elements_update_inst.h"
#include "scatter_update/scatter_elements_update_kernel_selector.h"
#include "scatter_update/scatter_elements_update_kernel_ref.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::scatter_update_axis convert_axis(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += rank;
    }
    auto cldnn_axis = axis;
    if (axis >= 2) {
        auto spatial_axis = axis - 2;
        const size_t default_dims = 4; // Default and minimum number of dimensions is 4
        auto spatial_size = std::max(rank, default_dims) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return kernel_selector::scatter_update_axis::BATCH;
        case 1: return kernel_selector::scatter_update_axis::FEATURE;
        case 2: return kernel_selector::scatter_update_axis::X;
        case 3: return kernel_selector::scatter_update_axis::Y;
        case 4: return kernel_selector::scatter_update_axis::Z;
        case 5: return kernel_selector::scatter_update_axis::W;
        default: OPENVINO_ASSERT(false, "[GPU] Unsupported scatter update axis");
    }
    return kernel_selector::scatter_update_axis::X;
}

kernel_selector::ScatterUpdateReduction convert_reduction_mode(const ScatterElementsUpdateOp::Reduction mode) {
    switch (mode) {
        case ScatterElementsUpdateOp::Reduction::NONE:
            return kernel_selector::ScatterUpdateReduction::NONE;
        case ScatterElementsUpdateOp::Reduction::SUM:
            return kernel_selector::ScatterUpdateReduction::SUM;
        case ScatterElementsUpdateOp::Reduction::PROD:
            return kernel_selector::ScatterUpdateReduction::PROD;
        case ScatterElementsUpdateOp::Reduction::MIN:
            return kernel_selector::ScatterUpdateReduction::MIN;
        case ScatterElementsUpdateOp::Reduction::MAX:
            return kernel_selector::ScatterUpdateReduction::MAX;
        case ScatterElementsUpdateOp::Reduction::MEAN:
            return kernel_selector::ScatterUpdateReduction::MEAN;
        default:
            OPENVINO_ASSERT(false, "[GPU] Invalid ScatterElementsUpdate::Reduction enum value");
    }
    return kernel_selector::ScatterUpdateReduction::NONE;
}
}  // namespace

struct scatter_elements_update_impl : typed_primitive_impl_ocl<scatter_elements_update> {
    using parent = typed_primitive_impl_ocl<scatter_elements_update>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::scatter_elements_update_kernel_selector;
    using kernel_params_t = kernel_selector::scatter_elements_update_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::scatter_elements_update_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scatter_elements_update_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<scatter_elements_update>();
        auto params = get_default_params<kernel_selector::scatter_elements_update_params>(impl_param);

        params.axis = convert_axis(primitive->axis, impl_param.get_input_layout(0).get_rank());
        params.mode = convert_reduction_mode(primitive->mode);
        params.use_init_val = primitive->use_init_val;

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
        return params;
    }
};

namespace detail {

attach_scatter_elements_update_impl::attach_scatter_elements_update_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i32};
    auto formats = {
            format::bfyx,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv32,
            format::bfzyx,
            format::b_fs_zyx_fsv16,
            format::b_fs_zyx_fsv32,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bfwzyx
    };

    implementation_map<scatter_elements_update>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<scatter_elements_update>::create<scatter_elements_update_impl>,
        types,
        formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::scatter_elements_update_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scatter_elements_update)
