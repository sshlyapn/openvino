// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct fully_connected_onednn : typed_primitive_onednn_impl<fully_connected, dnnl::inner_product_forward::desc> {
    using parent = typed_primitive_onednn_impl<fully_connected, dnnl::inner_product_forward::desc>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_onednn>(*this);
    }

    bool validate_impl(const typed_primitive_inst<fully_connected>& instance) const override {
        bool res = true;

        auto outer_id = _outer.id();
        auto data_type = instance.node.input().get_output_layout().data_type;

        // Integer signed/unsigned is ok for convoluiton
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(outer_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory()->get_layout().data_type,
                                                    "");

        return res;
    }

    std::unordered_map<int, dnnl::memory> get_arguments(fully_connected_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        {
            auto weights = instance.weights_memory();
            args.insert({DNNL_ARG_WEIGHTS, weights->get_onednn_memory(_pd.weights_desc(0))});
        }

        if (instance.bias_term()) {
            auto bias = instance.bias_memory();
            args.insert({DNNL_ARG_BIAS, bias->get_onednn_memory(_pd.weights_desc(1))});
        }

        return args;
    }

    static kernel_selector::WeightsReorderParams get_weights_reorder(const fully_connected_node& arg, const dnnl::primitive_desc& pd) {
        kernel_selector::WeightsReorderParams weights_reorder_params;
        auto& reorderKS = kernel_selector::ReorderWeightsKernelSelctor::Instance();
        kernel_selector::reorder_weights_params r_params;

        auto cldnn_prim = arg.get_primitive();
        auto weights_layout = arg.get_dependency(1).get_output_layout();
        cldnn::format out_fmt = onednn::convert_format(onednn::get_format_by_desc(pd.weights_desc(0)));
        kernel_selector::WeightsLayout reqLayout = to_weights_layout(out_fmt, false);

        // set engine info & forcing
        set_params(arg, r_params);
        r_params.layerID = arg.id() + "_reorder_";
        r_params.input = convert_weights_tensor(weights_layout, false);
        r_params.output = r_params.input.TransformIgnorePadding(reqLayout, r_params.input.GetDType(), 1, false);
        r_params.rotate_180 = false;

        kernel_selector::reorder_optional_params op;
        kernel_selector::KernelsData kernels_data = reorderKS.GetBestKernels(r_params, op);

        if (kernels_data.empty()) {
            throw std::runtime_error("No suitable kernel found for weights reorder from " +
                                        kernel_selector::toString(r_params.input.GetLayout()) + " to " +
                                        kernel_selector::toString(r_params.output.GetLayout()));
        }

        weights_reorder_params.engine = kernel_selector::WeightsReorderParams::Engine::GPU;
        weights_reorder_params.clKernel = std::make_shared<kernel_selector::clKernelData>(kernels_data[0].kernels[0]);
        weights_reorder_params.dest = r_params.output;

        return weights_reorder_params;
    }

    static std::shared_ptr<dnnl::inner_product_forward::desc> get_fully_connected_descriptor(const fully_connected_node& arg) {
        auto prim = arg.get_primitive();
        auto is_3d = prim->input_size == 3;

        auto& input = arg.get_dependency(0);
        auto& weights = arg.get_dependency(1);

        auto input_md = onednn::layout_to_memory_desc(input.get_output_layout(), dnnl::memory::format_tag::undef, false, is_3d);
        auto weights_md = onednn::layout_to_memory_desc(weights.get_output_layout(), dnnl::memory::format_tag::any);
        auto output_md = onednn::layout_to_memory_desc(arg.get_output_layout(), dnnl::memory::format_tag::ab, false, is_3d);

        if (arg.bias_term()) {
            auto bias_md = onednn::layout_to_memory_desc(arg.get_dependency(2).get_output_layout(), dnnl::memory::format_tag::any, true);
            return std::make_shared<dnnl::inner_product_forward::desc>(
                dnnl::prop_kind::forward_inference,
                input_md,
                weights_md,
                bias_md,
                output_md);
        } else {
            return std::make_shared<dnnl::inner_product_forward::desc>(
                dnnl::prop_kind::forward_inference,
                input_md,
                weights_md,
                output_md);
        }
    }

public:
    static primitive_impl* create(const fully_connected_node& arg) {
        std::cerr << "create onednn fc: " << arg.id() << std::endl;
        auto& engine = arg.get_program().get_engine();
        auto desc = get_fully_connected_descriptor(arg);
        auto prim = arg.get_primitive();

        if (prim->input_size == 3) {
            for (auto& fused_node : arg.get_fused_primitives()) {
                auto node = fused_node.node;
                if (node->is_type<eltwise>()) {
                    auto& dependency = arg.get_dependency(fused_node.dep_start_idx);
                    auto original_layout = dependency.get_output_layout();
                    onednn::treat_layout_as_bf(original_layout);
                    dependency.set_output_layout(original_layout, false);
                }
            }
        }

        auto attr = get_primitive_attributes(arg);
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};
        // DO we need to iterate over prim descs?

        return new fully_connected_onednn(arg, desc, attr, prim_desc, get_weights_reorder(arg, prim_desc));
    }
};

namespace detail {

attach_fully_connected_onednn::attach_fully_connected_onednn() {
    implementation_map<fully_connected>::add(impl_types::onednn, fully_connected_onednn::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        // std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        // std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        // std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        // std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        // std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        // std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        // std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        // std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),

        // std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        // std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        // std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        // std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
    });
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
