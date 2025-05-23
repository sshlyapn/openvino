// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/quantize.hpp"
#include "primitive_inst.h"
#include "data_inst.h"
#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<quantize> : public typed_program_node_base<quantize> {
    using parent = typed_program_node_base<quantize>;

    typed_program_node(std::shared_ptr<quantize> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    int get_levels() const { return get_primitive()->levels; }
    bool get_scale_shift_opt() const { return get_primitive()->scale_shift_opt; }
    bool get_need_pre_shift() const { return get_primitive()->need_pre_shift; }
    bool get_need_post_scale() const { return get_primitive()->need_post_scale; }
    bool get_need_post_shift() const { return get_primitive()->need_post_shift; }
    bool get_need_clamp() const { return get_primitive()->need_clamp; }
    bool get_need_min_clamp() const { return get_primitive()->need_min_clamp; }
    bool get_need_max_clamp() const { return get_primitive()->need_max_clamp; }
    bool get_per_tensor_input_scale() const { return get_primitive()->per_tensor_input_scale; }
    bool get_per_tensor_input_shift() const { return get_primitive()->per_tensor_input_shift; }
    bool get_per_tensor_input_range() const { return get_primitive()->per_tensor_input_range; }
    bool get_per_tensor_output_scale() const { return get_primitive()->per_tensor_output_scale; }
    bool get_per_tensor_output_shift() const { return get_primitive()->per_tensor_output_shift; }
    bool get_per_tensor_output_range() const { return get_primitive()->per_tensor_output_range; }
    float get_input_scale_val() const { return get_primitive()->in_scale; }
    float get_input_shift_val() const { return get_primitive()->in_shift; }
    float get_input_lo_val() const { return get_primitive()->in_lo; }
    float get_input_hi_val() const { return get_primitive()->in_hi; }
    float get_output_scale_val() const { return get_primitive()->out_scale; }
    float get_output_shift_val() const { return get_primitive()->out_shift; }
    float get_output_lo_val() const { return get_primitive()->out_lo; }
    float get_output_hi_val() const { return get_primitive()->out_hi; }

    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return quantize::create_fuse_params(typed_desc(), get_output_layout());
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using quantize_node = typed_program_node<quantize>;

template <>
class typed_primitive_inst<quantize> : public typed_primitive_inst_base<quantize> {
    using parent = typed_primitive_inst_base<quantize>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(quantize_node const& node, kernel_impl_params const& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(quantize_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(quantize_node const& node);

    typed_primitive_inst(network& network, quantize_node const& node);
};

using quantize_inst = typed_primitive_inst<quantize>;

}  // namespace cldnn
