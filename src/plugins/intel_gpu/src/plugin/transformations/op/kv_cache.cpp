// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "gather_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& beam_idx,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data, beam_idx})
    , m_concat_axis(concat_axis)
    , m_gather_axis(gather_axis)
    , m_indirect(true)
    , m_compressed(false)
    , m_output_type(output_type) {
    m_variable = past_variable;
    if (m_indirect)
        set_output_size(2);
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data})
    , m_concat_axis(concat_axis)
    , m_gather_axis(0)
    , m_indirect(false)
    , m_compressed(false)
    , m_output_type(output_type) {
    m_variable = past_variable;
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& beam_idx,
                 const Output<Node>& past_scale,
                 const Output<Node>& new_token_scale,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data, beam_idx, past_scale, new_token_scale})
    , m_concat_axis(concat_axis)
    , m_gather_axis(gather_axis)
    , m_indirect(true)
    , m_compressed(true)
    , m_output_type(output_type) {
    m_variable = past_variable;
    size_t out_ports = 3;
    set_output_size(out_ports);
    validate_and_infer_types();
}

bool KVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("concat_axis", m_concat_axis);
    visitor.on_attribute("gather_axis", m_gather_axis);
    visitor.on_attribute("indirect", m_indirect);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("compressed", m_compressed);
    return true;
}

void KVCache::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    std::vector<ov::PartialShape> input_shapes = {m_variable->get_info().data_shape, get_input_partial_shape(1)};
    if (m_indirect) {
        input_shapes.push_back(get_input_partial_shape(2));
    }

    if (m_compressed) {
        input_shapes.push_back(get_input_partial_shape(3));
        input_shapes.push_back(get_input_partial_shape(4));
    }

    auto shapes = shape_infer(this, input_shapes);
    size_t out_ports = 0;
    set_output_type(out_ports++, output_type, shapes[0]);
    // TODO: kv-cache compression is not supported for indirect kv cache
    if (m_indirect) {
        set_output_type(out_ports++, get_input_element_type(2), shapes[1]);
    }
    if (m_compressed) {
        set_output_type(out_ports++, get_input_element_type(3), shapes[2]);
    }
}

std::shared_ptr<Node> KVCache::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         m_variable,
                                         m_concat_axis,
                                         m_output_type);

    } else if (new_args.size() == 3) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         m_variable,
                                         m_concat_axis,
                                         m_gather_axis,
                                         m_output_type);
    } else {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         m_variable,
                                         m_concat_axis,
                                         m_gather_axis,
                                         m_output_type);
    }
}

std::vector<ov::PartialShape> shape_infer(const KVCache* op, std::vector<ov::PartialShape> input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.resize(op->get_output_size());

    // std::cout << "kv_cache shape infer " << op->get_output_size() << "\n";

    const auto& gather_axis = op->get_gather_axis();
    const auto& concat_axis = ov::util::normalize(op->get_concat_axis(), input_shapes[0].size());
    if (op->get_output_size() >= 2) {
        out_shapes[0] = input_shapes[0];
        out_shapes[0][gather_axis] = input_shapes[2][0];
        out_shapes[0][concat_axis] += input_shapes[1][concat_axis];

        std::vector<ov::Dimension> dims(out_shapes[0].size(), 1);
        dims[gather_axis] = out_shapes[0][gather_axis];
        dims[concat_axis] = out_shapes[0][concat_axis];
        out_shapes[1] = dims;

        // FIXME: indirect kv cache and compression are orthogonal feature. it can be selective.
        // If KV cache is compressed
        if (op->get_output_size() == 3){
            ov::PartialShape compression_scale_shape = input_shapes[3];
            compression_scale_shape[concat_axis] += input_shapes[4][concat_axis];
            out_shapes[2] = compression_scale_shape;

            // ov::PartialShape compression_scale_shape(std::vector<size_t>(out_shapes[0].size(), 1));
            // compression_scale_shape[0] = out_shapes[0][0];
            // compression_scale_shape[1] = out_shapes[0][1];
            // GPU_DEBUG_IF(cldnn::debug_configuration::get_instance()->enable_kv_cache_compression == 1) { // per-head compression
            //     compression_scale_shape[2] = out_shapes[0][2];
            // }
            // out_shapes[2] = compression_scale_shape;
        }
    } else {
        out_shapes[0] = input_shapes[0];
        out_shapes[0][concat_axis] += input_shapes[1][concat_axis];
    }

    return out_shapes;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
