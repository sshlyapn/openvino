// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/read_value.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

ReadValue::ReadValue(const std::shared_ptr<ov::op::util::Variable>& past_values) : Op() {
    m_variable = past_values;
    validate_and_infer_types();
}
ReadValue::ReadValue(const Output<Node>& variable_initializer, const std::shared_ptr<ov::op::util::Variable>& past_values) : Op({variable_initializer}) {
    m_variable = past_values;
    validate_and_infer_types();
}

bool ReadValue::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("variable_id", m_variable);

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);
    return true;
}

void ReadValue::validate_and_infer_types() {
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");
    const auto& variable_info = m_variable->get_info();
    const auto& variable_type = variable_info.data_type;
    const auto& variable_shape = variable_info.data_shape;

    // If no inputs provided, it means this ReadValue doesn't have initial subgraph. This is valid.
    if (get_input_size() > 0) {
        const auto& initial_type = get_input_element_type(0);
        const auto& initial_shape = get_input_partial_shape(0);

        // Variable's shape/type determine a permissible range of values for shape/type inferred from initial_subgraph.
        // If initial_subgraph is set, then we need to check that shape/type inferred from initial_subgraph
        // is within the permissible range.

        const auto compatible_type = variable_type.is_dynamic() || initial_type == variable_type;
        const auto compatible_shape = variable_shape.relaxes(initial_shape);

        if (get_friendly_name() == "ReadValue_179538") {
            std::cout << "input name" << get_input_node_shared_ptr(0)->get_friendly_name() << " " << get_input_node_shared_ptr(0)->get_type_name() << "\n";
        }

        OPENVINO_ASSERT(compatible_shape,
                        "The shape specified in the Variable has to extend (relax) the shape "
                        "inferred from the initializing subgraph.",
                        " Variable shape: ",
                        variable_shape,
                        " Initialization shape: ",
                        initial_shape);
        OPENVINO_ASSERT(compatible_type,
                        "The type specified in the Variable has to extend (relax) the type "
                        "inferred from the initializing subgraph.",
                        " Variable type: ",
                        variable_type,
                        " Initialization type: ",
                        initial_type, " ", get_friendly_name(), " ", std::string(get_type_name()));

        // workaround:
        // dynamic rank/type can be derived from the IRs generated via the prev versions of OV,
        // but dynamic rank/type are not supported in plugins,
        // so we are trying to fix them here using the rank/type of ReadValue 1st input, if it exists
        if (get_input_size() > 0 && variable_info.data_shape.rank().is_dynamic() &&
            variable_info.data_type.is_dynamic()) {
            set_output_type(0, initial_type, initial_shape);
            return;
        }
    }

    set_output_type(0, variable_type, variable_shape);
}

std::shared_ptr<Node> ReadValue::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
        switch (new_args.size()) {
    case 0:
        return std::make_shared<ReadValue>(m_variable);
    case 1:
        return std::make_shared<ReadValue>(new_args[0], m_variable);
    default:
        OPENVINO_THROW("Unable to clone ReadValue ",
                       this->get_friendly_name(),
                       " Incorrect number of inputs. Expected: 0 or 1. Actual: ",
                       new_args.size());
    }
}

CompressedReadValue::CompressedReadValue(const Output<Node>& variable_initializer,
                                         const Output<Node>& compressed_variable_initializer_scale,
                                         const std::shared_ptr<ov::op::util::Variable>& past_values) : ReadValue(variable_initializer, past_values) {
    const auto scales_input_idx = get_input_size();
    set_argument(scales_input_idx, compressed_variable_initializer_scale);
    validate_and_infer_types();
}

bool CompressedReadValue::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("variable_id", m_variable);

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);
    return true;
}

void CompressedReadValue::validate_and_infer_types() {
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");

    OPENVINO_ASSERT(get_input_size() == 2);
    ReadValue::validate_and_infer_types();

    const auto& scale_type = get_input_element_type(1);
    const auto& scale_shape = get_input_partial_shape(1);

    set_output_type(1, scale_type, scale_shape);
}

std::shared_ptr<Node> CompressedReadValue::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    OPENVINO_ASSERT(new_args.size() == 2, "Unable to clone CompressedReadValue ",
                    this->get_friendly_name(),
                    " Incorrect number of inputs. Expected: 2. Actual: ",
                    new_args.size());

    return std::make_shared<CompressedReadValue>(new_args[0], new_args[1], m_variable);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
