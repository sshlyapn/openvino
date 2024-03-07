// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/simple_math.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/core.hpp"

#include "intel_gpu/plugin/transformations_pipeline.hpp"

namespace ov {
namespace intel_gpu {

template<typename T>
static inline std::string vecToString(std::vector<T> vec) {
    if (vec.empty())
        return "";

    std::string res = std::to_string(vec[0]);
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + std::to_string(vec[i]);
    }
    return res;
}

template<>
inline std::string vecToString<std::string>(std::vector<std::string> vec) {
    if (vec.empty())
        return "";

    std::string res = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + vec[i];
    }
    return res;
}

class CustomLayerAttributeVisitor : public ov::AttributeVisitor {
public:
    CustomLayerAttributeVisitor() : m_values({}) { }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        OPENVINO_THROW("Attribute ", name, " can't be processed\n");
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        m_values[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<double>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }

    std::map<std::string, std::string> get_parameters() const {
        return m_values;
    }

protected:
    std::map<std::string, std::string> m_values;
};

static std::shared_ptr<ov::Model> make_prefill_subgraph(ov::element::Type type, std::int64_t num_heads = -1, std::int64_t num_kv_heads = -1, std::int64_t head_size = -1) {
    auto query = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1 /* batch */, -1 /* seq_len */, -1 /* queries per kv */, num_kv_heads, head_size}));
    query->set_friendly_name("query");
    auto key = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1 /* batch */, -1 /* seq_len */, num_kv_heads, 1, head_size}));
    key->set_friendly_name("key");
    auto value = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1 /* batch */, -1 /* seq_len */, num_kv_heads, 1, head_size}));
    value->set_friendly_name("value");
    auto mask = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1, -1, -1, -1, -1}));
    mask->set_friendly_name("mask");
    auto scale = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape({1}));
    scale->set_friendly_name("scale");

    // transpose Q, K and V to swap num_heads and seq_len dimensions
    auto permute_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({5}), {0, 3, 2, 1, 4});
    permute_const->set_friendly_name("permute_const");
    auto query_transposed = std::make_shared<ov::op::v1::Transpose>(query, permute_const);
    query_transposed->set_friendly_name("query_transposed");
    auto key_transposed = std::make_shared<ov::op::v1::Transpose>(key, permute_const);
    key_transposed->set_friendly_name("key_transposed");
    auto value_transposed = std::make_shared<ov::op::v1::Transpose>(value, permute_const);
    value_transposed->set_friendly_name("value_transposed");

    auto spda = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query_transposed, key_transposed, value_transposed, mask, scale, false);
    spda->set_friendly_name("sdpa");

    // transpose SPDA output to [batch, seq_len, num_q_per_kv, num_kv_heads, head_size] back
    auto spda_transposed = std::make_shared<ov::op::v1::Transpose>(spda, permute_const);

    return std::make_shared<ov::Model>(spda_transposed, ov::ParameterVector{query, key, value, mask, scale}, "spda_prefill_model");
}

void CreatePagedAttention(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {
    std::cout << "Create paged attention (id=" << op->get_friendly_name() << "), with " << op->get_input_size() << " inputs\n";

    auto config = p.get_config();
    config.set_property(ov::intel_gpu::max_dynamic_batch(1));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prefill_model = make_prefill_subgraph(op->get_output_element_type(0));
    TransformationsPipeline transformations(config, p.get_engine().get_device_info());
    transformations.apply(prefill_model);

    ProgramBuilder prog(prefill_model, p.get_engine(), config, false, false, p.get_task_executor(), p.get_compilation_context(), true);

    validate_inputs_count(op, {13});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::paged_attention(layer_type_name_ID(op), inputs);
    prim.prefill_stage = prog.get_compiled_program();

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    prim.output_paddings = get_output_paddings(op);

    OPENVINO_ASSERT(prim.num_outputs == 1, "[GPU] Unexpected outputs number");
    OPENVINO_ASSERT(prim.output_paddings[0] == cldnn::padding(), "[GPU] Unexpected output padding");

    p.add_primitive(*op, prim);
}

void CreateCustomOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, CustomLayerPtr customLayer) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    CustomLayerAttributeVisitor visitor;
    op->visit_attributes(visitor);
    auto params = visitor.get_parameters();

    // Handle defines
    std::string layerDefines;
    for (const auto& def : customLayer->Defines()) {
        std::string singleDefine("#define " + def.name + " " + def.prefix);
        if (params.find(def.param) != params.end()) {
            singleDefine += params.at(def.param);
        } else {
            singleDefine += def.default_value;
        }
        singleDefine += def.postfix + "\n";
        layerDefines.append(singleDefine);
    }

    // reserve
    std::vector<cldnn::input_info> reordered_inputs;
    reordered_inputs.resize(inputs.size());

    // Handle kernel parameters
    std::vector<cldnn::custom_gpu_primitive::arg_desc> kernelParameters;
    cldnn::format outputFormat(cldnn::format::any);
    for (const auto& param : customLayer->KernelParams()) {
        switch (param.type) {
        case CustomLayer::ParamType::Input: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].type = cldnn::custom_gpu_primitive::arg_input;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn::custom_gpu_primitive::arg_index>((param.portIndex >= static_cast<int>(inputs.size())) ? -1 : param.portIndex);

            // Handle input reorder
            if (param.portIndex < static_cast<int>(inputs.size()) && reordered_inputs[param.portIndex].pid.empty()) {
                // todo: add support for multiple reorders of the same input? (read as bfyx for one arg and yxfb for another)
                if (param.format != cldnn::format::any) {
                    auto reorderPrimName = inputs[param.portIndex].pid + "_" + op->get_friendly_name() + ProgramBuilder::m_preCustomLayerTag;
                    auto preprocessPrim = cldnn::reorder(
                        reorderPrimName,
                        inputs[param.portIndex],
                        param.format,
                        cldnn::element_type_to_data_type(op->get_input_element_type(param.portIndex)));

                    p.add_primitive(*op, preprocessPrim);
                    reordered_inputs[param.portIndex] = cldnn::input_info(reorderPrimName);
                } else {
                    reordered_inputs[param.portIndex] = inputs[param.portIndex];
                }
            }
            break;
        }
        case CustomLayer::ParamType::Output: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].type = cldnn::custom_gpu_primitive::arg_output;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn::custom_gpu_primitive::arg_index>((param.portIndex >= static_cast<int>(inputs.size())) ? -1 : param.portIndex);
            outputFormat = param.format;
            break;
        }
        default:
            OPENVINO_THROW("Invalid custom layer param type: ", param.type, " in operation: ", op->get_friendly_name());
        }
    }
    const std::string layerTitle("\n// Layer " + op->get_friendly_name() + " using Custom Layer " + customLayer->Name() + "\n");
    const std::string defineTitle("// Custom Layer User Defines\n");

    auto dims = op->get_output_shape(0);
    size_t N = (dims.size() > 0) ? dims[0] : 1;
    size_t C = (dims.size() > 1) ? dims[1] : 1;
    size_t H = (dims.size() > 2) ? dims[2] : 1;
    size_t W = (dims.size() > 3) ? dims[3] : 1;
    cldnn::tensor outputTensor = cldnn::tensor(cldnn::batch(N), cldnn::feature(C), cldnn::spatial(W, H));

    cldnn::layout outputLayout = cldnn::layout(cldnn::element_type_to_data_type(op->get_output_element_type(0)), outputFormat, outputTensor);

    // evaluate work sizes rules
    std::vector<size_t> gws, lws;

    // assume output tensor is dimension source by default
    int batchDim = outputTensor.batch[0];
    int featureDim = outputTensor.feature[0];
    int yDim = outputTensor.spatial[1];
    int xDim = outputTensor.spatial[0];
    int iidx = customLayer->InputDimSourceIndex();

    std::string genericLayerName = layer_type_name_ID(op);
    // if input index is greater than -1, take dimension from input
    if (iidx >= 0) {
        if (static_cast<size_t>(iidx) >= op->get_input_size())
            OPENVINO_THROW("Invalid input tensor for index: ", iidx);
        auto inputDims = op->get_input_shape(iidx);

        xDim = static_cast<int>(inputDims[inputDims.size() - 1]);
        yDim = dims.size() > 1 ? static_cast<int>(inputDims[inputDims.size() - 2]) : 0;
        featureDim = dims.size() > 2 ? static_cast<int>(inputDims[inputDims.size() - 3]) : 0;
        batchDim = dims.size() > 3 ? static_cast<int>(inputDims[inputDims.size() - 4]) : 0;
    }
    const std::map<char, int> vars = {
        { 'b', batchDim }  , { 'B', batchDim },
        { 'f', featureDim }, { 'F', featureDim },
        { 'y', yDim },       { 'Y', yDim },
        { 'x', xDim },       { 'X', xDim },
    };
    for (const auto& rule : customLayer->GlobalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        gws.push_back(expr.Evaluate());
    }
    for (const auto& rule : customLayer->LocalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        lws.push_back(expr.Evaluate());
    }

    auto customPrim = cldnn::custom_gpu_primitive(genericLayerName,
                                                  reordered_inputs,
                                                  { layerTitle, defineTitle, layerDefines, customLayer->KernelSource() },
                                                  customLayer->KernelEntry(),
                                                  kernelParameters,
                                                  customLayer->CompilerOptions(),
                                                  outputLayout,
                                                  gws,
                                                  lws);
    p.add_primitive(*op, customPrim);

    auto prevLayerName = genericLayerName;
    if (outputLayout.format != cldnn::format::any) {
        // Handle output reorder
        auto reorderPrimName = genericLayerName + ProgramBuilder::m_postCustomLayerTag;
        p.add_primitive(*op, cldnn::reorder(reorderPrimName,
                                            cldnn::input_info(genericLayerName),
                                            cldnn::format::get_default_format(op->get_output_shape(0).size()),
                                            customPrim.output_layout.data_type));
        prevLayerName = reorderPrimName;
    }
}

}  // namespace intel_gpu
}  // namespace ov
