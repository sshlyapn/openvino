// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/pass/manager.hpp>
#include <openvino/core/model.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/binary_convolution.hpp>
#include <openvino/op/convolution.hpp>
#include <plugin/transformations/binary_conv_to_conv.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"

#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;
using namespace ov;

TEST_F(TransformationTestsF, MarkDequantizationSubgraphTransformationFoldSubConst12) {
    // Input graph:           After transformation:
    //
    // Constant    Constant               Constant
    //    |U8         |U8                    |U8
    //    |           |                      |
    // Convert     Convert                Convert(DCF)  Constant
    //    |FP32   /F32                       |FP32      /FP32
    //    |      /                            \        /
    //   Subtract  Constant                    Subtract   Constant
    //    |FP32    /FP32                          |FP32     /FP32
    //    |       /                                \        /
    //   Multiply                                   Multiply
    //
    // After MarkDequantizationSubgraph all Subtract and Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also all 'Convert(DCF)' node before weights is marked with 'DisableConstantFolding' attribute
    // but Convert before Dequantization Sub const isn't because fold_subtract_const is set to true
    // Weights node is marked with 'KeepConstPrecision' attribute

    {
        auto weights = opset10::Constant::create(element::u8, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
        auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        model = std::make_shared<ov::Model>(ov::OutputVector{multiply});
    }

    manager.register_pass<pass::MarkDequantizationSubgraph>(element::TypeVector{element::u8}, true);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = opset10::Constant::create(element::u8, Shape{4, 16, 1, 1}, {3});
        enable_keep_const_precision(weights);
        auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
        pass::disable_constant_folding(convert);
        auto zero_point = opset10::Constant::create(element::f32, Shape{}, {127});
        auto subtract = std::make_shared<opset10::Subtract>(convert, zero_point);
        mark_as_dequantization_node(subtract);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        mark_as_dequantization_node(multiply);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{multiply});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
