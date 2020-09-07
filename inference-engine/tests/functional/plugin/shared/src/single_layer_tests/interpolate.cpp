// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/interpolate.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using ngraph::helpers::operator<<;

namespace LayerTestsDefinitions {

std::string InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<InterpolateLayerTestParams> obj) {
    InterpolateSpecificParams interpolateParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, targetShapes;
    std::string targetDevice;
    std::tie(interpolateParams, netPrecision, inputShapes, targetShapes, targetDevice) = obj.param;
    std::vector<size_t> padBegin, padEnd, axes;
    bool antialias;
    ngraph::op::v4::Interpolate::InterpolateMode mode;
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
    ngraph::op::v4::Interpolate::NearestMode nearestMode;
    double cubeCoef;
    std:tie(mode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef, axes) = interpolateParams;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "TS=" << CommonTestUtils::vec2str(targetShapes) << "_";
    result << "InterpolateMode=" << mode << "_";
    result << "CoordinateTransformMode=" << coordinateTransformMode << "_";
    result << "NearestMode=" << nearestMode << "_";
    result << "CubeCoef=" << cubeCoef << "_";
    result << "Antialias=" << antialias << "_";
    result << "PB=" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE=" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "Axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void InterpolateLayerTest::SetUp() {
    InterpolateSpecificParams interpolateParams;
    std::vector<size_t> inputShape, targetShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;

    std::tie(interpolateParams, netPrecision, inputShape, targetShape, targetDevice) = this->GetParam();
    std::vector<size_t> padBegin, padEnd, axes;
    bool antialias;
    ngraph::op::v4::Interpolate::InterpolateMode mode;
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
    ngraph::op::v4::Interpolate::NearestMode nearestMode;

    using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;
    ShapeCalcMode shape_calc_node = ShapeCalcMode::sizes;
    double cubeCoef;
    std:tie(mode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef, axes) = interpolateParams;

    for (auto& p : padBegin) {
        printf("Test padB: %lu\n", p);
    }
    for (auto& p : padEnd) {
        printf("Test padE: %lu\n", p);
    }

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto sizesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {targetShape.size()}, targetShape);
    auto sizesInput = std::make_shared<ngraph::opset3::Constant>(sizesConst);

    std::vector<float> scales(targetShape.size(), 1.0f);
    auto scalesConst = ngraph::opset3::Constant(ngraph::element::Type_t::f32, {scales.size()}, scales);
    auto scalesInput = std::make_shared<ngraph::opset3::Constant>(scalesConst);

    auto axesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {axes.size()}, axes);
    auto axesInput = std::make_shared<ngraph::opset3::Constant>(axesConst);

    ngraph::op::v4::Interpolate::InterpolateAttrs interpolateAttributes{mode, shape_calc_node, padBegin,
        padEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};
    auto interpolate = std::make_shared<ngraph::op::v4::Interpolate>(params[0],
                                                                     sizesInput,
                                                                     scalesInput,
                                                                     axesInput,
                                                                     interpolateAttributes);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(interpolate)};
    function = std::make_shared<ngraph::Function>(results, params, "interpolate");
}

TEST_P(InterpolateLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
