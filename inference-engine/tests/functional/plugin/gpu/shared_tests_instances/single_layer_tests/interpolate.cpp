// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> prc = {
        // InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
        {1, 4, 6, 6},
};

const std::vector<std::vector<size_t>> targetShapes = {
        {1, 4, 10, 10},
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> withoutNearestModes = {
        ngraph::op::v4::Interpolate::InterpolateMode::linear,
        ngraph::op::v4::Interpolate::InterpolateMode::cubic,
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> linearOnnxMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::nearest,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::simple,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
        ngraph::op::v4::Interpolate::NearestMode::floor,
        ngraph::op::v4::Interpolate::NearestMode::ceil,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
};

const std::vector<std::vector<size_t>> pads = {
        // {0, 0, 1, 2},
        {0, 0, 1, 1},
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<size_t>> defaultAxes = {
        {0, 1, 2, 3}
};

const std::vector<std::vector<size_t>> linearOnnxModeAxes = {
        {2, 3}
};

const auto interpolateCasesWithoutNearest = ::testing::Combine(
        ::testing::ValuesIn(withoutNearestModes),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes));

const auto interpolateCasesLinearOnnxMode = ::testing::Combine(
        ::testing::ValuesIn(linearOnnxMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(linearOnnxModeAxes));

const auto interpolateCasesNearesMode = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes));

INSTANTIATE_TEST_CASE_P(Interpolate_Basic, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(prc),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Interpolate_Nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearesMode,
        ::testing::ValuesIn(prc),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Interpolate_Linear_ONNX, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesLinearOnnxMode,
        ::testing::ValuesIn(prc),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    InterpolateLayerTest::getTestCaseName);

} // namespace
