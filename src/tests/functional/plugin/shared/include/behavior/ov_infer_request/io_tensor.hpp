// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <future>

#include "shared_test_classes/subgraph/basic_lstm.hpp"
#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

struct OVInferRequestIOTensorTest : public OVInferRequestTests {
    static std::string getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj);
    void SetUp() override;
    void TearDown() override;
    runtime::InferRequest req;
    ov::Output<const ov::Node> input;
    ov::Output<const ov::Node> output;
};

using OVInferRequestSetPrecisionParams = std::tuple<
        element::Type,                                                     // element type
        std::string,                                                       // Device name
        ov::AnyMap                                              // Config
>;
struct OVInferRequestIOTensorSetPrecisionTest : public testing::WithParamInterface<OVInferRequestSetPrecisionParams>,
                                                public CommonTestUtils::TestsCommon {
    static std::string getTestCaseName(const testing::TestParamInfo<OVInferRequestSetPrecisionParams>& obj);
    void SetUp() override;
    void TearDown() override;
    std::shared_ptr<ov::runtime::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function;
    runtime::CompiledModel execNet;
    runtime::InferRequest req;
    std::string         target_device;
    ov::AnyMap   config;
    element::Type       element_type;
};
} // namespace behavior
}  // namespace test
}  // namespace ov
