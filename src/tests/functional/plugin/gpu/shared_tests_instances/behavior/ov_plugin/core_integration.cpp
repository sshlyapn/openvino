// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include <openvino/runtime/intel_gpu/properties.hpp>

#ifdef _WIN32
#    include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
#    include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_config.hpp"
#include "gpu/gpu_context_api_ocl.hpp"

using namespace ov::test::behavior;

namespace {
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCommon,
        OVClassBasicTestP,
        ::testing::Values(std::make_pair("ov_intel_gpu_plugin", "GPU")));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassNetworkTestP, OVClassNetworkTestP, ::testing::Values("GPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_GOPS, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_TYPE, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetConfigTest,
        OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetAvailableDevices, OVClassGetAvailableDevices, ::testing::Values("GPU"));

//
// GPU specific metrics
//
using OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(deviceName, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)));
    uint64_t t = p;

    std::cout << "GPU device total memory size: " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE,
        ::testing::Values("GPU"));

using OVClassGetMetricTest_GPU_UARCH_VERSION = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_UARCH_VERSION, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(deviceName, GPU_METRIC_KEY(UARCH_VERSION)));
    std::string t = p;

    std::cout << "GPU device uarch: " << t << std::endl;
    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(UARCH_VERSION));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_UARCH_VERSION,
        ::testing::Values("GPU"));

using OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(deviceName, GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)));
    int t = p;

    std::cout << "GPU EUs count: " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT,
        ::testing::Values("GPU"));


using OVClassGetPropertyTest_GPU = OVClassBaseTestP;
TEST_P(OVClassGetPropertyTest_GPU, GetMetricSupporetedPropertiesAndPrintNoThrow) {
    ov::Core ie;

    std::vector<ov::PropertyName> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    std::cout << "SUPPORTED_PROPERTIES: ";
    for (const auto& prop : properties) {
        std::cout << prop << " ";
    }
    std::cout << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricAvailableDevicesAndPrintNoThrow) {
    ov::Core ie;

    std::vector<std::string> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::available_devices));

    std::cout << "AVAILABLE_DEVICES: ";
    for (const auto& prop : properties) {
        std::cout << prop << " ";
    }
    std::cout << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::available_devices);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricRangeForAsyncInferRequestsAndPrintNoThrow) {
    ov::Core ie;

    std::tuple<unsigned int, unsigned int, unsigned int> property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::range_for_async_infer_requests));

    std::cout << "RANGE_FOR_ASYNC_INFER_REQUESTS: " << std::get<0>(property) << " " <<
                                                       std::get<1>(property) << " " <<
                                                       std::get<2>(property) << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_async_infer_requests);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricRangeForStreamsAndPrintNoThrow) {
    ov::Core ie;

    std::tuple<unsigned int, unsigned int> property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::range_for_streams));

    std::cout << "RANGE_FOR_STREAMS: " << std::get<0>(property) << " " <<
                                          std::get<1>(property) << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_streams);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricOptimalBatchSizeAndPrintNoThrow) {
    ov::Core ie;

    unsigned int property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::optimal_batch_size));

    std::cout << "OPTIMAL_BATCH_SIZE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::optimal_batch_size);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricFullNameAndPrintNoThrow) {
    ov::Core ie;

    std::string property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::device::full_name));

    std::cout << "FULL_DEVICE_NAME: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricTypeAndPrintNoThrow) {
    ov::Core ie;

    ov::device::Type property = ov::device::Type::INTEGRATED;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::device::type));

    std::cout << "DEVICE_TYPE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::type);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricGopsAndPrintNoThrow) {
    ov::Core ie;

    std::map<ov::element::Type, float> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::device::gops));

    std::cout << "DEVICE_GOPS: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << "- " << prop.first << ": " << prop.second << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::gops);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricCapabilitiesAndPrintNoThrow) {
    ov::Core ie;

    std::vector<std::string> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::device::capabilities));

    std::cout << "OPTIMIZATION_CAPABILITIES: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << "- " << prop << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::capabilities);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricDeviceTotalMemSizeAndPrintNoThrow) {
    ov::Core ie;

    uint64_t property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::intel_gpu::device_total_mem_size));

    std::cout << "GPU_DEVICE_TOTAL_MEM_SIZE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::device_total_mem_size);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricUarchVersionAndPrintNoThrow) {
    ov::Core ie;

    std::string property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::intel_gpu::uarch_version));

    std::cout << "GPU_UARCH_VERSION: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::uarch_version);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricExecutionUnitsCountAndPrintNoThrow) {
    ov::Core ie;

    uint32_t property = 0;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::intel_gpu::execution_units_count));

    std::cout << "GPU_EXECUTION_UNITS_COUNT: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::execution_units_count);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricMemoryStatisticsAndPrintNoThrow) {
    ov::Core ie;

    std::map<std::string, uint64_t> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::intel_gpu::memory_statistics));

    std::cout << "GPU_MEMORY_STATISTICS: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << " " << prop.first << " - " << prop.second << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::memory_statistics);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricMaxBatchSizeAndPrintNoThrow) {
    ov::Core ie;

    uint32_t property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::intel_gpu::max_batch_size));

    std::cout << "GPU_MAX_BATCH_SIZE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::max_batch_size);
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetPropertyTest_GPU,
        ::testing::Values("GPU"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetConfigTest, OVClassGetConfigTest, ::testing::Values("GPU"));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest, ::testing::Values("GPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroExecutableNetworkGetMetricTest,
        OVClassLoadNetworkAfterCoreRecreateTest,
        ::testing::Values("GPU"));

// GetConfig / SetConfig for specific device

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice0Test, OVClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice1Test, OVClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.1")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice0Test, OVClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice1Test, OVClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.1")
);

// Several devices case

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestLoadNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestQueryNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestDefaultCore,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

// Set default device ID

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSetDefaultDeviceIDTest, OVClassSetDefaultDeviceIDTest,
        ::testing::Values(std::make_pair("GPU", "1"))
);

// Set config for all GPU devices

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSetGlobalConfigTest, OVClassSetGlobalConfigTest,
        ::testing::Values("GPU")
);
}  // namespace
