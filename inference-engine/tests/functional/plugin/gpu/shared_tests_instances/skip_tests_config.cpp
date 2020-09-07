// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            // Issues - 34059
            ".*BehaviorTests\\.pluginDoesNotChangeOriginalNetwork.*",
            //TODO: Issue: 34349
            R"(.*(IEClassLoadNetwork).*(QueryNetworkMULTIWithHETERONoThrow_V10|QueryNetworkHETEROWithMULTINoThrow_V10).*)",
            //TODO: Issue: 34748
            R"(.*(ComparisonLayerTest).*)",
            // Looks like an ngraph bug:
            R"(.*Interpolate_Basic.*tf_half_pixel_for_nn.*)",
    };
}