// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Parameter class
 * @file openvino/runtime/parameter.hpp
 */
#pragma once

#include <map>

#include "openvino/core/any.hpp"

namespace ov {
namespace runtime {
/**
 * @brief This type of map is commonly used to return set of loaded from inference plugin
 */
using ParamMap = std::map<std::string, Any>;
}  // namespace runtime
}  // namespace ov
