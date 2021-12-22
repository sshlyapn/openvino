// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_parameter.hpp>
#include <istream>
#include <map>
#include <openvino/core/any.hpp>
#include <openvino/runtime/common.hpp>
#include <openvino/runtime/config.hpp>
#include <openvino/runtime/parameter.hpp>
#include <ostream>
#include <string>

namespace ov {
runtime::ConfigMap any_copy(const runtime::ParamMap& config_map);

void any_lexical_cast(const Any& any, ov::Any& to);

}  // namespace ov
