// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "any_copy.hpp"

#include <sstream>

#include "openvino/runtime/config.hpp"

namespace ov {
runtime::ConfigMap any_copy(const ov::AnyMap& params) {
    auto to_config_string = [](const Any& any) -> std::string {
        if (any.is<bool>()) {
            return any.as<bool>() ? "YES" : "NO";
        } else {
            std::stringstream strm;
            any.print(strm);
            return strm.str();
        }
    };
    runtime::ConfigMap result;
    for (auto&& value : params) {
        result.emplace(value.first, to_config_string(value.second));
    }
    return result;
}

void any_lexical_cast(const ov::Any& from, ov::Any& to) {
    if (!from.is<std::string>()) {
        to = from;
    } else {
        auto str = from.as<std::string>();
        if (to.is<std::string>()) {
            to = from;
        } else if (to.is<bool>()) {
            if (str == "YES") {
                to = true;
            } else if (str == "NO") {
                to = false;
            } else {
                OPENVINO_UNREACHABLE("Unsupported lexical cast to bool from: ", str);
            }
        } else {
            std::stringstream strm(str);
            to.read(strm);
            if (strm.fail()) {
                OPENVINO_UNREACHABLE("Unsupported lexical cast to ", to.type_info().name(), " from: ", str);
            }
        }
    }
}
}  // namespace ov
