// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"

namespace cldnn {
namespace jit {

#define REGISTER_JIT(prim)                                \
    static detail::attach_##prim##_impl attach_##prim

void register_implementations() {
    REGISTER_JIT(subgraph);
}

}  // namespace cpu
}  // namespace cldnn
