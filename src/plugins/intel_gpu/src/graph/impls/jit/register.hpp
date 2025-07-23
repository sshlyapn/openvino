// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/subgraph.hpp"

namespace cldnn {
namespace jit {
void register_implementations();

namespace detail {

#define REGISTER_JIT(prim)        \
    struct attach_##prim##_impl { \
        attach_##prim##_impl();   \
    }

REGISTER_JIT(subgraph);

#undef REGISTER_JIT

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn
