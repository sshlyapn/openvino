// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/detection_output.hpp"
#include "intel_gpu/primitives/proposal.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/primitives/range.hpp"

namespace cldnn {
namespace cpu {
void register_implementations();

namespace detail {


#define REGISTER_CPU(prim)        \
    struct attach_##prim##_impl { \
        attach_##prim##_impl();   \
    }

REGISTER_CPU(assign);
REGISTER_CPU(proposal);
REGISTER_CPU(read_value);
REGISTER_CPU(non_max_suppression);
REGISTER_CPU(detection_output);
REGISTER_CPU(shape_of);
REGISTER_CPU(concatenation);
REGISTER_CPU(gather);
REGISTER_CPU(strided_slice);
REGISTER_CPU(range);

#undef REGISTER_CPU

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn
