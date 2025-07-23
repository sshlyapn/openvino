// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/subgraph.hpp"
#include "subgraph_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/jit/subgraph.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<subgraph>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_JIT(jit::Subgraph, shape_types::any)
    };

    return impls;
}

}  // namespace ov::intel_gpu
