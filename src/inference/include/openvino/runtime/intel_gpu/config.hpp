// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for GPU plugin
 *        To use in SetConfig() method of plugins
 *
 * @file gpu_config.hpp
 */
#pragma once

#include "openvino/runtime/config.hpp"

namespace ov {
namespace gpu {

/**
 * @brief Metric which defines size of memory in bytes available for the device. For iGPU it returns host memory size,
 * for dGPU - dedicated gpu memory size
 */
static constexpr Key<uint64_t, ConfigMutability::RO> device_total_mem_size{"DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief Metric to get microarchitecture identifier in major.minor.revision format
 */
static constexpr Key<std::string, ConfigMutability::RO> uarch_version{"UARCH_VERSION"};

/**
 * @brief Metric to get count of execution units for current GPU
 */
static constexpr Key<int32_t, ConfigMutability::RO> execution_units_count{"EXECUTION_UNITS_COUNT"};

/**
 * @brief This key instructs the GPU plugin to use throttle hints the OpenCL queue throttle hint
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
 * chapter 9.19. This option should be used with an unsigned integer value (1 is lowest energy consumption)
 * 0 means no throttle hint is set and default queue created.
 */
static constexpr Key<uint32_t> plugin_throttle{"PLUGIN_THROTTLE"};

/**
 * @brief Turning on this key enables to unroll recurrent layers such as TensorIterator or Loop with fixed iteration
 * count. This key is turned on by default. Turning this key on will achieve better inference performance for loops with
 * not too many iteration counts (less than 16, as a rule of thumb). Turning this key off will achieve better
 * performance for both graph loading time and inference time with many iteration counts (greater than 16). Note that
 * turning this key on will increase the graph loading time in proportion to the iteration counts.
 * Thus, this key should be turned off if graph loading time is considered to be most important target to optimize.*/
static constexpr Key<bool> enable_loop_unrolling{"ENABLE_LOOP_UNROLLING"};

}  // namespace gpu
}  // namespace ov
