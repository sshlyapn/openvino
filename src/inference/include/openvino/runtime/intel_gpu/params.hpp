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
namespace intel_gpu {

using gpu_handle_param = void*;

/**
 * @brief Enum to define context type
 */
enum class ContextType {
    OCL = 0,
    VA_SHARED = 1,
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ContextType& context_type) {
    switch (context_type) {
    case ContextType::OCL:
        return os << "OCL";
    case ContextType::VA_SHARED:
        return os << "VA_SHARED";
    default:
        throw ov::Exception{"Unsupported context type"};
    }
}

inline std::istream& operator>>(std::istream& is, ContextType& context_type) {
    std::string str;
    is >> str;
    if (str == "OCL") {
        context_type = ContextType::OCL;
    } else if (str == "VA_SHARED") {
        context_type = ContextType::VA_SHARED;
    } else {
        throw ov::Exception{"Unsupported context type: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief Shared device context type: can be either pure OpenCL (OCL)
 * or shared video decoder (VA_SHARED) context
 */
static constexpr Key<ContextType> context_type{"CONTEXT_TYPE"};

/**
 * @brief This key identifies OpenCL context handle
 * in a shared context or shared memory blob parameter map
 */
static constexpr Key<gpu_handle_param> context_type{"OCL_CONTEXT"};

/**
 * @brief This key identifies ID of device in OpenCL context
 * if multiple devices are present in the context
 */
static constexpr Key<int> context_type{"OCL_CONTEXT_DEVICE_ID"};

/**
 * @brief In case of multi-tile system,
 * this key identifies tile within given context
 */
static constexpr Key<int> context_type{"TILE_ID"};

/**
 * @brief This key identifies OpenCL queue handle in a shared context
 */
static constexpr Key<gpu_handle_param> context_type{"OCL_QUEUE"};

/**
 * @brief This key identifies video acceleration device/display handle
 * in a shared context or shared memory blob parameter map
 */
static constexpr Key<gpu_handle_param> context_type{"VA_DEVICE"};

/**
 * @brief Enum to define possible host task priorities
 */
enum class SharedMemType {
    OCL_BUFFER = 0,
    OCL_IMAGE2D = 1,
    USM_USER_BUFFER = 2,
    USM_HOST_BUFFER = 3,
    USM_DEVICE_BUFFER = 4,
    VA_SURFACE = 5,
    DX_BUFFER = 6
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SharedMemType& share_mem_type) {
    switch (share_mem_type) {
    case SharedMemType::OCL_BUFFER:
        return os << "OCL_BUFFER";
    case SharedMemType::OCL_IMAGE2D:
        return os << "OCL_IMAGE2D";
    case SharedMemType::USM_USER_BUFFER:
        return os << "USM_USER_BUFFER";
    case SharedMemType::USM_HOST_BUFFER:
        return os << "USM_HOST_BUFFER";
    case SharedMemType::USM_DEVICE_BUFFER:
        return os << "USM_DEVICE_BUFFER";
    case SharedMemType::VA_SURFACE:
        return os << "VA_SURFACE";
    case SharedMemType::DX_BUFFER:
        return os << "DX_BUFFER";
    default:
        throw ov::Exception{"Unsupported host task priority"};
    }
}

inline std::istream& operator>>(std::istream& is, SharedMemType& share_mem_type) {
    std::string str;
    is >> str;
    if (str == "OCL_BUFFER") {
        share_mem_type = SharedMemType::OCL_BUFFER;
    } else if (str == "OCL_IMAGE2D") {
        share_mem_type = SharedMemType::OCL_IMAGE2D;
    } else if (str == "USM_USER_BUFFER") {
        share_mem_type = SharedMemType::USM_USER_BUFFER;
    } else if (str == "USM_HOST_BUFFER") {
        share_mem_type = SharedMemType::USM_HOST_BUFFER;
    } else if (str == "USM_DEVICE_BUFFER") {
        share_mem_type = SharedMemType::USM_DEVICE_BUFFER;
    } else if (str == "VA_SURFACE") {
        share_mem_type = SharedMemType::VA_SURFACE;
    } else if (str == "DX_BUFFER") {
        share_mem_type = SharedMemType::DX_BUFFER;
    } else {
        throw ov::Exception{"Unsupported host task priority: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief This key identifies type of internal shared memory
 * in a shared memory blob parameter map.
 */
static constexpr Key<SharedMemType> context_type{"SHARED_MEM_TYPE"};

/**
 * @brief This key identifies OpenCL memory handle
 * in a shared memory blob parameter map
 */
static constexpr Key<gpu_handle_param> context_type{"MEM_HANDLE"};

/**
 * @brief This key identifies video decoder surface handle
 * in a shared memory blob parameter map
 */
#ifdef _WIN32
static constexpr Key<gpu_handle_param> context_type{"DEV_OBJECT_HANDLE"};
#else
static constexpr Key<uint32_t> context_type{"DEV_OBJECT_HANDLE"};
#endif

/**
 * @brief This key identifies video decoder surface plane
 * in a shared memory blob parameter map
 */
static constexpr Key<uint32_t> context_type{"VA_PLANE"};

}  // namespace intel_gpu
}  // namespace ov
