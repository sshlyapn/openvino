# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_ONEDNN_FOR_GPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_jit_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    LINK_LIBRARIES onednn_gpu_tgt openvino::snippets
)

ov_build_target_faster(${TARGET_NAME} PCH PCH_EXCLUDE detection_output.cpp)
