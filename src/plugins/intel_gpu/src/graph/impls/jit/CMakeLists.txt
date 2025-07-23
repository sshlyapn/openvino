# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_jit_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    LINK_LIBRARIES openvino::reference
)

ov_build_target_faster(${TARGET_NAME} PCH PCH_EXCLUDE detection_output.cpp)
