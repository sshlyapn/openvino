# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "async_gpu_detection")

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME async_gpu_detection
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES ie_samples_utils)

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS core imgproc imgcodecs QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV is disabled or not found, ${TARGET_NAME} will be built without OpenCV support")
else()
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES} ie_samples_utils)
    if(UNIX AND NOT APPLE)
        # Workaround issue that rpath-link is missing for PRIVATE dependencies
        # Fixed in cmake 3.16.0 https://gitlab.kitware.com/cmake/cmake/issues/19556
        target_link_libraries(${TARGET_NAME} INTERFACE "-Wl,-rpath-link,${OpenCV_INSTALL_PATH}/lib")
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
endif()
