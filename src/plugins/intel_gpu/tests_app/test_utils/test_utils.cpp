// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "float16.h"
#include <iostream>


namespace cldnn {
const cldnn::data_types type_to_data_type<FLOAT16>::value;
}  // namespace cldnn

using namespace cldnn;

namespace tests {

std::shared_ptr<cldnn::engine> create_test_engine() {
    auto ret = cldnn::engine::create(engine_types::ocl, runtime_types::ocl);
#ifdef ENABLE_ONEDNN_FOR_GPU
    if(ret->get_device_info().supports_immad)
        ret->create_onednn_engine({});
#endif
    return ret;
}

cldnn::engine& get_test_engine() {
    static std::shared_ptr<cldnn::engine> test_engine = nullptr;
    if (!test_engine) {
        test_engine = create_test_engine();
    }
    return *test_engine;
}

cldnn::stream_ptr get_test_stream_ptr() {
    static std::shared_ptr<cldnn::stream> test_stream = nullptr;
    if (!test_stream) {
        // Create OOO queue for test purposes. If in-order queue is needed in a test, then it should be created there explicitly
        ExecutionConfig cfg(ov::intel_gpu::queue_type(QueueTypes::out_of_order));
        test_stream = get_test_engine().create_stream(cfg);
    }
    return test_stream;
}

cldnn::stream& get_test_stream() {
    return *get_test_stream_ptr();
}

double default_tolerance(data_types dt) {
    switch (dt) {
    case data_types::f16:
        return 1e-3;
    case data_types::f32:
        return 1e-5;
    case data_types::i8:
    case data_types::u8:
        return 1.5;
    default:
        IE_THROW() << "Unknown";
    }
    IE_THROW() << "Unknown";
}

}  // namespace tests
