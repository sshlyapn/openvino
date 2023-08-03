// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_user_event.hpp"
#include <list>

using namespace cldnn::ocl;

void ocl_user_event::set_impl() {
    // we simulate "wrapper_cast" here to cast from cl::Event to cl::UserEvent which both wrap the same cl_event
    // casting is valid as long as cl::UserEvent does not add any members to cl::Event (which it shouldn't)
    static_assert(sizeof(cl::UserEvent) == sizeof(cl::Event) && alignof(cl::UserEvent) == alignof(cl::Event),
                  "cl::UserEvent does not match cl::Event");
    static_cast<cl::UserEvent&&>(get()).setStatus(CL_COMPLETE);
    _duration = std::unique_ptr<cldnn::instrumentation::profiling_period_basic>(
        new cldnn::instrumentation::profiling_period_basic(_timer.uptime()));
    _timestamp_end = openvino::itt::timestamp();
}

bool ocl_user_event::get_profiling_info_impl(std::list<cldnn::instrumentation::profiling_interval>& info) {
    if (_duration == nullptr) {
        return false;
    }

    auto period = std::make_shared<instrumentation::profiling_period_basic>(_duration->value());
    info.push_back({ instrumentation::profiling_stage::executing, period });
    return true;
}

std::pair<uint64_t, uint64_t> ocl_user_event::get_host_timestamps(const stream& s) const {
    return {_timestamp_begin, _timestamp_end};
}

void ocl_user_event::wait_impl() {
    if (!_set) {
        throw std::runtime_error("[CLDNN] ocl_user_event::wait_impl is called before marking event handle as complete");
    }

    if (_event.get() != nullptr) {
        _event.wait();
    }
}

bool ocl_user_event::is_set_impl() {
    if (_event.get() != nullptr) {
        return _event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
    }
    return true;
}
