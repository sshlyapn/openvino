// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "ocl/ocl_memory.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {

memory::memory(engine* engine, const layout& layout, allocation_type type, bool reused)
    : _engine(engine), _layout(layout), _bytes_count(_layout.bytes_count()), _type(type), _reused(reused) {
    if (!_reused && _engine) {
        _engine->add_memory_used(_bytes_count, type);
        GPU_DEBUG_LOG << "Allocate " << _bytes_count << " bytes of " << type << " allocation type"
                      << " (current=" << _engine->get_used_device_memory(type) << ";"
                      << " max=" << _engine->get_max_used_device_memory(type) << ")" << std::endl;
    }
}

memory::~memory() {
    if (!_reused && _engine) {
        try {
            _engine->subtract_memory_used(_bytes_count, _type);
        } catch (...) {}
        GPU_DEBUG_LOG << "Free " << _bytes_count << " bytes of " << _type << " allocation type"
                      << " (current=" << _engine->get_used_device_memory(_type) << ";"
                      << " max=" << _engine->get_max_used_device_memory(_type) << ")" << std::endl;
    }
}

std::unique_ptr<surfaces_lock> surfaces_lock::create(engine_types engine_type, std::vector<memory::ptr> mem, const stream& stream) {
    switch (engine_type) {
    case engine_types::ocl: return std::unique_ptr<ocl::ocl_surfaces_lock>(new ocl::ocl_surfaces_lock(mem, stream));
    default: throw std::runtime_error("Unsupported engine type in surfaces_lock::create");
    }
}

ov::Shape MemoryStatistic::shape_math(const ov::Shape& shape1, const ov::Shape& shape2, math_op op) {
    std::vector<size_t> result;

    OPENVINO_ASSERT(shape1.size() == shape2.size());

    for (size_t i = 0; i < shape1.size(); i++) {
        if (op == math_op::SUB && shape1[i] < shape2[i])
            return std::vector<size_t>();

        if (op == math_op::SUB)
            result.push_back(shape1[i] - shape2[i]);
        else if (op == math_op::SUM)
            result.push_back(shape1[i] + shape2[i]);
        else if (op == math_op::MUL)
            result.push_back(shape1[i] * shape2[i]);
    }

    return result;
}

void MemoryStatistic::add_shape(ov::Shape& shape) {
    if (shapes.size() >= deque_size) {
        shapes.pop_front();
    }
    shapes.push_back(shape);
}

bool MemoryStatistic::can_preallocate(size_t current_buffer_size, size_t desired_buffer_size) {
    auto device_mem_usage = _engine->get_used_device_memory(cldnn::allocation_type::usm_device);

    if (desired_buffer_size <= current_buffer_size)
        return true;

    float ration = static_cast<float>(desired_buffer_size) / static_cast<float>(current_buffer_size);

    if (device_mem_usage * ration >= _engine->get_device_info().max_global_mem_size * 0.95) {
        std::cout << "MEMORY LIMIT!!\n";
    }

    return device_mem_usage * ration < _engine->get_device_info().max_global_mem_size * 0.95;
}

std::pair<bool, ov::Shape> MemoryStatistic::predict_preallocated_shape_size(const std::string& id, ov::Shape& current_shape, bool can_reuse_buffer) {
    add_shape(current_shape);

    if (can_reuse_buffer)
        return {false, {}};

    if (shapes.size() == deque_size) {
        std::vector<ov::Shape> diffs;
        for (size_t i = deque_size - 1; i >= deque_size - 2; --i) {
            auto result = shape_math(shapes[i], shapes[i - 1], math_op::SUB);
            if (result.empty())
                break;
            diffs.push_back(result);
        }

        if (diffs[0] == diffs[1] && diffs.size() == 2) {
            const auto iters = 10;
            std::vector<size_t> mul(diffs[0].size(), iters);
            auto diff = shape_math(diffs[0], mul, math_op::MUL);
            auto new_shape = shape_math(current_shape, diff, math_op::SUM);
            return {true, new_shape};
        } else {
            const auto ratio = 1.1f;
            auto current_shape_size = ov::shape_size(current_shape);
            ov::Shape new_shape_size(current_shape.size(), 1);
            new_shape_size[0] = static_cast<size_t>(current_shape_size * ratio);
            // for (size_t i = 0; i < shapes.size(); i++)
            //     std::cout << i << ". " << shapes[i] << "\n";
            // std::cout << "Use 10% increase: " << current_shape << " -> " << new_shape_size << "\n";
            return {true, new_shape_size};
        }
    }
    return {false, {}};
}

}  // namespace cldnn
