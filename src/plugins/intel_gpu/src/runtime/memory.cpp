// Copyright (C) 2018-2024 Intel Corporation
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

namespace {
std::string convert_element_new(int64_t i) {
    std::stringstream ss;
    ss << std::setw(4) << i;
    return ss.str();
}

std::string convert_element_new(int32_t i) {
    std::stringstream ss;
    ss << std::setw(4) << i;
    return ss.str();
}


std::string convert_element_new(float f) {
    std::stringstream ss;
    ss << std::setw(8) << std::fixed << std::setprecision(4) << f;
    return ss.str();
}


std::string convert_element_new(ov::float16 h) {
    std::stringstream ss;
    ss << std::setw(8) << std::fixed << std::setprecision(4) << static_cast<float>(h);
    return ss.str();
}

size_t get_x_pitch(const layout& layout) {
    try {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    } catch (...) {
        // When spatial size of x=0, x_pitch is meaningless
        return 0;
    }
}

template <class T>
void format_memory(const memory::ptr mem, stream& stream, std::string name = "") {
    std::stringstream ss;
    ss << "Memory content of " << mem->buffer_ptr() << " (name=" << name << ")" << ":" << "\n";
    ss << "Layout: " << mem->get_layout().to_short_string() << "\n";

    mem_lock<T, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(mem->get_layout());

    auto&& size = mem->get_layout().get_tensor();


    std::stringstream buffer_content;
    auto shape = mem->get_layout().get_shape();
    if (shape.size() == 4 && shape[0] == 1 && shape[3] == 1) {
        ss << "(      y:    ): ";
        auto alignment = ov::element::Type(mem->get_layout().data_type).is_real() ? 8 : 4;
        for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y) {
            ss << std::setw(alignment) << y;
        }
        ss << "\n";

        for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f) {
            buffer_content << "(" << std::setw(3) << 0 << ", " << std::setw(2) << f << ", " << std::setw(3) << "" << "): ";
            cldnn::tensor t(cldnn::group(0), cldnn::batch(0), cldnn::feature(f), cldnn::spatial(0, 0, 0, 0));
            size_t input_it = mem->get_layout().get_linear_offset(t);
            for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y, input_it += 1) {
                buffer_content << convert_element_new(mem_ptr[input_it]);
            }
            buffer_content << "\n";
        }
    } else {
        ss << "(      i:    ): ";
        auto alignment = ov::element::Type(mem->get_layout().data_type).is_real() ? 8 : 4;
        for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x) {
            ss << std::setw(alignment) << x;
        }
        ss << "\n";

        for (cldnn::tensor::value_type g = 0; g < size.group[0]; ++g) {
            for (cldnn::tensor::value_type b = 0; b < size.batch[0]; ++b) {
                for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f) {
                    for (cldnn::tensor::value_type w = 0; w < size.spatial[3]; ++w) {
                        for (cldnn::tensor::value_type z = 0; z < size.spatial[2]; ++z) {
                            for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y) {
                                buffer_content << "(" << std::setw(3) << b << ", " << std::setw(2) << f << ", " << std::setw(3) << y << "): ";
                                cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                                size_t input_it = mem->get_layout().get_linear_offset(t);

                                for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                    buffer_content << convert_element_new(mem_ptr[input_it]);
                                }
                                buffer_content << "\n";
                            }
                        }
                    }
                }
            }
        }
    }

    const size_t prime_number = 2654435761; // magic number to reduce hash collision rate.
    auto seed = hash_combine(prime_number, buffer_content.str());

    ss << buffer_content.str() << "\n";
    ss << "Hash: " << seed << "\n";
    ss << "End of memory\n";

    GPU_DEBUG_TRACE_DETAIL << ss.str() << "\n";
}
} // namespace

void memory::print_memory(stream& stream, layout data_layout, std::string name, bool add_paddings) const {
    GPU_DEBUG_TRACE_DETAIL << "Original layout:\n" << data_layout << "\n";
    if (add_paddings) {
        auto padded_dims = data_layout.get_padded_dims();
        ov::PartialShape vec;
        for (size_t i = 0; i < padded_dims.size(); i++)
            vec.push_back(padded_dims[i]);
        data_layout.set_partial_shape(vec);
        data_layout.data_padding = padding();
    }

    // Reinterpret buffer to represent actual data layout
    auto actual_mem = this->get_engine()->reinterpret_buffer(*this, data_layout);

    if (data_layout.count() == 0) {
        GPU_DEBUG_TRACE_DETAIL << name <<  " is empty buffer\n";
        return;
    }

    auto mem_dt = actual_mem->get_layout().data_type;
    if (mem_dt == cldnn::data_types::f32)
        format_memory<float>(actual_mem, stream, name);
    else if (mem_dt == cldnn::data_types::f16)
        format_memory<ov::float16>(actual_mem, stream, name);
    else if (mem_dt == cldnn::data_types::i64)
        format_memory<int64_t>(actual_mem, stream, name);
    else if (mem_dt == cldnn::data_types::i32)
        format_memory<int32_t>(actual_mem, stream, name);
    else if (mem_dt == cldnn::data_types::i8)
        format_memory<int8_t>(actual_mem, stream, name);
    else if (mem_dt == cldnn::data_types::u8)
        format_memory<uint8_t>(actual_mem, stream, name);
    else if (mem_dt == cldnn::data_types::u8)
        format_memory<uint8_t>(actual_mem, stream, name);
    else
        std::cout << "Dump for this data type is not supported: " << mem_dt << std::endl;
}


MemoryTracker::MemoryTracker(engine* engine, void* buffer_ptr, size_t buffer_size, allocation_type alloc_type)
    : m_engine(engine)
    , m_buffer_ptr(buffer_ptr)
    , m_buffer_size(buffer_size)
    , m_alloc_type(alloc_type) {
    if (m_engine) {
        m_engine->add_memory_used(m_buffer_size, m_alloc_type);
        GPU_DEBUG_LOG << "Allocate " << m_buffer_size << " bytes of " << m_alloc_type << " allocation type ptr = " << m_buffer_ptr
                      << " (current=" << m_engine->get_used_device_memory(m_alloc_type) << ";"
                      << " max=" << m_engine->get_max_used_device_memory(m_alloc_type) << ")" << std::endl;
    }
}

MemoryTracker::~MemoryTracker() {
    if (m_engine) {
        try {
            m_engine->subtract_memory_used(m_buffer_size, m_alloc_type);
        } catch (...) {}
        GPU_DEBUG_LOG << "Free " << m_buffer_size << " bytes of " << m_alloc_type << " allocation type ptr = " << m_buffer_ptr
                      << " (current=" << m_engine->get_used_device_memory(m_alloc_type) << ";"
                      << " max=" << m_engine->get_max_used_device_memory(m_alloc_type) << ")" << std::endl;
    }
}

memory::memory(engine* engine, const layout& layout, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker)
    : _engine(engine), _layout(layout), _bytes_count(_layout.bytes_count()), m_mem_tracker(mem_tracker), _type(type) {
}

std::unique_ptr<surfaces_lock> surfaces_lock::create(engine_types engine_type, std::vector<memory::ptr> mem, const stream& stream) {
    switch (engine_type) {
    case engine_types::sycl:
    case engine_types::ocl:
        return std::unique_ptr<ocl::ocl_surfaces_lock>(new ocl::ocl_surfaces_lock(mem, stream));
    default: throw std::runtime_error("Unsupported engine type in surfaces_lock::create");
    }
}

}  // namespace cldnn
