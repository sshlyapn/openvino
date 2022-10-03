// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/crop.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

static constexpr float threshold_int8 = 1.f;
static constexpr float threshold_fp16 = 1e-1;
static constexpr float threshold_fp32 = 3e-4;

struct gemm_base_test_params;

template <typename input0_type, typename input1_type, typename input2_type, typename output_type, typename accumulator_type>
void execute(gemm_base_test_params p);

struct gemm_base_test_params {
    size_t m_size;
    size_t n_size;
    size_t k_size;
    size_t b0_num;
    size_t f0_num;
    size_t b1_num;
    size_t f1_num;
    size_t b2_num;
    size_t f2_num;
    size_t b_out_num;
    size_t f_out_num;
    bool transpose_input0;
    bool transpose_input1;
    float alpha;
    float beta;
    data_types input0_type;
    data_types input1_type;
    data_types input2_type;
    data_types output_type;
    std::vector <int> range0;
    std::vector <int> range1;
    std::vector <int> range2;
    std::string kernel_name;
};

#define CASE_GEMM_INT8_ONEDNN_1 1, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_NN_TRANSPOSITION_ONEDNN 32, 16, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

int main() {
    execute<int8_t, int8_t, float, float, int32_t>(gemm_base_test_params{CASE_GEMM_INT8_ONEDNN_1, ""});
    execute<FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16>(gemm_base_test_params{CASE_GEMM_FP16_NN_TRANSPOSITION_ONEDNN, ""});

    return 0;
}

inline size_t getGemmIndex(size_t x, size_t y, size_t f, size_t b, size_t x_size, size_t y_size, size_t f_num, size_t b_num,
                            size_t x_pitch, size_t y_pitch, size_t f_pitch, size_t b_pitch) {
    return (x % x_size) * x_pitch + (y % y_size) * y_pitch + (f % f_num) * f_pitch + (b % b_num) * b_pitch;
}

template <typename input0_type, typename input1_type, typename input2_type, typename output_type, typename accumulator_type>
void execute(gemm_base_test_params p) {
    static std::shared_ptr<cldnn::engine> engine = nullptr;
    const auto enable_profiling = true;
    if (!engine) {
        cldnn::queue_types queue_type = cldnn::queue_types::out_of_order;
        std::string sources_dumps_dir = "";
        priority_mode_types priority_mode = priority_mode_types::disabled;
        throttle_mode_types throttle_mode = throttle_mode_types::disabled;
        bool use_memory_pool = true;
        bool use_unified_shared_memory = true;
        auto engine_config = engine_configuration(enable_profiling, queue_type, sources_dumps_dir, priority_mode, throttle_mode, use_memory_pool, use_unified_shared_memory);
        engine = cldnn::engine::create(engine_types::ocl, runtime_types::ocl, engine_config);
    }

    auto y0_size = p.m_size;
    auto y0_pitch = p.k_size;
    auto x0_size = p.k_size;
    auto x0_pitch = 1;
    auto f0_pitch = y0_size * x0_size;
    auto b0_pitch = p.f0_num * f0_pitch;

    auto y1_size = p.k_size;
    auto y1_pitch = p.n_size;
    auto x1_size = p.n_size;
    auto x1_pitch = 1;
    auto f1_pitch = y1_size * x1_size;
    auto b1_pitch = p.f1_num * f1_pitch;

    auto y2_size = p.m_size;
    auto y2_pitch = p.n_size;
    auto x2_size = p.n_size;
    auto x2_pitch = 1;
    auto f2_pitch = y2_size * x2_size;
    auto b2_pitch = p.f2_num * f2_pitch;

    auto y_out_size = p.m_size;
    auto y_out_pitch = p.n_size;
    auto x_out_size = p.n_size;
    auto x_out_pitch = 1;
    auto f_out_pitch = y_out_size * x_out_size;
    auto b_out_pitch = p.f_out_num * f_out_pitch;

    if (p.transpose_input0) {
        y0_size = p.k_size;
        y0_pitch = p.m_size;
        x0_size = p.m_size;
        x0_pitch = 1;
    }

    if (p.transpose_input1) {
        y1_size = p.n_size;
        y1_pitch = p.k_size;
        x1_size = p.k_size;
        x1_pitch = 1;
    }

    auto input0_size = tensor((int)p.b0_num, (int)p.f0_num, (int)x0_size, (int)y0_size);
    VVVVF<input0_type> input0_data = generate_random_4d<input0_type>(p.b0_num, p.f0_num, x0_size, y0_size, p.range0[0], p.range0[1], p.range0[2]);
    auto input0_data_bfyx = flatten_4d(format::bfyx, input0_data);
    auto input0_mem = engine->allocate_memory({ p.input0_type, format::bfyx, input0_size });
    set_values(input0_mem, input0_data_bfyx);

    auto input1_size = tensor((int)p.b1_num, (int)p.f1_num, (int)x1_size, (int)y1_size);
    VVVVF<input1_type> input1_data = generate_random_4d<input1_type>(p.b1_num, p.f1_num, x1_size, y1_size, p.range1[0], p.range1[1], p.range1[2]);
    auto input1_data_bfyx = flatten_4d(format::bfyx, input1_data);
    auto input1_mem = engine->allocate_memory({ p.input1_type, format::bfyx, input1_size });
    set_values(input1_mem, input1_data_bfyx);

    auto input2_size = tensor((int)p.b2_num, (int)p.f2_num, (int)x2_size, (int)y2_size);
    VVVVF<input2_type> input2_data = generate_random_4d<input2_type>(p.b2_num, p.f2_num, x2_size, y2_size, p.range2[0], p.range2[1], p.range2[2]);
    auto input2_data_bfyx = flatten_4d(format::bfyx, input2_data);
    auto input2_mem = engine->allocate_memory({ p.input2_type, format::bfyx, input2_size });
    set_values(input2_mem, input2_data_bfyx);

    std::vector<output_type> out_data(p.b_out_num * p.f_out_num * p.m_size * p.n_size);

    for (size_t b = 0; b < p.b_out_num; ++b) {
        for (size_t f = 0; f < p.f_out_num; ++f) {
            for (size_t y = 0; y < p.m_size; ++y) {
                for (size_t x = 0; x < p.n_size; ++x) {
                    size_t input2_data_index = getGemmIndex(x, y, f, b, x2_size, y2_size, p.f2_num, p.b2_num, x2_pitch, y2_pitch, f2_pitch, b2_pitch);
                    size_t out_data_index = getGemmIndex(x, y, f, b, x_out_size, y_out_size, p.f_out_num, p.b_out_num,
                                                            x_out_pitch, y_out_pitch, f_out_pitch, b_out_pitch);
                    accumulator_type acc = 0;

                    for (size_t k = 0; k < p.k_size; ++k) {
                        size_t input0_data_index = getGemmIndex(k * (!p.transpose_input0) + y * p.transpose_input0, y * (!p.transpose_input0) +
                        k * p.transpose_input0, f, b, x0_size, y0_size, p.f0_num, p.b0_num, x0_pitch, y0_pitch, f0_pitch, b0_pitch);
                        size_t input1_data_index = getGemmIndex(x * (!p.transpose_input1) + k * p.transpose_input1, k * (!p.transpose_input1) +
                        x * p.transpose_input1, f, b, x1_size, y1_size, p.f1_num, p.b1_num, x1_pitch, y1_pitch, f1_pitch, b1_pitch);
                        acc += (accumulator_type)input0_data_bfyx[input0_data_index] * (accumulator_type)input1_data_bfyx[input1_data_index];
                    }

                    out_data[out_data_index] = (output_type)acc;
                    out_data[out_data_index] *= (output_type)p.alpha;
                    if (p.beta)
                        out_data[out_data_index] += (output_type)p.beta * (output_type)input2_data_bfyx[input2_data_index];
                }
            }
        }
    }

    topology topology;
    topology.add(input_layout("input0", input0_mem->get_layout()));
    topology.add(input_layout("input1", input1_mem->get_layout()));
    if (p.beta != 0) {
        topology.add(input_layout("input2", input2_mem->get_layout()));
        topology.add(gemm("gemm_prim", { "input0", "input1", "input2" }, p.output_type, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
    } else {
        topology.add(gemm("gemm_prim", { "input0", "input1" }, p.output_type, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
    }
    topology.add(reorder("reorder_bfyx", "gemm_prim", format::bfyx, data_types::f32));

    build_options options;
    implementation_desc gemm_impl = { format::bfyx, p.kernel_name };
    options.set_option(build_option::force_implementations({ {"gemm_prim", gemm_impl} }));

    network network(*engine, topology, options);
    network.set_input_data("input0", input0_mem);
    network.set_input_data("input1", input1_mem);
    if (p.beta != 0) {
        network.set_input_data("input2", input2_mem);
    }
    auto outputs = network.execute();
    auto output = outputs.at("reorder_bfyx").get_memory();
    mem_lock<float> output_ptr(output, get_test_stream());

    if (enable_profiling) {
        auto calc_time = [](const primitive_id& id, const event::ptr ev) {
            cldnn::instrumentation::profiling_info cldnnInfo{id, ev->get_profiling_info()};
            long long time = 0;
            for (auto &interval : cldnnInfo.intervals) {
                using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
                time += std::chrono::duration_cast<duration_t>(interval.value->value()).count();
            }
            return time;
        };

        auto executed_primitives = network.get_executed_primitives();
        auto gemm = executed_primitives.find("gemm_prim") != executed_primitives.end() ? executed_primitives.find("gemm_prim") :
                                                                                         executed_primitives.find("reorder_bfyx");
        if (gemm == executed_primitives.end())
            throw std::runtime_error("Couldn't find requested primitive\n");

        auto time = calc_time(gemm->first, gemm->second);
        std::cout << "Gemm time: " << time << "us" << std::endl;
    }

    data_types data_type = type_to_data_type<input0_type>::value;
    ASSERT_EQ(output_ptr.size(), (size_t)(p.b_out_num * p.f_out_num * p.m_size * p.n_size));
    if (data_type == data_types::i8 || data_type == data_types::u8) {
        for (size_t i = 0; i < out_data.size(); ++i) {
            ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_int8) << "index = " << i;
        }
    } else if (data_type == data_types::f16) {
        for (size_t i = 0; i < out_data.size(); ++i) {
            ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_fp16) << "index = " << i;
        }
    } else {
        for (size_t i = 0; i < out_data.size(); ++i) {
            ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_fp32) << "index = " << i;
        }
    }
}
