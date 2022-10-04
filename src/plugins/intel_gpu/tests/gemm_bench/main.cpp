// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

static constexpr float threshold_int8 = 1.f;
static constexpr float threshold_fp16 = 1e-1;
static constexpr float threshold_fp32 = 3e-4;

enum class matmul_type {
    gemm,
    fc
};

std::ostream& operator<<(std::ostream& stream,
                         const matmul_type& type) {
    if (type == matmul_type::gemm)
        stream << "gemm";
    else if (type == matmul_type::fc)
        stream << "fc";
    else
        throw std::runtime_error("Unsupported matmul type");
    return stream;
 }

struct gemm_params;

template <typename input0_type, typename input1_type, typename input2_type, typename output_type>
void execute(gemm_params p, matmul_type prim_type = matmul_type::gemm);

struct gemm_params {
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
    std::vector <int> range0;
    std::vector <int> range1;
    std::vector <int> range2;
    std::string kernel_name;
};

#define CASE_0 256, 256, 256, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_1 512, 1024, 256, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

int main() {
    execute<uint8_t, int8_t, int8_t, int8_t>(gemm_params{CASE_0, ""}, matmul_type::gemm);
    execute<int8_t, int8_t, int8_t, int8_t>(gemm_params{CASE_0, ""}, matmul_type::gemm);
    execute<int8_t, int8_t, int8_t, int8_t>(gemm_params{CASE_0, "gemm_ref"}, matmul_type::gemm);
    execute<int8_t, int8_t, float, float>(gemm_params{CASE_0, "gemm_ref"}, matmul_type::gemm);
    execute<int8_t, int8_t, float, float>(gemm_params{CASE_0, "gemm_mmad_int8_slm"}, matmul_type::gemm);

    execute<FLOAT16, FLOAT16, FLOAT16, FLOAT16>(gemm_params{CASE_1, "gemm_ref"}, matmul_type::gemm);
    execute<FLOAT16, FLOAT16, FLOAT16, FLOAT16>(gemm_params{CASE_1, "gemm_tiled_opt"}, matmul_type::gemm);

    execute<FLOAT16, FLOAT16, FLOAT16, FLOAT16>(gemm_params{CASE_1, "fully_connected_gpu_bf_tiled"}, matmul_type::fc);
    execute<FLOAT16, FLOAT16, FLOAT16, FLOAT16>(gemm_params{CASE_1, "fully_connected_gpu_bfyx_ref"}, matmul_type::fc);

    execute<int8_t, int8_t, int8_t, int8_t>(gemm_params{CASE_1, "fully_connected_gpu_imad"}, matmul_type::fc);
    execute<int8_t, int8_t, int8_t, int8_t>(gemm_params{CASE_1, "fully_connected_gpu_bfyx_ref"}, matmul_type::fc);
    return 0;
}

template <typename input0_type, typename input1_type, typename input2_type, typename output_type>
void execute(gemm_params p, matmul_type prim_type) {
    const bool enable_profiling = true;
    const bool use_onednn = false;

    static std::shared_ptr<cldnn::engine> engine = nullptr;
    if (!engine) {
        cldnn::queue_types queue_type = use_onednn ? cldnn::queue_types::in_order : cldnn::queue_types::out_of_order;
        std::string sources_dumps_dir = "";
        priority_mode_types priority_mode = priority_mode_types::disabled;
        throttle_mode_types throttle_mode = throttle_mode_types::disabled;
        bool use_memory_pool = true;
        bool use_unified_shared_memory = true;
        auto engine_config = engine_configuration(enable_profiling, queue_type, sources_dumps_dir, priority_mode, throttle_mode, use_memory_pool, use_unified_shared_memory);
        engine = cldnn::engine::create(engine_types::ocl, runtime_types::ocl, engine_config);
    }

    data_types input0_dt = type_to_data_type<input0_type>::value;
    data_types input1_dt = type_to_data_type<input1_type>::value;
    data_types input2_dt = type_to_data_type<input2_type>::value;
    data_types output_dt = type_to_data_type<output_type>::value;

    auto y0_size = p.m_size;
    auto x0_size = p.k_size;

    auto y1_size = p.k_size;
    auto x1_size = p.n_size;

    auto y2_size = p.m_size;
    auto x2_size = p.n_size;

    if (p.transpose_input0) {
        y0_size = p.k_size;
        x0_size = p.m_size;
    }

    if (p.transpose_input1) {
        y1_size = p.n_size;
        x1_size = p.k_size;
    }

    memory::ptr input0_mem;
    memory::ptr input1_mem;
    memory::ptr input2_mem;

    topology topology;
    if (prim_type == matmul_type::gemm) {
        auto input0_size = tensor((int)p.b0_num, (int)p.f0_num, (int)x0_size, (int)y0_size);
        VVVVF<input0_type> input0_data = generate_random_4d<input0_type>(p.b0_num, p.f0_num, x0_size, y0_size, p.range0[0], p.range0[1], p.range0[2]);
        auto input0_data_bfyx = flatten_4d(format::bfyx, input0_data);
        input0_mem = engine->allocate_memory({ input0_dt, format::bfyx, input0_size });
        set_values(input0_mem, input0_data_bfyx);

        auto input1_size = tensor((int)p.b1_num, (int)p.f1_num, (int)x1_size, (int)y1_size);
        VVVVF<input1_type> input1_data = generate_random_4d<input1_type>(p.b1_num, p.f1_num, x1_size, y1_size, p.range1[0], p.range1[1], p.range1[2]);
        auto input1_data_bfyx = flatten_4d(format::bfyx, input1_data);
        input1_mem = engine->allocate_memory({ input1_dt, format::bfyx, input1_size });
        set_values(input1_mem, input1_data_bfyx);

        auto input2_size = tensor((int)p.b2_num, (int)p.f2_num, (int)x2_size, (int)y2_size);
        VVVVF<input2_type> input2_data = generate_random_4d<input2_type>(p.b2_num, p.f2_num, x2_size, y2_size, p.range2[0], p.range2[1], p.range2[2]);
        auto input2_data_bfyx = flatten_4d(format::bfyx, input2_data);
        input2_mem = engine->allocate_memory({ input2_dt, format::bfyx, input2_size });
        set_values(input2_mem, input2_data_bfyx);

        topology.add(input_layout("input0", input0_mem->get_layout()));
        topology.add(input_layout("input1", input1_mem->get_layout()));
        if (p.beta != 0) {
            topology.add(input_layout("input2", input2_mem->get_layout()));
            topology.add(gemm("matmul_prim", { "input0", "input1", "input2" }, output_dt, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
        } else {
            topology.add(gemm("matmul_prim", { "input0", "input1" }, output_dt, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
        }
        topology.add(reorder("reorder_bfyx", "matmul_prim", format::bfyx, data_types::f32));
    } else if (prim_type == matmul_type::fc) {
        auto input0_size = tensor((int)p.m_size, (int)p.k_size, 1, 1);
        VVVVF<input0_type> input0_data = generate_random_4d<input0_type>(p.b0_num, p.f0_num, 1, 1, p.range0[0], p.range0[1], p.range0[2]);
        auto input0_data_bfyx = flatten_4d(format::bfyx, input0_data);
        input0_mem = engine->allocate_memory({ input0_dt, format::bfyx, input0_size });
        set_values(input0_mem, input0_data_bfyx);

        auto weights_size = tensor((int)p.n_size, (int)p.k_size, 1, 1);
        VVVVF<input1_type> weights_data = generate_random_4d<input1_type>(p.n_size, p.k_size, 1, 1, p.range1[0], p.range1[1], p.range1[2]);
        auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
        input1_mem = engine->allocate_memory({ input1_dt, format::bfyx, weights_size });
        set_values(input1_mem, weights_data_bfyx);

        topology.add(input_layout("input0", input0_mem->get_layout()));
        topology.add(data("weights", input1_mem));
        topology.add(fully_connected("matmul_prim", "input0", "weights", "", output_dt));
        topology.add(reorder("reorder_bfyx", "matmul_prim", format::bfyx, data_types::f32));
    } else {
        throw std::runtime_error("Unsupported primitive type for benchmarking");
    }

    build_options options;

    if (use_onednn) {
        implementation_desc matmul_impl = { format::bfyx, "", impl_types::onednn };
        options.set_option(build_option::force_implementations({ {"matmul_prim", matmul_impl} }));
    } else {
        implementation_desc matmul_impl = { format::bfyx, p.kernel_name };
        options.set_option(build_option::force_implementations({ {"matmul_prim", matmul_impl} }));
    }

    options.set_option(build_option::optimize_data(true));

    network network(*engine, topology, options);
    if (prim_type == matmul_type::gemm) {
        network.set_input_data("input0", input0_mem);
        network.set_input_data("input1", input1_mem);
        if (p.beta != 0) {
            network.set_input_data("input2", input2_mem);
        }
    } else if (prim_type == matmul_type::fc) {
        network.set_input_data("input0", input0_mem);
    }

    auto outputs = network.execute();
    outputs.at("reorder_bfyx").get_memory();

    size_t iteration_num = 1000;
    size_t total_time_us = 0;
    std::string kernel_name;

    if (enable_profiling) {
        for (size_t i = 0; i < iteration_num; i++) {
            auto outputs = network.execute();
            outputs.at("reorder_bfyx").get_memory();

            auto calc_time = [](const primitive_id& id, const event::ptr ev) {
                cldnn::instrumentation::profiling_info cldnnInfo{id, ev->get_profiling_info()};
                long long time = 0;
                for (auto &interval : cldnnInfo.intervals) {
                    using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
                    if (interval.stage == instrumentation::profiling_stage::executing)
                        time += std::chrono::duration_cast<duration_t>(interval.value->value()).count();
                }
                return time;
            };

            auto executed_primitives = network.get_executed_primitives();
            auto gemm = executed_primitives.find("matmul_prim") != executed_primitives.end() ? executed_primitives.find("matmul_prim") :
                                                                                               executed_primitives.find("reorder_bfyx");
            if (gemm == executed_primitives.end())
                throw std::runtime_error("Couldn't find profiling info for requested primitive\n");

            kernel_name = network.get_implementation_info(gemm->first);

            total_time_us += calc_time(gemm->first, gemm->second);
        }
        std::cout << prim_type << " kernel (" << kernel_name << ") time is " << total_time_us / iteration_num << "us" << std::endl;
    }
}
