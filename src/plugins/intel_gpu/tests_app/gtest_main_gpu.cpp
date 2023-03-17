// Copyright 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <string>

#include "../../intel_gpu/include/intel_gpu/runtime/device_query.hpp"
#include "test_utils/test_utils.h"

#include "test_utils/test_utils.h"

#include "network_test.h"
#include <intel_gpu/runtime/utils.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/fully_connected.hpp"
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "fully_connected_inst.h"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <cctype>
#include <string>

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opencl_helper_instance.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/runtime/device_query.hpp>
#include <memory>

// static size_t img_size = 800;
static std::string kernel_code =
    "__attribute__((intel_reqd_sub_group_size(16)))"
    "__attribute__((reqd_work_group_size(16, 1, 1)))"
    "void kernel simple_reorder(const __global uchar* src, __global float* dst) {"
    "    uint gid = get_global_id(0);"
    "    dst[gid] = convert_float(src[gid]) * 0.33f;"
    "}";

static std::string test =
    "\n#ifndef INTEL_INTERNAL_DEBUG_H_INCLUDED "
    "\n#define INTEL_INTERNAL_DEBUG_H_INCLUDED "
    "\nulong __attribute__((overloadable)) intel_get_cycle_counter( void ); "
    "\n#endif "
    ""
    "\n__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))"
    "\nvoid kernel touch_data(__global uint* dst, const __global int* test_data) {"
    "\n    int sum = 0;"
    "\n    int val = 0;"
    "\n    int next_id = 0;"
    "\n    const uint gid = get_global_id(0);"
    "\n    __attribute__((opencl_unroll_hint(1)))"
    "\n    for (uint j = 0; j < LOAD_ITERATIONS; j++) {"
    "\n        // val = intel_sub_group_block_read(test_data + (next_id));"
    "\n        val = test_data[next_id + gid];"
    "\n        sum += val;"
    "\n        next_id = val;"
    "\n    }"
    "\n    dst[gid] = sum;"
    "\n}"
    ""
    "\n__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))"
    "\nvoid kernel latency_test(__global uint* dst, const __global int* test_data, int iteration_num) {"
    "\n    int sum = 0;"
    "\n    int val = 0;"
    "\n    int next_id = 0;"
    "\n    ulong timer_start = 0;"
    "\n    ulong timer_end = 0, total_time = 0, time = 0;"
    "\n    ulong max_time = 0, min_time = -1;"
    "\n    const uint gid = get_global_id(0);"
    "\n    __attribute__((opencl_unroll_hint(1)))"
    "\n#ifdef USE_PRE_HEAT"
    "\n    for (uint iter = 0; iter < iteration_num + 1; iter++) {"
    "\n#else"
    "\n    for (uint iter = 0; iter < iteration_num; iter++) {"
    "\n#endif"
    "\n        next_id = 0;"
    "\n        __attribute__((opencl_unroll_hint(1)))"
    "\n        for (uint j = 0; j < LOAD_ITERATIONS; j++) {"
    "\n            timer_start = intel_get_cycle_counter();"
    "\n            val = test_data[next_id + gid];"
    "\n            // val = intel_sub_group_block_read(test_data + (next_id));"
    "\n            sum += val;"
    "\n            timer_end = intel_get_cycle_counter();"
    "\n            time = timer_end - timer_start;"
    "\n            max_time = max(max_time, time);"
    "\n            min_time = min(min_time, time);"
    "\n            total_time += time;"
    "\n            next_id = val;"
    "\n#ifdef USE_PRE_HEAT"
    "\n            if (iter == 0) {"
    "\n                total_time = 0;"
    "\n                max_time = 0;"
    "\n                min_time = -1;"
    "\n            }"
    "\n#endif"
    "\n        }"
    "\n    }"
    "\n    dst[gid] = sum;"
    "\n    dst[0] = total_time;"
    "\n    dst[1] = max_time;"
    "\n    dst[2] = min_time;"
    "\n}";

using time_interval = std::chrono::microseconds;
static std::string time_suffix = "us";

void mem_perf_test_to_host_and_back_to_device(size_t buffer_size_b = 64, int mode = 2, int iteration_num = 10);

void mem_perf_test_to_host_and_back_to_device(size_t buffer_size_b, int mode, int iteration_num) {
    std::cout << "Run with size " << buffer_size_b / 1024.0 << "KB mode=" << mode << " iters=" << iteration_num << "\n";
    auto ocl_instance = std::make_shared<OpenCL>();
    auto& ctx = ocl_instance->_context;
    auto& device = ocl_instance->_device;

    if (!ocl_instance->_supports_usm)
        exit(0);

    const float gpu_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() / 1000.0;

    const bool use_two_kernels = mode == 1;
    const bool use_pre_heat_in_kernel = mode == 2;

    const size_t gws = 16;
    const size_t lws = 16;
    const size_t simd = 16;
    const size_t test_data_size = buffer_size_b;
    const size_t profiling_data_size = sizeof(uint32_t) * simd;
    const size_t load_instuction_count = test_data_size / simd / sizeof(uint32_t);
    const std::string touch_data_kernel = "touch_data";
    const std::string kernel_name = "latency_test";
    std::string build_options = "-DLOAD_ITERATIONS=" + std::to_string(load_instuction_count) + " "
                              + "-DSUB_GROUP_SIZE=" + std::to_string(simd) + " ";
    if (use_pre_heat_in_kernel)
        build_options += "-DUSE_PRE_HEAT=1 ";

    std::string used_kernels = kernel_name;
    if (use_two_kernels)
        used_kernels = touch_data_kernel + " " + used_kernels;

    std::cout << "gpu_frequency=" << gpu_frequency << "GHz" << std::endl;
    std::cout << "gws=" << gws << " lws=" << lws << " simd=" << simd << " kernels=[" << used_kernels << "] use_pre_heat_in_kernel=" << use_pre_heat_in_kernel << std::endl;
    std::cout << "test_data_size=" << test_data_size / 1024.0 << "KB profiling_data_size=" << profiling_data_size << "B "
              << "total_load_instructions_number=" << load_instuction_count << std::endl;

    cl::Program program(ctx, test);
    program.build({device}, build_options.c_str());
    std::stringstream dump_file;

    cl::Kernel latency_kernel_cl(program, kernel_name.c_str());
    cl::KernelIntel latency_kernel(latency_kernel_cl, *ocl_instance->_usm_helper);

    std::vector<int> test_data_buffer;
    for (size_t i = 0; i < test_data_size / sizeof(int); i++) {
        test_data_buffer.push_back(static_cast<int>(((i / simd) + 1) * simd));
    }

    cl::UsmMemory test_data_buffer_device(*ocl_instance->_usm_helper);
    test_data_buffer_device.allocateDevice(test_data_size);

    cl::UsmMemory output_buffer_device(*ocl_instance->_usm_helper);
    output_buffer_device.allocateDevice(profiling_data_size);

    cl::UsmMemory output_buffer_host(*ocl_instance->_usm_helper);
    output_buffer_host.allocateHost(profiling_data_size);

    cl::CommandQueue queue(ctx, device);

    ocl_instance->_usm_helper->enqueue_memcpy(queue, test_data_buffer_device.get(), test_data_buffer.data(), test_data_size, true);

    latency_kernel.setArgUsm(0, output_buffer_device);
    latency_kernel.setArgUsm(1, test_data_buffer_device);
    latency_kernel.setArg(2, iteration_num);

    cl::Event ev1;
    std::vector<cl::Event> wait_list1;

    // Flush all caches
    queue.finish();

    if (use_two_kernels) {
        cl::Kernel cl_kernel1(program, touch_data_kernel.c_str());
        cl::KernelIntel kernel1(cl_kernel1, *ocl_instance->_usm_helper);

        kernel1.setArgUsm(0, output_buffer_device);
        kernel1.setArgUsm(1, test_data_buffer_device);

        queue.enqueueNDRangeKernel(kernel1, cl::NDRange(), cl::NDRange(gws), cl::NDRange(lws), nullptr, &ev1);
        wait_list1.push_back(ev1);
    }

    cl::Event ev2;
    queue.enqueueNDRangeKernel(latency_kernel, cl::NDRange(), cl::NDRange(gws), cl::NDRange(lws), &wait_list1, &ev2);
    cl::WaitForEvents({ev2});

    ocl_instance->_usm_helper->enqueue_memcpy(queue, output_buffer_host.get(), output_buffer_device.get(), test_data_size, true);

    uint32_t* profiling_res = static_cast<uint32_t*>(output_buffer_host.get());
    auto const clcs = profiling_res[0];
    auto const max = profiling_res[1];
    auto const max_ns = profiling_res[1] / gpu_frequency;
    auto const min = profiling_res[2];
    auto const min_ns = profiling_res[2] / gpu_frequency;
    std::cout << "max=" << max << "(" << max_ns << "ns)" << " min=" << min << "(" << min_ns << "ns)" << std::endl;
    std::cout << "clcs=" << clcs << " clcs_per_load=" << clcs / load_instuction_count / iteration_num << " latency_per_load=" << clcs / load_instuction_count / gpu_frequency / iteration_num << "ns" << std::endl;
    exit(0);
}

using namespace tests;

void run_dynamic(size_t seq_len);
void run_static(size_t seq_len);

static double get_exectime_fc(const std::shared_ptr<event>& event)
{
    using namespace std::chrono;
    double avg_time = 0.0;
    auto intervals = event->get_profiling_info();
    for (const auto& q : intervals)
    {
        if (q.stage != instrumentation::profiling_stage::executing) {
            continue;
        }
        avg_time = duration_cast<duration<double, microseconds::period>>(q.value->value()).count();
        break;
    }
    return avg_time;
}


void run_dynamic(size_t seq_len) {


    auto& engine = get_test_engine();

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::queue_type(cldnn::QueueTypes::out_of_order));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::enable_profiling(true));

    std::vector<std::string> times;

    {
        double exectime_total = 0;

        std::vector<size_t> batch_sizes = { 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            22, 23, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 47, 51, 53, 59, 64, 71};

        batch_sizes = {8, 32, 128};
        batch_sizes = {};

        for (size_t i = 512; i > 128; i-=16)
            batch_sizes.push_back(i);

        for (size_t i = 128; i >= 8; i-=4)
            batch_sizes.push_back(i);

        batch_sizes = {seq_len};

        // batch_sizes = { 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        //     22, 23, 26, 27, 28, 29, 31, 32};

        std::vector<std::pair<size_t, size_t>> weights_sizes = {{768, 768}, /* {3072, 768}, */ {768, 3072}, {9, 768}};
        weights_sizes = {{768, 768}};

        data_types data_type = cldnn::data_types::f16;

        for (const auto& weights : weights_sizes) {
            for (const auto& batch_size : batch_sizes) {
                float* data_vec;
                cldnn::memory_ptr input_data_mem;
                std::vector<float> input_data_vec;
                cldnn::memory_ptr bias_mem;

                const int32_t input_f = weights.second, weight_b = weights.first;

                auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(), ov::Dimension(), input_f }, data_types::f32,format::bfyx };

                auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx});
                auto weights_data_vec = generate_random_1d<float>(weight_b * input_f, 0, 1);

                set_values(weights_data, weights_data_vec);

                bias_mem = engine.allocate_memory({ ov::PartialShape{ weight_b }, data_types::f32,format::bfyx});
                auto bias_data_vec = generate_random_1d<float>(weight_b, 0, 1);

                set_values(bias_mem, bias_data_vec);

                cldnn::topology topology{
                    input_layout("input", input_dyn_layout),
                    reorder("input_reorder", input_info("input"), format::bfyx, data_type),
                    data("weights", weights_data),
                    reorder("weights_reorder", input_info("weights"), format::bfyx, data_type),
                    data("bias", bias_mem),
                    reorder("bias_reorder", input_info("bias"), format::bfyx, data_type),
                    fully_connected("fc", input_info("input_reorder"), "weights_reorder", "bias_reorder", padding(), 3)
                };

                network network(engine, topology, config);

                auto input_actual_layout = layout{ ov::PartialShape{ 1, (ov::Dimension::value_type)batch_size, input_f }, data_types::f32,format::bfyx};
                input_data_mem = engine.allocate_memory(input_actual_layout);
                input_data_vec = generate_random_1d<float>(batch_size * input_f, 0, 1);
                set_values(input_data_mem, input_data_vec);
                data_vec = input_data_vec.data();
                network.set_input_data("input", input_data_mem);

                double exectime = 0;
                const size_t iters_num = 100;
                for (size_t iter = 0; iter < iters_num; iter++) {
                    auto outputs = network.execute();

                    auto output_prim_mem = outputs.begin()->second.get_memory();

                    auto out_l = network.get_output_layout(outputs.begin()->first);

                    cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

                    for (int b = 0; b < (int)batch_size; b++) {
                        for (int ofm = 0; ofm < weight_b; ofm++) {
                            auto acc = 0.f;
                            for (int ifm = 0; ifm < input_f; ifm++) {
                                acc += weights_data_vec[ofm * input_f + ifm] * data_vec[b * input_f + ifm];
                            }
                            if (acc != output_ptr[b * weight_b + ofm] && data_type != data_types::f16) {
                                std::cerr << "Error for b=" << b << " f=" << ofm << std::endl;
                                OPENVINO_ASSERT(false);
                            }
                        }
                    }

                    exectime += get_exectime_fc(network.get_primitive_event("fc"));
                    OPENVINO_ASSERT(network.get_primitive("fc")->get_impl()->is_dynamic());
                }

                exectime /= iters_num;

                std::stringstream time_str;
                time_str << "Time for " << batch_size << "x" << input_f << " -> " << batch_size << "x" << weight_b << " " << exectime << "us" << " impl=" << network.get_primitive("fc")->get_implementation_name();

                times.push_back(time_str.str());

                std::cout << time_str.str() << std::endl;
                exectime_total += exectime;
            }
        }

        std::cout << std::endl;

        for (auto& t : times)
            std::cout << t << std::endl;

        std::cout << "Total avg time: " << (exectime_total / (batch_sizes.size() * weights_sizes.size())) << "us" << std::endl;
        std::cout << "Total time for: " << exectime_total << "us" << std::endl;
    }
}

void run_static(size_t seq_len) {
auto& engine = get_test_engine();

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::queue_type(cldnn::QueueTypes::in_order));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::enable_profiling(true));

    std::vector<std::string> times;

    double exectime_total = 0;

    std::vector<size_t> batch_sizes = { 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        22, 23, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 47, 51, 53, 59, 64, 71};

    batch_sizes = {8, 32, 128};
    batch_sizes = {};

    for (size_t i = 512; i > 128; i-=16)
        batch_sizes.push_back(i);

    for (size_t i = 128; i >= 8; i-=4)
        batch_sizes.push_back(i);

    batch_sizes = {seq_len};
    std::vector<std::pair<size_t, size_t>> weights_sizes = {{768, 768}, {3072, 768}, {768, 3072}, {9, 768}};
    weights_sizes = {{768, 768}};

    for (const auto& weights : weights_sizes) {
        for (const auto& batch_size : batch_sizes) {
            float* data_vec;
            cldnn::memory_ptr input_data_mem;
            std::vector<float> input_data_vec;

            const int32_t input_f = weights.second, weight_b = weights.first;

            auto input_static_layout = layout{ ov::PartialShape{ 1, (ov::Dimension::value_type)batch_size, input_f }, data_types::f32,format::bfyx };

            auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx});
            auto weights_data_vec = generate_random_1d<float>(weight_b * input_f, 0, 1);

            set_values(weights_data, weights_data_vec);

            cldnn::topology topology{
                input_layout("input", input_static_layout),
                data("weights", weights_data),
                fully_connected("fc", input_info("input"), "weights", "", padding(), 3)
            };

            // options.set_option(cldnn::build_option::allow_new_shape_infer(true));
            network network(engine, topology, config);

            input_data_mem = engine.allocate_memory(input_static_layout);
            input_data_vec = generate_random_1d<float>(batch_size * input_f, 0, 1);
            set_values(input_data_mem, input_data_vec);
            data_vec = input_data_vec.data();
            network.set_input_data("input", input_data_mem);

            double exectime = 0;
            const size_t iters_num = 1000;
            for (size_t iter = 0; iter < iters_num; iter++) {
                auto outputs = network.execute();
                OPENVINO_ASSERT(outputs.size() == size_t(1));
                OPENVINO_ASSERT(outputs.begin()->first == "fc");

                auto output_prim_mem = outputs.begin()->second.get_memory();

                auto out_l = network.get_output_layout(outputs.begin()->first);

                cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

                for (int b = 0; b < (int)batch_size; b++) {
                    for (int ofm = 0; ofm < weight_b; ofm++) {
                        auto acc = 0.f;
                        for (int ifm = 0; ifm < input_f; ifm++) {
                            acc += weights_data_vec[ofm * input_f + ifm] * data_vec[b * input_f + ifm];
                        }
                        if (acc != output_ptr[b * weight_b + ofm])
                            std::cerr << "Error for b=" << b << " f=" << ofm << std::endl;
                        OPENVINO_ASSERT(acc == output_ptr[b * weight_b + ofm]);
                    }
                }

                exectime += get_exectime_fc(network.get_primitive_event("fc"));
                OPENVINO_ASSERT(!network.get_primitive("fc")->get_impl()->is_dynamic());
            }

            exectime /= iters_num;

            std::stringstream time_str;
            time_str << "Time for " << batch_size << "x" << input_f << " -> " << batch_size << "x" << weight_b << " " << exectime << "us" << " impl=" << network.get_primitive("fc")->get_implementation_name();

            times.push_back(time_str.str());

            std::cout << time_str.str() << std::endl;
            exectime_total += exectime;
        }
    }

    std::cout << std::endl;

    for (auto& t : times)
        std::cout << t << std::endl;

    std::cout << "Total avg time: " << (exectime_total / (batch_sizes.size() * weights_sizes.size())) << "us" << std::endl;
    std::cout << "Total time for: " << exectime_total << "us" << std::endl;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        mem_perf_test_to_host_and_back_to_device();
    } else if (argc == 2) {
        size_t buffer_size_b = atoi(argv[1]);
        mem_perf_test_to_host_and_back_to_device(buffer_size_b);
    } else if (argc >= 3) {

        std::string unit = argv[2];
        std::transform(unit.begin(), unit.end(), unit.begin(),
            [](unsigned char c){ return std::toupper(c); });

        size_t buffer_size_b = atoi(argv[1]);
        if (unit == "B")
            buffer_size_b *= 1;
        else if (unit == "KB")
            buffer_size_b *= 1024;
        else if (unit == "MB")
            buffer_size_b *= 1024 * 1024;
        else
            std::cout << "Unsupported unit\n";

        size_t mode = 2;
        if (argc >= 4) {
            mode = atoi(argv[3]);
        }
        size_t iters = 10;
        if (argc >= 5) {
            iters = atoi(argv[4]);
        }
        mem_perf_test_to_host_and_back_to_device(buffer_size_b, mode, iters);
    }
    exit(0);


    bool dynamic_model = true;
    size_t seq_len = 8;

    // start modification

    std::cout << "Run for dynamic=" << dynamic_model << " seq_len=" << seq_len << std::endl;

    if (dynamic_model) {
        printf("Running dynamic() from %s", __FILE__);
        printf("Running dynamic() from %s", __FILE__);
        run_dynamic(seq_len);
        exit(0);
    } else {
        printf("Running static() from %s", __FILE__);
        run_static(seq_len);
        exit(0);
    }

    return 0;
}
