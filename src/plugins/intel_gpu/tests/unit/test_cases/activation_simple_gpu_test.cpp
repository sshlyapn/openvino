// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include "activation_inst.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

TEST(activation_f32_fw_gpu, dynamic) {
    auto& engine = get_test_engine();

    ov::PartialShape in_shape  = { 1, 1, 4, 2 };
    layout in_layout { ov::PartialShape::dynamic(in_shape.size()), data_types::f32, format::bfyx };

    auto input = engine.allocate_memory({ in_shape, data_types::f32, format::bfyx });
    set_values(input, { -0.12f, 0.56f, 0.45f, -0.789f, 42.f, 0.999f, 0.7899f, 0.f});

    std::vector<activation_func> funcs = {
        activation_func::gelu,
        activation_func::relu,
        activation_func::hyperbolic_tan,
        activation_func::sqrt
    };

    for (auto func : funcs) {
        topology topology(input_layout("input", in_layout));
        topology.add(activation("activation", input_info("input"), func));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topology, config);

        network.set_input_data("input", input);

        auto inst = network.get_primitive("activation");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
        cldnn::mem_lock<float> input_ptr(input, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();

        ASSERT_EQ(output_layout.format, format::bfyx);
        ASSERT_EQ(y_size, 4);
        ASSERT_EQ(x_size, 2);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
            switch (func) {
            case activation_func::gelu:
                ASSERT_NEAR(0.5f * static_cast<float>(input_ptr[i]) * (1.f + std::erf(static_cast<float>((input_ptr[i])) / std::sqrt(2.0f))),
                            output_ptr[i], 1e-5f);
                break;
            case activation_func::relu:
                ASSERT_EQ(std::max(input_ptr[i], static_cast<float>(0)), output_ptr[i]);
                break;
            case activation_func::hyperbolic_tan:
                    ASSERT_FLOAT_EQ(std::tanh(static_cast<float>(input_ptr[i])), output_ptr[i]);
                break;
            case activation_func::sqrt:
                    if (input_ptr[i] >= 0) {
                        ASSERT_FLOAT_EQ(std::sqrt(static_cast<float>(input_ptr[i])), output_ptr[i]);
                    }
                    break;
            default:
                break;
            }
        }
    }
}

TEST(activation_f32_fw_gpu, not_basic_yxfb) {
    //  Input:
    //  1  0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0
    //
    //  Output:
    //  0, 1, 0, 0, 0,
    //  1, 0, 0, 0, 0,
    //  0, 0, 0, 1, 0,
    //  0, 0, 0, 0, 1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
      0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
      3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
      1.0f, 1.0f, 1.0f, -1.0f, 0.0f });
    VF<float> output_vec = {
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f };

    topology topology(
        input_layout("input", input->get_layout()),
        activation("not", input_info("input"), activation_func::negation));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, erf_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
                 0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            activation("not", input_info("input"), activation_func::erf));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        ASSERT_FLOAT_EQ(std::erf(input_ptr[i]), output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, hard_sigmoid_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    activation_additional_params params = { 1.0f, 0.5f };
    set_values(input,
               { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
                 0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            activation("not", input_info("input"), activation_func::hard_sigmoid, params));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = std::fmax(0.0f, std::fmin(1.0f, params.a * input_ptr[i] + params.b));
        ASSERT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, reciprocal_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input->get_layout()),
            activation("not", input_info("input"), activation_func::reciprocal));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = 1 / input_ptr[i];
        ASSERT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, selu_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    activation_additional_params params = { 1.0f, 0.5f };
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input->get_layout()),
            activation("not", input_info("input"), activation_func::selu, params));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = input_ptr[i] <= 0 ? params.b * (params.a * std::exp(input_ptr[i]) - params.a) :
                                        params.b * input_ptr[i];
        ASSERT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, softplus_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input->get_layout()),
            activation("not", input_info("input"), activation_func::softplus));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = std::log(std::exp(input_ptr[i]) + 1);
        ASSERT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, softsign_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input->get_layout()),
            activation("not", input_info("input"), activation_func::softsign));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = input_ptr[i] / (1 + std::abs(input_ptr[i]));
        ASSERT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f16_fw_gpu, softsign_basic_yxfb) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::yxfb, {1, 1, 2, 2}});
    set_values(input, {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.5f)});
    VF<FLOAT16> output_vec = {FLOAT16(0.5f), FLOAT16(0.66650391f), FLOAT16(0.75f), FLOAT16(0.81835938f)};

    topology topology(input_layout("input", input->get_layout()),
                      activation("not", input_info("input"), activation_func::softsign));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<FLOAT16> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<FLOAT16> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, sign_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input->get_layout()),
            activation("not", input_info("input"), activation_func::sign));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = input_ptr[i] > 0 ? 1.0f : input_ptr[i] < 0 ? -1.0f : 0.0f;
        ASSERT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, pow_basic_yxfb) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    set_values(input,
    { 1.0f, 2.0f, 3.0f, 4.0f });
    VF<float> output_vec = { 1.0f, 4.0f, 9.0f, 16.0f };

    topology topology(
        input_layout("input", input->get_layout()),
        activation("pow", input_info("input"), activation_func::pow, { 2.0f, 0.0f }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "pow");

    auto output_memory = outputs.at("pow").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f16_fw_gpu, pow_basic_yxfb) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 1, 1, 2, 2 } });
    set_values(input,
        { FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.5f) });
    VF<FLOAT16> output_vec = { FLOAT16(1.0f), FLOAT16(8.0f), FLOAT16(27.0f), FLOAT16(91.125f) };

    topology topology(
        input_layout("input", input->get_layout()),
        activation("pow", input_info("input"), activation_func::pow, { FLOAT16(3.0f), FLOAT16(0.0f) }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "pow");

    auto output_memory = outputs.at("pow").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<FLOAT16> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_yxfb) {
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
      2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
      3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
      1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f };

    topology topology(
        input_layout("input", input->get_layout()),
        activation("relu", input_info("input"), activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0 }, 0 }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_bfzyx) {
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  2 -1 -4  5  2
    //  2  1 -3  2 -2
    //  4 -3  2  4 -1
    //  1  2  1 -2  2
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1
    //
    //  2   -0.5 -2    5    2
    //  2    1   -1.5  2   -1
    //  4   -1.5  2    4   -0.5
    //  1    2    1   -1    2
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 1, 5, 4, 2 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
        2.0f, -1.0f, -4.0f, 5.0f, 2.0f,
        2.0f, 1.0f, -3.0f, 2.0f, -2.0f,
        4.0f, -3.0f, 2.0f, 4.0f, -1.0f,
        1.0f, 2.0f, 1.0f, -2.0f, 2.0f });
    VF<float> output_vec = {
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f,
        2.0f, -0.5f, -2.0f, 5.0f, 2.0f,
        2.0f, 1.0f, -1.5f, 2.0f, -1.0f,
        4.0f, -1.5f, 2.0f, 4.0f, -0.5f,
        1.0f, 2.0f, 1.0f, -1.0f, 2.0f };

    topology topology(
        input_layout("input", input->get_layout()),
        activation("relu", input_info("input"), activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0, 0 }, 0 }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfzyx);
    ASSERT_EQ(z_size, 2);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, basic_yxfb_all_functions)
{
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  a: 0.5, b: 2.5
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 5, 4 } });
    auto input_params = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    set_values(input,
    { 0.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });

    std::vector<activation_func> funcs = {
        activation_func::none,
        activation_func::logistic,
        activation_func::hyperbolic_tan,
        activation_func::relu,
        activation_func::relu_negative_slope,
        activation_func::clamp,
        activation_func::softrelu,
        activation_func::abs,
        activation_func::linear,
        activation_func::square,
        activation_func::sqrt,
        activation_func::elu,
        activation_func::sin,
        activation_func::sinh,
        activation_func::cos,
        activation_func::cosh,
        activation_func::exp,
        activation_func::negation,
        activation_func::log2,
        activation_func::tan,
        activation_func::negative,
        activation_func::abs,
        activation_func::swish,
        activation_func::hswish,
        activation_func::mish,
        activation_func::gelu,
        activation_func::hsigmoid
    };

    activation_additional_params params = { 0.5f, 2.5f };
    set_values(input_params, { params.a, params.b });

    for (uint8_t i = 0 ; i < 2 ; i++)
    {
        for (auto func : funcs)
        {
            topology topology(input_layout("input", input->get_layout()));

            if (i == 0)
            {
                topology.add(activation("activation", input_info("input"), func, params));
            }
            else
            {
                topology.add(data("input_params", input_params));
                topology.add(activation("activation", input_info("input"), "input_params", func));
            }

            network network(engine, topology, get_test_default_config(engine));
            network.set_input_data("input", input);
            auto outputs = network.execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "activation");

            auto output_memory = outputs.at("activation").get_memory();
            auto output_layout = output_memory->get_layout();
            cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
            cldnn::mem_lock<float> input_ptr(input, get_test_stream());

            int y_size = output_layout.spatial(1);
            int x_size = output_layout.spatial(0);
            int f_size = output_layout.feature();
            int b_size = output_layout.batch();
            ASSERT_EQ(output_layout.format, format::yxfb);
            ASSERT_EQ(y_size, 4);
            ASSERT_EQ(x_size, 5);
            ASSERT_EQ(f_size, 1);
            ASSERT_EQ(b_size, 1);

            for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
            {
                switch (func)
                {
                case activation_func::none:
                    ASSERT_FLOAT_EQ(input_ptr[i], output_ptr[i]);
                    break;
                case activation_func::logistic:
                    ASSERT_FLOAT_EQ(1.f / (1.f + std::exp((float)-input_ptr[i])), output_ptr[i]);
                    break;
                case activation_func::hyperbolic_tan:
                    ASSERT_FLOAT_EQ(std::tanh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::relu:
                    ASSERT_FLOAT_EQ(std::fmax((float)input_ptr[i], 0.f), output_ptr[i]);
                    break;
                case activation_func::clamp:
                    ASSERT_FLOAT_EQ(std::fmin((float)std::fmax((float)input_ptr[i], params.a), params.b), output_ptr[i]);
                    break;
                case activation_func::softrelu:
                    ASSERT_FLOAT_EQ(std::log(1.f + std::exp((float)input_ptr[i])), output_ptr[i]);
                    break;
                case activation_func::abs:
                    ASSERT_FLOAT_EQ(std::fabs(input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::linear:
                    ASSERT_FLOAT_EQ((params.a*input_ptr[i] + params.b), output_ptr[i]);
                    break;
                case activation_func::square:
                    ASSERT_FLOAT_EQ((input_ptr[i] * input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::sqrt:
                    if (input_ptr[i] >= 0)
                    {
                        ASSERT_FLOAT_EQ(std::sqrt((float)input_ptr[i]), output_ptr[i]);
                    }
                    break;
                case activation_func::elu:
                    ASSERT_FLOAT_EQ(std::fmax((float)input_ptr[i], 0.0f) +
                                    params.a*(std::exp(std::fmin((float)input_ptr[i], 0.0f)) - 1), output_ptr[i]);
                    break;
                case activation_func::sin:
                    ASSERT_FLOAT_EQ(std::sin((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::sinh:
                    ASSERT_FLOAT_EQ(std::sinh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::cos:
                    ASSERT_FLOAT_EQ(std::cos((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::cosh:
                    ASSERT_FLOAT_EQ(std::cosh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::exp:
                    ASSERT_FLOAT_EQ(std::exp((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::negation:
                    ASSERT_FLOAT_EQ((float)(!input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::log2:
                    if (input_ptr[i] > 0) //logarithm exist only for positive real values
                    {
                        ASSERT_FLOAT_EQ(std::log2((float)input_ptr[i]), output_ptr[i]);
                    }
                    break;
                case activation_func::tan:
                    ASSERT_FLOAT_EQ(std::tan((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::negative:
                    ASSERT_FLOAT_EQ(-((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::swish:
                    ASSERT_FLOAT_EQ((float)input_ptr[i] / (1.f + std::exp((float)(-params.a * input_ptr[i]))), output_ptr[i]);
                    break;
                case activation_func::hswish:
                    ASSERT_FLOAT_EQ((float)input_ptr[i] * std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f, output_ptr[i]);
                    break;
                case activation_func::mish:
                    ASSERT_NEAR((float)input_ptr[i] * std::tanh(std::log(1.f + std::exp((float)input_ptr[i]))),
                                output_ptr[i], 1e-5f);
                    break;
                case activation_func::gelu:
                    ASSERT_NEAR(0.5f * (float)input_ptr[i] * (1.f + std::erf((float)(input_ptr[i]) / std::sqrt(2.0f))),
                                output_ptr[i], 1e-5f);
                    break;
                case activation_func::hsigmoid:
                    ASSERT_FLOAT_EQ(std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f, output_ptr[i]);
                    break;
                default:
                    break;
                }
            }
        }
    }
}

TEST(activation_f16_fw_gpu, basic_bfyx_all_functions)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 2, 4 } });
    auto input_params = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { FLOAT16(-4.5f), FLOAT16(-2.5f), FLOAT16(-1.5f), FLOAT16(0.5f),
                        FLOAT16(0.9f),  FLOAT16(1.5f),  FLOAT16(2.0f),  FLOAT16(2.5f) });

    std::vector<activation_func> funcs = {
        activation_func::linear,
        activation_func::mish,
        activation_func::hswish,
        activation_func::hsigmoid,
        activation_func::round_half_to_even,
        activation_func::round_half_away_from_zero
    };

    activation_additional_params params = { 3.f, 2.f };
    set_values(input_params, { FLOAT16(params.a), FLOAT16(params.b) });

    for (uint8_t i = 0 ; i < 2 ; i++) {
        for (auto func : funcs) {
            topology topology(input_layout("input", input->get_layout()));

            if (i == 0) {
                topology.add(activation("activation", input_info("input"), func, params));
            } else {
                topology.add(data("input_params", input_params));
                topology.add(activation("activation", input_info("input"), "input_params", func));
            }

            network network(engine, topology, get_test_default_config(engine));
            network.set_input_data("input", input);
            auto outputs = network.execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "activation");

            auto output_memory = outputs.at("activation").get_memory();
            auto output_layout = output_memory->get_layout();
            cldnn::mem_lock<FLOAT16> output_ptr(output_memory, get_test_stream());
            cldnn::mem_lock<FLOAT16> input_ptr(input, get_test_stream());

            int y_size = output_layout.spatial(1);
            int x_size = output_layout.spatial(0);
            int f_size = output_layout.feature();
            int b_size = output_layout.batch();
            ASSERT_EQ(output_layout.format, format::bfyx);
            ASSERT_EQ(y_size, 4);
            ASSERT_EQ(x_size, 2);
            ASSERT_EQ(f_size, 1);
            ASSERT_EQ(b_size, 1);

            for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
                switch (func) {
                case activation_func::linear: {
                    VF<FLOAT16> output_vec = {FLOAT16(-11.5f), FLOAT16(-5.5f), FLOAT16(-2.5f), FLOAT16(3.5f),
                                              FLOAT16(4.7f), FLOAT16(6.5f), FLOAT16(8.0f), FLOAT16(9.5f)};
                    ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
                    break;
                }
                case activation_func::mish:
                    ASSERT_NEAR((FLOAT16)((float)input_ptr[i] * std::tanh(std::log(1.f + std::exp((float)input_ptr[i])))),
                        output_ptr[i], 1e-2f);
                    break;
                case activation_func::hswish:
                    ASSERT_NEAR((FLOAT16)((float)input_ptr[i] * std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f),
                        output_ptr[i], 1e-3f);
                    break;
                case activation_func::hard_sigmoid:
                    ASSERT_NEAR((FLOAT16)(std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f),
                        output_ptr[i], 1e-3f);
                    break;
                case activation_func::round_half_to_even:
                    ASSERT_FLOAT_EQ((FLOAT16)std::rint((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::round_half_away_from_zero:
                    ASSERT_FLOAT_EQ((FLOAT16)std::round((float)input_ptr[i]), output_ptr[i]);
                    break;
                default:
                    break;
                }
            }
        }
    }
}

TEST(activation_f32_fw_gpu, basic_yxfb_asin_acos_log_atan)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 4 } });
    set_values(input, { 0.12f, 0.56f, 0.45f, 0.789f, 0.546f, 0.999f, 0.7899f, 0.6677f});

    std::vector<activation_func> funcs = {
        activation_func::asin,
        activation_func::acos,
        activation_func::log,
        activation_func::log2,
        activation_func::atan,
        activation_func::asin,
        activation_func::asinh,
        activation_func::atanh
    };

    for (auto func : funcs)
    {
        topology topology(input_layout("input", input->get_layout()));
        topology.add(activation("activation", input_info("input"), func));

        network network(engine, topology, get_test_default_config(engine));
        network.set_input_data("input", input);
        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
        cldnn::mem_lock<float> input_ptr(input, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::yxfb);
        ASSERT_EQ(y_size, 4);
        ASSERT_EQ(x_size, 2);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
        {
            switch (func)
            {
            case activation_func::asin:
                ASSERT_FLOAT_EQ(std::asin((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::acos:
                ASSERT_FLOAT_EQ(std::acos((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::log:
                ASSERT_FLOAT_EQ(std::log((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::log2:
                ASSERT_FLOAT_EQ(std::log2((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::atan:
                ASSERT_FLOAT_EQ(std::atan((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::asinh:
                ASSERT_FLOAT_EQ(std::asinh((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::acosh:
                ASSERT_FLOAT_EQ(std::acosh((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::atanh:
                ASSERT_FLOAT_EQ(std::atanh((float)input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_f32_fw_gpu, relu_basic_acosh_yxfb) {
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //  The expected size is the same as input - the output padding is set to 0, 0
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Slope: 0.5

    auto &engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {1, 1, 5, 4}});

    set_values(input,
               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                2.0f, 2.0f, 3.0f, 4.0f, 6.0f,
                3.0f, 3.0f, 3.0f, 5.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    topology topology(
            input_layout("input", input->get_layout()),
            reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })),
            activation("relu", input_info("reorder"), activation_func::acosh, {0.5f, 0.f}, padding{ { 0, 0, 0, 0 }, 0 }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int i = 0; i < x_size * y_size * f_size * b_size; ++i) {
        ASSERT_FLOAT_EQ(std::acosh(input_ptr[i]), output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_input_padding_yxfb) {
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //  The expected size is the same as in put - the output padding is set to 0, 0
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });

    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
         1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
         2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
         3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
         1.0f, 1.0f, 1.0f, -0.5f, 1.0f};

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })),
        activation("relu", input_info("reorder"), activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0 }, 0 }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_input_padding_bfzyx) {
    //  Input Padding: 0x2x1 (zyx format) out of the reorder layer
    //  The expected size is the same as input - the output padding is set to 0, 0
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 1, 5, 4, 2 } });
    set_values(input,
    {
        1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f,

        1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f,
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f };

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 2, 1, 0 }, 0 })),
        activation("relu", input_info("reorder"), activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0, 0 }, 0 }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();

    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfzyx);
    ASSERT_EQ(z_size, 2);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_output_padding_yxfb) {
    //  Output Padding: 3x3 (yx format)
    //
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  Slope: 0.5
    //
    //  Output:
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    1   -1   -1.5  4    5    0    0    0
    //  0    0    0    2    2    3    4   -3    0    0    0
    //  0    0    0    3   -1.5  3    5    1    0    0    0
    //  0    0    0    1    1    1   -0.5  1    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.5f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 3.0f, 4.0f, -3.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 3.0f, -1.5f, 3.0f, 5.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    topology topology(
        input_layout("input", input->get_layout()),
        activation("relu", input_info("input"), activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 3, 3 }, 0 }));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory->get_layout();
    auto output_size = output_layout.get_buffer_size();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_size.spatial[1];
    int x_size = output_size.spatial[0];
    int f_size = output_size.feature[0];
    int b_size = output_size.batch[0];
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 10);
    ASSERT_EQ(x_size, 11);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, basic_yxfb_floor_ceil)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 4 } });
    set_values(input, { 0.01f, 0.99f, -0.01f, -0.99f, 1.1f, 1.0f, 0.0f, -1.1f });

    std::vector<activation_func> funcs = {
        activation_func::floor,
        activation_func::ceil
    };

    for (auto func : funcs)
    {
        topology topology(input_layout("input", input->get_layout()));
        topology.add(activation("activation", input_info("input"), func));

        network network(engine, topology, get_test_default_config(engine));
        network.set_input_data("input", input);
        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
        cldnn::mem_lock<float> input_ptr(input, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::yxfb);
        ASSERT_EQ(y_size, 4);
        ASSERT_EQ(x_size, 2);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
        {
            switch (func)
            {
            case activation_func::floor:
                ASSERT_FLOAT_EQ(std::floor((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::ceil:
                ASSERT_FLOAT_EQ(std::ceil((float)input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_i8_fw_gpu, basic_yxfb_all_funcs)
{
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::i8, format::yxfb,{ 2, 2, 2, 2 } });

    std::vector<int8_t> input_vec = {
        1,   0,  5,   1,
        2,   0,  6,  -5,
        3,   0, -7,  12,
        4,   0, -8,   8
    };
    set_values(input, input_vec);

    // functions valid for int8 type input
    std::vector<activation_func> funcs = {
        activation_func::none,
        activation_func::negative,
        activation_func::negation
    };

    for (auto func : funcs)
    {
        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(activation("activation", input_info("input"), func));

        network network(engine, topology, get_test_default_config(engine));
        network.set_input_data("input", input);
        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());
        cldnn::mem_lock<int8_t> input_ptr(input, get_test_stream());

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
            switch (func) {
            case activation_func::none:
                ASSERT_EQ((int8_t)input_ptr[i], output_ptr[i]);
                break;
            case activation_func::negative:
                ASSERT_EQ(-((int8_t)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::negation:
                ASSERT_EQ(!((int8_t)input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_i8_fw_gpu, clamp_basic_bfzyx)
{
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::i8, format::bfzyx, { 1, 2, 2, 2, 2 } });

    std::vector<int8_t> input_vec = {
        1,   0,  5,   1,
        2,   0,  6,  -5,
        3,   0, -7,  12,
        4,   0, -8,   8
    };
    set_values(input, input_vec);

    topology topology;
    activation_additional_params params = {-136, 136};
    topology.add(input_layout("input", input->get_layout()));
    topology.add(activation("activation", input_info("input"), activation_func::clamp, params));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "activation");

    auto output_memory = outputs.at("activation").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<int8_t> input_ptr(input, get_test_stream());

    for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
        ASSERT_EQ(std::min(std::max(static_cast<int32_t>(input_ptr[i]), static_cast<int32_t>(params.a)), static_cast<int32_t>(params.b)), static_cast<int32_t>(output_ptr[i]));
    }
}

TEST(activation_i32_fw_gpu, basic_yxfb_i32_funcs) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::i32, format::yxfb,{ 2, 2, 2, 2 } });

    std::vector<int32_t> input_vec = {
        1,   0,  5,   1,
        2,   0,  6,  -5,
        3,   0, -7,  12,
        4,   0, -8,   8
    };
    set_values(input, input_vec);

    // functions valid for int8 type input
    std::vector<activation_func> funcs = {
        activation_func::none,
        activation_func::negative,
        activation_func::negation,
        activation_func::relu,
        activation_func::clamp,
        activation_func::floor
    };

    for (auto func : funcs) {
        topology topology;
        activation_additional_params params = {0.0, 1.0};
        topology.add(input_layout("input", input->get_layout()));
        topology.add(activation("activation", input_info("input"), func, params));

        network network(engine, topology, get_test_default_config(engine));
        network.set_input_data("input", input);
        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());
        cldnn::mem_lock<int32_t> input_ptr(input, get_test_stream());

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
            switch (func) {
            case activation_func::none:
                ASSERT_EQ((int32_t)input_ptr[i], output_ptr[i]);
                break;
            case activation_func::negative:
                ASSERT_EQ(-((int32_t)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::negation:
                ASSERT_EQ(!((int32_t)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::relu:
                ASSERT_EQ(std::max(static_cast<int32_t>(input_ptr[i]), 0), output_ptr[i]);
                break;
            case activation_func::clamp:
                ASSERT_EQ(std::min(std::max(input_ptr[i], static_cast<int32_t>(params.a)), static_cast<int32_t>(params.b)), output_ptr[i]);
                break;
            case activation_func::floor:
                ASSERT_EQ((int32_t)std::floor(input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_f32_fw_gpu, b_fs_yx_fsv16_prelu) {
    constexpr int b = 1;
    constexpr int f = 17;
    constexpr int x = 2;
    constexpr int y = 2;

    auto& eng = get_test_engine();

    auto in_lay = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(b, f, x, y));
    auto params_lay = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1, f, 1, 1));

    auto in_mem = eng.allocate_memory(in_lay);
    auto params_mem = eng.allocate_memory(params_lay);

    auto in_data = generate_random_4d<float>(b, f, y, x, -1, 1);
    auto params_data = generate_random_1d<float>(f, -1, 1);

    set_values(params_mem, params_data);

    auto topo = cldnn::topology(
        cldnn::input_layout("in", in_lay),
        cldnn::reorder("in_fsv16", input_info("in"), cldnn::format::b_fs_yx_fsv16, cldnn::data_types::f32),
        cldnn::data("actv_params", params_mem),
        cldnn::activation("actv", input_info("in_fsv16"), "actv_params", cldnn::activation_func::relu_negative_slope),
        cldnn::reorder("out", input_info("actv"), cldnn::format::bfyx, cldnn::data_types::f32)
    );

    cldnn::network net(eng, topo, get_test_default_config(eng));
    set_values(in_mem, flatten_4d(format::bfyx, in_data));
    net.set_input_data("in", in_mem);

    auto result = net.execute();
    auto out_mem = result.at("out").get_memory();

    std::vector<float> expected = flatten_4d(format::bfyx, in_data);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] < 0.f) {
            expected[i] = expected[i] * params_data[i / (x * y) % f];
        }
    }

    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());
    ASSERT_EQ(expected.size(), out_ptr.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(expected[i], out_ptr[i]) << "at i=" << i;
    }
}

using activation_random_test_params = std::tuple<data_types,
                                                 format::type,                  // input_format
                                                 tensor,                        // input_size
                                                 activation_func,               // func_type
                                                 activation_additional_params,  // additional_params
                                                 padding,
                                                 impl_types,
                                                 bool>;

struct activation_random_test : testing::TestWithParam<activation_random_test_params>
{
    bool enable_profiling = false;

    size_t get_x_pitch(layout& layout) {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = generate_random_4d<T>(b, f, y, x, min, max, k);
        mem_lock<T> ptr{mem, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<FLOAT16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_outputs(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();
        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        cldnn::mem_lock<T> ref_ptr(out_ref, get_test_stream());
        cldnn::mem_lock<T> opt_ptr(out_opt, get_test_stream());

        auto ref_x_pitch = get_x_pitch(output_lay);
        auto opt_x_pitch = get_x_pitch(opt_output_lay);

        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(0, yi, 0, 0));
                    auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                    auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_val = ref_ptr[ref_out_offset + xi * ref_x_pitch];
                        auto opt_out_val = opt_ptr[opt_out_offset + xi * opt_x_pitch];
                        if (ref_out_val != opt_out_val) {
                            ASSERT_NEAR(ref_out_val, opt_out_val, 1e-4);
                        }
                    }
                }
            }
        }
    }

    void execute_compare(const activation_random_test_params& params, bool check_result) {
        auto& engine = get_test_engine();

        data_types input_type;
        format::type input_format;
        tensor input_size;
        activation_func func_type;
        activation_additional_params additional_params;
        padding padd;
        impl_types impl_type;
        bool is_caching_test;
        std::tie(input_type, input_format, input_size, func_type, additional_params, padd, impl_type, is_caching_test) = params;
        auto in_layout = layout(input_type, format::bfyx, input_size);

        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);

        /// bfyx
        cldnn::topology topo;
        topo.add(input_layout("in", in_layout));
        auto prim = activation("activation", input_info("in"), func_type);
        prim.additional_params = additional_params;
        topo.add(prim);

        ExecutionConfig config = get_test_default_config(engine,
                                    ov::intel_gpu::custom_outputs(std::vector<std::string>{"activation"}));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("in", in_mem);

        // first execution of ref
        auto result = net->execute();
        auto output = result.at("activation").get_memory();

        cldnn::topology topo_opt;
        topo_opt.add(input_layout("in", in_layout));
        topo_opt.add(reorder("in_to_input_type", input_info("in"), input_format, input_type));
        auto prim_opt = activation("activation_blocked", input_info("in_to_input_type"), func_type);
        prim_opt.additional_params = additional_params;
        topo_opt.add(prim_opt);
        // force output format to input format.
        topo_opt.add(reorder("res_to_input_format", input_info("activation_blocked"), input_format, input_type));

        auto activation_impl_desc = ov::intel_gpu::ImplementationDesc();
        activation_impl_desc.output_format = input_format;
        ExecutionConfig config_opt = get_test_default_config(engine,
                                        {ov::intel_gpu::custom_outputs(std::vector<std::string>{"activation_blocked", "res_to_input_format"}),
                                         ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"activation_blocked", {input_format, "activation_ref", impl_type}}})});

        network net_opt(engine, topo_opt, config_opt);

        // Use in_mem from ref network
        net_opt.set_input_data("in", in_mem);

        // first execution of opt
        auto result_opt = net_opt.execute();
        auto output_opt = result_opt.at("res_to_input_format").get_memory();

        if (check_result == true) {
            // Check data_types
            if (input_type == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (input_type == data_types::f16) {
                compare_outputs<FLOAT16>(output, output_opt);
            } else if (input_type == data_types::i8) {
                compare_outputs<int8_t>(output, output_opt);
            } else if (input_type == data_types::u8) {
                compare_outputs<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(input_type);
            }
        }
    }
};

TEST_P(activation_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true);
}

const auto reluParams = testing::ValuesIn(std::vector<activation_random_test_params>{
    {data_types::i8, format::b_fs_yx_fsv32, {1, 32, 5, 5}, activation_func::relu, {}, {}, impl_types::any, false},
    {data_types::i8, format::bs_fs_yx_bsv32_fsv32, {32, 32, 5, 5}, activation_func::relu, {}, {}, impl_types::any, false},
    {data_types::f16, format::bs_fs_yx_bsv32_fsv16, {32, 32, 5, 5}, activation_func::relu, {}, {}, impl_types::any, false},
    {data_types::i8, format::bs_fs_yx_bsv32_fsv32, {16, 16, 5, 5}, activation_func::relu, {}, {}, impl_types::any, false},
    {data_types::f16, format::bs_fs_yx_bsv32_fsv16, {16, 16, 5, 5}, activation_func::relu, {}, {}, impl_types::any, false},
});

INSTANTIATE_TEST_SUITE_P(relu_activation_blocked_tests, activation_random_test, reluParams);

const std::vector<data_types> dataTypes = {data_types::f16, data_types::f32};
const std::vector<format::type> types = {format::b_fs_yx_fsv2,
                                         format::b_fs_zyx_fsv2,
                                         format::bs_fs_yx_bsv32_fsv32,
                                         format::bs_fs_yx_bsv32_fsv16};

// TODO: need to investigate input for commented activation functions
const std::vector<activation_func> activationFunctions = {activation_func::none,
                                                          activation_func::logistic,
                                                          activation_func::gelu,
                                                          activation_func::hyperbolic_tan,
                                                          activation_func::relu,
                                                          activation_func::relu_negative_slope,
                                                          activation_func::clamp,
                                                          activation_func::softrelu,
                                                          activation_func::abs,
                                                          activation_func::linear,
                                                          activation_func::square,
//                                                          activation_func::sqrt,
                                                          activation_func::elu,
                                                          activation_func::sin,
//                                                          activation_func::asin,
                                                          activation_func::sinh,
//                                                          activation_func::asinh,
                                                          activation_func::cos,
//                                                          activation_func::acos,
                                                          activation_func::cosh,
//                                                          activation_func::acosh,
//                                                          activation_func::log,
//                                                          activation_func::log2,
                                                          activation_func::exp,
                                                          activation_func::tan,
                                                          activation_func::atan,
//                                                          activation_func::atanh,
                                                          activation_func::floor,
                                                          activation_func::ceil,
                                                          activation_func::negative,
                                                          activation_func::negation,
                                                          activation_func::pow,
                                                          activation_func::reciprocal,
                                                          activation_func::erf,
                                                          activation_func::hard_sigmoid,
                                                          activation_func::hsigmoid,
                                                          activation_func::selu,
                                                          activation_func::sign,
                                                          activation_func::softplus,
                                                          activation_func::swish,
                                                          activation_func::hswish,
                                                          activation_func::mish,
                                                          activation_func::round_half_to_even,
                                                          activation_func::round_half_away_from_zero,
                                                          activation_func::gelu_tanh,
                                                          activation_func::softsign};

const std::vector<tensor> inputShapes = {
    {1, 32, 5, 5},
    {32, 32, 5, 5},
    {16, 16, 5, 5},
};

INSTANTIATE_TEST_SUITE_P(
    fp_activation_blocked_tests0,
    activation_random_test,
    ::testing::Combine(::testing::Values(dataTypes[0]),
                       ::testing::Values(types[0]),
                       ::testing::Values(inputShapes[0]),
                       ::testing::ValuesIn(activationFunctions),
                       ::testing::Values(activation_additional_params{}),
                       ::testing::Values(padding{}),
                       ::testing::Values(false),
                       ::testing::Values(impl_types::any)));
INSTANTIATE_TEST_SUITE_P(
    fp_activation_blocked_tests1,
    activation_random_test,
    ::testing::Combine(::testing::Values(dataTypes[1]),
                       ::testing::Values(types[3]),
                       ::testing::Values(inputShapes[2]),
                       ::testing::ValuesIn(activationFunctions),
                       ::testing::Values(activation_additional_params{}),
                       ::testing::Values(padding{}),
                       ::testing::Values(false),
                       ::testing::Values(impl_types::any)));
INSTANTIATE_TEST_SUITE_P(
    fp_activation_blocked_tests2,
    activation_random_test,
    ::testing::Combine(::testing::Values(dataTypes[0]),
                       ::testing::Values(types[2]),
                       ::testing::Values(inputShapes[1]),
                       ::testing::ValuesIn(activationFunctions),
                       ::testing::Values(activation_additional_params{}),
                       ::testing::Values(padding{}),
                       ::testing::Values(false),
                       ::testing::Values(impl_types::any)));
INSTANTIATE_TEST_SUITE_P(
    fp_activation_blocked_tests3,
    activation_random_test,
    ::testing::Combine(::testing::ValuesIn(dataTypes),
                       ::testing::ValuesIn(types),
                       ::testing::ValuesIn(inputShapes),
                       ::testing::Values(activationFunctions.front()),
                       ::testing::Values(activation_additional_params{}),
                       ::testing::Values(padding{}),
                       ::testing::Values(false),
                       ::testing::Values(impl_types::any)));
INSTANTIATE_TEST_SUITE_P(
    fp_activation_blocked_tests4,
    activation_random_test,
    ::testing::Combine(::testing::ValuesIn(dataTypes),
                       ::testing::ValuesIn(types),
                       ::testing::ValuesIn(inputShapes),
                       ::testing::Values(activationFunctions.back()),
                       ::testing::Values(activation_additional_params{}),
                       ::testing::Values(padding{}),
                       ::testing::Values(false),
                       ::testing::Values(impl_types::any)));
INSTANTIATE_TEST_SUITE_P(
    export_import,
    activation_random_test,
    ::testing::Combine(::testing::Values(dataTypes[0]),
                       ::testing::Values(types[0]),
                       ::testing::Values(inputShapes[0]),
                       ::testing::Values(activationFunctions.back()),
                       ::testing::Values(activation_additional_params{}),
                       ::testing::Values(padding{}),
                       ::testing::Values(true),
                       ::testing::Values(impl_types::any)));

INSTANTIATE_TEST_SUITE_P(
    cpu_impls,
    activation_random_test,
    ::testing::Combine(::testing::Values(data_types::f16),
                       ::testing::Values(format::bfyx),
                       ::testing::Values(tensor{1, 3, 2, 4}),
                       ::testing::ValuesIn({ activation_func::relu, activation_func::abs, activation_func::gelu,
                                             activation_func::round_half_to_even, activation_func::clamp, activation_func::pow,
                                             activation_func::negative, activation_func::swish }),
                       ::testing::Values(activation_additional_params{2.0f, 3.5f}),
                       ::testing::Values(padding{}),
                       ::testing::Values(true),
                       ::testing::Values(impl_types::cpu)));
