// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/slice.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "convolution_inst.h"

#include <random>
#include <algorithm>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {

const std::string no_bias = "";

TEST(convolution_f32_fw_gpu, basic_convolution_no_bias_dynamic_osv16) {
    auto& engine = get_test_engine();

    ov::Shape in0_shape = { 1, 1, 4, 5 };

    auto in0_dyn_layout = layout{ov::PartialShape::dynamic(in0_shape.size()), data_types::f32, format::bfyx};
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 } });

    set_values(weights, {
        1.0f, 2.0f, 1.0f,
        2.0f, 1.0f, 2.0f
    });

    topology topology(
        input_layout("input", in0_dyn_layout),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, 1, { 2, 1 }, {1, 1}, {0, 0}, {0, 0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    // first execute
    {
        auto input0 = engine.allocate_memory({ in0_shape, data_types::f32, format::bfyx });
        set_values(input0, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            2.0f, 2.0f, 3.0f, 4.0f, 6.0f,
            3.0f, 3.0f, 3.0f, 5.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        });
        network.set_input_data("input", input0);

        auto inst = network.get_primitive("conv");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "conv");

        auto output_memory = outputs.at("conv").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::bfyx);
        ASSERT_EQ(y_size, 2);
        ASSERT_EQ(x_size, 3);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        VVF<float> output_vec = {
            { 20.0f, 27.0f, 38.0f },
            { 17.0f, 19.0f, 19.0f }
        };
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
            }
        }
    }

    // second execute
    {
        in0_shape = { 1, 1, 6, 4 };
        auto input0 = engine.allocate_memory({ in0_shape, data_types::f32, format::bfyx });
        set_values(input0, {
            1.0f, 2.0f, 3.0f, 4.0f,
            2.0f, 2.0f, 3.0f, 4.0f,
            3.0f, 3.0f, 3.0f, 5.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            5.0f, 4.0f, 3.0f, 2.0f,
            4.0f, 4.0f, 3.0f, 3.0f,
        });
        network.set_input_data("input", input0);

        auto inst = network.get_primitive("conv");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "conv");

        auto output_memory = outputs.at("conv").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::bfyx);
        ASSERT_EQ(y_size, 3);
        ASSERT_EQ(x_size, 2);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        VVF<float> output_vec = {
            { 20.f, 27.f },
            { 17.f, 19.f },
            { 34.f, 29.f }
        };
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
            }
        }
    }
}

struct conv_params {
    ov::Shape in_shape;
    ov::Shape wei_shape;
    ov::Strides stride;
    ov::Strides dilation;
    ov::CoordinateDiff pad_begin;
    ov::CoordinateDiff pad_end;
};

class conv_osv16_tests : public testing::TestWithParam<conv_params> {};
TEST_P(conv_osv16_tests, basic_convolution_no_bias) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto in0_layout = layout{p.in_shape, data_types::f32, format::bfyx};
    auto input0 = engine.allocate_memory({ p.in_shape, data_types::f32, format::bfyx });
    auto weights = engine.allocate_memory({p.wei_shape, data_types::f32, format::bfyx});

    topology topology(
        input_layout("input", in0_layout),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, 1, p.stride, p.dilation, p.pad_begin, p.pad_end, false));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bfyx, "convolution_gpu_bfyx_os_iyx_osv16", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl } }));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);


    network.set_input_data("input", input0);
    auto outputs = network.execute();
}

INSTANTIATE_TEST_SUITE_P(smoke, conv_osv16_tests,
    testing::ValuesIn(std::vector<conv_params>{
        {
            ov::Shape{1, 8, 4, 6},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 14, 14},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 32, 32},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 60, 60},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 64, 64},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 110, 111},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
    }));

class conv_osv16_tests_dyn : public testing::TestWithParam<conv_params> {};
TEST_P(conv_osv16_tests_dyn, basic_convolution_no_bias) {
    auto& engine = get_test_engine();
    auto p = GetParam();

    auto calc_ref = [&](memory::ptr input, memory::ptr weights) {
        auto in0_layout = layout{p.in_shape, data_types::f32, format::bfyx};

        topology topology_ref(
            input_layout("input", in0_layout),
            data("weights", weights),
            convolution("conv", input_info("input"), "weights", no_bias, 1, p.stride, p.dilation, p.pad_begin, p.pad_end, false));

        ExecutionConfig config = get_test_default_config(engine);
        ov::intel_gpu::ImplementationDesc conv_impl = { format::bfyx, "convolution_gpu_bfyx_os_iyx_osv16", impl_types::ocl };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl } }));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network_ref(engine, topology_ref, config);
        network_ref.set_input_data("input", input);
        auto outputs_ref = network_ref.execute();

        return outputs_ref.at("conv").get_memory();
    };

    auto in0_layout = layout{ov::PartialShape{ov::Dimension(), ov::Dimension(p.in_shape[1]), ov::Dimension(), ov::Dimension()}, data_types::f32, format::bfyx};
    auto input0 = engine.allocate_memory({ p.in_shape, data_types::f32, format::bfyx });
    auto weights = engine.allocate_memory({p.wei_shape, data_types::f32, format::bfyx});

    topology topology(
        input_layout("input", in0_layout),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, 1, p.stride, p.dilation, p.pad_begin, p.pad_end, false));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bfyx, "convolution_gpu_bfyx_os_iyx_osv16", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl } }));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("input", input0);
    auto outputs = network.execute();

    auto output_memory = outputs.at("conv").get_memory();
    auto output_memory_ref = calc_ref(input0, weights);

    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> output_ptr_ref(output_memory_ref, get_test_stream());

    ASSERT_EQ(outputs.at("conv").get_layout(), output_memory_ref->get_layout());
    for (size_t i = 0; i < output_ptr.size(); i++) {
        ASSERT_EQ(output_ptr[i], output_ptr_ref[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, conv_osv16_tests_dyn,
    testing::ValuesIn(std::vector<conv_params>{
        {
            ov::Shape{1, 8, 4, 6},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 14, 14},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 32, 32},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 60, 60},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 64, 64},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 110, 111},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0}
        },
        {
            ov::Shape{1, 8, 110, 111},    // input_layout
            ov::Shape{16, 8, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::CoordinateDiff{1, 1}
        },
        {
            ov::Shape{1, 8, 110, 111},    // input_layout
            ov::Shape{16, 8, 5, 5},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::CoordinateDiff{1, 1}
        },

// 2,640,32,32

        {
            ov::Shape{2, 640, 32, 32},    // input_layout
            ov::Shape{640, 640, 3, 3},   // weight layout
            ov::Strides{1, 1},
            ov::Strides{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::CoordinateDiff{1, 1}
        },
    }));

template<typename T>
class SliceTest : public ::testing::Test {
public:
    static std::vector<T> GenInput(int size) {
        std::vector<T> result;
        for (int i = 0; i < size; i++)
            result.push_back(i);
        return result;
    }

    void execute(bool is_caching_test) {
        assert(input_shape_.size() == 4 || input_shape_.size() == 5);
        format input_format = input_shape_.size() == 4 ? format::bfyx : format::bfzyx;
        layout data_layout ( input_type_, input_format, tensor{input_shape_} );
        std::vector<T> input_vals = GenInput(static_cast<int>(data_layout.get_linear_size()));
        memory::ptr input = engine_.allocate_memory(data_layout);
        set_values(input, input_vals);
        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("start", start_));
        topology.add(data("stop", stop_));
        topology.add(data("step", step_));
        std::vector<input_info> inputs { input_info("input"), input_info("start"), input_info("stop"), input_info("step") };
        if (axes_) {
            topology.add(data("axes", axes_));
            inputs.push_back(input_info("axes"));
        }
        topology.add(slice("slice", inputs, tensor{output_shape_}));

        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "slice");

        auto output = outputs.at("slice").get_memory();

        cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), expected_output_.size());
        for (size_t i = 0; i < output_ptr.size(); ++i)
            ASSERT_TRUE(are_equal(expected_output_[i], output_ptr[i], 2e-3));
    }

    data_types DataType() const;

protected:
    engine& engine_ = get_test_engine();
    std::vector<std::int32_t> input_shape_;
    data_types input_type_ {DataType()};
    memory::ptr start_;
    memory::ptr stop_;
    memory::ptr step_;
    memory::ptr axes_;
    std::vector<std::int32_t> output_shape_;
    std::vector<T> expected_output_;
};

template<>
data_types SliceTest<float>::DataType() const {return data_types::f32;}

template<>
data_types SliceTest<int>::DataType() const { return data_types::i32; }

template<>
data_types SliceTest<long long>::DataType() const { return data_types::i64; }

using testing::Types;
typedef Types<float, int, long long> DataTypes;
TYPED_TEST_SUITE(SliceTest, DataTypes);

TYPED_TEST(SliceTest, bfyx_positive_step) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, {0, 1, 0, 1});
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, { 1, 2, 5, 100 });
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1201, 1211, 1221, 1231, 1241, 1301, 1311, 1321, 1331, 1341,
            1401, 1411, 1421, 1431, 1441, 1501, 1511, 1521, 1531, 1541,
            1601, 1611, 1621, 1631, 1641, 1701, 1711, 1721, 1731, 1741,
            1801, 1811, 1821, 1831, 1841, 1901, 1911, 1921, 1931, 1941,
            2001, 2011, 2021, 2031, 2041, 2101, 2111, 2121, 2131, 2141
    };
    this->execute(false);
}

TYPED_TEST(SliceTest, bfyx_negative_step) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 1, 2, 5, 100 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {0, 1, 0, 1});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { -1, -1, -1, -10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1799, 1789, 1779, 1769, 1759, 1699, 1689, 1679, 1669, 1659,
            1599, 1589, 1579, 1569, 1559, 1499, 1489, 1479, 1469, 1459,
            1399, 1389, 1379, 1369, 1359, 1299, 1289, 1279, 1269, 1259,
            1199, 1189, 1179, 1169, 1159, 1099, 1089, 1079, 1069, 1059,
             999,   989,  979, 969,  959,  899,  889,  879,  869,  859
    };
    this->execute(false);
}

TYPED_TEST(SliceTest, bfzyx) {
    this->input_shape_ = { 2, 3, 10, 12, 5 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 0, 0, 0, 0, 0 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {1, 2, 2, 2, 2});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 1, 1 });
    this->output_shape_ = { 1, 2, 2, 2, 2 };
    this->expected_output_ = {
              0,   1,  10,  11, 120, 121, 130, 131,
            600, 601, 610, 611, 720, 721, 730, 731
    };
    this->execute(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TYPED_TEST(SliceTest, bfyx_positive_step_cached) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, {0, 1, 0, 1});
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, { 1, 2, 5, 100 });
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1201, 1211, 1221, 1231, 1241, 1301, 1311, 1321, 1331, 1341,
            1401, 1411, 1421, 1431, 1441, 1501, 1511, 1521, 1531, 1541,
            1601, 1611, 1621, 1631, 1641, 1701, 1711, 1721, 1731, 1741,
            1801, 1811, 1821, 1831, 1841, 1901, 1911, 1921, 1931, 1941,
            2001, 2011, 2021, 2031, 2041, 2101, 2111, 2121, 2131, 2141
    };
    this->execute(true);
}

TYPED_TEST(SliceTest, bfyx_negative_step_cached) {
    this->input_shape_ = { 1, 2, 100, 12 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 1, 2, 5, 100 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {0, 1, 0, 1});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { -1, -1, -1, -10 });
    this->output_shape_ = { 1, 1, 5, 10 };
    this->expected_output_ = {
            1799, 1789, 1779, 1769, 1759, 1699, 1689, 1679, 1669, 1659,
            1599, 1589, 1579, 1569, 1559, 1499, 1489, 1479, 1469, 1459,
            1399, 1389, 1379, 1369, 1359, 1299, 1289, 1279, 1269, 1259,
            1199, 1189, 1179, 1169, 1159, 1099, 1089, 1079, 1069, 1059,
             999,   989,  979, 969,  959,  899,  889,  879,  869,  859
    };
    this->execute(true);
}
#endif
TYPED_TEST(SliceTest, bfzyx_cached) {
    this->input_shape_ = { 2, 3, 10, 12, 5 };
    this->start_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->start_, { 0, 0, 0, 0, 0 });
    this->stop_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->stop_, {1, 2, 2, 2, 2});
    this->step_ = this->engine_.allocate_memory({ data_types::i64, format::bfzyx, { 5, 1, 1, 1 } });
    set_values<int64_t>(this->step_, { 1, 1, 1, 1, 1 });
    this->output_shape_ = { 1, 2, 2, 2, 2 };
    this->expected_output_ = {
              0,   1,  10,  11, 120, 121, 130, 131,
            600, 601, 610, 611, 720, 721, 730, 731
    };
    this->execute(true);
}

} // anonymous namespace
