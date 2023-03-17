// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include <gtest/gtest.h>
#include "test_utils/test_utils.h"

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/tensor.hpp>

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/primitive.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

using namespace cldnn;

namespace tests {

// =====================================================================================================================
// Typed comparison

// =====================================================================================================================
// Reference tensor
struct reference_tensor {
    virtual void compare(cldnn::memory::ptr actual) = 0;
};

template <typename T, size_t N>
struct reference_tensor_typed : reference_tensor {};

template <typename T>
struct reference_tensor_typed<T, 1> : reference_tensor {
    using vector_type = VF<T>;
    reference_tensor_typed(vector_type data) : reference(std::move(data)) {}

    void compare(cldnn::memory::ptr actual) override {
        cldnn::mem_lock<T> ptr(actual, get_test_stream());

        for (size_t bi = 0; bi < reference.size(); ++bi) {
            auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(0), cldnn::spatial(0, 0, 0, 0));
            size_t offset = actual->get_layout().get_linear_offset(coords);
            auto& ref = reference[bi];
            auto& val = ptr[offset];
            TYPED_ASSERT_EQ(ref, val) << " at bi=" << bi;
        }
    }

    void fill_memory(cldnn::memory::ptr mem) {
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(0), cldnn::spatial(0, 0, 0, 0));
            size_t offset = mem->get_layout().get_linear_offset(coords);
            ptr[offset] = reference[bi];
        }
    }

    cldnn::tensor get_shape() {
        return cldnn::tensor(cldnn::batch(reference.size()));
    }

    vector_type reference;
};

template <typename T>
struct reference_tensor_typed<T, 2> : reference_tensor {
    using vector_type = VVF<T>;
    reference_tensor_typed(vector_type data) : reference(std::move(data)) {}

    void compare(cldnn::memory::ptr actual) override {
        cldnn::mem_lock<T> ptr(actual, get_test_stream());
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(0, 0, 0, 0));
                size_t offset = actual->get_layout().get_linear_offset(coords);
                auto& ref = reference[bi][fi];
                auto& val = ptr[offset];
                TYPED_ASSERT_EQ(ref, val) << "at bi=" << bi << " fi=" << fi;
            }
        }
    }

    void fill_memory(cldnn::memory::ptr mem) {
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(0, 0, 0, 0));
                size_t offset = mem->get_layout().get_linear_offset(coords);
                ptr[offset] = reference[bi][fi];
            }
        }
    }

    cldnn::tensor get_shape() {
        return cldnn::tensor(cldnn::batch(reference.size()), cldnn::feature(reference[0].size()));
    }

    vector_type reference;
};

template <typename T>
struct reference_tensor_typed<T, 4> : reference_tensor {
    using vector_type = VVVVF<T>;
    reference_tensor_typed(vector_type data) : reference(std::move(data)) {}
    void compare(cldnn::memory::ptr actual) override {
        cldnn::mem_lock<T> ptr(actual, get_test_stream());
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                for (size_t yi = 0; yi < reference[0][0].size(); ++yi) {
                    for (size_t xi = 0; xi < reference[0][0][0].size(); ++xi) {
                        auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(xi, yi, 0, 0));
                        size_t offset = actual->get_layout().get_linear_offset(coords);
                        auto& ref = reference[bi][fi][yi][xi];
                        auto& val = ptr[offset];
                        TYPED_ASSERT_EQ(ref, val) << "at bi=" << bi << " fi=" << fi << " yi=" << yi << " xi=" << xi;
                    }
                }
            }
        }
    }

    void fill_memory(cldnn::memory::ptr mem) {
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                for (size_t yi = 0; yi < reference[0][0].size(); ++yi) {
                    for (size_t xi = 0; xi < reference[0][0][0].size(); ++xi) {
                        auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(xi, yi, 0, 0));
                        size_t offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = reference[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    cldnn::tensor get_shape() {
        return cldnn::tensor(cldnn::batch(reference.size()),
                             cldnn::feature(reference[0].size()),
                             cldnn::spatial(reference[0][0][0].size(), reference[0][0].size()));
    }

    vector_type reference;
};

// =====================================================================================================================
// Reference calculations
template <typename InputT>
struct fully_connected_accumulator {
    using type = float;
};

template <>
struct fully_connected_accumulator<uint8_t> {
    using type = int;
};

template <>
struct fully_connected_accumulator<int8_t> {
    using type = int;
};

template <typename OutputT,
          typename InputT,
          typename WeightsT,
          typename BiasT,
          typename AccT = typename fully_connected_accumulator<InputT>::type>
VVF<OutputT> fully_connected_reference_typed(VVVVF<InputT>& input, VVVVF<WeightsT>& weights, VF<BiasT>& bias) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();
    size_t output_f = weights.size();
    auto output = VVF<OutputT>(output_b, VF<OutputT>(output_f));
    for (size_t bi = 0; bi < output_b; ++bi) {
        for (size_t ofi = 0; ofi < output_f; ++ofi) {
            AccT acc = static_cast<AccT>(0);
            for (size_t ifi = 0; ifi < input_f; ++ifi) {
                for (size_t yi = 0; yi < input_y; ++yi) {
                    for (size_t xi = 0; xi < input_x; ++xi) {
                        acc += static_cast<AccT>(input[bi][ifi][yi][xi]) * static_cast<AccT>(weights[ofi][ifi][yi][xi]);
                    }
                }
            }
            output[bi][ofi] = static_cast<OutputT>(acc) + static_cast<OutputT>(bias[ofi]);
        }
    }
    return output;
}

template <typename OutputT,
          typename InputT,
          typename WeightsT,
          typename BiasT,
          typename AccT = typename fully_connected_accumulator<InputT>::type>
VVVVF<OutputT> fully_connected_reference_typed_3d(VVVVF<InputT>& input, VVVVF<WeightsT>& weights, VF<BiasT>& bias) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();        // input is assumed to be bfyx
    size_t output_f = weights.size();    // weights is assumed to be bfyx
    VVVVF<OutputT> output(output_b, VVVF<OutputT>(input_f, VVF<OutputT>(output_f, VF<OutputT>(1))));
    OutputT res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t n = 0; n < input_f; ++n) {
            for (size_t f = 0; f < output_f; ++f) {
                res = bias[f];
                for (size_t y = 0; y < input_y; ++y) {
                    for (size_t x = 0; x < input_x; ++x) {
                        res += (OutputT)input[b][n][y][x] * (OutputT)weights[f][y][0][0];
                    }
                }
                output[b][n][f][0] = (OutputT)res;
            }
        }
    }
    return output;
}

// =====================================================================================================================
// Network test
struct reference_node_interface {
    using ptr = std::shared_ptr<reference_node_interface>;

    virtual reference_tensor& get_reference() = 0;
    virtual cldnn::primitive_id get_id() = 0;
    virtual ~reference_node_interface() = default;
};

template <typename T, size_t N>
struct reference_node : reference_node_interface {
    using ptr = std::shared_ptr<reference_node>;

    reference_node(cldnn::primitive_id id, reference_tensor_typed<T, N> data)
        : id(id), reference(std::move(data)) {}

    cldnn::primitive_id id;
    reference_tensor_typed<T, N> reference;

    reference_tensor& get_reference() override { return reference; }
    cldnn::primitive_id get_id() override { return id; }
};

// =====================================================================================================================
// Random data generation
template <typename T>
struct type_test_ranges {
    static constexpr int min = -1;
    static constexpr int max = 1;
    static constexpr int k = 8;
};

template <>
struct type_test_ranges<uint8_t> {
    static constexpr int min = 0;
    static constexpr int max = 255;
    static constexpr int k = 1;
};

template <>
struct type_test_ranges<int8_t> {
    static constexpr int min = -127;
    static constexpr int max = 127;
    static constexpr int k = 1;
};

template <typename T>
VF<T> generate_smart_random_1d(size_t a) {
    return generate_random_1d<T>(a, type_test_ranges<T>::min, type_test_ranges<T>::max, type_test_ranges<T>::k);
}

template <typename T>
VVF<T> generate_smart_random_2d(size_t a, size_t b) {
    return generate_random_2d<T>(a, b, type_test_ranges<T>::min, type_test_ranges<T>::max, type_test_ranges<T>::k);
}

template <typename T>
VVVVF<T> generate_smart_random_4d(size_t a, size_t b, size_t c, size_t d) {
    return generate_random_4d<T>(a, b, c, d, type_test_ranges<T>::min, type_test_ranges<T>::max, type_test_ranges<T>::k);
}

}  // namespace tests
