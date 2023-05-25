// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "activation_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/power.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/elu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/prelu.hpp"
#include "ngraph/op/clamp.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/asinh.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/acosh.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/atanh.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/hard_sigmoid.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/selu.hpp"
#include "ngraph/op/softplus.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/swish.hpp"
#include "ngraph/op/hswish.hpp"
#include "ngraph/op/mish.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/hsigmoid.hpp"
#include "ngraph/op/round.hpp"


namespace cldnn {
namespace cpu {

struct activation_impl : public typed_primitive_impl<activation> {
    using parent = typed_primitive_impl<activation>;
    using parent::parent;

    activation_func activation_function;
    activation_additional_params params;

    std::shared_ptr<ov::op::Op> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<activation_impl>(*this);
    }

    activation_impl() : parent("activation_cpu_impl") {}

    explicit activation_impl(const activation_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<activation>());
        const auto& node = arg.as<activation>();
        activation_function = node.get_primitive()->activation_function;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << make_data(&activation_function, sizeof(activation_func));
        ob << make_data(&params, sizeof(activation_additional_params));
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> make_data(&activation_function, sizeof(activation_func));
        ib >> make_data(&params, sizeof(activation_additional_params));
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, activation_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "activation::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);

        if (!op) {
            switch (activation_function) {
            case activation_func::pow:
                op = std::make_shared<ov::op::v1::Power>(); break;
            case activation_func::hyperbolic_tan:
                op = std::make_shared<ov::op::v0::Tanh>(); break;
            case activation_func::elu:
                op = std::make_shared<ov::op::v0::Elu>(); break;
            case activation_func::logistic:
                op = std::make_shared<ov::op::v0::Sigmoid>(); break;
            case activation_func::relu:
                op = std::make_shared<ov::op::v0::Relu>(); break;
            case activation_func::relu_negative_slope:
                op = std::make_shared<ov::op::v0::PRelu>(); break;
            case activation_func::clamp: {
                auto clamp_op = std::make_shared<ov::op::v0::Clamp>();
                clamp_op->set_min(params.a);
                clamp_op->set_max(params.b);
                op = clamp_op;
                break;
            }
            case activation_func::exp:
                op = std::make_shared<ov::op::v0::Exp>(); break;
            case activation_func::negation:
                op = std::make_shared<ov::op::v1::LogicalNot>(); break;
            case activation_func::asin:
                op = std::make_shared<ov::op::v0::Asin>(); break;
            case activation_func::asinh:
                op = std::make_shared<ov::op::v3::Asinh>(); break;
            case activation_func::acos:
                op = std::make_shared<ov::op::v0::Acos>(); break;
            case activation_func::acosh:
                op = std::make_shared<ov::op::v3::Acosh>(); break;
            case activation_func::atan:
                op = std::make_shared<ov::op::v0::Atan>(); break;
            case activation_func::atanh:
                op = std::make_shared<ov::op::v3::Atanh>(); break;
            case activation_func::abs:
                op = std::make_shared<ov::op::v0::Abs>(); break;
            case activation_func::floor:
                op = std::make_shared<ov::op::v0::Floor>(); break;
            case activation_func::ceil:
                op = std::make_shared<ov::op::v0::Ceiling>(); break;
            case activation_func::erf:
                op = std::make_shared<ov::op::v0::Erf>(); break;
            case activation_func::hard_sigmoid:
                op = std::make_shared<ov::op::v0::HardSigmoid>(); break;
            case activation_func::log:
                op = std::make_shared<ov::op::v0::Log>(); break;
            case activation_func::negative:
                op = std::make_shared<ov::op::v0::Negative>(); break;
            case activation_func::selu:
                op = std::make_shared<ov::op::v0::Selu>(); break;
            case activation_func::softplus:
                op = std::make_shared<ov::op::v4::SoftPlus>(); break;
            case activation_func::tan:
                op = std::make_shared<ov::op::v0::Tan>(); break;
            case activation_func::sin:
                op = std::make_shared<ov::op::v0::Sin>(); break;
            case activation_func::sinh:
                op = std::make_shared<ov::op::v0::Sinh>(); break;
            case activation_func::cos:
                op = std::make_shared<ov::op::v0::Cos>(); break;
            case activation_func::cosh:
                op = std::make_shared<ov::op::v0::Cosh>(); break;
            case activation_func::swish:
                op = std::make_shared<ov::op::v4::Swish>(); break;
            case activation_func::hswish:
                op = std::make_shared<ov::op::v4::HSwish>(); break;
            case activation_func::mish:
                op = std::make_shared<ov::op::v4::Mish>(); break;
            case activation_func::gelu:
            case activation_func::gelu_tanh: {
                auto gelu_op = std::make_shared<ov::op::v7::Gelu>();
                auto approximation_mode =
                    activation_function == cldnn::activation_func::gelu ? ov::op::GeluApproximationMode::ERF
                                                                        : ov::op::GeluApproximationMode::TANH;
                gelu_op->set_approximation_mode(approximation_mode);
                op = gelu_op;
                break;
            }
            case activation_func::sign:
                op = std::make_shared<ov::op::v0::Sign>(); break;
            case activation_func::hsigmoid:
                op = std::make_shared<ov::op::v5::HSigmoid>(); break;
            case activation_func::round_half_to_even:
            case activation_func::round_half_away_from_zero: {
                auto round_op = std::make_shared<ov::op::v5::Round>();
                auto round_mode =
                    activation_function == cldnn::activation_func::round_half_to_even ? ov::op::v5::Round::RoundMode::HALF_TO_EVEN
                                                                                      : ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO;
                round_op->set_mode(round_mode);
                op = round_op;
                break;
            }
            default:
                OPENVINO_THROW("[GPU] Couldn't create activation operation: unsupported activation type ",
                               "(", static_cast<size_t>(activation_function), ") for primitive with id", instance.id());
            }

            OPENVINO_ASSERT(op->has_evaluate(), "[GPU] Couldn't find evaluate() function for activation ",
                                                "primitive with id ", instance.id());
        }

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        // TODO: consider to re-implement lock in more exception-safetest way
        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(input_mem_ptrs[i]->get_layout(), input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        if (activation_function == activation_func::pow) {
            input_host_tensors.push_back(ov::Tensor(ov::element::Type_t::f32, {}, &params.a));
        } else if (activation_function == activation_func::relu_negative_slope) {
            if (input_host_tensors.size() < 2) {
                input_host_tensors.push_back(ov::Tensor(ov::element::Type_t::f32, {}, &params.a));
            }
        } else if (activation_function == activation_func::swish) {
            if (params.a != 1.0f)
                input_host_tensors.push_back(ov::Tensor(ov::element::Type_t::f32, {}, &params.a));
        }

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);
        output_host_tensors.push_back(make_tensor(output_mem_ptr->get_layout(), output_lock.data()));

        op->evaluate(output_host_tensors, input_host_tensors);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const activation_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<activation_impl>();
    }
};


namespace detail {

attach_activation_impl::attach_activation_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32,
        data_types::i64,
    };

    implementation_map<activation>::add(impl_types::cpu, shape_types::static_shape, activation_impl::create, types, formats);
    implementation_map<activation>::add(impl_types::cpu, shape_types::dynamic_shape, activation_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::activation_impl)
