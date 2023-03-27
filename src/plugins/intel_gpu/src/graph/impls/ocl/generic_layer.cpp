// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "generic_layer_inst.h"

namespace cldnn {
namespace ocl {

struct generic_layer_impl : typed_primitive_impl<generic_layer> {
    using parent = typed_primitive_impl<generic_layer>;
    using parent::parent;

    kernel_selector::cl_kernel_data _cl_kernel_data;
    kernel::ptr _kernel;
    kernel_id _kernel_id;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generic_layer_impl>(*this);
    }

    generic_layer_impl() : parent() {}

    generic_layer_impl(const generic_layer_impl& other)
    : _cl_kernel_data(other._cl_kernel_data)
    , _kernel(nullptr)
    , _kernel_id(other._kernel_id) {
        if (other._kernel == nullptr) {
            throw std::runtime_error("Can't copy generic_layer_impl node: kernel is nullptr");
        }
        _kernel = other._kernel->clone();
    }

    generic_layer_impl(kernels_cache& cache, const kernel_impl_params& params)
        : _cl_kernel_data()
        , _kernel(nullptr) {
        auto reorder_params = params.typed_desc<generic_layer>()->params;
        auto casted_params = std::dynamic_pointer_cast<WeightsReorderParamsOCL>(reorder_params);
        OPENVINO_ASSERT(casted_params, "[GPU] Invalid weights reorder parameters type for ", params.desc->id, " node");
        _cl_kernel_data = *casted_params->cl_kernel;
        _kernel_id = cache.set_kernel_source(_cl_kernel_data.code.kernelString, false);
    }

    generic_layer_impl(const kernel_impl_params& params)
        : _cl_kernel_data()
        , _kernel(nullptr)
        , _kernel_id() {
        auto reorder_params = params.typed_desc<generic_layer>()->params;
        auto casted_params = std::dynamic_pointer_cast<WeightsReorderParamsOCL>(reorder_params);
        OPENVINO_ASSERT(casted_params, "[GPU] Invalid weights reorder parameters type for ", params.desc->id, " node");
        _cl_kernel_data = *casted_params->cl_kernel;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob <<_cl_kernel_data;
        ob << _kernel_id;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> _cl_kernel_data;
        ib >> _kernel_id;
    }

    void init_kernels(const kernels_cache& kernels_cache) override {
        _kernel = kernels_cache.get_kernel(_kernel_id);
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        kernel_strings.push_back(_cl_kernel_data.code.kernelString);
        return kernel_strings;
    }

    void set_kernels(std::map<const std::string, kernel::ptr>& kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Unexpected kernels number for generic_layer");
        _kernel_id = kernels.begin()->first;
        _kernel = kernels.begin()->second;
    }

    void set_arguments_impl(generic_layer_inst& instance) override {
        kernel_arguments_data args;
        args.scalars = &_cl_kernel_data.params.scalars;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        args.outputs.push_back(instance.output_memory_ptr());

        set_arguments_impl(instance, args);
    }

    void set_arguments_impl(generic_layer_inst& instance, kernel_arguments_data& args) override {
        stream& stream = instance.get_network().get_stream();
        stream.set_arguments(*_kernel, _cl_kernel_data.params, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, generic_layer_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        args.scalars = &_cl_kernel_data.params.scalars;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        args.outputs.push_back(instance.output_memory_ptr());
        return stream.enqueue_kernel(*_kernel, _cl_kernel_data.params, args, events, true);
    }

    static std::unique_ptr<primitive_impl> create(const kernel_impl_params& params) {
        return make_unique<generic_layer_impl>(params);
    }
};

static std::unique_ptr<primitive_impl> create(const generic_layer_node& arg, const kernel_impl_params& params) {
    return make_unique<generic_layer_impl>(arg.get_program().get_kernels_cache(), params);
}


namespace detail {
attach_generic_layer_impl::attach_generic_layer_impl() {
    implementation_map<generic_layer>::add(cldnn::impl_types::ocl, create, {});

    WeightsReordersFactory::add(cldnn::impl_types::ocl, shape_types::static_shape, generic_layer_impl::create);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::generic_layer_impl)
