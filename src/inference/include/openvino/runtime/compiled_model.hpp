// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides CompiledModel class
 *
 * @file openvino/runtime/compiled_model.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/runtime/config.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace InferenceEngine {
class IExecutableNetworkInternal;
}  // namespace InferenceEngine
namespace ov {
namespace runtime {

class Core;
class InferRequest;

/**
 * @brief This is an interface of an executable network
 */
class OPENVINO_RUNTIME_API CompiledModel {
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs CompiledModel from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Plugin to use. This is required to ensure that CompiledModel can work properly even if plugin
     * object is destroyed.
     */
    CompiledModel(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& impl,
                  const std::shared_ptr<void>& so);
    friend class ov::runtime::Core;
    friend class ov::runtime::InferRequest;

    void get_config(const std::string& name, ov::Any& to, const ConfigMutability mutability) const;

public:
    /**
     * @brief A default constructor.
     */
    CompiledModel() = default;

    /**
     * @brief Destructor presereves unload order of implementation object and reference to library
     */
    ~CompiledModel();

    /**
     * @brief Get executable graph information from a device
     *
     * @return Model containing Executable Graph Info
     */
    std::shared_ptr<const Model> get_runtime_model() const;

    /**
     * @brief Get inputs of executable graph
     *
     * @return vector of inputs
     */
    std::vector<ov::Output<const ov::Node>> inputs() const;

    /**
     * @brief Get input of executable graph
     *
     * @return Model input or throw ov::Exception in case of several outputs
     */
    ov::Output<const ov::Node> input() const;

    /**
     * @brief Get input of executable graph
     *
     * @param i input index
     * @return Model input or throw ov::Exception if input wasn't found
     */
    ov::Output<const ov::Node> input(size_t i) const;

    /**
     * @brief Get input of executable graph
     *
     * @param tensor_name The input tensor name
     * @return Model output or throw ov::Exception if input wasn't found
     */
    ov::Output<const ov::Node> input(const std::string& tensor_name) const;

    /**
     * @brief Get outputs of executable graph model
     *
     * @return vector of outputs
     */
    std::vector<ov::Output<const ov::Node>> outputs() const;

    /**
     * @brief Get output of executable graph
     *
     * @return Model output or throw ov::Exception in case of several outputs
     */
    ov::Output<const ov::Node> output() const;

    /**
     * @brief Get output of executable graph
     *
     * @param i output index
     * @return Model output or throw ov::Exception if output wasn't found
     */
    ov::Output<const ov::Node> output(size_t i) const;

    /**
     * @brief Get output of executable graph
     *
     * @param tensor_name The output tensor name
     * @return Model output or throw ov::Exception if output wasn't found
     */
    ov::Output<const ov::Node> output(const std::string& tensor_name) const;

    /**
     * @brief Creates an inference request object used to infer the model.
     *
     * The created request has allocated input and output blobs (that can be changed later).
     *
     * @return InferRequest object
     */
    InferRequest create_infer_request();

    /**
     * @brief Exports the current executable network.
     *
     * @see Core::import_model
     *
     * @param model_stream Model output stream
     */
    void export_model(std::ostream& model_stream);

    /**
     * @brief Sets configuration for current executable network
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void set_config(const AnyMap& config);

    /**
     * @brief Sets configuration for current executable network
     *
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param configs Optional pack of pairs: (config parameter name, config parameter value)
     * @return nothing
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<void, Configs...> set_config(Configs&&... configs) {
        set_config(AnyMap{std::forward<Configs>(configs)...});
    }

    /** @brief Gets configuration for current executable network.
     *
     * The method is responsible to extract information
     * which affects executable network execution. The list of supported configuration values can be extracted via
     * CompiledModel::get_metric with the SUPPORTED_CONFIG_KEYS key, but some of these keys cannot be changed
     * dynamically, e.g. DEVICE_ID cannot changed if an executable network has already been compiled for particular
     * device.
     *
     * @param name config key, can be found in ie_plugin_config.hpp
     * @return Configuration parameter value
     */
    Any get_config(const std::string& name) const;

    /**
     * @brief Gets configuration dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via set_config method.
     *
     * @tparam T - type of returned value
     * @param key  - config key object.
     * @return Value of config corresponding to config key.
     */
    template <typename T, ConfigMutability mutability>
    util::EnableIfRaedableConfig<T, mutability> get_config(const ov::Key<T, mutability>& key) const {
        auto to = Any::make<T>();
        get_config(key.str(), to, mutability);
        return to.template as<T>();
    }

    /**
     * @brief Gets general runtime metric for an executable network.
     *
     * It can be model name, actual device ID on
     * which executable network is running or all other properties which cannot be changed dynamically.
     *
     * @param name metric name to request
     * @return Metric parameter value
     */
    Any get_metric(const std::string& name) const;

    /**
     * @brief Gets general runtime metric for dedicated hardware.
     *
     * The method is needed to request common device properties
     * which are executable network agnostic. It can be device name, temperature, other devices-specific values.
     *
     * @tparam T - type of returned value
     * @param key - metric key object.
     * @return Metric value corresponding to metric key.
     */
    template <typename T, ConfigMutability mutability>
    util::EnableIfRaedableConfig<T, mutability> get_metric(const ov::Key<T, mutability>& key) const {
        auto to = Any::make<T>();
        get_config(key.str(), to, mutability);
        return to.template as<T>();
    }

    /**
     * @brief Returns pointer to plugin-specific shared context
     * on remote accelerator device that was used to create this CompiledModel
     * @return A context
     */
    RemoteContext get_context() const;

    /**
     * @brief Checks if current CompiledModel object is not initialized
     * @return true if current CompiledModel object is not initialized, false - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current CompiledModel object is initialized
     * @return true if current CompiledModel object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept;
};

}  // namespace runtime
}  // namespace ov
