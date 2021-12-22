// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime Core class C++ API
 *
 * @file openvino/runtime/core.hpp
 */
#pragma once

#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_plugin_config.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/op.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/parameter.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/tensor.hpp"

namespace InferenceEngine {
class IExtension;
}  // namespace InferenceEngine

namespace ov {

namespace runtime {

/**
 * @brief This class represents OpenVINO runtime Core entity.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class OPENVINO_RUNTIME_API Core {
    class Impl;
    std::shared_ptr<Impl> _impl;

    void get_config(const std::string& device_name,
                    const std::string& name,
                    ov::Any& to,
                    const ConfigMutability ConfigMutability) const;

public:
    /** @brief Constructs OpenVINO Core instance using XML configuration file with
     * plugins description.
     *
     * See register_plugins for more details.
     *
     * @param xml_config_file A path to .xml file with plugins to load from. If XML configuration file is not specified,
     * then default OpenVINO Runtime plugins are loaded from the default plugin.xml file.
     */
    explicit Core(const std::string& xml_config_file = {});

    /**
     * @brief Returns plugins version information
     *
     * @param device_name Device name to identify plugin
     * @return A vector of versions
     */
    std::map<std::string, Version> get_versions(const std::string& device_name) const;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Reads models from IR and ONNX formats
     * @param model_path path to model
     * @param bin_path path to data file
     * For IR format (*.bin):
     *  * if path is empty, will try to read bin file with the same name as xml and
     *  * if bin file with the same name was not found, will load IR without weights.
     * For ONNX format (*.onnx):
     *  * bin_path parameter is not used.
     * @return Model
     */
    std::shared_ptr<ov::Model> read_model(const std::wstring& model_path, const std::wstring& bin_path = {}) const;
#endif

    /**
     * @brief Reads models from IR and ONNX formats
     * @param model_path path to model
     * @param bin_path path to data file
     * For IR format (*.bin):
     *  * if path is empty, will try to read bin file with the same name as xml and
     *  * if bin file with the same name was not found, will load IR without weights.
     * For ONNX format (*.onnx):
     *  * bin_path parameter is not used.
     * @return Model
     */
    std::shared_ptr<ov::Model> read_model(const std::string& model_path, const std::string& bin_path = {}) const;
    /**
     * @brief Reads models from IR and ONNX formats
     * @param model string with model in IR or ONNX format
     * @param weights shared pointer to constant tensor with weights
     * Reading ONNX models doesn't support loading weights from data tensors.
     * @note Created model object shares the weights with `weights` object.
     * So, do not create `weights` on temporary data which can be later freed, since the model
     * constant data becomes to point to invalid memory.
     * @return Model
     */
    std::shared_ptr<ov::Model> read_model(const std::string& model, const Tensor& weights) const;

    /**
     * @brief Creates an executable network from a model object and loads model to default device.
     *
     * Users can create as many executable networks as they need and use
     * them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model Model object acquired from Core::read_model
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model, const ParamMap& config = {});

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file to default device.
     *
     * Users can create as many executable networks as they need and use
     * them simultaneously (up to the limitation of the hardware resources)
     *
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model Model object acquired from Core::read_model
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * load operation
     *
     * @return An executable network reference
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<CompiledModel, Configs...> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                      Configs&&... configs) {
        return compile_model(model, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Creates an executable network from a model object.
     *
     * Users can create as many executable networks as they need and use
     * them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model Model object acquired from Core::read_model
     * @param device_name Name of device to load model to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model,
                                const std::string& device_name,
                                const ParamMap& config = {});

    /**
     * @brief Creates an executable network from a model object.
     *
     * Users can create as many executable networks as they need and use
     * them simultaneously (up to the limitation of the hardware resources)
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model Model object acquired from Core::read_model
     * @param device_name Name of device to load model to
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * load operation
     * @return An executable network reference
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<CompiledModel, Configs...> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                      const std::string& device_name,
                                                                      Configs&&... configs) {
        return compile_model(model, device_name, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file and load model to default device.
     *
     * This can be more efficient than using read_model + compile_model(Model) flow
     * especially for cases when caching is enabled and cached model is available
     *
     * @param model_path path to model
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation/
     *
     * @return An executable network reference
     */
    CompiledModel compile_model(const std::string& model_path, const ParamMap& config = {});

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file to default device.
     *
     * This can be more efficient than using read_model + compile_model(Model) flow
     * especially for cases when caching is enabled and cached model is available
     *
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model_path path to model
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * load operation
     *
     * @return An executable network reference
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<CompiledModel, Configs...> compile_model(const std::string& model_path,
                                                                      Configs&&... configs) {
        return compile_model(model_path, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file
     *
     * This can be more efficient than using read_model + compile_model(Model) flow
     * especially for cases when caching is enabled and cached model is available
     *
     * @param model_path path to model
     * @param device_name Name of device to load model to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation/
     *
     * @return An executable network reference
     */
    CompiledModel compile_model(const std::string& model_path,
                                const std::string& device_name,
                                const ParamMap& config = {});

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file
     *
     * This can be more efficient than using read_model + compile_model(Model) flow
     * especially for cases when caching is enabled and cached model is available
     *
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model_path path to model
     * @param device_name Name of device to load model to
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * load operation
     *
     * @return An executable network reference
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<CompiledModel, Configs...> compile_model(const std::string& model_path,
                                                                      const std::string& device_name,
                                                                      Configs&&... configs) {
        return compile_model(model_path, device_name, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Creates an executable network from a network object within a specified remote context.
     * @param model Model object acquired from Core::read_model
     * @param context Pointer to RemoteContext object
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network object
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model,
                                const RemoteContext& context,
                                const ParamMap& config = {});

    /**
     * @brief Creates an executable network from a network object within a specified remote context.
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model Model object acquired from Core::read_model
     * @param context Pointer to RemoteContext object
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * load operation
     * @return An executable network object
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<CompiledModel, Configs...> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                      const RemoteContext& context,
                                                                      Configs&&... configs) {
        return compile_model(model, context, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Registers extension
     * @deprecated This method is deprecated. Please use other add_extension methods
     * @param extension Pointer to already loaded extension
     */
    OPENVINO_DEPRECATED("Please use add_extension(ov::Extension) or add_extension(path_to_library) instead.")
    void add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension);

    /**
     * @brief Registers extension
     * @param library_path path to library with ov::Extension
     */
    void add_extension(const std::string& library_path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Registers extension
     * @param library_path path to library with ov::Extension
     */
    void add_extension(const std::wstring& library_path);
#endif

    /**
     * @brief Registers extension
     * @param extension Pointer to base extension
     */
    void add_extension(const std::shared_ptr<ov::Extension>& extension);

    /**
     * @brief Registers extensions
     * @param extensions Vector of loaded base extensions
     */
    void add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions);

    /**
     * @brief Registers extension
     * @param extension Extension class which is inherited from ov::Extension class
     */
    template <class T, typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
    }

    /**
     * @brief Registers extensions
     * @param extension Extension class which is inherited from ov::Extension class
     * @param args list of extensions
     */
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension, Targs... args) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
        add_extension(args...);
    }

    /**
     * @brief Registers custom operation
     */
    template <class T, typename std::enable_if<std::is_base_of<ov::op::Op, T>::value, bool>::type = true>
    void add_extension() {
        std::shared_ptr<ov::Extension> ext = std::make_shared<ov::OpExtension<T>>();
        add_extension(ext);
    }

    /**
     * @brief Registers custom operations
     */
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::op::Op, T>::value && sizeof...(Targs), bool>::type = true>
    void add_extension() {
        std::shared_ptr<ov::Extension> ext = std::make_shared<ov::OpExtension<T>>();
        add_extension(ext);
        if (sizeof...(Targs) > 0)
            add_extension<Targs...>();
    }

    /**
     * @brief Creates an executable network from a previously exported one
     * @param model_stream Model stream
     * @param device_name Name of device load executable network on
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return An executable network reference
     */
    CompiledModel import_model(std::istream& model_stream, const std::string& device_name, const ParamMap& config = {});

    /**
     * @brief Creates an executable network from a previously exported one
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model_stream Model stream
     * @param device_name Name of device load executable network on
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * load operation
     * @return An executable network reference
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<CompiledModel, Configs...> import_model(std::istream& model_stream,
                                                                     const std::string& device_name,
                                                                     Configs&&... configs) {
        return import_model(model_stream, device_name, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Creates an executable network from a previously exported one within a specified
     * remote context.
     *
     * @param model_stream Model stream
     * @param context Pointer to RemoteContext object
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    CompiledModel import_model(std::istream& model_stream, const RemoteContext& context, const ParamMap& config = {});

    /**
     * @brief Creates an executable network from a previously exported one within a specified
     * remote context.
     *
     * @param model_stream Model stream
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param context Pointer to RemoteContext object
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * load operation
     * @return An executable network reference
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<CompiledModel, Configs...> import_model(std::istream& model_stream,
                                                                     const RemoteContext& context,
                                                                     Configs&&... configs) {
        return import_model(model_stream, context, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Query device if it supports specified model with specified configuration
     *
     * @param device_name A name of a device to query
     * @param model Model object to query
     * @param config Optional map of pairs: (config parameter name, config parameter value)
     * @return An object containing a map of pairs a operation name -> a device name supporting this operation.
     */
    SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                const std::string& device_name,
                                const ParamMap& config = {}) const;

    /**
     * @brief Query device if it supports specified model with specified configuration
     *
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param device_name A name of a device to query
     * @param model Model object to query
     * @param configs Optional pack of pairs: (config parameter name, config parameter value) relevant only for this
     * query operation
     * @return An object containing a map of pairs a operation name -> a device name supporting this operation.
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<SupportedOpsMap, Configs...> query_model(const std::shared_ptr<const ov::Model>& model,
                                                                      const std::string& device_name,
                                                                      Configs&&... configs) const {
        return query_model(model, device_name, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Sets configuration for all the
     * registered devices, acceptable keys can be found in ie_plugin_config.hpp
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void set_config(const ParamMap& config);

    /**
     * @brief Sets configuration for all the
     * registered devices, acceptable keys can be found in ie_plugin_config.hpp
     *
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param configs Optional pack of pairs: (config parameter name, config parameter value)
     * @return nothing
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<void, Configs...> set_config(Configs&&... configs) {
        set_config(ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Sets configuration for device, acceptable keys can be found in ie_plugin_config.hpp
     *
     * @param device_name An name of a device.
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void set_config(const std::string& device_name, const ParamMap& config);

    /**
     * @brief Sets configuration for device, acceptable keys can be found in ie_plugin_config.hpp
     *
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param device_name An name of a device.
     * @param configs Optional pack of pairs: (config parameter name, config parameter value)
     * @return nothing
     */
    template <typename... Configs>
    util::EnableIfAllConfigs<void, Configs...> set_config(const std::string& device_name, Configs&&... configs) {
        set_config(device_name, ParamMap{std::forward<Configs>(configs)...});
    }

    /**
     * @brief Gets configuration dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via set_config method.
     *
     * @param device_name  - A name of a device to get a configuration value.
     * @param name  - config key.
     * @return Value of config corresponding to config key.
     */
    Any get_config(const std::string& device_name, const std::string& name) const;

    /**
     * @brief Gets configuration dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via set_config method.
     *
     * @tparam T - type of returned value
     * @param deviceName  - A name of a device to get a configuration value.
     * @param key  - config key.
     * @return Value of config corresponding to config key.
     */
    template <typename T, ConfigMutability mutability>
    util::EnableIfRaedableConfig<T, mutability> get_config(const std::string& deviceName,
                                                           const ov::Key<T, mutability>& key) const {
        auto to = Any::make<T>();
        get_config(deviceName, key.str(), to, mutability);
        return to.template as<T>();
    }

    /**
     * @brief Gets general runtime metric for dedicated hardware.
     *
     * The method is needed to request common device properties.
     * It can be device name, temperature, other devices-specific values.
     *
     * @param device_name - A name of a device to get a metric value.
     * @param name - metric name to request.
     * @return Metric value corresponding to metric key.
     */
    Any get_metric(const std::string& device_name, const std::string& name) const;

    /**
     * @brief Gets general runtime metric for dedicated hardware.
     *
     * The method is needed to request common device properties
     * which are executable network agnostic. It can be device name, temperature, other devices-specific values.
     *
     * @tparam T - type of returned value
     * @tparam mutability - mutability of configuration
     * @param deviceName - A name of a device to get a metric value.
     * @param key - metric key to request.
     * @return Metric value corresponding to metric key.
     */
    template <typename T, ConfigMutability mutability>
    util::EnableIfRaedableConfig<T, mutability> get_metric(const std::string& deviceName,
                                                           const ov::Key<T, mutability>& key) const {
        auto to = Any::make<T>();
        get_config(deviceName, key.str(), to, mutability);
        return to.template as<T>();
    }

    /**
     * @brief Returns devices available for inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    std::vector<std::string> get_available_devices() const;

    /**
     * @brief Register new device and plugin which implement this device inside OpenVINO Runtime.
     *
     * @param plugin_name A name of plugin. Depending on platform `plugin_name` is wrapped with shared library suffix
     * and prefix to identify library full name
     *
     * @param device_name A device name to register plugin for. If device name is not specified, then it's taken from
     * plugin itself.
     */
    void register_plugin(const std::string& plugin_name, const std::string& device_name);

    /**
     * @brief Unloads previously loaded plugin with a specified name from OpenVINO Runtime
     * The method is needed to remove plugin instance and free its resources. If plugin for a
     * specified device has not been created before, the method throws an exception.
     *
     * @param device_name Device name identifying plugin to remove from OpenVINO Runtime
     */
    void unload_plugin(const std::string& device_name);

    /** @brief Registers plugin to OpenVINO Runtime Core instance using XML configuration file with
     * plugins description.
     *
     *  XML file has the following structure:
     *
     * ```xml
     * <ie>
     *     <plugins>
     *         <plugin name="" location="">
     *             <extensions>
     *                 <extension location=""/>
     *             </extensions>
     *             <properties>
     *                 <property key="" value=""/>
     *             </properties>
     *         </plugin>
     *     </plugins>
     * </ie>
     * ```
     *
     * - `name` identifies name of device enabled by plugin
     * - `location` specifies absolute path to dynamic library with plugin. A path can also be relative to inference
     * engine shared library. It allows to have common config for different systems with different configurations.
     * - Properties are set to plugin via the `set_config` method.
     * - Extensions are set to plugin via the `add_extension` method.
     *
     * @param xml_config_file A path to .xml file with plugins to register.
     */
    void register_plugins(const std::string& xml_config_file);

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param device_name Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    RemoteContext create_context(const std::string& device_name, const ParamMap& params);

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param device_name Name of a device to create new shared context on.
     * @param params Pack of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    template <typename... Params>
    util::EnableIfAllConfigs<RemoteContext, Params...> create_context(const std::string& device_name,
                                                                      Params&&... params) {
        return create_context(device_name, ParamMap{std::forward<Params>(params)...});
    }

    /**
     * @brief Get a pointer to default(plugin-supplied) shared context object for specified accelerator device.
     * @param device_name  - A name of a device to get create shared context from.
     * @return A shared pointer to a default remote context.
     */
    RemoteContext get_default_context(const std::string& device_name);
};
}  // namespace runtime
}  // namespace ov
