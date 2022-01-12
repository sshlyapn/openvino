// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for OpenVINO runtime plugins
 *        To use in set_config, compile_model, import_model methods
 *
 * @file openvino/runtime/config.hpp
 */
#pragma once

#include <istream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "ie_precision.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {

/**
 * @brief Enum to define configuration value mutability
 */
enum class ConfigMutability {
    RO,  //!< Read only configuration values can not be passed as input parameter
    RW,  //!< Read/Write configuration key may change readability in runtime
    WO,  //!< Write only configuration values could not be obtained from openvino objects
};

/** @cond INTERNAL */
namespace util {
template <typename... Args>
struct AllConfigs;

template <typename T, typename... Args>
struct AllConfigs<T, Args...> {
    constexpr static const bool value =
        std::is_convertible<T, std::pair<std::string, ov::Any>>::value && AllConfigs<Args...>::value;
};

template <typename T>
struct AllConfigs<T> {
    constexpr static const bool value = std::is_convertible<T, std::pair<std::string, ov::Any>>::value;
};

template <typename T, typename... Args>
using EnableIfAllConfigs = typename std::enable_if<AllConfigs<Args...>::value, T>::type;

template <typename T, ConfigMutability mutability>
using EnableIfRaedableConfig =
    typename std::enable_if<mutability == ConfigMutability::RO || mutability == ConfigMutability::RW, T>::type;

/**
 * @brief This class us used to bind configuration key string with paramter type
 * @tparam T type of parameter used to pass or get configuration
 */
template <typename T, ConfigMutability mutability_ = ConfigMutability::RW>
struct BaseKey {
    using value_type = T;                                  //!< Configuration parameter type
    constexpr static const auto mutability = mutability_;  //!< Configuration parameter readability

    /**
     * @brief Constructs configuration key variable
     * @param str_ configuration key string
     */
    constexpr BaseKey(const char* str_) : _str{str_} {}

    /**
     * @brief return configuration key string
     * @return Pointer to const string key representation
     */
    const char* str() const {
        return _str;
    }

    /**
     * @brief compares configuration key string
     * @return true if string is the same
     */
    bool operator==(const std::string& str) const {
        return _str == str;
    }

private:
    const char* _str = nullptr;
};
template <typename T, ConfigMutability M>
inline std::ostream& operator<<(std::ostream& os, const BaseKey<T, M>& key) {
    return os << key.str();
}
}  // namespace util
/** @endcond */

/**
 * @brief This class us used to bind configuration key string with paramter type
 * @tparam T type of parameter used to pass or get configuration
 */
template <typename T, ConfigMutability mutability_ = ConfigMutability::RW>
struct Key : public util::BaseKey<T, mutability_> {
    using util::BaseKey<T, mutability_>::BaseKey;

    struct Pair : std::pair<std::string, Any> {
        using std::pair<std::string, Any>::pair;
#ifdef OPENVINO_DEV
        operator Any() && {
            return std::move(second);
        }
#endif
    };

    /**
     * @brief Constructs configuration
     * @tparam Args configuration paramter constructor arguments types
     * @param args configuration paramter constructor arguments
     * @return Pair of string key representation and type erased paramter.
     */
    template <typename... Args>
    inline Pair operator()(Args&&... args) const {
        return {this->str(), Any::make<T>(std::forward<Args>(args)...)};
    }
};

/**
 * @brief This class us used to bind configuration key string with read only paramter type
 * @tparam T type of parameter used to pass or get configuration
 */
template <typename T>
struct Key<T, ConfigMutability::RO> : public util::BaseKey<T, ConfigMutability::RO> {
    using util::BaseKey<T, ConfigMutability::RO>::BaseKey;
/** @cond INTERNAL */
#ifdef OPENVINO_DEV
    template <typename... Args>
    inline Any operator()(Args&&... args) const {
        return Any::make<T>(std::forward<Args>(args)...);
    }
#endif
    /** @endcond */
};

class KeyVector : public Key<std::vector<std::string>, ConfigMutability::RO> {
    static const char* get_str(const char* str) {
        return str;
    }
    static const char* get_str(const std::string& str) {
        return str.c_str();
    }
    static const std::vector<std::string>& get_str(const std::vector<std::string>& arg) {
        return arg;
    }
    template <typename T, ConfigMutability M>
    static const char* get_str(const Key<T, M>& key) {
        return key.str();
    }

public:
    using Key<std::vector<std::string>, ConfigMutability::RO>::Key;
/** @cond INTERNAL */
#ifdef OPENVINO_DEV
    template <typename... Args>
    inline Any operator()(Args&&... args) const {
        return std::vector<std::string>{get_str(std::forward<Args>(args))...};
    }
#endif
    /** @endcond */
};

/**
 * @brief Metric to get a std::vector<std::string> of available device IDs
 */
static constexpr KeyVector available_devices{"AVAILABLE_DEVICES"};

/**
 * @brief Metric to get a std::vector<std::string> of supported metrics.
 *
 * This can be used as an executable network metric as well.
 *
 * Each of the returned device metrics can be passed to Core::GetMetric, executable network metrics
 * can be passed to ExecutableNetwork::GetMetric.
 *
 */
static constexpr KeyVector supported_metrics{"SUPPORTED_METRICS"};

/**
 * @brief Metric to get a std::vector<std::string> of supported config keys.
 *
 * This can be used as an executable network metric as well.
 *
 * Each of the returned device configuration keys can be passed to Core::SetConfig, Core::GetConfig, and
 * Core::LoadNetwork, configuration keys for executable networks can be passed to ExecutableNetwork::SetConfig and
 * ExecutableNetwork::GetConfig.
 *
 */
static constexpr KeyVector supported_config_keys{"SUPPORTED_CONFIG_KEYS"};

/**
 * @brief Metric to get a std::vector<std::string> of optimization options per device.
 */
static constexpr KeyVector optimization_capabilities{"OPTIMIZATION_CAPABILITIES"};

/**
 * @brief Metric which defines support of import/export functionality by plugin
 */
static constexpr Key<bool, ConfigMutability::RO> import_export_support{"IMPORT_EXPORT_SUPPORT"};

/**
 * @brief Metric to get a name of model_name
 */
static constexpr Key<std::string, ConfigMutability::RO> model_name{"NETWORK_NAME"};

/**
 * @brief Metric to get an unsigned integer value of optimal number of executable network infer requests.
 */
static constexpr Key<uint32_t, ConfigMutability::RO> optimal_number_of_infer_requests{
    "OPTIMAL_NUMBER_OF_INFER_REQUESTS"};

namespace hint {

/**
 * @brief Enum to define possible performance measure hints
 */
enum class ModelPriority {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 3,
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ModelPriority& model_priority) {
    switch (model_priority) {
    case ModelPriority::LOW:
        return os << "LOW";
    case ModelPriority::MEDIUM:
        return os << "MEDIUM";
    case ModelPriority::HIGH:
        return os << "HIGH";
    default:
        throw ov::Exception{"Unsupported performance measure hint"};
    }
}

inline std::istream& operator>>(std::istream& is, ModelPriority& model_priority) {
    std::string str;
    is >> str;
    if (str == "LOW") {
        model_priority = ModelPriority::LOW;
    } else if (str == "MEDIUM") {
        model_priority = ModelPriority::MEDIUM;
    } else if (str == "HIGH") {
        model_priority = ModelPriority::HIGH;
    } else {
        throw ov::Exception{"Unsupported model priority: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief High-level OpenVINO model priority hint
 * Defines what model should be provided with more performant bounded resource first
 */
static constexpr Key<ModelPriority> model_priority{"MODEL_PRIORITY"};

/**
 * @brief Enum to define possible performance mode hints
 */
enum class PerformanceMode {
    LATENCY = 0,
    THROUGHPUT = 1,
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const PerformanceMode& performance_mode) {
    switch (performance_mode) {
    case PerformanceMode::LATENCY:
        return os << "LATENCY";
    case PerformanceMode::THROUGHPUT:
        return os << "THROUGHPUT";
    default:
        throw ov::Exception{"Unsupported performance measure hint"};
    }
}

inline std::istream& operator>>(std::istream& is, PerformanceMode& performance_mode) {
    std::string str;
    is >> str;
    if (str == "LATENCY") {
        performance_mode = PerformanceMode::LATENCY;
    } else if (str == "THROUGHPUT") {
        performance_mode = PerformanceMode::THROUGHPUT;
    } else {
        throw ov::Exception{"Unsupported performance measure: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief High-level OpenVINO Performance Hints
 * unlike low-level config keys that are individual (per-device), the hints are somthing that every device accepts
 * and turns into device-specific settings
 */
static constexpr Key<PerformanceMode> performance_mode{"PERFORMANCE_HINT"};

/**
 * @brief (Optional) config key that backs the (above) Performance Hints
 * by giving additional information on how many inference requests the application will be keeping in flight
 * usually this value comes from the actual use-case (e.g. number of video-cameras, or other sources of inputs)
 */
static constexpr Key<uint32_t> num_requests{"PERFORMANCE_HINT_NUM_REQUESTS"};

/**
 * @brief Enum to define possible performance hints
 */
enum class CalculationMode {
    PERFORMANCE = 0,
    ACCURACY = 1,
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const CalculationMode& calculation_mode) {
    switch (calculation_mode) {
    case CalculationMode::PERFORMANCE:
        return os << "PERFORMANCE";
    case CalculationMode::ACCURACY:
        return os << "ACCURACY";
    default:
        throw ov::Exception{"Unsupported calculation mode hint"};
    }
}

inline std::istream& operator>>(std::istream& is, CalculationMode& calculation_mode) {
    std::string str;
    is >> str;
    if (str == "PERFORMANCE") {
        calculation_mode = CalculationMode::PERFORMANCE;
    } else if (str == "ACCURACY") {
        calculation_mode = CalculationMode::ACCURACY;
    } else {
        throw ov::Exception{"Unsupported calculation mode: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief Defines runtime precision mode
 */
static constexpr Key<element::Type> precision_mode{"PRECISION_MODE"};

}  // namespace hint

/**
 * @brief The name for setting performance counters option.
 *
 * It is passed to Core::SetConfig(), this option should be used with values:
 * ov::runtime::config::value::yes or ov::runtime::config::value::no
 */
static constexpr Key<bool> enable_profiling{"PERF_COUNT"};

namespace log {

/**
 * @brief Enum to define possible affinity template hints
 */
enum class Level {
    NO = -1,      //!< disable any loging
    ERR = 0,      //!< error events that might still allow the application to continue running
    WARNING = 1,  //!< potentially harmful situations which may further lead to ERROR
    INFO = 2,     //!< informational messages that display the progress of the application at coarse-grained level
    DEBUG = 3,    //!< fine-grained events that are most useful to debug an application.
    TRACE = 4,    //!< finer-grained informational events than the DEBUG
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Level& level) {
    switch (level) {
    case Level::NO:
        return os << "NO";
    case Level::ERR:
        return os << "LOG_ERROR";
    case Level::WARNING:
        return os << "LOG_WARNING";
    case Level::INFO:
        return os << "LOG_INFO";
    case Level::DEBUG:
        return os << "LOG_DEBUG";
    case Level::TRACE:
        return os << "LOG_TRACE";
    default:
        throw ov::Exception{"Unsupported log level"};
    }
}

inline std::istream& operator>>(std::istream& is, Level& level) {
    std::string str;
    is >> str;
    if (str == "NO") {
        level = Level::NO;
    } else if (str == "LOG_ERROR") {
        level = Level::ERR;
    } else if (str == "LOG_WARNING") {
        level = Level::WARNING;
    } else if (str == "LOG_INFO") {
        level = Level::INFO;
    } else if (str == "LOG_DEBUG") {
        level = Level::DEBUG;
    } else if (str == "LOG_TRACE") {
        level = Level::TRACE;
    } else {
        throw ov::Exception{"Unsupported log level: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief the key for setting desirable log level.
 */
static constexpr Key<Level> level{"LOG_LEVEL"};
}  // namespace log

/**
 * @brief This key defines the directory which will be used to store any data cached by plugins.
 *
 * The underlying cache structure is not defined and might differ between OpenVINO releases
 * Cached data might be platform / device specific and might be invalid after OpenVINO version change
 * If this key is not specified or value is empty string, then caching is disabled.
 * The key might enable caching for the plugin using the following code:
 *
 * @code
 * ie.set_config("GPU", ov::cache_dir("cache/")); // enables cache for GPU plugin
 * @endcode
 *
 * The following code enables caching of compiled network blobs for devices where import/export is supported
 *
 * @code
 * ie.set_config(ov::cache_dir("cache/")); // enables models cache
 * @endcode
 */
static constexpr Key<std::string> cache_dir{"CACHE_DIR"};

namespace device {

/**
 * @brief the key for setting of required device to execute on
 * values: device id starts from "0" - first device, "1" - second device, etc
 */
static constexpr Key<std::string> id{"DEVICE_ID"};

/**
 * @brief Type for device Priorities config option, with comma-separated devices listed in the desired priority
 */
struct Priorities : public Key<std::string> {
private:
    template <typename H, typename... T>
    static inline std::string concat(const H& head, T&&... tail) {
        return head + std::string{','} + concat(std::forward<T>(tail)...);
    }

    template <typename H>
    static inline std::string concat(const H& head) {
        return head;
    }

public:
    using Key<std::string>::Key;

    /**
     * @brief Constructs device priorities
     * @tparam Args configuration paramter constructor arguments types
     * @param args configuration paramter constructor arguments
     * @return Pair of string key representation and type erased paramter.
     */
    template <typename... Args>
    inline std::pair<std::string, Any> operator()(Args&&... args) const {
        return {str(), Any{concat(std::forward<Args>(args)...)}};
    }
};

/**
 * @brief Device Priorities config option, with comma-separated devices listed in the desired priority
 */
static constexpr Priorities priorities{"MULTI_DEVICE_PRIORITIES"};

/**
 * @brief Type for key to pass set of configuration values to specified device
 */
struct Config {
    /**
     * @brief Constructs configuration
     * @param device_name device plugin alias
     * @param config set of configuration values with names
     * @return Pair of string key representation and type erased paramter.
     */
    inline std::pair<std::string, Any> operator()(const std::string& device_name, const AnyMap& config) const {
        return {device_name, config};
    }

    /**
     * @brief Constructs configuration
     * @tparam Configs Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param device_name device plugin alias
     * @param configs Optional pack of pairs: (config parameter name, config parameter value)
     * @return Pair of string key representation and type erased paramter.
     */
    template <typename... Configs>
    inline util::EnableIfAllConfigs<std::pair<std::string, Any>, Configs...> operator()(const std::string& device_name,
                                                                                        Configs&&... configs) const {
        return {device_name, AnyMap{std::pair<std::string, Any>{configs...}}};
    }
};

/**
 * @brief Key to pass set of configuration values to specified device
 */
static constexpr Config config;

/**
 * @brief Metric to get a std::string value representing a full device name.
 */
static constexpr Key<std::string, ConfigMutability::RO> full_name{"FULL_DEVICE_NAME"};

/**
 * @brief Metric which defines the device architecture.
 */
static constexpr Key<std::string, ConfigMutability::RO> architecture{"DEVICE_ARCHITECTURE"};

/**
 * @brief Enum to define possible device types
 */
enum class Type {
    INTEGRATED = 0,  //!<  Device is integrated into host system
    DISCRETE = 1,    //!<  Device is not integrated into host system
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Type& device_type) {
    switch (device_type) {
    case Type::DISCRETE:
        return os << "discrete";
    case Type::INTEGRATED:
        return os << "integrated";
    default:
        throw ov::Exception{"Unsupported device separability type"};
    }
}

inline std::istream& operator>>(std::istream& is, Type& device_type) {
    std::string str;
    is >> str;
    if (str == "discrete") {
        device_type = Type::DISCRETE;
    } else if (str == "integrated") {
        device_type = Type::INTEGRATED;
    } else {
        throw ov::Exception{"Unsupported device type: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief Metric to get a type of device. See Type enum definition for possible return values
 */
static constexpr Key<Type, ConfigMutability::RO> type{"DEVICE_TYPE"};

/**
 * @brief Metric which defines Giga OPS per second count (GFLOPS or GIOPS) for a set of precisions supported by
 * specified device
 */
static constexpr Key<std::map<ie::Precision, float>, ConfigMutability::RO> gops{"DEVICE_GOPS"};

/**
 * @brief  Metric to get a float of device thermal
 */
static constexpr Key<float, ConfigMutability::RO> thermal{"DEVICE_THERMAL"};

}  // namespace device

/**
 * @brief The key for enabling of dumping the topology with details of layers and details how
 * this network would be executed on different devices to the disk in GraphViz format.
 */
static constexpr Key<bool, ConfigMutability::RW> dump_graph_dot{"HETERO_DUMP_GRAPH_DOT"};

/**
 * @brief The key with the list of device targets used to fallback unsupported layers
 * by HETERO plugin
 */
static constexpr device::Priorities target_fallback{"TARGET_FALLBACK"};

namespace execution {

/**
 * @brief The number of executor logical partitions
 */
static constexpr Key<int32_t, ConfigMutability::RW> streams{"STREAMS"};

/**
 * @brief Maximum number of concurent tasks executed by execututor
 */
static constexpr Key<int32_t, ConfigMutability::RW> concurrency{"CONCURRENCY"};

/**
 * @brief Enum to define possible affinity patterns
 */
enum class Affinity {
    NO = -1,   //!<  Disable threads affinity pinning
    CORE = 0,  //!<  Pin threads to cores, best for static benchmarks
    NUMA = 1,  //!<  Pin threads to NUMA nodes, best for real-life, contented cases. On the Windows and MacOS* this
               //!<  option behaves as CORE
    HYBRID_AWARE = 2,  //!< Let the runtime to do pinning to the cores types, e.g. prefer the "big" cores for latency
                       //!< tasks. On the hybrid CPUs this option is default
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Affinity& affinity) {
    switch (affinity) {
    case Affinity::NO:
        return os << "NO";
    case Affinity::CORE:
        return os << "CORE";
    case Affinity::NUMA:
        return os << "NUMA";
    case Affinity::HYBRID_AWARE:
        return os << "HYBRID_AWARE";
    default:
        throw ov::Exception{"Unsupported affinity pattern"};
    }
}

inline std::istream& operator>>(std::istream& is, Affinity& affinity) {
    std::string str;
    is >> str;
    if (str == "NO") {
        affinity = Affinity::NO;
    } else if (str == "CORE") {
        affinity = Affinity::CORE;
    } else if (str == "NUMA") {
        affinity = Affinity::NUMA;
    } else if (str == "HYBRID_AWARE") {
        affinity = Affinity::HYBRID_AWARE;
    } else {
        throw ov::Exception{"Unsupported affinity pattern: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief The name for setting CPU affinity per thread option.
 * @note The setting is ignored, if the OpenVINO compiled with OpenMP and any affinity-related OpenMP's
 * environment variable is set (as affinity is configured explicitly)
 */
static constexpr Key<Affinity> affinity{"AFFINITY"};
}  // namespace execution
}  // namespace ov
