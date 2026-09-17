#pragma once

#include <torch/extension.h>
#include <atomic>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <set>
#include <string>
#include <unordered_map>

/**
 * @file device.h
 * @brief Singleton backend manager for Intel NPU acceleration via oneAPI Level Zero.
 */

/**
 * @class NPUBackend
 * @brief Singleton class managing OpenVINO Core, Level Zero ZeroContext, and thread-safe Model Cache.
 *
 * Provides lifecycle management for Intel NPU hardware acceleration:
 * - Direct Level Zero `ze_context_handle_t` wrapping (`ov::intel_npu::level_zero::ZeroContext`).
 * - Thread-safe LRU in-memory compiled model and infer request cache.
 * - Hardware property probing (`NPU_TURBO`, `NPU_DISABLE_IDLE_MEMORY_PRUNING`, `NPU_RUN_INFERENCES_SEQUENTIALLY`).
 * - Zero-copy memory binding from PyTorch tensor memory pointers (`data_ptr()`).
 */
class NPUBackend {
public:
    /**
     * @brief Get the global singleton instance of NPUBackend.
     * @return Reference to the singleton NPUBackend.
     */
    static NPUBackend& getInstance();

    // Non-copyable and non-assignable
    NPUBackend(const NPUBackend&) = delete;
    void operator=(const NPUBackend&) = delete;

    /**
     * @brief Retrieve the global OpenVINO Core instance.
     */
    ov::Core& getCore();

    /**
     * @brief Retrieve the global OpenVINO RemoteContext.
     */
    ov::RemoteContext& getContext();

    /**
     * @brief Retrieve the native Intel Level Zero ZeroContext wrapper.
     * @throws std::runtime_error if ZeroContext is uninitialized.
     */
    ov::intel_npu::level_zero::ZeroContext& getZeroContext();

    /**
     * @brief Check whether native Level Zero ZeroContext initialization succeeded.
     */
    bool hasZeroContext() const;

    /**
     * @brief Retrieve a compiled model from cache, or compile and cache it.
     * @param key Unique cache key identifier.
     * @param model OpenVINO model topology.
     * @return Compiled model executable.
     */
    ov::CompiledModel getOrCompileModel(const std::string& key, std::shared_ptr<ov::Model> model);

    /**
     * @brief Retrieve or construct a cached InferRequest for the given model key.
     * @param key Unique cache key identifier.
     * @return OpenVINO InferRequest handle.
     */
    ov::InferRequest getOrCachedInferRequest(const std::string& key);

    /**
     * @brief Configure persistent disk cache directory for compiled binary blobs.
     * @param path Absolute directory path.
     */
    void setCacheDir(const std::string& path);

    /**
     * @brief Purge all in-memory compiled models and infer requests.
     */
    void clearCache();

    /**
     * @brief Set a driver-level configuration property key-value pair.
     */
    void setProperty(const std::string& key, const std::string& value);

    /**
     * @brief Set performance hint ("LATENCY", "THROUGHPUT", or "CUMULATIVE_THROUGHPUT").
     */
    void setPerformanceHint(const std::string& hint);

    /**
     * @brief Configure execution device for eager ops ("NPU" or "CPU").
     */
    void setEagerDevice(const std::string& device);

    /**
     * @brief Query whether a specific hardware property key is supported by the NPU driver.
     */
    bool isPropertySupported(const std::string& key) const;

    /**
     * @brief Retrieve a cache-key suffix describing the active compile configuration.
     *
     * Eager binaries are cached by op/shape/dtype; the performance hint and
     * eager device also affect compilation, so they must participate in the
     * key to avoid reusing a stale binary after a hint change.
     */
    std::string getCompileConfigKey() const;

    /**
     * @brief Retrieve the configured eager execution device ("NPU" or "CPU").
     */
    std::string getEagerDevice() const;

    /**
     * @brief Retrieve a comma-separated list of all driver-supported property keys.
     */
    std::string getSupportedPropertiesList() const;

    /**
     * @brief Get the monotonic cache mutation version counter.
     */
    uint64_t getCacheVersion() const;

    /**
     * @brief Log formatted messages with the [Intel NPU] prefix.
     */
    template <typename... Args>
    void log(const std::string& fmt, Args... args) {
        std::cout << "[Intel NPU] " << fmt << "\n";
    }

    /**
     * @brief Check if an Intel NPU device is detected and accessible.
     */
    bool isAvailable();

private:
    NPUBackend();  ///< Private constructor for singleton
    ~NPUBackend() = default;

    struct CacheEntry {
        ov::CompiledModel compiled_model;
        std::list<std::string>::iterator list_it;

        CacheEntry() = default;
        CacheEntry(ov::CompiledModel m, std::list<std::string>::iterator it)
            : compiled_model(std::move(m)), list_it(it) {
        }
    };

    std::unique_ptr<ov::Core> m_core;
    ov::RemoteContext m_context;
    std::unique_ptr<ov::intel_npu::level_zero::ZeroContext> m_zero_context;
    std::map<std::string, CacheEntry> m_model_cache;
    std::list<std::string> m_access_order;
    const size_t m_max_cache_size = 200;

    mutable std::mutex m_mutex;
    bool m_is_available;
    bool m_has_context = false;               ///< True when RemoteContext init succeeded
    bool m_has_zero_context = false;          ///< True when ZeroContext init succeeded
    std::string m_eager_device = "CPU";       ///< Target device for eager ops (CPU fallback for tiny ops)
    std::set<std::string> m_supported_props;  ///< Populated by probeProperties()
    ov::hint::PerformanceMode m_performance_hint = ov::hint::PerformanceMode::LATENCY;
    std::atomic<uint64_t> m_cache_version{0};

    void probeProperties();  ///< Queries ov::supported_properties at initialization
};

// --- C-API Wrappers for PyBind11 Python Bindings ---
bool is_npu_available();
void initialize_npu();
void set_npu_cache_dir(const std::string& path);
void set_npu_property(const std::string& key, const std::string& value);
void set_npu_performance_hint(const std::string& hint);
void set_npu_eager_device(const std::string& device);
void set_npu_turbo(bool enable);
void set_npu_sda(bool enable);
void clear_cpp_model_cache();