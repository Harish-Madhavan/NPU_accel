#pragma once
#include <torch/extension.h>

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <set>
#include <string>
#include <unordered_map>

/**
 * NPUBackend: Singleton class to manage OpenVINO Core and Model Cache.
 * Ensures thread safety and proper resource management.
 */
class NPUBackend {
public:
    static NPUBackend& getInstance();

    // Delete copy constructor and assignment operator
    NPUBackend(const NPUBackend&) = delete;
    void operator=(const NPUBackend&) = delete;

    // Accessors
    ov::Core& getCore();
    ov::RemoteContext& getContext();

    // Model Caching
    ov::CompiledModel getOrCompileModel(const std::string& key, std::shared_ptr<ov::Model> model);
    ov::InferRequest getOrCachedInferRequest(const std::string& key);
    void setCacheDir(const std::string& path);
    void setProperty(const std::string& key, const std::string& value);
    void setPerformanceHint(const std::string& hint);
    void setEagerDevice(const std::string& device);

    // Property probing
    bool isPropertySupported(const std::string& key) const;
    std::string getSupportedPropertiesList() const;

    // Logging
    template <typename... Args>
    void log(const std::string& fmt, Args... args) {
        // Simple logger for now, can be replaced with spdlog later
        // Use a lock if writing to shared stream
        std::cout << "[Intel NPU] " << fmt << "\n";
    }

    // Utilities
    bool isAvailable();

private:
    NPUBackend();  // Private constructor
    ~NPUBackend() = default;

    struct CacheEntry {
        ov::CompiledModel compiled_model;
        std::list<std::string>::iterator list_it;

        CacheEntry() = default;
        CacheEntry(ov::CompiledModel m, std::list<std::string>::iterator it)
            : compiled_model(std::move(m)), list_it(it) {
        }
        // NOTE: InferRequest is NOT cached here.
        // A fresh InferRequest is created per call (see getOrCachedInferRequest)
        // to avoid shared-mutable-state races on concurrent inference.
    };

    std::unique_ptr<ov::Core> m_core;
    ov::RemoteContext m_context;
    std::map<std::string, CacheEntry> m_model_cache;
    std::list<std::string> m_access_order;
    const size_t m_max_cache_size = 200;

    std::mutex m_mutex;
    bool m_is_available;
    bool m_has_context = false;               // True when RemoteContext init succeeded
    std::string m_eager_device = "CPU";       // Default to CPU for tiny ops
    std::set<std::string> m_supported_props;  // Populated by probeProperties()
    ov::hint::PerformanceMode m_performance_hint = ov::hint::PerformanceMode::LATENCY;

    void probeProperties();  // Queries ov::supported_properties at init
    };

    // C-API wrappers for Python bindings
    bool is_npu_available();
    void initialize_npu();
    void set_npu_cache_dir(const std::string& path);
    void set_npu_property(const std::string& key, const std::string& value);
    void set_npu_performance_hint(const std::string& hint);
    void set_npu_eager_device(const std::string& device);