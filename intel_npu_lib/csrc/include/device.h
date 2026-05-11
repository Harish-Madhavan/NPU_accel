#pragma once
#include <torch/extension.h>
#include <openvino/openvino.hpp>
#include <string>
#include <memory>
#include <map>
#include <list>
#include <mutex>
#include <iostream>

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
    
    // Model Caching
    ov::CompiledModel getOrCompileModel(const std::string& key, std::shared_ptr<ov::Model> model);
    void setCacheDir(const std::string& path);

    // Logging
    template<typename... Args>
    void log(const std::string& fmt, Args... args) {
        // Simple logger for now, can be replaced with spdlog later
        // Use a lock if writing to shared stream
        std::cout << "[Intel NPU] " << fmt << "\n";
    }

    // Utilities
    bool isAvailable();

private:
    NPUBackend(); // Private constructor
    ~NPUBackend() = default;

    struct CacheEntry {
        ov::CompiledModel compiled_model;
        std::list<std::string>::iterator list_it;

        CacheEntry() = default;
        CacheEntry(ov::CompiledModel m, std::list<std::string>::iterator it)
            : compiled_model(m), list_it(it) {}
    };

    std::unique_ptr<ov::Core> m_core;
    std::map<std::string, CacheEntry> m_model_cache;
    std::list<std::string> m_access_order;
    const size_t m_max_cache_size = 200;
    
    std::mutex m_mutex;
    bool m_is_available;
};

// C-API wrappers for Python bindings
bool is_npu_available();
void initialize_npu();
void set_npu_cache_dir(const std::string& path);