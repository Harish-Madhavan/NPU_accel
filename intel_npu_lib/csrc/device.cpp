#include "include/device.h"
#include <iostream>
#include <vector>

NPUBackend& NPUBackend::getInstance() {
    static NPUBackend* instance = new NPUBackend();
    return *instance;
}

NPUBackend::NPUBackend() {
    try {
        m_core = std::make_unique<ov::Core>();
        m_is_available = false;
        
        std::vector<std::string> devices = m_core->get_available_devices();
        for (const auto& device : devices) {
            if (device.find("NPU") != std::string::npos) {
                m_is_available = true;
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Intel NPU] Error during initialization: " << e.what() << std::endl;
        m_is_available = false;
    }
}

ov::Core& NPUBackend::getCore() {
    return *m_core;
}

bool NPUBackend::isAvailable() {
    return m_is_available;
}

ov::CompiledModel NPUBackend::getOrCompileModel(const std::string& key, std::shared_ptr<ov::Model> model) {
    std::lock_guard<std::mutex> lock(m_mutex);

    auto it = m_model_cache.find(key);
    if (it != m_model_cache.end()) {
        // Key found, move to front of access order (most recent)
        m_access_order.erase(it->second.list_it);
        m_access_order.push_front(key);
        it->second.list_it = m_access_order.begin();
        return it->second.compiled_model;
    }

    // If key not in cache, we need to compile it.
    std::string device = "NPU";
    if (!m_is_available) {
        device = "CPU"; 
    }

    if (!model) {
        throw std::runtime_error("Model pointer is null but key not found in cache: " + key);
    }

    // Compile model
    ov::CompiledModel compiled = m_core->compile_model(model, device);

    // Evict if cache is full
    if (m_model_cache.size() >= m_max_cache_size) {
        std::string oldest_key = m_access_order.back();
        m_access_order.pop_back();
        m_model_cache.erase(oldest_key);
    }

    // Add new entry to front
    m_access_order.push_front(key);
    m_model_cache[key] = {compiled, m_access_order.begin()};

    return compiled;
}

void NPUBackend::setCacheDir(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_mutex);
    try {
        m_core->set_property(ov::cache_dir(path));
    } catch (const std::exception& e) {
        std::cerr << "[Intel NPU] Failed to set cache directory: " << e.what() << std::endl;
    }
}

// Wrappers
bool is_npu_available() {
    return NPUBackend::getInstance().isAvailable();
}

void initialize_npu() {
    // Just trigger singleton creation
    NPUBackend::getInstance();
}

void set_npu_cache_dir(const std::string& path) {
    NPUBackend::getInstance().setCacheDir(path);
}
