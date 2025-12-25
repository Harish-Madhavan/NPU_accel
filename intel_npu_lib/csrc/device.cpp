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
        return it->second;
    }

    // Cache cleanup policy: Simple LRU-approximation or Size-limit
    if (m_model_cache.size() > 200) {
        // Clear half of cache to avoid thrashing? Or just clear all for simplicity in Phase 1
        // log("Cache limit reached, clearing model cache.");
        m_model_cache.clear();
    }

    std::string device = "NPU";
    // If NPU not available, fallback to CPU for testing/safety? 
    // For now strict NPU as per library name, or let OpenVINO handle it.
    if (!m_is_available) {
        // Warn once?
        device = "CPU"; 
    }

    if (!model) {
        throw std::runtime_error("Model pointer is null but key not found in cache: " + key);
    }

    // log("Compiling model for key: " + key);
    ov::CompiledModel compiled = m_core->compile_model(model, device);
    m_model_cache[key] = compiled;
    return compiled;
}

// Wrappers
bool is_npu_available() {
    return NPUBackend::getInstance().isAvailable();
}

void initialize_npu() {
    // Just trigger singleton creation
    NPUBackend::getInstance();
}
