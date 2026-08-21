#include "include/device.h"

#include <iostream>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------
// Singleton accessor
// ---------------------------------------------------------------------------

NPUBackend& NPUBackend::getInstance() {
    static NPUBackend instance;
    return instance;
}

// ---------------------------------------------------------------------------
// Property probing helpers
// ---------------------------------------------------------------------------

void NPUBackend::probeProperties() {
    try {
        auto props = m_core->get_property("NPU", ov::supported_properties);
        for (const auto& p : props) {
            m_supported_props.insert(std::string(p));
        }
        std::cout << "[Intel NPU] Driver supports " << m_supported_props.size() << " properties."
                  << std::endl;
    } catch (const std::exception& e) {
        // Older drivers may not implement ov::supported_properties; carry on.
        std::cerr << "[Intel NPU] Could not probe supported properties: " << e.what() << std::endl;
    }
}

bool NPUBackend::isPropertySupported(const std::string& key) const {
    // If probing failed (empty set) we optimistically allow everything.
    if (m_supported_props.empty()) return true;
    return m_supported_props.count(key) > 0;
}

std::string NPUBackend::getSupportedPropertiesList() const {
    std::ostringstream ss;
    for (const auto& p : m_supported_props) ss << p << " ";
    return ss.str();
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

NPUBackend::NPUBackend() {
    try {
        m_core = std::make_unique<ov::Core>();
        m_is_available = false;

        std::vector<std::string> devices = m_core->get_available_devices();
        for (const auto& device : devices) {
            if (device.find("NPU") == std::string::npos) continue;

            m_is_available = true;

            // 1. Acquire Level Zero RemoteContext (ze_context_handle_t wrapper).
            try {
                m_context = m_core->get_default_context("NPU");
                m_has_context =
                    false;  // Let OpenVINO manage memory copies dynamically for eager mode
                std::cout << "[Intel NPU] Level Zero RemoteContext acquired." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Intel NPU] RemoteContext unavailable: " << e.what() << std::endl;
            }

            // 2. Probe which property keys this driver version accepts.
            probeProperties();

            // 3. Build and apply the global device property map.
            //    Only include keys the driver actually supports.
            ov::AnyMap global_props;
            global_props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;

            // Driver-version-specific optimisation keys (accepted in some OV builds).
            struct PropSpec {
                std::string key;
                std::string val;
            };
            static const PropSpec L0_PROPS[] = {
                {"NPU_BACKEND_TYPE", "LEVEL_ZERO"},
                {"NPU_USE_SDA", "YES"},
                {"NPU_TURBO", "YES"},
            };
            for (const auto& ps : L0_PROPS) {
                if (isPropertySupported(ps.key)) {
                    global_props[ps.key] = ps.val;
                    std::cout << "[Intel NPU] Setting " << ps.key << "=" << ps.val << std::endl;
                }
            }

            try {
                m_core->set_property("NPU", global_props);
            } catch (const std::exception& e) {
                std::cerr << "[Intel NPU] set_property failed: " << e.what()
                          << ". Falling back to LATENCY only." << std::endl;
                m_core->set_property("NPU", {{ov::hint::performance_mode.name(), "LATENCY"}});
            }

            break;  // Only one NPU expected.
        }
    } catch (const std::exception& e) {
        std::cerr << "[Intel NPU] Initialization error: " << e.what() << std::endl;
        m_is_available = false;
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

ov::Core& NPUBackend::getCore() {
    return *m_core;
}
ov::RemoteContext& NPUBackend::getContext() {
    return m_context;
}
bool NPUBackend::isAvailable() {
    return m_is_available;
}

void NPUBackend::setPerformanceHint(const std::string& hint) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (hint == "THROUGHPUT") {
        m_performance_hint = ov::hint::PerformanceMode::THROUGHPUT;
    } else if (hint == "CUMULATIVE_THROUGHPUT") {
        m_performance_hint = ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
    } else {
        m_performance_hint = ov::hint::PerformanceMode::LATENCY;
    }
}

void NPUBackend::setEagerDevice(const std::string& device) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_eager_device = device;
}

// ---------------------------------------------------------------------------
// Model cache — double-checked locking so compilation runs outside the mutex
// ---------------------------------------------------------------------------

ov::CompiledModel NPUBackend::getOrCompileModel(const std::string& key,
                                                std::shared_ptr<ov::Model> model) {
    // --- Fast path: cache hit (holds lock only for the lookup) ---
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_model_cache.find(key);
        if (it != m_model_cache.end()) {
            m_access_order.erase(it->second.list_it);
            m_access_order.push_front(key);
            it->second.list_it = m_access_order.begin();
            return it->second.compiled_model;
        }
    }

    // --- Slow path: compile outside the mutex (can take 30-40 s) ---
    if (!model) throw std::runtime_error("Model pointer is null but key not in cache: " + key);

    ov::CompiledModel compiled;
    if (m_eager_device == "NPU") {
        ov::AnyMap compile_props = {
            {ov::hint::performance_mode.name(), m_performance_hint},
            {ov::hint::inference_precision.name(), ov::element::f16},
        };

        // Add compilation-mode tuning only when the driver supports it.
        if (isPropertySupported("NPU_COMPILATION_MODE_CONFIG")) {
            compile_props["NPU_COMPILATION_MODE_CONFIG"] = "enable-se-ptrs-operations=true";
        }

        try {
            // Prefer context-based compile to activate the USM/SDA pipeline.
            if (m_has_context) {
                compiled = m_core->compile_model(model, m_context, compile_props);
            } else {
                compiled = m_core->compile_model(model, m_eager_device, compile_props);
            }
        } catch (const std::exception& e) {
            std::cerr << "[Intel NPU] Compile with hints failed (" << e.what()
                      << "), retrying bare." << std::endl;
            compiled = m_core->compile_model(model, m_eager_device);
        }
    } else {
        compiled = m_core->compile_model(model, m_eager_device);
    }

    // --- Insert under lock (double-check: another thread may have compiled) ---
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_model_cache.find(key);
        if (it != m_model_cache.end()) {
            // Another thread beat us; discard our copy and return theirs.
            return it->second.compiled_model;
        }
        if (m_model_cache.size() >= m_max_cache_size) {
            m_model_cache.erase(m_access_order.back());
            m_access_order.pop_back();
            m_cache_version.fetch_add(1, std::memory_order_relaxed);
        }
        m_access_order.push_front(key);
        m_model_cache[key] = {compiled, m_access_order.begin()};
    }
    return compiled;
}

// ---------------------------------------------------------------------------
// InferRequest — thread-local reuse, no per-call allocation on the hot path
// ---------------------------------------------------------------------------

ov::InferRequest NPUBackend::getOrCachedInferRequest(const std::string& key) {
    // Each thread owns its own InferRequest per compiled model.
    // ov::InferRequest is NOT thread-safe; sharing would corrupt inference state.
    // Thread-local storage gives us zero-overhead reuse for the common
    // single-threaded case, and full isolation for concurrent callers.
    thread_local std::unordered_map<std::string, ov::InferRequest> tl_requests;
    thread_local uint64_t tl_cache_version = 0;

    uint64_t current_version = m_cache_version.load(std::memory_order_relaxed);
    if (tl_cache_version != current_version) {
        tl_requests.clear();
        tl_cache_version = current_version;
    }

    auto tl_it = tl_requests.find(key);
    if (tl_it != tl_requests.end()) {
        return tl_it->second;  // Hot path: already have a request for this thread.
    }

    // Slow path: fetch compiled model under lock, create request outside it.
    ov::CompiledModel compiled;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_model_cache.find(key);
        if (it == m_model_cache.end())
            throw std::runtime_error("Attempted to get InferRequest for non-compiled model: " +
                                     key);
        compiled = it->second.compiled_model;
    }

    tl_requests[key] = compiled.create_infer_request();
    return tl_requests[key];
}

// ---------------------------------------------------------------------------
// Cache / property setters
// ---------------------------------------------------------------------------

void NPUBackend::setCacheDir(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_mutex);
    try {
        m_core->set_property(ov::cache_dir(path));
    } catch (const std::exception& e) {
        std::cerr << "[Intel NPU] Failed to set cache directory: " << e.what() << std::endl;
    }
}

void NPUBackend::setProperty(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!isPropertySupported(key)) {
        std::cerr << "[Intel NPU] Property '" << key
                  << "' is not supported by this driver; ignoring." << std::endl;
        return;
    }
    try {
        m_core->set_property("NPU", {{key, value}});
    } catch (const std::exception& e) {
        std::cerr << "[Intel NPU] Failed to set property " << key << ": " << e.what() << std::endl;
    }
}

void NPUBackend::clearCache() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_model_cache.clear();
    m_access_order.clear();
    m_cache_version.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// C-API wrappers
// ---------------------------------------------------------------------------

uint64_t NPUBackend::getCacheVersion() const {
    return m_cache_version.load(std::memory_order_relaxed);
}

bool is_npu_available() {
    return NPUBackend::getInstance().isAvailable();
}
void initialize_npu() {
    NPUBackend::getInstance();
}
void set_npu_cache_dir(const std::string& path) {
    NPUBackend::getInstance().setCacheDir(path);
}
void set_npu_property(const std::string& key, const std::string& value) {
    NPUBackend::getInstance().setProperty(key, value);
}
void set_npu_performance_hint(const std::string& hint) {
    NPUBackend::getInstance().setPerformanceHint(hint);
}
void set_npu_eager_device(const std::string& device) {
    NPUBackend::getInstance().setEagerDevice(device);
}
void clear_cpp_model_cache() {
    NPUBackend::getInstance().clearCache();
}
