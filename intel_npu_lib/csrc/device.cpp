#include "include/device.h"
#include <iostream>
#include <vector>
#include <string>

#include <openvino/openvino.hpp>

// Global state to track if initialized
static bool g_npu_initialized = false;

bool is_npu_available() {
    try {
        ov::Core core;
        std::vector<std::string> devices = core.get_available_devices();
        for (const auto& device : devices) {
            if (device.find("NPU") != std::string::npos) {
                return true;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "OpenVINO Error: " << e.what() << std::endl;
        return false;
    } catch (...) {
        return false;
    }
    return false;
}

void initialize_npu() {
    if (!g_npu_initialized) {
         // std::cout << "[Intel NPU Lib] Initializing OpenVINO Core..." << std::endl;
         // ov::Core core; 
         // We don't keep the core globally in this simple example, 
         // but in a real driver you would cache the device context.
        g_npu_initialized = true;
    }
}