#pragma once
#include <torch/extension.h>

// Checks if the NPU device is available via the driver
bool is_npu_available();

// Initialize the NPU device (context creation, etc.)
void initialize_npu();
