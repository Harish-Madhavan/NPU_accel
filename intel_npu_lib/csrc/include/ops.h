#pragma once
#include <torch/extension.h>

// Stub for NPU add operation
torch::Tensor npu_add(torch::Tensor a, torch::Tensor b);
