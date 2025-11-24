#pragma once
#include <torch/extension.h>

// NPU operations
torch::Tensor npu_add(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_sub(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_mul(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_div(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_matmul(torch::Tensor a, torch::Tensor b);
