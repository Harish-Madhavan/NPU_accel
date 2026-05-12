#pragma once
#include <torch/extension.h>

// NPU operations
torch::Tensor npu_add(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_sub(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_neg(torch::Tensor a);
torch::Tensor npu_mul(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_div(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_matmul(torch::Tensor a, torch::Tensor b);
torch::Tensor npu_relu(torch::Tensor a);
torch::Tensor npu_gelu(torch::Tensor a);
torch::Tensor npu_silu(torch::Tensor a);
torch::Tensor npu_rmsnorm(torch::Tensor input, torch::Tensor weight, float epsilon);
torch::Tensor npu_layer_norm(torch::Tensor input, std::vector<int64_t> normalized_shape, torch::Tensor weight, torch::Tensor bias, float epsilon);
torch::Tensor npu_softmax(torch::Tensor a, int64_t dim);
torch::Tensor npu_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor npu_transpose(torch::Tensor input, std::vector<int64_t> permutation);
torch::Tensor npu_reshape(torch::Tensor input, std::vector<int64_t> shape);
torch::Tensor npu_cat(std::vector<torch::Tensor> tensors, int64_t dim);
torch::Tensor npu_stack(std::vector<torch::Tensor> tensors, int64_t dim);
torch::Tensor npu_mean(torch::Tensor input, std::vector<int64_t> dim, bool keepdim);

torch::Tensor npu_scaled_dot_product_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor attn_mask,
    double dropout_p,
    bool is_causal,
    double scale
);

// CV Ops
torch::Tensor npu_conv2d(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    std::vector<int64_t> stride, 
    std::vector<int64_t> padding, 
    std::vector<int64_t> dilation, 
    int64_t groups
);

torch::Tensor npu_max_pool2d(
    torch::Tensor input, 
    std::vector<int64_t> kernel_size, 
    std::vector<int64_t> stride, 
    std::vector<int64_t> padding, 
    std::vector<int64_t> dilation,
    bool ceil_mode
);
