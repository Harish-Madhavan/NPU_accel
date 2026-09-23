#pragma once

#include <torch/extension.h>
#include <vector>

/**
 * @file ops.h
 * @brief Native C++ operator declarations accelerated on Intel NPU via oneAPI Level Zero.
 */

// ===========================================================================
// 1. Element-wise Math & Binary Operations
// ===========================================================================

/** @brief Element-wise addition on Intel NPU (c = a + b). */
torch::Tensor npu_add(torch::Tensor a, torch::Tensor b);

/** @brief Element-wise subtraction on Intel NPU (c = a - b). */
torch::Tensor npu_sub(torch::Tensor a, torch::Tensor b);

/** @brief Element-wise negation on Intel NPU (c = -a). */
torch::Tensor npu_neg(torch::Tensor a);

/** @brief Element-wise multiplication on Intel NPU (c = a * b). */
torch::Tensor npu_mul(torch::Tensor a, torch::Tensor b);

/** @brief Element-wise division on Intel NPU (c = a / b). */
torch::Tensor npu_div(torch::Tensor a, torch::Tensor b);

// ===========================================================================
// 2. Matrix, Linear, & Attention Operations
// ===========================================================================

/** @brief Hardware accelerated matrix multiplication (C = A @ B). */
torch::Tensor npu_matmul(torch::Tensor a, torch::Tensor b);

/** @brief Hardware accelerated linear projection (Y = X @ W^T + b). */
torch::Tensor npu_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

/** @brief Scaled Dot-Product Attention (SDPA) with causal masking and scaling. */
torch::Tensor npu_scaled_dot_product_attention(torch::Tensor query, torch::Tensor key,
                                               torch::Tensor value, torch::Tensor attn_mask,
                                               double dropout_p, bool is_causal, double scale);

/** @brief Backward derivatives for SDPA (grad_query, grad_key, grad_value).
 *
 * Recomputes attention probabilities with Level Zero MatMul/Softmax kernels
 * (mirroring the forward scale/mask/causal configuration) and applies the
 * softmax-Jacobian chain rule with Level Zero MatMul and elementwise kernels.
 * dropout_p is accepted for signature parity; the forward pass does not apply
 * dropout, so the backward pass does not differentiate through it either.
 */
std::vector<torch::Tensor> npu_scaled_dot_product_attention_backward(torch::Tensor grad_output,
                                                                      torch::Tensor query, torch::Tensor key,
                                                                      torch::Tensor value, torch::Tensor attn_mask,
                                                                      double dropout_p, bool is_causal, double scale);

// ===========================================================================
// 3. Activations & Normalizations
// ===========================================================================

/** @brief Rectified Linear Unit activation (y = max(0, x)). */
torch::Tensor npu_relu(torch::Tensor a);

/** @brief Gaussian Error Linear Unit activation (GELU). */
torch::Tensor npu_gelu(torch::Tensor a);

/** @brief Sigmoid Linear Unit / Swish activation (y = x * sigmoid(x)). */
torch::Tensor npu_silu(torch::Tensor a);

/** @brief Softmax activation along specified dimension. */
torch::Tensor npu_softmax(torch::Tensor a, int64_t dim);

/** @brief Root Mean Square Normalization for LLM backbones. */
torch::Tensor npu_rmsnorm(torch::Tensor input, torch::Tensor weight, float epsilon);

/** @brief Layer Normalization with learnable affine weights. */
torch::Tensor npu_layer_norm(torch::Tensor input, std::vector<int64_t> normalized_shape,
                             torch::Tensor weight, torch::Tensor bias, float epsilon);

// ===========================================================================
// 4. Tensor Reshaping, Indexing, & Manipulation
// ===========================================================================

/** @brief Permute tensor dimensions. */
torch::Tensor npu_transpose(torch::Tensor input, std::vector<int64_t> permutation);

/** @brief Reshape tensor to target shape. */
torch::Tensor npu_reshape(torch::Tensor input, std::vector<int64_t> shape);

/** @brief Squeeze specified dimensions of size 1. */
torch::Tensor npu_squeeze(torch::Tensor input, std::vector<int64_t> dims);

/** @brief Unsqueeze specified dimensions. */
torch::Tensor npu_unsqueeze(torch::Tensor input, std::vector<int64_t> dims);

/** @brief Concatenate tensors along specified dimension. */
torch::Tensor npu_cat(std::vector<torch::Tensor> tensors, int64_t dim);

/** @brief Stack tensors along new dimension. */
torch::Tensor npu_stack(std::vector<torch::Tensor> tensors, int64_t dim);

/** @brief Reduce mean along specified dimensions. */
torch::Tensor npu_mean(torch::Tensor input, std::vector<int64_t> dim, bool keepdim);

/** @brief Gather tensor slices along dim based on index tensor. */
torch::Tensor npu_index_select(torch::Tensor input, int64_t dim, torch::Tensor index);

/** @brief Token / embedding lookup (Gather axis=0). */
torch::Tensor npu_embedding(torch::Tensor weight, torch::Tensor indices);

// ===========================================================================
// 5. Computer Vision & LLM-Specific Operations
// ===========================================================================

/** @brief Hardware accelerated 2D convolution. */
torch::Tensor npu_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         std::vector<int64_t> stride, std::vector<int64_t> padding,
                         std::vector<int64_t> dilation, int64_t groups);

/** @brief Backward derivatives for Conv2d (grad_input, grad_weight, grad_bias).
 *
 * grad_input runs as a transposed-convolution Level Zero graph
 * (ConvolutionBackpropData); grad_weight runs as an im2col Gather + MatMul
 * Level Zero graph with a block-diagonal groups mask; grad_bias is a
 * ReduceSum Level Zero graph. Entries for unrequested gradients are
 * returned as undefined tensors (same contract as npu_linear_backward).
 */
std::vector<torch::Tensor> npu_conv2d_backward(torch::Tensor grad_output, torch::Tensor input,
                                               torch::Tensor weight, std::vector<int64_t> stride,
                                               std::vector<int64_t> padding, std::vector<int64_t> dilation,
                                               int64_t groups, bool needs_input_grad,
                                               bool needs_weight_grad, bool needs_bias_grad);

/** @brief 2D Max pooling operation. */
torch::Tensor npu_max_pool2d(torch::Tensor input, std::vector<int64_t> kernel_size,
                             std::vector<int64_t> stride, std::vector<int64_t> padding,
                             std::vector<int64_t> dilation, bool ceil_mode);

/** @brief In-place NPU KV-cache scatter update for autoregressive generation. */
torch::Tensor npu_update_kv_cache(torch::Tensor cache, torch::Tensor new_kv,
                                  torch::Tensor position);

/** @brief Hardware accelerated Rotary Position Embedding (RoPE) for LLMs. */
torch::Tensor npu_rotary_embedding(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);

/** @brief Weight-only quantized linear transformation (MatMul + Scale + ZP + Bias). */
torch::Tensor npu_quantized_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor scale,
                                   torch::Tensor zero_point, torch::Tensor bias);

// ===========================================================================
// 6. Training & Backward Autograd Operations
// ===========================================================================

/** @brief Backward derivative for MatMul (grad_a = grad_out @ B^T, grad_b = A^T @ grad_out). */
std::vector<torch::Tensor> npu_matmul_backward(torch::Tensor grad_output, torch::Tensor a, torch::Tensor b);

/** @brief Backward derivative for Linear layer (grad_input, grad_weight, grad_bias). */
std::vector<torch::Tensor> npu_linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                               bool needs_input_grad, bool needs_weight_grad, bool needs_bias_grad);

/** @brief Backward derivative for ReLU (grad_out where x > 0). */
torch::Tensor npu_relu_backward(torch::Tensor grad_output, torch::Tensor input);

/** @brief Exact backward derivative for GELU (cdf + x * pdf). */
torch::Tensor npu_gelu_backward(torch::Tensor grad_output, torch::Tensor input);

/** @brief Exact backward derivative for SiLU (sigmoid * (1 + x * (1 - sigmoid))). */
torch::Tensor npu_silu_backward(torch::Tensor grad_output, torch::Tensor input);

/** @brief Backward derivative for Softmax (y * (grad_out - sum(grad_out * y))). */
torch::Tensor npu_softmax_backward(torch::Tensor grad_output, torch::Tensor output, int64_t dim);

/** @brief Backward derivative for RMSNorm (grad_input, grad_weight). */
std::vector<torch::Tensor> npu_rmsnorm_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, float epsilon);

/** @brief Mean Squared Error loss forward reduction. */
torch::Tensor npu_mse_loss(torch::Tensor pred, torch::Tensor target, int64_t reduction);

/** @brief Backward gradient for Mean Squared Error loss. */
torch::Tensor npu_mse_loss_backward(torch::Tensor grad_output, torch::Tensor pred, torch::Tensor target, int64_t reduction);

/** @brief Cross Entropy Loss forward reduction. */
torch::Tensor npu_cross_entropy_loss(torch::Tensor pred, torch::Tensor target, int64_t reduction);

/** @brief Backward gradient for Cross Entropy Loss. */
torch::Tensor npu_cross_entropy_loss_backward(torch::Tensor grad_output, torch::Tensor pred, torch::Tensor target, int64_t reduction);

/** @brief L1 Loss (Mean Absolute Error) forward reduction. */
torch::Tensor npu_l1_loss(torch::Tensor pred, torch::Tensor target, int64_t reduction);

/** @brief Backward gradient for L1 Loss. */
torch::Tensor npu_l1_loss_backward(torch::Tensor grad_output, torch::Tensor pred, torch::Tensor target, int64_t reduction);

/** @brief Embedding gradient scatter (index_add). */
torch::Tensor npu_embedding_backward(torch::Tensor grad_output, torch::Tensor indices, int64_t num_embeddings);

// ===========================================================================
// 7. Elementwise transcendentals, clamping, selection & triangular ops
// ===========================================================================

/** @brief Element-wise sine on Intel NPU. */
torch::Tensor npu_sin(torch::Tensor a);

/** @brief Element-wise cosine on Intel NPU. */
torch::Tensor npu_cos(torch::Tensor a);

/** @brief Element-wise exponential on Intel NPU. */
torch::Tensor npu_exp(torch::Tensor a);

/** @brief Element-wise square root on Intel NPU. */
torch::Tensor npu_sqrt(torch::Tensor a);

/** @brief Element-wise absolute value on Intel NPU. */
torch::Tensor npu_abs(torch::Tensor a);

/** @brief Element-wise sign on Intel NPU. */
torch::Tensor npu_sign(torch::Tensor a);

/** @brief Element-wise reciprocal square root (x^-0.5 via Power). */
torch::Tensor npu_rsqrt(torch::Tensor a);

/** @brief Element-wise power (broadcasting binary). */
torch::Tensor npu_pow(torch::Tensor a, torch::Tensor exponent);

/** @brief Clamp to [min_val, max_val]. */
torch::Tensor npu_clamp(torch::Tensor a, double min_val, double max_val);

/** @brief Select elements from a/b by boolean condition. */
torch::Tensor npu_where(torch::Tensor condition, torch::Tensor a, torch::Tensor b);

/** @brief Upper-triangular part (zeroed below diagonal); multiply runs on NPU. */
torch::Tensor npu_triu(torch::Tensor a, int64_t diagonal);

/** @brief Backward derivatives for sin/cos/exp/sqrt/abs/rsqrt (go * local deriv). */
torch::Tensor npu_sin_backward(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor npu_cos_backward(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor npu_exp_backward(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor npu_sqrt_backward(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor npu_abs_backward(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor npu_rsqrt_backward(torch::Tensor grad_output, torch::Tensor input);

/** @brief Backward derivative for pow w.r.t. the base (exponent grads use CPU fallback). */
torch::Tensor npu_pow_backward(torch::Tensor grad_output, torch::Tensor a, torch::Tensor exponent);

/** @brief Backward derivative for clamp (gradient passes only inside [min, max]). */
torch::Tensor npu_clamp_backward(torch::Tensor grad_output, torch::Tensor input, double min_val,
                                 double max_val);

/** @brief Backward derivatives for where (grad_a, grad_b) via mask multiply. */
std::vector<torch::Tensor> npu_where_backward(torch::Tensor grad_output, torch::Tensor condition);

/** @brief Backward derivative for triu (triu is its own gradient mask). */
torch::Tensor npu_triu_backward(torch::Tensor grad_output, int64_t diagonal);

/** @brief MaxPool2d backward for tiled pools (stride == kernel, no padding).
 *
 * Gradient routes through the argmax mask recomputed as
 * Equal(input, Tile(MaxPool(input))). Exact ties share the gradient
 * (torch routes to the first max); overlapping/padded/dilated pools throw
 * (Python routes those to the CPU fallback).
 */
torch::Tensor npu_max_pool2d_backward(torch::Tensor grad_output, torch::Tensor input,
                                      std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
                                      std::vector<int64_t> padding);

// ===========================================================================
// 8. Hardware-Fused Optimizer Steps
// ===========================================================================

/** @brief Fused Adam optimizer step on NPU (updates param, exp_avg, and exp_avg_sq). */
std::vector<torch::Tensor> npu_adam_step(torch::Tensor param, torch::Tensor grad,
                                         torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
                                         double lr, double beta1, double beta2, double eps,
                                         double weight_decay, int64_t step);

/** @brief Fused SGD optimizer step with momentum buffers on NPU. */
std::vector<torch::Tensor> npu_sgd_step(torch::Tensor param, torch::Tensor grad,
                                        torch::Tensor momentum_buffer, double lr,
                                        double momentum, double weight_decay,
                                        double dampening, bool nesterov, bool has_momentum_buffer);
