#include <torch/extension.h>

#include "include/device.h"
#include "include/ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("is_npu_available", &is_npu_available, "Check if NPU is available");
    m.def("npu_add", &npu_add, "NPU accelerated add");
    m.def("npu_sub", &npu_sub, "NPU accelerated subtract");
    m.def("npu_neg", &npu_neg, "NPU accelerated negative");
    m.def("npu_mul", &npu_mul, "NPU accelerated multiply");
    m.def("npu_div", &npu_div, "NPU accelerated divide");
    m.def("npu_matmul", &npu_matmul, "NPU accelerated matrix multiplication");
    m.def("npu_relu", &npu_relu, "NPU accelerated ReLU");
    m.def("npu_gelu", &npu_gelu, "NPU GELU");
    m.def("npu_silu", &npu_silu, "NPU SiLU");
    m.def("npu_rmsnorm", &npu_rmsnorm, "NPU RMSNorm");
    m.def("npu_layer_norm", &npu_layer_norm, "NPU LayerNorm");
    m.def("npu_softmax", &npu_softmax, "NPU accelerated Softmax");
    m.def("npu_linear", &npu_linear, "NPU accelerated Linear (MatMul + Bias)");
    m.def("npu_transpose", &npu_transpose, "NPU accelerated Transpose");
    m.def("npu_reshape", &npu_reshape, "NPU accelerated Reshape");
    m.def("npu_squeeze", &npu_squeeze, "NPU accelerated Squeeze");
    m.def("npu_unsqueeze", &npu_unsqueeze, "NPU accelerated Unsqueeze");
    m.def("npu_cat", &npu_cat, "NPU accelerated Concat");
    m.def("npu_stack", &npu_stack, "NPU accelerated Stack");
    m.def("npu_mean", &npu_mean, "NPU accelerated Mean");
    m.def("npu_index_select", &npu_index_select, "NPU accelerated Index Select");
    m.def("npu_scaled_dot_product_attention", &npu_scaled_dot_product_attention,
          "NPU Scaled Dot Product Attention");
    m.def("npu_conv2d", &npu_conv2d, "NPU accelerated Conv2d");
    m.def("npu_max_pool2d", &npu_max_pool2d, "NPU accelerated MaxPool2d");
    m.def("npu_update_kv_cache", &npu_update_kv_cache, "NPU accelerated KV cache update");
    m.def("npu_rotary_embedding", &npu_rotary_embedding,
          "NPU accelerated Rotary Position Embedding (RoPE)");
    m.def("npu_quantized_linear", &npu_quantized_linear,
          "NPU accelerated weight-only Quantized Linear (MatMul + Scale + ZP + Bias)");
    m.def("npu_embedding", &npu_embedding, "NPU accelerated Embedding Lookup");
    m.def("npu_embedding_backward", &npu_embedding_backward, "NPU accelerated Embedding Backward");
    m.def("npu_matmul_backward", &npu_matmul_backward, "NPU accelerated MatMul Backward");
    m.def("npu_linear_backward", &npu_linear_backward, "NPU accelerated Linear Backward");
    m.def("npu_relu_backward", &npu_relu_backward, "NPU accelerated ReLU Backward");
    m.def("npu_gelu_backward", &npu_gelu_backward, "NPU accelerated GELU Backward");
    m.def("npu_silu_backward", &npu_silu_backward, "NPU accelerated SiLU Backward");
    m.def("npu_softmax_backward", &npu_softmax_backward, "NPU accelerated Softmax Backward");
    m.def("npu_rmsnorm_backward", &npu_rmsnorm_backward, "NPU accelerated RMSNorm Backward");
    m.def("npu_mse_loss", &npu_mse_loss, "NPU accelerated MSE Loss");
    m.def("npu_mse_loss_backward", &npu_mse_loss_backward, "NPU accelerated MSE Loss Backward");
    m.def("npu_adam_step", &npu_adam_step, "NPU accelerated Adam Optimizer Step");
    m.def("npu_sgd_step", &npu_sgd_step, "NPU accelerated SGD Optimizer Step");
    m.def("set_cache_dir", &set_npu_cache_dir, "Set OpenVINO disk cache directory");
    m.def("set_property", &set_npu_property, "Set global NPU property (Level Zero optimizations)");
    m.def("set_performance_hint", &set_npu_performance_hint,
          "Set performance hint (LATENCY, THROUGHPUT)");
    m.def("set_eager_device", &set_npu_eager_device, "Set device for eager operations (CPU, NPU)");
    m.def("clear_cpp_model_cache", &clear_cpp_model_cache, "Clear in-memory C++ model cache");
    m.def("get_cache_version", []() {
        return NPUBackend::getInstance().getCacheVersion();
    }, "Get model cache version counter");
}
