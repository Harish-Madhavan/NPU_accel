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
  m.def("npu_softmax", &npu_softmax, "NPU accelerated Softmax");
  m.def("npu_linear", &npu_linear, "NPU accelerated Linear (MatMul + Bias)");
  m.def("npu_transpose", &npu_transpose, "NPU accelerated Transpose");
  m.def("npu_reshape", &npu_reshape, "NPU accelerated Reshape");
  m.def("npu_conv2d", &npu_conv2d, "NPU accelerated Conv2d");
  m.def("npu_max_pool2d", &npu_max_pool2d, "NPU accelerated MaxPool2d");
}
