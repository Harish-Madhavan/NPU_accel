#include <torch/extension.h>
#include "include/device.h"
#include "include/ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_npu_available", &is_npu_available, "Check if NPU is available");
  m.def("npu_add", &npu_add, "NPU accelerated add");
  m.def("npu_sub", &npu_sub, "NPU accelerated subtract");
  m.def("npu_mul", &npu_mul, "NPU accelerated multiply");
  m.def("npu_div", &npu_div, "NPU accelerated divide");
  m.def("npu_matmul", &npu_matmul, "NPU accelerated matrix multiplication");
}
