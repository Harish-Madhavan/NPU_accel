#include <torch/extension.h>
#include "include/device.h"
#include "include/ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_npu_available", &is_npu_available, "Check if NPU is available");
  m.def("npu_add", &npu_add, "NPU accelerated add");
}
