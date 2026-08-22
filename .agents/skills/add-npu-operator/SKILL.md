---
name: add-npu-operator
description: Step-by-step procedural guide for implementing and exposing a new hardware-accelerated operator for Intel NPU.
---

# Guide: Adding a New Operator to Intel NPU Acceleration

Follow this 7-step checklist whenever implementing a new operator.

## 1. C++ Header Declaration
Declare the operator signature in `intel_npu_lib/csrc/include/ops.h`:
```cpp
torch::Tensor npu_my_op(torch::Tensor a, torch::Tensor b);
```

## 2. C++ Kernel Implementation
Implement the operator using OpenVINO opset and zero-copy tensor binding in `intel_npu_lib/csrc/ops.cpp`:
```cpp
torch::Tensor npu_my_op(torch::Tensor a, torch::Tensor b) {
    // 1. Ensure contiguous layout
    a = a.contiguous();
    b = b.contiguous();

    // 2. Build OpenVINO Parameter nodes and operation node
    auto param_a = std::make_shared<ov::opset13::Parameter>(...);
    auto param_b = std::make_shared<ov::opset13::Parameter>(...);
    auto op_node = std::make_shared<ov::opset13::MyOp>(param_a, param_b);

    // 3. Compile or retrieve from LRU cache
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op_node}, ov::ParameterVector{param_a, param_b});
    auto compiled = NPUBackend::getInstance().getOrCompileModel(cache_key, model);

    // 4. Bind memory and execute
    ...
    return out_tensor;
}
```

## 3. PyBind11 Binding
Expose the function in `intel_npu_lib/csrc/bindings.cpp`:
```cpp
m.def("npu_my_op", &npu_my_op, "NPU accelerated MyOp");
```

## 4. Python Functional Dispatch & Fallback
Add the function with C++ dispatch and CPU fallback in `intel_npu_lib/src/intel_npu_acceleration/_functional/`:
```python
def my_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is not None and not _is_proxy(a, b):
        try:
            return _C.npu_my_op(a, b)
        except Exception:
            pass
    return torch.my_op(a, b)
```

## 5. Autograd Integration (If Gradient-Aware)
If the operator requires backward backprop support, implement a `torch.autograd.Function` in `src/intel_npu_acceleration/autograd/functions.py`.

## 6. Register in Registry
Register the operation in `src/intel_npu_acceleration/registry.py` and export in `functional.py` and `__init__.py`.

## 7. Build and Add Unit Tests
1. Rebuild: `python setup.py build_ext --inplace`
2. Add unit test in `tests/test_new_ops.py` or dedicated test file.
3. Run `pytest` and verify 100% green status.
