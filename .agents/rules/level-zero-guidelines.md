# Level Zero Driver & C++ Backend Guidelines

## Intent
Guide AI agents in modifying C++ Level Zero driver integration and OpenVINO operator graph nodes.

## Rules
1. **Thread Safety & Singleton Access**:
   - Always access OpenVINO Core and Level Zero contexts via `NPUBackend::getInstance()`.
   - Never create independent `ov::Core` instances in operator kernels.

2. **Zero-Copy Memory Binding**:
   - When binding PyTorch input and output tensors to OpenVINO tensors, use continuous memory pointers:
     ```cpp
     ov::Tensor ov_tensor(ov_element_type, ov_shape, tensor.data_ptr());
     ```
   - Ensure tensors are `.contiguous()` before invoking C++ backend functions.

3. **Level Zero Property Invariants**:
   - Maintain the standard `L0_PROPS` table in `device.cpp`:
     - `NPU_TURBO = YES`
     - `NPU_DISABLE_IDLE_MEMORY_PRUNING = YES`
     - `NPU_RUN_INFERENCES_SEQUENTIALLY = NO`
     - `NPU_DEFER_WEIGHTS_LOAD = YES`
     - `NPU_QDQ_OPTIMIZATION = YES`
     - `NPU_QDQ_OPTIMIZATION_AGGRESSIVE = YES`
     - `NPU_COMPILATION_MODE_PARAMS = optimization-level=2`
     - `MODEL_PRIORITY = HIGH`

4. **Always Rebuild Extension After C++ Edits**:
   - Run `python setup.py build_ext --inplace` from `intel_npu_lib/`.
