# Detailed Roadmap: Intel NPU Acceleration Library

This roadmap outlines the strategic direction for transforming the prototype into a robust, production-grade library. The focus is on streamlining architecture, expanding test coverage beyond LLMs, and preparing for diverse AI workloads.

## Phase 1: Housekeeping & Modernization (Immediate)
*Goal: Remove technical debt, clean up the codebase, and enforce strict coding standards.*

- [x] **Build System Cleanup**:
    - [x] Remove redundant `intel_npu_lib/CMakeLists.txt` (build is fully handled by `setup.py` and `setuptools`).
    - [x] Clean up root directory artifacts (`build/` folder).
- **Code Quality**:
    - [x] **Logging**: Replace `std::cout` and `print()` debugging with a proper logging framework (spdlog for C++, `logging` module for Python).
    - [ ] **Formatting**: Enforce `clang-format` for C++ and `ruff`/`black` for Python.
    - [x] **Type Safety**: Add type hints to all Python functions and run `mypy` validation.
- **C++ Refactoring**:
    - [x] **Singleton Pattern**: Encapsulate global state (`g_core`, `g_model_cache`) into a thread-safe `NPUBackend` singleton class.
    - [x] **RAII**: Ensure all OpenVINO objects are managed with smart pointers (already mostly done, but verify global maps).

## Phase 2: Architecture Refinement (Short-term)
*Goal: Modularize the compiler and improve error handling.*

- [x] **Compiler Modularization (`compiler.py`)**:
    - [x] Extract `OpRegistry` into `registry.py` to decouple operator mapping from graph building.
    - [x] Extract `OVGraphBuilder` into `graph_builder.py`.
    - [x] Create a dedicated `frontend.py` for the user-facing `compile` API.
- [x] **Error Handling**:
    - [x] Implement a custom C++ Exception class (`NPUException`) that translates to a specific Python error.
    - [x] Ensure clear error messages when an operator is not supported (instead of generic "Function not implemented").
- [x] **Dynamic Shapes**:
    - [x] Move away from "Constant Folding" shape tensors where possible. Use OpenVINO's dynamic shape support (`-1` dimensions) more aggressively to avoid recompiling for every new sequence length.

## Phase 3: Expanded Testing & QA (Medium-term)
*Goal: Ensure reliability across different workloads and edge cases.*

- [ ] **Test Infrastructure**:
    - [ ] **Parametrized Tests**: specific tests for:
        - Input shapes (Scalar, 1D, 2D, 3D, 4D).
        - Data types (FP32, FP16 - ensuring auto-casting works).
        - Edge cases (Zero-sized tensors, unaligned memory).
- [x] **New Workloads**:
    - [x] **Computer Vision**: Implement and test a standard CNN (MNIST/ResNet).
        - [x] Add `npu_conv2d` and `npu_maxpool2d` operators.
    - [ ] **Encoder Models**: Test a BERT-like encoder (Self-Attention without causal mask).
- [ ] **Stress Testing**:
    - [ ] **Memory Leak Check**: Run inference in a loop for 10,000+ iterations to verify stable memory usage.
    - [ ] **Context Switching**: Test interleaving calls between two different compiled models.

## Phase 4: Advanced Features & Optimization (Long-term)
*Goal: maximize performance and hardware utilization.*

- [ ] **Stateful Execution**:
    - [ ] **KV Cache**: Investigate OpenVINO `ReadValue`/`Assign` ops to keep KV cache on the NPU memory, avoiding the expensive "Slice -> Concat -> Copy back" loop.
- [ ] **Quantization**:
    - [ ] Add support for `int8` weight loading and execution.
- [ ] **Async Inference**:
    - [ ] Expose OpenVINO's async API to Python to allow overlapping CPU pre-processing with NPU execution.

## Phase 5: CI/CD & Documentation
- [ ] **CI Pipeline**: Create GitHub Actions to build wheel and run `pytest`.
- [ ] **Benchmarks**: Create `benchmarks/` folder with scripts to track latency (ms) and throughput (tok/s) across commits.
- [ ] **API Documentation**: Auto-generate docs from docstrings (Sphinx/MkDocs).

## Phase 6: Training & Fine-Tuning Support (Future)
- [ ] **Backward Pass Implementation**: Map PyTorch autograd gradients to OpenVINO backward ops (if available) or implement custom C++ backward kernels.
- [ ] **Optimizer Offloading**: Offload Adam/SGD parameter updates to the NPU.
