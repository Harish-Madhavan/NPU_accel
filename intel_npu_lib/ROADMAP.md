# 🗺️ Strategic Roadmap: Intel NPU Acceleration Library

This document outlines the engineering roadmap and strategic milestones to transition the **Intel NPU Acceleration Library** into a world-class, production-grade deep learning compiler. The library focuses on optimizing execution architectures, minimizing CPU bottlenecks, and achieving maximum NPU compute core saturation.

---

## 📊 Roadmap Phase Summary

| Phase | Strategic Objective | Target Complexity | Primary Performance Metric | Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Housekeeping & Modernization | 🟢 Low (2/5) | 📉 Stability & Maintainability | **100% Completed** |
| **2** | Compiler Architecture Refinement | 🟡 Medium (3/5) | ⚡ Zero Compilation Latency | **100% Completed** |
| **3** | Workload QA & Expanded Testing | 🟡 Medium (3/5) | 🧪 100% Core Operator Coverage | **80% In Progress** |
| **4** | Stateful Decoders & Hardware Offloading | 🔴 High (4.5/5) | 🚀 **2.0x - 2.5x** LLM Generation Speedup | **75% In Progress** |
| **5** | CI/CD Infrastructure & API Docs | 🟢 Low (2/5) | 📈 Development Lifecycle velocity | **50% In Progress** |
| **6** | Hardware autograd & Backpropagation | 🔴 High (5/5) | 🏋️ Native On-Device Fine-tuning | **0% Backlog** |

---

## 🟢 Phase 1: Housekeeping & Modernization (Immediate)
*Goal: Eliminate technical debt, modernize build systems, and enforce C++/Python type safety.*

- [x] **Build System Cleanup**:
    - [x] Consolidate C++ builds into a single unified `setup.py` utilizing Python's modern `setuptools` build pipeline (purged legacy `CMakeLists.txt`).
    - [x] Autoclean workspace roots from temporary target directory builds.
- [x] **Code Quality & Diagnostics**:
    - [x] **Logging**: Swapped legacy print diagnostics for structured C++ and Python logging macros.
    - [x] **Formatters**: Integrated `clang-format` rules for C++ and `ruff` formatting rules for Python.
    - [x] **Static Type Checkers**: Added fully comprehensive type annotations across functional APIs and validated via `mypy`.
- [x] **C++ Extension Refactoring**:
    - [x] **Thread-Safe Singleton Contexts**: Encapsulated global device caches and core context managers inside a thread-safe `NPUBackend` singleton wrapper.
    - [x] **RAII Memory Management**: Wrapped raw pointers into C++ standard smart pointers (`std::shared_ptr` / `std::unique_ptr`) to avoid resource leaks.

---

## 🟢 Phase 2: Compiler Architecture Refinement (Short-term)
*Goal: Modularize the tracing compiler, enhance diagnostic logging, and implement dynamic shape bucketing.*

- [x] **Compiler Decoupling**:
    - [x] Isolated operator converters into a registration-based [registry.py](file:///E:/Git%20Repo/NPU_accel/intel_npu_lib/src/intel_npu_acceleration/registry.py) file.
    - [x] Modularized graph translation logic into a dedicated [graph_builder.py](file:///E:/Git%20Repo/NPU_accel/intel_npu_lib/src/intel_npu_acceleration/graph_builder.py) file.
    - [x] Created clean user-facing compile entry points in [frontend.py](file:///E:/Git%20Repo/NPU_accel/intel_npu_lib/src/intel_npu_acceleration/frontend.py).
- [x] **High-Fidelity Diagnostic Errors**:
    - [x] Implemented python-derived `NPUCompilationError` exceptions with clear, context-specific failure causes to simplify diagnostics.
- [x] **Dynamic Shape Bucketing**:
    - [x] Engineered `NPUDynamicGraphModule` to automatically match variable-length input sequences to static compiled NPU buckets, eliminating driver-level recompilation penalties while preserving execution speed.

---

## 🟡 Phase 3: Expanded Testing & QA (Medium-term)
*Goal: Guarantee production-grade reliability under complex topologies and strict mathematical testing.*

- [ ] **Test Infrastructure**:
    - [ ] **Parametrized Multi-Dimension Tests**: Build parameterized matrix tests validating input shapes from 1D up to 5D across dynamic types (FP32, FP16, INT8, UINT8) to verify precision auto-casting.
    - [ ] **Boundary Conditions**: Add QA testing for extreme boundaries, including zero-sized inputs, unaligned arrays, and u8 image boundary clamping.
- [x] **Diversified Model Workloads**:
    - [x] **Computer Vision (CNNs)**: Fully verified compiling and executing standard CNN blocks (ResNet, Conv2D, MaxPool2d, BatchNorm2d) on NPU hardware.
    - [x] **Transformer Encoders**: Successfully compiled and verified BERT-style self-attention modules under `strict=True` compilation mode.
- [ ] **Stress and Lifecycle Testing**:
    - [ ] **Memory Leak Audits**: Run continuous inference loops for 10,000+ generations, logging NPU virtual memory parameters to verify stable hardware state lifetimes.
    - [ ] **Context-Switching Stability**: Assert correct outputs and stable memory footprints when dynamically interleaving execution between multiple distinct compiled NPU models.

---

## 🔴 Phase 4: Advanced Features & Optimization (Long-term)
*Goal: Eliminate data pipeline host synchronization penalties, offload image transformations, and compress network representations.*

- [x] **Stateful NPU KV-Caches (`NPUStatefulKVCache`)**:
    - [x] **Stateful Registers**: Implemented OpenVINO's `ReadValue` and `Assign` operators in the compiled graph to maintain KV state directly inside NPU hardware registers, dropping CPU host-to-device context synchronization to **zero**.
    - [x] **Hardware Precision Compliance**: Encapsulated state register indexing as floating-point float32 variables, casting to `int32` only inside transient address calculation nodes to satisfy strict NPU hardware constraints.
    - [ ] **Multi-Layer State Pipelines**: Automatically map multi-decoder stateful caches and custom attention modules during full model tracing.
- [x] **Asynchronous Execution Streams**:
    - [x] **Multi-Request Buffering**: Exposed OpenVINO's async infer queue to Python, enabling multi-request buffering to overlap CPU pre-processing with NPU execution.
    - [x] **Round-Robin Output Double-Buffering**: Built automatic double-buffering per execution stream to ensure 100% memory safety for concurrent inferences.
    - [ ] **Weak-Reference Output Leases**: Transition output double-buffering to a scalable leased buffer pool using weak references for arbitrary concurrent execution bounds.
- [ ] **INT8 Post-Training Quantization (PTQ)**:
    - [ ] **NNCF Compression Integration**: Integrate OpenVINO's Neural Network Compression Framework (NNCF) to compile model weights into native Intel NPU low-precision matrix arithmetic (DP4A/VNNI) representations, yielding **up to a 2x speedup** over FP16.
- [x] **Hardware Pre-Post Processing (PPP) Offloading**:
    - [x] **Image Pre-Processing Offloading**: Baked NHWC-to-NCHW transposition, precision casting, and mean/scale normalizations directly into the compiled graph inputs, resolving host CPU pre-processing starvation.
    - [ ] **Dynamic Pre-Post Processing (PPP)**: Add support for dynamic hardware-level image cropping, resizing, and padding inside the NPU PPP pipeline.

---

## 🟡 Phase 5: CI/CD Infrastructure & API Documentation
*Goal: Automate build verifications and scale developer documentation.*

- [x] **CI Pipeline Automation**:
    - [x] Setup multi-platform GitHub Actions building distributable wheels and running `pytest` suites on Windows and Ubuntu runners.
- [ ] **Regression Performance Profilers**:
    - [ ] Create a dedicated `benchmarks/` pipeline to capture latency profiles (ms), memory usage, and throughput (tokens/second) across every commit to prevent performance regressions.
- [ ] **Automated API Documentation**:
    - [ ] Implement MkDocs or Sphinx pipelines to auto-generate responsive, modern developer documentation directly from codebase docstrings.

---

## 🔴 Phase 6: Training & Backpropagation Support (Future)
*Goal: Unlock low-power training and fine-tuning capabilities directly on client AI PCs.*

- [ ] **Hardware-Accelerated Gradients**:
    - [ ] Register backward-pass graph converters mapping standard PyTorch autograd gradients to high-performance OpenVINO backward operations or custom C++ backpropagation kernels.
- [ ] **Optimizer State Offloading**:
    - [ ] Offload optimizer weight updates (SGD, Adam) to run directly on the NPU compute cores to avoid host memory round-trips during model fine-tuning.
