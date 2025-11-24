# Roadmap: Intel NPU Acceleration Library (OpenVINO Backend)

This document outlines the steps required to transform this skeleton library into a functional accelerator using **OpenVINO**. This approach abstracts low-level driver details and allows running operations on the NPU (and CPU/GPU) using a unified API.

## 1. Architecture Overview
The library uses OpenVINO Runtime (`openvino` C++ API) to execute PyTorch operators on the NPU.

**Flow:**
1.  **PyTorch Op called** (e.g., `npu_add`).
2.  **Wrap Tensors:** Wrap PyTorch memory (CPU pointer) into `ov::Tensor` to avoid copies where possible.
3.  **Graph Construction:** 
    *   *Option A (Eager-like):* Construct a tiny OpenVINO model for the specific operation on-the-fly (using `ov::opset`).
    *   *Option B (Graph):* Capture a sub-graph (TorchScript/FX) and convert it to OpenVINO IR.
4.  **Execution:** Send to NPU via `ov::InferRequest`.

## 2. Integration Steps

### Step A: Install Dependencies
No manual SDK installation is required! The build system now automatically handles OpenVINO.

1.  **Build System:** We use `pyproject.toml` to define build dependencies.
2.  **Runtime:** When you install this library, `openvino` will be installed automatically via pip.
3.  **Linking:** `setup.py` is configured to find the OpenVINO C++ headers and libraries directly from the installed Python package.

### Step B: Build and Install
Simply run:
```bash
pip install .
```
This will:
1.  Install `torch` and `openvino` (if missing).
2.  Locate the OpenVINO libraries.
3.  Compile the C++ extension linking against them.

### Step C: Implement Device Management (`csrc/device.cpp`)
1.  Include `<openvino/openvino.hpp>`.
2.  Instantiate `ov::Core`.
3.  Check `core.get_available_devices()` for "NPU".

### Step D: Implement Kernels (`csrc/ops.cpp`)
For a simple `add` operation:
1.  **Create Op:** Use `ov::op::v1::Add`.
2.  **Build Model:** Create `ov::Model` from the op.
3.  **Compile:** `core.compile_model(model, "NPU")`.
4.  **Infer:**
    *   Create `ov::Tensor` from PyTorch data pointers: `ov::Tensor(type, shape, ptr)`.
    *   `infer_request.set_input_tensor(...)`.
    *   `infer_request.infer()`.

## 3. Future: Zero-Copy Optimization
OpenVINO supports "remote tensors" (using Level Zero backing) for zero-copy networking between PyTorch and NPU. This requires advanced usage of `ov::intel_gpu::ocl::ClContext` or NPU equivalents when they become public API.