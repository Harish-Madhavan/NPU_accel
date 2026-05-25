# 🛠️ Developer & Contributor Guide

This guide details how to build, test, and contribute to the **Intel NPU Acceleration Library**. It includes adding custom operators, local building diagnostics, and the CI/CD integration framework.

---

## 🏗️ Local Compilation & Windows Builder

To build the native C++ library (`_C.pyd`) locally on Windows, we provide an optimized command script: **`build_npu.bat`** in the workspace root.

This script automatically locates Visual Studio 2022's build tools (`vcvars64.bat`), configures the compiler SDK environments, and compiles the C++ codebase:
```cmd
# Execute local builder from the workspace root
.\build_npu.bat
```

### 🔍 Compilation Logs & Success Validation
*   All compilation outputs and warnings are redirected to `intel_npu_lib/build_log.txt`.
*   The builder monitors compilation return codes (`%ERRORLEVEL%`). It prints a success message (`[SUCCESS] C++ Extension built...`) or blocks and alerts you of errors.

---

## 🧩 Adding New NPU Operators (OpenVINO Converters)

Adding support for a new PyTorch operator involves registering a **converter** that maps the PyTorch FX graph node into standard OpenVINO Opsets. All converters reside in [converters.py](file:///E:/Git%20Repo/NPU_accel/intel_npu_lib/src/intel_npu_acceleration/converters.py).

We support three types of converter registrations:

### 1. Function Converters
Maps standard functions (like `torch.sigmoid` or `torch.nn.functional.silu`).
```python
from .registry import OpRegistry
import openvino.opset13 as ops

@OpRegistry.register_function(torch.sigmoid, torch.nn.functional.sigmoid)
def convert_sigmoid(builder: OVGraphBuilder, node, args, kwargs):
    # Retrieve the OpenVINO input node
    inp = builder.get_input_or_constant(args[0])
    # Build and return the standard OpenVINO node
    return ops.sigmoid(inp)
```

### 2. Tensor Method Converters
Allows the operator to compile successfully when called as a method on a PyTorch Tensor object (e.g. `x.sigmoid()`).
```python
@OpRegistry.register_method("sigmoid")
def convert_sigmoid_method(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.sigmoid(inp)
```

### 3. PyTorch Module Converters
Maps PyTorch layer modules (such as `torch.nn.Sigmoid` or `torch.nn.ReLU6`).
```python
@OpRegistry.register_module(torch.nn.Sigmoid)
def convert_sigmoid_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    # submod contains the PyTorch layer instance (e.g., to read submod.negative_slope)
    inp = builder.get_input_or_constant(args[0])
    return ops.sigmoid(inp)
```

---

## 🧪 Testing Guidelines

We utilize `pytest` to run our test suite. All tests are located in `intel_npu_lib/tests/`.

### 1. Test Best Practices
*   **Contiguity**: Always ensure input tensors are contiguous (`.contiguous()`) during tests.
*   **Auto-Casting**: NPU handles calculations in FP16, so compare outputs using `torch.allclose(out_npu, out_expected, rtol=1e-2, atol=1e-2)`.
*   **Strict Verification**: Set `strict=True` inside `npu.compile(model, input, strict=True)` to guarantee your new operator is natively compiled to NPU instructions instead of silently falling back to the CPU!

### 2. Execute Local Test Suite
```bash
cd intel_npu_lib

# Run all 38 tests
python -m pytest

# Run a specific test with log printing
python -m pytest tests/test_basic.py -v -s
```

---

## 🌐 Production CI/CD Pipeline

Our GitHub Actions workflow [.github/workflows/ci.yml](file:///E:/Git%20Repo/NPU_accel/.github/workflows/ci.yml) manages code quality, linting, building, and test coverages:

### 1. CI Build Matrix
*   **Operating Systems**: Windows (`windows-latest`) and Linux (`ubuntu-latest`).
*   **Python Versions**: `3.10`, `3.11`, and `3.12`.

### 2. Compilation Caching
We leverage **`ccache`** to cache compiled C++ object files. This speeds up recurring pull request tests by eliminating redundant C++ compilation, dropping build times from minutes to seconds.

### 3. Artifact Archiving
The workflow automatically builds PEP 517 standard `.whl` distributable binary packages and generates XML/HTML test coverage reports, archiving them as downloadable GHA workflow artifacts.
