# Intel NPU Acceleration Library Documentation

The `intel_npu_lib` is a custom PyTorch extension designed to accelerate tensor operations and neural network inference by offloading execution to Intel Neural Processing Units (NPUs) using OpenVINO.

## Architecture Overview

The library operates on two primary levels:
1. **Eager Mode Execution (`functional.py` & `csrc/ops.cpp`)**: Overrides basic PyTorch functions to execute directly on the NPU via C++ bindings. This uses OpenVINO's `Core::compile_model` with `LATENCY` and `f16` hints. It features true zero-copy outputs where OpenVINO writes results directly to pre-allocated `torch::Tensor` memory.
2. **Graph Compilation Mode (`frontend.py` & `converters.py`)**: Uses `torch.fx` to trace entire models (like LLMs or Vision models). It builds a holistic OpenVINO graph, avoiding Python dispatch overhead and enabling massive operator fusion.

---

## Supported Operations Parity Matrix

The following table lists the operations supported by the NPU library, comparing their availability in Eager Mode (direct C++ dispatch) vs. Graph Mode (FX compiled).

### 1. Element-wise & Math Operations
| PyTorch Operation | Eager Mode (`functional.py`) | Graph Mode (`converters.py`) | Notes |
| :--- | :---: | :---: | :--- |
| `torch.add` / `+` | ✅ | ✅ | Automatic type promotion handled in Python. |
| `torch.sub` / `-` | ✅ | ✅ | Automatic type promotion handled in Python. |
| `torch.mul` / `*` | ✅ | ✅ | Automatic type promotion handled in Python. |
| `torch.div` / `/` | ✅ | ✅ | |
| `torch.neg` / `-` | ✅ | ✅ | |
| `operator.floordiv` (`//`) | ❌ | ✅ | Handled via Floor(Divide(A, B)) in OpenVINO. |
| `torch.pow` / `**` | ❌ | ✅ | |
| `torch.sin`, `torch.cos` | ❌ | ✅ | |
| `torch.rsqrt` | ❌ | ✅ | |
| `torch.clamp`, `hardtanh` | ❌ | ✅ | Maps to `ops.maximum` and `ops.minimum`. |
| `torch.where` | ❌ | ✅ | |
| `torch.triu` | ❌ | ✅ | Uses complex slicing and `range` thresholds. |

### 2. Matrix & Neural Network Operations
| PyTorch Operation | Eager Mode (`functional.py`) | Graph Mode (`converters.py`) | Notes |
| :--- | :---: | :---: | :--- |
| `torch.matmul`, `torch.mm` | ✅ | ✅ | |
| `torch.nn.functional.linear` | ✅ | ✅ | Decomposed to `MatMul + Add` in FX graph. |
| `torch.nn.functional.conv2d` | ✅ | ❌ | Currently eager-only dispatch support. |
| `torch.nn.functional.max_pool2d` | ✅ | ❌ | |
| `torch.nn.functional.scaled_dot_product_attention` | ✅ | ✅ | Natively uses OpenVINO's SDPA node. |
| `torch.nn.functional.dropout` | ✅ | ✅ | Treated as Identity (No-op) during inference. |

### 3. Activations & Normalization
| PyTorch Operation | Eager Mode (`functional.py`) | Graph Mode (`converters.py`) | Notes |
| :--- | :---: | :---: | :--- |
| `torch.relu`, `F.relu` | ✅ | ✅ | |
| `torch.nn.functional.gelu` | ✅ | ✅ | Supports both `erf` and `tanh` approximations. |
| `torch.nn.functional.silu` | ✅ | ✅ | Maps to OpenVINO `swish` operation. |
| `torch.softmax`, `F.softmax` | ✅ | ✅ | |
| `rmsnorm` (Custom) | ✅ | ✅ | Manually mapped using Variance, Mean, Divide. |
| `torch.nn.functional.layer_norm` | ✅ | ✅ | Manually mapped using Variance, Mean, Divide. |

### 4. Tensor Manipulation, Shape, & Indexing
| PyTorch Operation | Eager Mode (`functional.py`) | Graph Mode (`converters.py`) | Notes |
| :--- | :---: | :---: | :--- |
| `torch.reshape`, `view` | ✅ | ✅ | |
| `torch.transpose` | ✅ | ✅ | |
| `torch.cat`, `torch.stack` | ✅ | ✅ | |
| `torch.mean` | ✅ | ✅ | |
| `torch.zeros`, `ones`, `full` | ❌ | ✅ | Supports dynamic sizing from FX Nodes. |
| `torch.arange` | ❌ | ✅ | |
| `torch.clone` | ❌ | ✅ | Identity operation in graph. |
| `operator.getitem` (Read) | ❌ | ✅ | Massive strided-slice support for all patterns. |
| `operator.setitem` (Write) | ❌ | ✅ | Maps to `ScatterNDUpdate` or `Concat` slices. |
| `torch.index_select` | ❌ | ✅ | Maps to `Gather`. |
| `index_copy` / `update_kv_cache` | ✅ | ✅ | Maps to `ScatterUpdate`. Highly optimized for LLMs. |
| `builtins.getattr` | ❌ | ✅ | Intercepts `.shape` and `.device` calls. |

---

## Build & Continuous Integration

### Build Acceleration
The C++ core (`csrc/`) requires OpenVINO headers. 
* **Windows**: Compiles with `/O2` and `/MP` for multi-core aggressive optimization.
* **Linux**: Compiles with `-O3`.
To build locally, use `ninja` for maximum speed:
```bash
pip install ninja
python setup.py build_ext --inplace
```

### CI Pipeline
The GitHub Actions CI pipeline runs across `windows-latest` and `ubuntu-latest` against `Python 3.10` and `3.11`. It uses:
1. **Pip Caching**: Avoids re-downloading PyTorch/OpenVINO wheels.
2. **Ninja**: Greatly speeds up C++ compilation time.
3. **Pytest-Xdist**: Executes the test suite in parallel (`-n auto`).
4. **Pytest-Cov**: Generates line-by-line coverage reports for the `intel_npu_acceleration` Python module.
