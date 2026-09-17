# 🏁 Quick-Start & Environment Setup Guide

This guide details how to install, configure, and verify the **Intel NPU Acceleration Library** on Windows and Linux, exploring the core execution models and disk caching.

---

## 💻 Hardware & Software Setup

### 1. Prerequisite Compilers
Because the library builds a native C++ extension (`_C.pyd` on Windows / `_C.so` on Linux), you must have a C++ compiler configured:
*   **Windows 11**: Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/) and select **"Desktop development with C++"** in the installer.
*   **Linux (Ubuntu 22.04+)**: Run `sudo apt update && sudo apt install build-essential python3-dev ninja-build` to configure GCC and Ninja.

### 2. Install Package in Development Mode
Editable installation is recommended for developers so that changes in Python source files are reflected instantly:
```bash
# Clone and navigate
git clone https://github.com/Harish-Madhavan/NPU_accel.git
cd NPU_accel/intel_npu_lib

# Install Python packages and inplace C++ library
# (CPU torch keeps the download small; install it first)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openvino "numpy>=1.24"

# Windows: configures MSVC automatically (supports --clean / --no-install)
#   ..\build_npu.bat
# Any OS:
python setup.py build_ext --inplace
pip install --no-build-isolation --no-deps -e .
```

Build knobs: `NPU_NO_BUILD_EXT=1` skips the native extension (pure-Python
fallback), `NPU_VERBOSE_BUILD=1` prints OpenVINO discovery details, and
`OPENVINO_DIR`/`INTEL_OPENVINO_DIR` point at a non-PyPI toolkit install.
Single source of truth: package metadata lives in `pyproject.toml`;
`setup.py` only defines how to build the `_C` extension.

### 3. Verification
Verify that the package can bind to the C++ Level Zero driver and load successfully:
```python
import intel_npu_acceleration as npu

print(f"Is NPU backend available: {npu.is_available()}")
```
```bash
# Equivalent terminal diagnostics (also installed as `intel-npu-info`):
python -m intel_npu_acceleration --info
```

---

## ⚡ Eager vs. Graph Execution Modes

The library provides two distinct ways to leverage Intel NPU hardware:

### 1. Eager Execution (Granular Ops)
For testing individual mathematical kernels, processing small inputs, or experimenting with standalone parameters, you can dispatch functional operators directly:
```python
import torch
import intel_npu_acceleration as npu

a = torch.randn(4, 4, dtype=torch.float16)
b = torch.randn(4, 4, dtype=torch.float16)

# Execute isolated MatMul on NPU
c = npu.matmul(a, b)

# Apply Sigmoid activation on NPU
d = npu.sigmoid(c)
```
> [!NOTE]
> Eager operations carry a minor overhead from the Level Zero driver queue scheduling (~0.1 ms). For deep neural networks, **Graph Compilation** is recommended.

### 2. Graph Compilation (Flipped Modules)
The most performant execution model. A single call traces your `nn.Module` using PyTorch `torch.fx`, fuses adjacent operators, and compiles them into a single binary executable:
```python
import torch
import intel_npu_acceleration as npu

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Trace, optimize, and compile for NPU
model = MLP().eval()
x_example = torch.randn(1, 256)
npu_model = npu.compile(model, x_example)

# Run ultra-fast fused inference
out = npu_model(x_example)
```

---

## 💾 Model Caching (Disk Persistence)

NPU model compilation takes several seconds because OpenVINO compiles the mathematical graph into Intel Level Zero hardware instructions. To prevent this compilation startup latency on subsequent runs, the library utilizes **Disk Caching**:

### 1. Default Behavior
By default, the library creates a `./npu_cache` directory in your current working directory. The cached files contain the compiled OpenVINO IR binary (`.blob` files), which loads in **under 1 millisecond** on future startups.

### 2. Configuring Cache Path
You can customize the cache path through either the Python API or environment variables:

**Via Python API:**
```python
import intel_npu_acceleration as npu

npu.set_cache_dir("C:/my_ai_models/npu_cache")
```

**Via Environment Variable:**
```bash
# Windows Command Prompt
set INTEL_NPU_CACHE_DIR=C:\my_ai_models\npu_cache

# PowerShell
$env:INTEL_NPU_CACHE_DIR="C:\my_ai_models\npu_cache"

# Linux
export INTEL_NPU_CACHE_DIR="/home/user/my_ai_models/npu_cache"
```
(`NPU_CACHE_DIR` is accepted as an alias. Explicit env vars take precedence
over the default `./npu_cache` and are created on demand at import.)
