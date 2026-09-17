# 🚀 Intel NPU Acceleration Library

![Intel NPU Acceleration Banner](docs/assets/banner.png)

[![Intel](https://img.shields.io/badge/Intel-OpenVINO-blue)](https://github.com/openvinotoolkit/openvino)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![NPU](https://img.shields.io/badge/Hardware-Intel--NPU-brightgreen)](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html)

**Intel NPU Acceleration** is a high-performance, drop-in PyTorch backend extension that bridges standard PyTorch workflows and the dedicated **Neural Processing Unit (NPU)** found in modern Intel® Core™ Ultra processors.

> **Zero Learning Curve**: Designed to require **no knowledge beyond standard PyTorch**. All low-level Intel oneAPI Level Zero optimizations (Turbo frequency boost, aggressive QDQ operator fusion, zero-copy buffer leasing, multi-stream parallel command queues, and high-priority micro-scheduler dispatch) are **automatically enabled out of the box**.

---

## 📑 Table of Contents
- [✨ Out-of-the-Box Optimizations](#-out-of-the-box-optimizations)
- [💻 System Requirements](#-system-requirements)
- [🛠️ Installation](#-installation)
- [🚀 Quick Start (Pure PyTorch)](#-quick-start-pure-pytorch)
- [🏋️ Seamless Training on NPU](#️-seamless-training-on-npu)
- [💾 Stateful LLM KV-Caching](#-stateful-llm-kv-caching)
- [📋 Supported Operators](#-supported-operators)
- [🧪 Testing & Verification](#-testing--verification)

---

## ✨ Out-of-the-Box Optimizations

When you use `intel_npu_acceleration`, the following hardware optimizations are applied automatically by default:

* **⚡ Automatic Level Zero Turbo Boost:** Operates NPU matrix execution units at peak turbo frequencies.
* **🛡️ Driver Memory Pool Preservation:** Disables idle memory pruning (`NPU_DISABLE_IDLE_MEMORY_PRUNING=YES`) to eliminate TLB invalidation and deallocation latency.
* **🚀 Concurrent Hardware Command Queues:** Unlocks parallel non-blocking execution across multi-stream hardware queues (`NPU_RUN_INFERENCES_SEQUENTIALLY=NO`).
* **🧠 Aggressive Operator & Quantization Fusion:** Applies optimization level 2 compilation (`NPU_QDQ_OPTIMIZATION_AGGRESSIVE=YES`).
* **🔄 Zero-Copy Buffer Leasing:** Reuses pre-allocated device memory to eliminate host-to-device copying overhead.
* **📦 Standard PyTorch Device Parity:** Familiar `is_available()`, `device_count()`, `get_device_name()`, `empty_cache()`, and `synchronize()` APIs.

---

## 💻 System Requirements

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Processor** | Intel® Core™ Ultra (Series 1 or newer) | Intel® Core™ Ultra 7 / 9 |
| **NPU Driver** | Intel® NPU Driver version 31.0.100.x | Latest version from [Intel Support](https://www.intel.com/content/www/us/en/support/articles/000095856/processors.html) |
| **OS** | Windows 11 (64-bit) / Ubuntu 22.04 LTS | Windows 11 |
| **Python** | 3.10 | 3.11+ / 3.12+ / 3.13+ |
| **PyTorch** | 2.1.0 | 2.2.0+ / 2.6.0+ |
| **OpenVINO** | 2023.3 | 2024.x+ |

---

## 🛠️ Installation

### Prerequisites
- **Python** 3.10+ (3.11 / 3.12 recommended)
- **Intel NPU driver** 31.0.100.x+ ([Intel Support](https://www.intel.com/content/www/us/en/support/articles/000095856/processors.html))
- **Windows**: Visual Studio 2022 with *Desktop development with C++* (provides `cl.exe`), or run `build_npu.bat` which locates it via `vswhere`
- **Linux**: `sudo apt install build-essential python3-dev`

### Option A — Editable install (recommended for development)
```bash
# Clone the repository
git clone https://github.com/Harish-Madhavan/NPU_accel.git
cd NPU_accel/intel_npu_lib

# CPU-only PyTorch is enough (much smaller download than the default CUDA wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openvino "numpy>=1.24"

# Windows (configures MSVC automatically and builds the C++ extension):
#   ..\build_npu.bat
# Any OS / manual build:
python setup.py build_ext --inplace
pip install --no-build-isolation --no-deps -e .
```

Environment knobs for the build:
| Variable | Effect |
| :--- | :--- |
| `NPU_NO_BUILD_EXT=1` | Skip the C++ extension (pure-Python / CPU-fallback install) |
| `NPU_VERBOSE_BUILD=1` | Verbose OpenVINO header/lib discovery logging |
| `OPENVINO_DIR` / `INTEL_OPENVINO_DIR` | Extra hint for a non-PyPI OpenVINO toolkit install |

### Option B — Build a wheel
```bash
cd NPU_accel/intel_npu_lib
pip install build
python -m build --wheel --no-isolation
pip install dist/*.whl
```

### Verify NPU Availability
```python
import intel_npu_acceleration as npu

print(f"NPU Available : {npu.is_available()}")
print(f"Device Name   : {npu.get_device_name()}")
print(f"Device Count  : {npu.device_count()}")
```
Or from a terminal (also available as `intel-npu-info` after install):
```bash
python -m intel_npu_acceleration --info
```

### Troubleshooting
| Symptom | Fix |
| :--- | :--- |
| `cl.exe not found` / MSVC error | Install VS 2022 *Desktop development with C++*, or run from a Developer Prompt / `build_npu.bat` |
| OpenVINO headers/libs not found (stub-mode warning) | `pip install "openvino>=2024.0.0"`, or set `OPENVINO_DIR` to a toolkit install |
| `pip install -e .` downloads a huge CUDA torch | Pre-install CPU torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Compiled-model cache grows unbounded | Set `INTEL_NPU_CACHE_DIR` (or `NPU_CACHE_DIR`) to a dedicated folder; prune with `python -m intel_npu_acceleration --clear-cache` |
| No NPU hardware | Library still imports and falls back to CPU; `npu.is_available()` returns `False` |

---

## 🚀 Quick Start (Pure PyTorch)

### 1. Standard PyTorch Syntax (`model.to("npu")` & `torch.compile`)

Use your normal PyTorch patterns. The library automatically routes execution to the Intel NPU:

```python
import torch
import intel_npu_acceleration  # Auto-registers 'npu' device, torch.npu, and Dynamo backend!

# Option A: Standard PyTorch device movement (.to("npu"))
model = torch.nn.Linear(128, 10).to("npu")
x = torch.randn(2, 128, device="npu")
out = model(x)

# Option B: PyTorch 2.x torch.compile
compiled_model = torch.compile(model, backend="npu")
out2 = compiled_model(x)

# Option C: Universal PyTorch Accelerator (PyTorch 2.4+)
assert torch.accelerator.is_available()
torch.accelerator.synchronize()
```

### 2. Standard `torch.npu` APIs

```python
import torch
import intel_npu_acceleration

# Device inspection & memory management
print(f"NPU Available : {torch.npu.is_available()}")
print(f"Device Name   : {torch.npu.get_device_name(0)}")
print(f"Device Count  : {torch.npu.device_count()}")

# Level Zero command streams and synchronization
with torch.npu.device(0):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = model(x)
    stream.synchronize()

torch.npu.synchronize()
torch.npu.empty_cache()

# Automatic Mixed Precision
with torch.autocast(device_type="npu", dtype=torch.float16):
    out = model(x)
```

---

## 🏋️ Seamless Training on NPU

Both forward inference and backward autograd passes are accelerated on the NPU, paired with fused hardware optimizers (`NPUAdam`, `NPUSGD`):

```python
import torch
import intel_npu_acceleration as npu
from intel_npu_acceleration.optim import NPUAdam

# Build model using NPU drop-in layers (or standard PyTorch modules)
model = torch.nn.Sequential(
    npu.nn.Linear(128, 64),
    npu.nn.RMSNorm(64),
    npu.nn.Linear(64, 10),
)

criterion = npu.nn.MSELoss()
optimizer = NPUAdam(model.parameters(), lr=1e-3)

x = torch.randn(16, 128)
target = torch.randn(16, 10)

# Training loop: forward, backward gradients, and parameter updates all run on NPU
for epoch in range(10):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()      # Level Zero accelerated backward pass!
    optimizer.step()     # Fused NPU hardware optimizer step!
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## 💾 Stateful LLM KV-Caching

Eliminates CPU-to-NPU bus memory transfer latency by maintaining autoregressive KV-cache tensors directly inside the NPU hardware registers:

```python
import torch
import intel_npu_acceleration as npu

# Instantiate stateful cache for autoregressive decoding
kv_cache = npu.nn.NPUStatefulKVCache(
    batch_size=1, max_seq_len=2048, num_heads=32, head_dim=64
)

# Compile stateful model with zero CPU host-to-device cache transfer overhead
npu_model = npu.compile(kv_cache, torch.randn(1, 1, 32, 64), strict=True)
```

---

## 📋 Supported Operators

For a full operator matrix comparing Eager Mode, Graph Mode, and Autograd Backward passes, see [SUPPORTED_OPS.md](SUPPORTED_OPS.md).

---

## 🧪 Testing & Verification

Run the full automated test suite:
```powershell
pytest
```
* **Status**: 148 tests passing with 0 warnings (upstream `torch.jit.trace` deprecation warnings from OpenVINO conversion are suppressed both in-library and via pytest `filterwarnings`).
