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

```bash
# Clone the repository
git clone https://github.com/Harish-Madhavan/NPU_accel.git
cd NPU_accel/intel_npu_lib

# Install in editable mode
pip install -e .
```

### Verify NPU Availability
```python
import intel_npu_acceleration as npu

print(f"NPU Available : {npu.is_available()}")
print(f"Device Name   : {npu.get_device_name()}")
print(f"Device Count  : {npu.device_count()}")
```

---

## 🚀 Quick Start (Pure PyTorch)

### 1. PyTorch 2.x `torch.compile` (Standard Syntax)

You write standard PyTorch code. The library handles all NPU kernel compilation and Level Zero optimizations under the hood:

```python
import torch
import intel_npu_acceleration  # Automatically registers the 'npu' backend!

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 64)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = MyModel().eval()
x = torch.randn(1, 128)

# Pure PyTorch:
compiled_model = torch.compile(model, backend="npu")
output = compiled_model(x)
```

### 2. One-Line Accelerator / Module Extension

```python
# Option A: One-line accelerate helper
compiled_model = npu.accelerate(model)

# Option B: Direct module method
compiled_model = model.compile_npu(mode="reduce-overhead")
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
* **Status**: 120 test suites passing, 34 subtests passing with 0 warnings.
