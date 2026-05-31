# 🚀 Intel NPU Acceleration Library

![Intel NPU Acceleration Banner](docs/assets/banner.png)

[![Intel](https://img.shields.io/badge/Intel-OpenVINO-blue)](https://github.com/openvinotoolkit/openvino)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![NPU](https://img.shields.io/badge/Hardware-Intel--NPU-brightgreen)](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html)

**Intel NPU Acceleration** is a high-performance PyTorch extension that bridges the gap between Python-based deep learning workflows and the dedicated **Neural Processing Unit (NPU)** found in modern Intel® Core™ Ultra processors.

By leveraging the **OpenVINO™ Runtime** backend, this library allows developers to seamlessly offload compute-intensive operations, such as Large Language Model (LLM) inference and computer vision pre-processing, to the NPU, freeing up the CPU and GPU while maintaining exceptional power efficiency.

---

## 📑 Table of Contents
- [✨ Key Features](#-key-features)
- [💻 System Requirements](#-system-requirements)
- [🛠️ Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📋 Supported Operators](#-supported-operators)
- [📂 Benchmarking](#-benchmarking)
- [🏗️ Architecture Deep Dive](#-architecture-deep-dive)
- [⚙️ Configuration & Environment](#-configuration--environment)
- [❓ Troubleshooting](#-troubleshooting)
- [🗺️ Roadmap & Contributing](#-roadmap--contributing)

---

## ✨ Key Features

-   **🎯 Zero-Effort Acceleration:** Compile entire `torch.nn.Module` objects with a single line of code.
-   **⚡ Dual Execution Modes:**
     -   **Graph Compilation:** Traces PyTorch models with `torch.fx`, optimizes the graph, and executes it as a fused OpenVINO executable.
     -   **Eager Ops:** Optimized C++ kernels for individual operations, accessible directly from Python.
-   **💾 Stateful NPU KV-Caches:** Offloads autoregressive LLM state and position indexing directly to NPU internal hardware registers (`ReadValue` and `Assign`), eliminating CPU-to-NPU bus latency.
-   **🛡️ Double-Buffered Output Safety:** Alternates outputs between two pre-allocated buffers per stream, guaranteeing 100% memory safety in multi-stream or concurrent executions without allocation overhead.
-   **🖼️ Hardware Pre-Post Processing (PPP):** Bakes layout transpositions (`NHWC` $\leftrightarrow$ `NCHW`), type casting, and mean/scale normalizations directly into the compiled graph, avoiding host CPU starvation.
-   **🧠 Compiler Performance Intelligence:** Proactively analyzes configuration at compile-time and prints warnings/guidance to help users maximize NPU compute core utilization.
-   **💾 Intelligent Caching:** Automatically caches compiled models to disk to ensure lightning-fast startup times.

---

## 💻 System Requirements

To ensure optimal performance and compatibility, please verify your system meets the following requirements:

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Processor** | Intel® Core™ Ultra (Series 1 or newer) | Intel® Core™ Ultra 7 / 9 |
| **NPU Driver** | Intel® NPU Driver version 31.0.100.x | Latest version from [Intel Support](https://www.intel.com/content/www/us/en/support/articles/000095856/processors.html) |
| **OS** | Windows 11 (64-bit) / Ubuntu 22.04 LTS | Windows 11 |
| **Python** | 3.10 | 3.11+ |
| **PyTorch** | 2.1.0 | 2.2.0+ |
| **OpenVINO** | 2023.3 | 2024.x |

---

## 🛠️ Installation

The library manages its own core dependencies, but requires a C++ build environment for the initial extension compilation.

### 1. Configure Build Environment
-   **Windows:** Install [Visual Studio 2022](https://visualstudio.microsoft.com/vs/community/) with "Desktop development with C++".
-   **Linux:** `sudo apt update && sudo apt install build-essential python3-dev`

### 2. Install the Package
```bash
# Clone the repository
git clone https://github.com/Harish-Madhavan/NPU_accel.git
cd NPU_accel/intel_npu_lib

# Install in editable mode (recommended for developers)
pip install -e .
```

### 3. Verify Installation
```python
import intel_npu_acceleration as npu
print(f"Intel NPU Available: {npu.is_available()}")
```

---

## 🚀 Quick Start

### 1. PyTorch Native Compilation (PyTorch 2.0+ torch.compile)

The library integrates natively with PyTorch 2.x. You can compile your models using standard `torch.compile` by setting the backend to `"npu"`:

```python
import torch
import intel_npu_acceleration as npu_lib

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 64)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))

model = MyModel().eval()
x = torch.randn(1, 128)

# Natively compile model
compiled_model = torch.compile(model, backend="npu")
output = compiled_model(x)

# Pass compiler options dynamically via options dict
compiled_opt = torch.compile(
    model,
    backend="npu",
    options={"clone_outputs": False, "performance_hint": "THROUGHPUT"},
)
output_opt = compiled_opt(x)
```

### 2. Frontend Compiler API

You can also compile your models explicitly using the library's `compile` API. This traces the computational graph under a specified example input and maps fused operators directly to the NPU:

```python
import torch
import intel_npu_acceleration as npu

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 64)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))

model = MyModel().eval()
example_input = torch.randn(1, 128)

# Compile using frontend API
npu_model = npu.compile(model, example_input)

with torch.no_grad():
    output = npu_model(example_input)
```

### 2. Stateful KV-Caching (Autoregressive LLM Inference)

Keep LLM context memory directly inside the NPU hardware registers, avoiding high bus latency.

```python
import torch
import torch.nn as nn
import intel_npu_acceleration as npu

class StatefulLLMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate stateful cache: batch_size=1, max_seq_len=1024, heads=8, dim=64
        self.kv_cache = npu.functional.NPUStatefulKVCache(
            batch_size=1, max_seq_len=1024, num_heads=8, head_dim=64
        )

    def forward(self, new_token_kv):
        return self.kv_cache(new_token_kv)

model = StatefulLLMDecoder().eval()
x_example = torch.randn(1, 1, 8, 64) # single token input

# Compile stateful model
npu_model = npu.compile(model, x_example, strict=True)

# Autoregressive generation loop (0 CPU host-to-device cache copies!)
for step in range(50):
    new_token = torch.randn(1, 1, 8, 64)
    cache_output = npu_model(new_token)

# Reset hardware state registers for the next prompt/sequence
npu_model.reset_states()
```

### 3. NPU Hardware Pre-Post Processing (PPP) Offloading

Bake layout and color conversions directly into the compiled NPU model.

```python
import torch
import intel_npu_acceleration as npu

# Compile with preprocess config baked in
preprocess_config = {
    "input": {
        "layout": "NHWC",          # User feeds uint8 NHWC image tensor (0..255)
        "element_type": "u8",
        "model_layout": "NCHW",    # NPU transposes and casts to float32 NCHW in hardware
        "mean": [123.675, 116.28, 103.53],
        "scale": [58.395, 57.12, 57.375],
    }
}

# Model expects standard float32 NCHW format
model = MyVisionModel().eval()
x_example = torch.randn(1, 3, 224, 224)
npu_model = npu.compile(model, x_example, preprocess_config=preprocess_config)

# Raw image uint8 NHWC tensor feed (Zero CPU overhead!)
img_tensor = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
output = npu_model(img_tensor)
```

### 4. Low-Level Eager Execution

For more granular control, use accelerated operators directly on PyTorch Tensors.

```python
import torch
import intel_npu_acceleration as npu

a = torch.randn(10, 10)
b = torch.randn(10, 10)

# Matrix multiplication on NPU
c = npu.matmul(a, b)

# Element-wise addition on NPU
d = npu.add(c, a)

# RMSNorm on NPU
weight = torch.ones(10)
e = npu.rmsnorm(d, weight, eps=1e-6)
```

---

## 📋 Supported Operators

The library supports a wide range of operators through its **Graph Compiler**. If an operator is not supported by the NPU, the compiler will raise a `NPUCompilationError` during compilation (or cleanly fall back to CPU if `strict=False`).

| Category | Operators |
| :--- | :--- |
| **Arithmetic** | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`, `Rsqrt`, `Mean` |
| **Linear Algebra** | `MatMul`, `Linear`, `Transpose`, `MM` |
| **Activations** | `ReLU`, `GELU`, `SiLU` (Swish), `Hardsigmoid`, `Hardswish`, `Softmax` |
| **Vision** | `Conv2d`, `MaxPool2d`, `BatchNorm2d` |
| **LLM Specific** | `SDPA` (Attention), `RMSNorm`, `KV-Cache Update`, `Sin/Cos` (RoPE), `StatefulKVCache` |
| **Tensor Ops** | `Reshape` (View), `Cat`, `Stack`, `IndexSelect`, `Where`, `Triu` |
| **Misc** | `Embedding`, `Clone`, `Full`, `Arange`, `GetItem` (Slicing) |

---

## 📂 Benchmarking

We provide a specialized stress-test script for measuring NPU throughput (TOPS) and latency.

```bash
python examples/benchmark_npu_matmul.py --size 4096 --iters 100 --dtype float16
```
This script compares NPU performance against the standard PyTorch CPU implementation and reports:
-   **Average Latency (ms)**
-   **Throughput (GOPS/TOPS)**
-   **Speedup factor over CPU**

---

## 🏗️ Architecture Deep Dive

The library operates through three distinct layers:

1.  **Frontend (Python/FX):** 
    Uses `torch.fx.symbolic_trace` via custom `NPUTracer` to capture the PyTorch model as a graph. Overrides leaf-module tracing to compile specialized modules like `NPUStatefulKVCache` atomically.
2.  **Bridge (Python/OpenVINO):**
    Iterates through the FX graph nodes and translates them into **OpenVINO Opsets**. Bakes pre-post processing configurations (type casts, transpositions, normalizations) directly into graph inputs.
3.  **Backend (C++/Core):**
    A thin, high-performance wrapper around the **OpenVINO C++ Runtime API**. Manages the singleton NPU device context, executes asynchronous requests, coordinates zero-copy double-buffering, and interacts with the disk-based cache.

---

## ⚙️ Configuration & Environment

The library can be configured via environment variables or direct API calls:

-   **Disk Cache:** By default, compiled models are stored in a `./npu_cache` directory.
    -   Override via environment: `INTEL_NPU_CACHE_DIR=/path/to/cache`
    -   Override via API: `npu.set_cache_dir("/path/to/cache")`
-   **Logging Level:**
    -   Set `LOGLEVEL=DEBUG` to see detailed compilation logs and operator mapping info.

---

## ❓ Troubleshooting

### 1. `ImportError: Could not load C++ extension`
This usually means the C++ extension wasn't built correctly or dependencies are missing.
-   **Solution:** Re-run `pip install -e .` and ensure you have a working C++ compiler. Check if `openvino` is installed.

### 2. `NPU device not found`
The `npu.is_available()` returns `False`.
-   **Solution:** Ensure you are on an Intel Core Ultra processor and have the latest **Intel NPU Driver** installed from the Intel website. Virtual machines often do NOT expose the NPU to the guest OS.

### 3. `NPUCompilationError: Function ... not supported`
You are trying to compile a model that contains an operator not yet implemented in our registry.
-   **Solution:** Check the [Supported Operators](#-supported-operators) table. You can contribute a new converter in `src/intel_npu_acceleration/converters.py`.

---

## 🌟 Featured Workload: TinyLlama

Our [TinyLlama example](../examples/tiny_llama.py) demonstrates:
-   **Flash-Attention like performance** using the `SDPA` operator.
-   **Stateful KV-cache offloading** using `NPUStatefulKVCache`.
-   **High-speed generation** on low-power Intel NPU hardware.

---

*Intel, the Intel logo, and OpenVINO are trademarks of Intel Corporation.*
