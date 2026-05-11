# 🚀 Intel NPU Acceleration Library

![Intel NPU Acceleration Banner](docs/assets/banner.png)

[![Intel](https://img.shields.io/badge/Intel-OpenVINO-blue)](https://github.com/openvinotoolkit/openvino)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![NPU](https://img.shields.io/badge/Hardware-Intel--NPU-brightgreen)](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html)

**Intel NPU Acceleration** is a high-performance PyTorch extension that bridges the gap between Python-based deep learning workflows and the dedicated **Neural Processing Unit (NPU)** found in modern Intel® Core™ Ultra processors.

By leveraging the **OpenVINO™ Runtime** backend, this library allows developers to seamlessly offload compute-intensive operations, such as Large Language Model (LLM) inference, to the NPU, freeing up the CPU and GPU for other tasks while maintaining exceptional power efficiency.

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
-   **🤖 LLM-First Optimization:** Specialized support for **RMSNorm**, **RoPE**, and **KV-Cache** management, enabling high-performance LLM inference on edge devices.
-   **🔄 Seamless Integration:** Works with standard PyTorch `Tensor` objects and requires no changes to your existing model definition.
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

### 1. Model Compilation (Recommended)

The `compile` API is the most powerful way to use the library. It analyzes your model's computational graph and fuses operations for the NPU.

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

# Compile the model
# The example_input is used to infer tensor shapes and data types
npu_model = npu.compile(model, example_input)

# High-performance inference
with torch.no_grad():
    output = npu_model(example_input)

### 2. Low-Level Eager Execution

For more granular control or for testing individual kernels, you can use NPU-accelerated operators directly on PyTorch Tensors. This mode avoids the tracer/compiler overhead for one-off operations.

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
```

---

## 📋 Supported Operators

The library supports a wide range of operators through its **Graph Compiler**. If an operator is not supported by the NPU, the compiler will raise a `NPUCompilationError` during the compilation phase.

| Category | Operators |
| :--- | :--- |
| **Arithmetic** | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`, `Rsqrt`, `Mean` |
| **Linear Algebra** | `MatMul`, `Linear`, `Transpose`, `MM` |
| **Activations** | `ReLU`, `GELU`, `SiLU` (Swish), `Softmax` |
| **Vision (Beta)** | `Conv2d`, `MaxPool2d`, `BatchNorm2d` |
| **LLM Specific** | `SDPA` (Attention), `RMSNorm`, `KV-Cache Update`, `Sin/Cos` (RoPE) |
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
    Uses `torch.fx.symbolic_trace` to capture the PyTorch model as a graph. It propagates shapes through the graph to ensure all tensor dimensions are resolved before compilation.
2.  **Bridge (Python/OpenVINO):**
    Iterates through the FX graph nodes and translates them into **OpenVINO Opsets**. This layer handles complex transformations, such as converting PyTorch's `scaled_dot_product_attention` into an optimized sequence of OpenVINO kernels.
3.  **Backend (C++/Core):**
    A thin, high-performance wrapper around the **OpenVINO C++ Runtime API**. It manages the singleton NPU device context, handles asynchronous execution requests, and interacts with the disk-based model cache.

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
You are trying to compile a model that contains an operator not yet implemented in our converter registry.
-   **Solution:** Check the [Supported Operators](#-supported-operators) table. You can contribute a new converter in `src/intel_npu_acceleration/converters.py`.

---

## 🌟 Featured Workload: TinyLlama

Want to see the library in action? Our [TinyLlama example](../examples/tiny_llama.py) demonstrates:
-   **Flash-Attention like performance** using the `SDPA` operator.
-   **Efficient state management** with the custom `update_kv_cache` functional op.
-   **High-speed generation** on low-power NPU hardware.

---

## 🗺️ Roadmap & Contributing

We welcome contributions! If you'd like to help expand the library:
1.  **Check the [ROADMAP.md](ROADMAP.md)** for planned features and technical debt.
2.  **Add a new operator:** Check `registry.py` and `converters.py` for examples.
3.  **Report bugs:** Open an issue with a reproducing script and your hardware specs.

---

*Intel, the Intel logo, and OpenVINO are trademarks of Intel Corporation.*


