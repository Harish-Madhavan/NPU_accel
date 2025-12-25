# Intel NPU Acceleration Library

This library provides PyTorch acceleration support for Intel NPUs (Neural Processing Units) using the OpenVINO™ Runtime backend. It integrates seamlessly as a PyTorch extension, allowing users to offload compute-intensive operators and entire neural network subgraphs to the NPU.

## Key Features

*   **Dual Execution Modes:**
    *   **Eager Execution:** Direct access to highly optimized C++ kernels for individual operators (e.g., `npu_matmul`, `npu_add`).
    *   **Graph Compilation:** A `torch.fx` based compiler that traces PyTorch models, converts them to OpenVINO intermediate representation (IR), and executes them efficiently on the NPU.
*   **Seamless Integration:** Designed to work with standard PyTorch tensors and modules.
*   **LLM Ready:** Includes support for key Large Language Model operations like RMSNorm, RoPE components (Sin/Cos/Rotate), and KV-Cache updates.

## Supported Operators

The library currently supports the following operations via its compiler and C++ backend:

*   **Arithmetic:** `Add`, `Sub`, `Mul`, `Div`, `FloorDiv`, `Pow`, `Neg`, `Rsqrt`, `Mean`.
*   **Linear Algebra:** `MatMul` (`mm`), `Linear`.
*   **Activations:** `ReLU`, `GELU`, `SiLU` (Swish).
*   **Tensor Manipulation:** `Transpose`, `Reshape` (`view`), `Cat`, `Stack`, `Clone`, `Full`, `Arange`.
*   **Indexing & Control Flow:** `Where`, `IndexSelect`, `Triu` (for causal masks), `GetItem` (Slicing/Gather), `SetItem` (Update).
*   **Advanced:** `RMSNorm`, `Softmax`, `Embedding`.

## Installation

The package manages its own dependencies (including OpenVINO).

```bash
pip install .
```

*Note: Ensure you have a C++ compiler installed (e.g., MSVC on Windows, GCC/Clang on Linux) as this package builds a C++ extension.*

## Usage

### 1. Graph Compilation (Recommended)

The most efficient way to use the library is to compile standard PyTorch modules. This fuses operations and reduces Python overhead.

```python
import torch
import intel_npu_acceleration
from intel_npu_acceleration.compiler import compile_to_npu

# Define a standard PyTorch module
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 64)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = MyModel().eval()
input_data = torch.randn(1, 128)

# Compile the model for NPU
# We provide an example input to trace the graph and infer shapes
npu_model = compile_to_npu(model, input_data)

# Run inference on NPU
output = npu_model(input_data)
print("NPU Output:", output)
```

### 2. Eager Execution

You can also access individual NPU-accelerated operators directly.

```python
import torch
import intel_npu_acceleration._C as npu_ops

a = torch.randn(10, 10)
b = torch.randn(10, 10)

# Run matrix multiplication on NPU
c = npu_ops.npu_matmul(a, b)

# Run element-wise add
d = npu_ops.npu_add(c, a)
```

## Architecture

This library bridges PyTorch and OpenVINO:
1.  **C++ Extension (`csrc/`):** Implements the Python bindings and the interface to the OpenVINO C++ Runtime. It handles tensor wrapping (zero-copy where possible) and dynamic model caching.
2.  **Compiler (`compiler.py`):** Uses `torch.fx` to trace the Python code, captures the execution graph, and rebuilds it using OpenVINO opsets. This graph is then compiled once and executed repeatedly.

