# Intel NPU Acceleration Library

This library provides PyTorch acceleration support for Intel NPUs (Neural Processing Units).
It integrates as a PyTorch extension, allowing users to offload supported operators to the NPU.

## Installation

```bash
pip install .
```

## Usage

```python
import torch
import intel_npu_acceleration

# Check if NPU is available
print(f"NPU Available: {intel_npu_acceleration.is_available()}")

# Example usage (once ops are implemented)
# x = torch.randn(10).to("npu")
```
