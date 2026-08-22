# Intel NPU Acceleration Library Documentation

The `intel_npu_acceleration` library is a high-performance PyTorch backend extension designed to accelerate tensor operations, deep learning training, and neural network inference by offloading execution directly to Intel® Neural Processing Units (NPUs) via oneAPI Level Zero and OpenVINO™.

---

## 🏛️ Architecture Overview

The library operates on three unified levels:

1. **Native PyTorch 2.x TorchDynamo Backend (`torch.compile(..., backend="npu")`)**:
   - Registered under `backend="npu"` and `backend="intel_npu"`.
   - Translates PyTorch compile modes:
     - `mode="reduce-overhead"` $\rightarrow$ Latency-optimized execution, zero-copy buffer leasing, hardware Turbo boost.
     - `mode="max-autotune"` $\rightarrow$ High-throughput pipelining, 4 parallel hardware streams, FP16 precision fusion.
     - `dynamic=True` $\rightarrow$ Dynamic shape compilation with dead placeholder symbol pruning.

2. **C++ Level Zero Driver Integration (`csrc/device.cpp` & `csrc/ops.cpp`)**:
   - Binds directly to the Intel Level Zero NPU runtime driver (`ov::intel_npu::level_zero::ZeroContext`, `ze_context_handle_t`).
   - Hardware Turbo clock frequency boost (`NPU_TURBO=YES`).
   - Unified Level Zero memory preservation (`NPU_DISABLE_IDLE_MEMORY_PRUNING=YES`) to eliminate TLB invalidation.
   - Non-blocking concurrent hardware command queues (`NPU_RUN_INFERENCES_SEQUENTIALLY=NO`).
   - High-priority hardware micro-scheduler dispatch (`MODEL_PRIORITY=HIGH`).

3. **End-to-End Training & Autograd Backend (`intel_npu_acceleration.autograd` & `optim`)**:
   - Both forward AND backward autograd passes run accelerated on the NPU.
   - Hardware-accelerated optimizers (`NPUAdam`, `NPUSGD`) execute parameter updates directly on device memory.
   - Drop-in PyTorch layers (`intel_npu_acceleration.nn`).

---

## 📋 Supported Operations Parity Matrix

### 1. Element-wise & Math Operations
| PyTorch Operation | Eager C++ Level Zero | Graph Mode (`torch.compile`) | Autograd Backward on NPU | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `torch.add` / `+` | ✅ | ✅ | ✅ | Automatic dtype promotion and unbroadcasting |
| `torch.sub` / `-` | ✅ | ✅ | ✅ | Automatic dtype promotion and unbroadcasting |
| `torch.mul` / `*` | ✅ | ✅ | ✅ | Automatic dtype promotion and unbroadcasting |
| `torch.div` / `/` | ✅ | ✅ | ✅ | |
| `torch.neg` / `-` | ✅ | ✅ | ✅ | |
| `torch.pow` / `**` | ✅ | ✅ | ✅ | |
| `torch.sin`, `torch.cos` | ✅ | ✅ | ✅ | |
| `torch.rsqrt` | ✅ | ✅ | ✅ | |
| `torch.clamp`, `hardtanh` | ✅ | ✅ | ✅ | |
| `torch.where` | ✅ | ✅ | ✅ | |
| `torch.triu` | ✅ | ✅ | ✅ | |

### 2. Matrix, Linear, & Vision Operations
| PyTorch Operation | Eager C++ Level Zero | Graph Mode (`torch.compile`) | Autograd Backward on NPU | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `torch.matmul`, `torch.mm` | ✅ | ✅ | ✅ | Level Zero accelerated $g_A = g_{out} B^T, g_B = A^T g_{out}$ |
| `torch.nn.functional.linear` | ✅ | ✅ | ✅ | Level Zero accelerated $g_x = g_{out} W, g_W = g_{out}^T x, g_b = \sum g_{out}$ |
| `torch.nn.functional.conv2d` | ✅ | ✅ | ✅ | Multi-group and strided 2D convolution |
| `torch.nn.functional.max_pool2d` | ✅ | ✅ | ✅ | |
| `torch.nn.functional.avg_pool2d` | ✅ | ✅ | ✅ | |
| `scaled_dot_product_attention` | ✅ | ✅ | ✅ | Native hardware SDPA with causal masking support |

### 3. Activations & Normalizations
| PyTorch Operation | Eager C++ Level Zero | Graph Mode (`torch.compile`) | Autograd Backward on NPU | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `torch.relu`, `F.relu` | ✅ | ✅ | ✅ | Level Zero conditional selection derivative |
| `torch.nn.functional.gelu` | ✅ | ✅ | ✅ | Exact error function & normal distribution derivative |
| `torch.nn.functional.silu` | ✅ | ✅ | ✅ | Exact sigmoid product derivative |
| `torch.softmax`, `F.softmax` | ✅ | ✅ | ✅ | Level Zero accelerated softmax backward without CPU fallback |
| `rmsnorm` | ✅ | ✅ | ✅ | Root mean square normalization forward and backward |
| `torch.nn.functional.layer_norm` | ✅ | ✅ | ✅ | Affine-weight layer normalization |

### 4. Indexing, Embeddings, & Losses
| PyTorch Operation | Eager C++ Level Zero | Graph Mode (`torch.compile`) | Autograd Backward on NPU | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `torch.nn.functional.embedding` | ✅ | ✅ | ✅ | Hardware gather and index gradient scatter |
| `torch.index_select` | ✅ | ✅ | ✅ | |
| `torch.reshape`, `view` | ✅ | ✅ | ✅ | Zero-copy descriptor manipulation |
| `torch.transpose` | ✅ | ✅ | ✅ | |
| `torch.squeeze`, `unsqueeze` | ✅ | ✅ | ✅ | |
| `torch.cat`, `torch.stack` | ✅ | ✅ | ✅ | |
| `torch.mean` | ✅ | ✅ | ✅ | |
| `torch.zeros`, `ones`, `full` | ✅ | ✅ | ✅ | |
| `update_kv_cache` | ✅ | ✅ | ✅ | Stateful NPU KV-cache manipulation |
| `mse_loss` | ✅ | ✅ | ✅ | Hardware-accelerated MSE loss forward and backward |

### 5. Hardware Optimizers
| Optimizer | Module | Description |
| :--- | :--- | :--- |
| `NPUAdam` | `intel_npu_acceleration.optim.NPUAdam` | Fused first moment, second moment, bias correction, and weight update kernel on NPU |
| `NPUSGD` | `intel_npu_acceleration.optim.NPUSGD` | Momentum-buffered SGD update kernel on NPU |

---

## 🛠️ Build & Verification

```bash
# Build C++ Level Zero Extension
cd intel_npu_lib
python setup.py build_ext --inplace

# Run Full Test Suite
pytest
```
