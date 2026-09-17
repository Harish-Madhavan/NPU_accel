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
   - Hardware Turbo clock frequency boost (`NPU_TURBO=YES`, live via `npu.enable_turbo()`).
   - Unified Level Zero memory preservation (`NPU_DISABLE_IDLE_MEMORY_PRUNING=YES`) to eliminate TLB invalidation.
   - Non-blocking concurrent hardware command queues (`NPU_RUN_INFERENCES_SEQUENTIALLY=NO`).
   - High-priority hardware micro-scheduler dispatch (`MODEL_PRIORITY=HIGH`).
   - Eager kernels compile with the same Level Zero property set as graph mode (gated on driver support); eager binaries are keyed by op/shape/dtype **plus** performance hint and eager device so hint changes can't reuse stale binaries.

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
| `cross_entropy_loss` | ✅ | ✅ | ✅ | 1D indices & 2D probabilities, forward & backward on NPU |
| `l1_loss` | ✅ | ✅ | ✅ | Mean Absolute Error loss forward and sign backward on NPU |
| `bce_with_logits_loss` | ✅ | ✅ | ✅ | Binary cross entropy with logits forward & backward |

### 5. Hardware Optimizers & Training Utilities
| Tool / Optimizer | Module | Description |
| :--- | :--- | :--- |
| `NPUAdam` | `intel_npu_acceleration.optim.NPUAdam` | Fused first moment, second moment, bias correction, and weight update kernel on NPU |
| `NPUSGD` | `intel_npu_acceleration.optim.NPUSGD` | Momentum-buffered SGD update kernel on NPU |
| `clip_grad_norm_` | `intel_npu_acceleration.optim.clip_grad_norm_` | Gradient clipping matching PyTorch standard `clip_grad_norm_` |

### 6. First-Class PyTorch Backend APIs (`torch.device`, `torch.npu`, `torch.accelerator`)
| API Surface | Support | Description |
| :--- | :---: | :--- |
| `torch.device("npu")` / `torch.device("npu:0")` | ✅ | Full PrivateUse1 device typing and device string parsing |
| `torch.npu.is_available()` | ✅ | Detects physical Intel Core Ultra NPU hardware availability |
| `torch.npu.device_count()`, `current_device()` | ✅ | Device enumeration and active device index tracking |
| `torch.npu.get_device_name()`, `get_device_properties()` | ✅ | Device naming and properties container with attribute & dict access |
| `torch.npu.synchronize()`, `empty_cache()` | ✅ | Level Zero command queue synchronization and cache purge |
| `torch.npu.Stream()`, `torch.npu.Event()` | ✅ | Stream command queue management and timing events |
| `torch.npu.stream(s)`, `torch.npu.device(d)` | ✅ | Stream and device context managers |
| `torch.npu.amp.autocast("npu")` | ✅ | Automatic mixed precision (FP16 / BF16) execution |
| `torch.backends.npu.*` | ✅ | Performance flags (`allow_tf32`, `flash_sdp_enabled`, `version`) |
| `torch.accelerator.*` (PyTorch 2.4+) | ✅ | Universal accelerator bridge (`synchronize`, `empty_cache`, `current_accelerator`, `streams`) |
| `torch.Tensor.to("npu")`, `torch.Tensor.npu()` | ✅ | Zero-copy Unified System Memory (USM) tensor management |
| `torch.Tensor.is_npu` | ✅ | Tensor device type introspection property |
| `torch.nn.Module.to("npu")`, `model.npu()` | ✅ | Automated graph compilation and hardware execution |

### 7. Quantization & Model Compression
| Technique | Support | Details |
| :--- | :---: | :--- |
| `quantize(model, mode="int8")` | ✅ | INT8 symmetric weights with per-channel row-wise or per-tensor scales |
| `quantize(model, mode="int4")` | ✅ | 4-bit asymmetric packed weights (2 nibbles per uint8 byte) with zero-point offsets |
| `quantized_linear` | ✅ | Hardware-accelerated linear kernel supporting FP32, FP16, INT8, and packed INT4 |
| Compiler Auto-Dequantization | ✅ | Native OpenVINO compiler automatic unpack for INT8 and packed INT4 during `compile_to_npu` |


---

## 🛠️ Build & Verification

```bash
# Build C++ Level Zero Extension
cd intel_npu_lib
python setup.py build_ext --inplace

# Run Full Test Suite (148 tests, 0 warnings)
pytest
```

> Note: `torch.jit.trace` / `trace_method` `FutureWarning`s emitted internally by
> `openvino.convert_model` are suppressed in `frontend/compiler.py` via targeted
> `warnings.catch_warnings()` and in `pyproject.toml` via `filterwarnings`, so both
> library users and `pytest` runs stay warning-free.
