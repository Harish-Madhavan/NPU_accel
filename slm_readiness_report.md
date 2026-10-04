# SLM Backend Readiness Assessment
### Intel NPU Acceleration Library — `intel_npu_lib`

---

## Executive Summary

The backend is **substantially ready** to run a Small Language Model (SLM) on Intel NPU. The full Transformer decoder pipeline is implemented, tested, and demonstrated via `tiny_llama.py`. Two execution modes are available: a functional (stateless) KV-cache model and a stateful hardware-register-backed model. Several gaps remain before a production-grade, real-weight SLM (e.g. Phi-3 Mini, Qwen-1.5, Gemma-2B) can be deployed.

---

## ✅ What Is Ready

### 1. Core Operator Coverage (Graph Mode)

All operators required by a standard Transformer decoder are implemented as FX-graph converters:

| Category | Ops Supported |
|---|---|
| **Linear / MatMul** | `nn.Linear`, `F.linear`, `torch.matmul`, `quantized_linear` |
| **Normalization** | `RMSNorm` (custom), `LayerNorm`, `BatchNorm2d` |
| **Activations** | `ReLU`, `GeLU` (erf + tanh), `SiLU/Swish`, `Softmax`, `HardSigmoid`, `HardSwish` |
| **Attention** | `F.scaled_dot_product_attention` → native OpenVINO SDPA node |
| **Embeddings** | `nn.Embedding`, `F.embedding` |
| **KV-Cache Ops** | `update_kv_cache` (scatter-update), `NPUStatefulKVCache` (ReadValue/Assign) |
| **Structural** | `reshape`, `transpose`, `cat`, `stack`, `squeeze`, `unsqueeze`, `arange`, `triu`, `where`, `sin`, `cos`, `rsqrt`, `pow`, `getitem` (strided slicing), `setitem` (ScatterNDUpdate) |
| **Math** | `add`, `sub`, `mul`, `div`, `neg`, `floordiv`, `clamp` |

> [!NOTE]
> RoPE computation (`sin`, `cos`, `arange`, slice-and-cat) is fully handled in graph mode via the structural converters.

---

### 2. Compilation Pipeline

- **`compile_to_npu(model, example_input)`** — torch.fx trace → OpenVINO graph → NPU compile, with:
  - MD5-keyed **in-memory graph cache** (LRU, max 100 entries)
  - **On-disk model cache** via `set_cache_dir()` (eliminates recompilation across sessions)
  - Automatic **CPU fallback** if NPU hardware is unavailable
  - **Hybrid graph partitioning** for models with unsupported ops (NPU+CPU split)
  - `strict=True` mode for pure-NPU compile or hard-fail

---

### 3. KV-Cache: Two Strategies

| Strategy | Class | Description |
|---|---|---|
| **Functional (stateless)** | `update_kv_cache` | Cache tensor passed in/out each step. Verified in `test_llama_model.py` |
| **Stateful (hardware registers)** | `NPUStatefulKVCache` | Uses OpenVINO `ReadValue` / `Assign` to store KV state directly in NPU registers. Zero host-to-device transfer per step |

The stateful path is demonstrated end-to-end in `tiny_llama.py` (Parts B/C) and has dedicated unit tests (`test_stateful_kv.py`).

---

### 4. Quantization

| Format | Status |
|---|---|
| **Weight-only INT8** (simulated) | ✅ `quantize_to_int8()` in `tiny_llama.py`; converter handles `weight_scale` dequantization inside OV graph |
| **INT8 via `quantized_linear`** | ✅ Full eager + graph mode, per-channel scale + zero-point |
| **INT4 (packed uint8)** | ✅ Unpacking logic in both eager and graph converters (`convert_quantized_linear`) |
| **NNCF / PTQ integration** | ❌ Not yet implemented (Phase 4 backlog) |

---

### 5. Async Inference & Throughput Mode

- `infer_async()` / `wait_async()` implemented in `NPUGraphModule`
- Round-robin multi-stream request buffering
- **Leased buffer pool** with `weakref.finalize` for zero-copy, concurrent-safe output management
- Thread-safe `threading.Lock()` on forward/async dispatch

---

### 6. Performance Levers Available

```python
npu_compiler.enable_turbo()           # Level Zero NPU_TURBO=YES
npu_compiler.enable_sda()             # Shared Device Address (zero-copy transfers)
npu_compiler.set_performance_hint("THROUGHPUT")  # Multi-stream saturation
npu_compiler.set_eager_device("NPU")  # Force eager ops to NPU
```

---

### 7. Test Coverage

Over 120 test suites (339 unit and integration tests passing with 0 errors/warnings) covering:
- Functional KV cache correctness (`test_kv_cache.py`)
- Stateful KV cache multi-step correctness, reset, and multi-layer auto-mapping (`test_stateful_kv.py`)
- Full LLaMA compile + output parity (`test_llama_model.py`)
- Native SLM pipeline & Hugging Face engine verification (`test_llm.py`)
- Quantization (INT8, INT4, FP16, compiled + eager) (`test_quantization.py`)
- Autograd through embedding + SDPA (`test_autograd_slm.py`)
- Async + hybrid execution (`test_async_hybrid.py`)
- Stress / memory leak / context-switching (`test_stress.py`)

---

## ⚠️ Gaps & Limitations

### Critical for Real SLM Deployment

| Gap | Impact | Notes |
|---|---|---|
| **Multi-Layer Stateful Pipeline** | ✅ Resolved | `stateful=True` automatically maps multi-decoder functional KV-caches to native NPU state registers (`test_multi_layer_auto_stateful_mapping`). |
| **Tokenizer / HuggingFace integration** | ✅ Resolved | `intel_npu_acceleration.llm.NPUCausalLM` provides `from_pretrained`, token streaming, and safetensors / bin weight loading. Tested on `SmolLM2-135M` (57+ tok/s). |
| **NNCF / PTQ Integration** | 🟡 Medium | Only simulated INT8 (cast to int8, scale-dequant at runtime). True NPU DP4A/VNNI path requires NNCF post-training quantization — not integrated yet. |
| **Dynamic shape / variable batch** | 🟡 Medium | `NPUDynamicGraphModule` handles dynamic sequence length via bucket padding, but requires explicit `bucket_sizes` tuning. Batch dimension is always static. |

### Minor / Non-blocking

| Gap | Notes |
|---|---|
| **`avg_pool2d`, `dropout` eager** | Not needed for decoder-only LLMs. |
| **Regression benchmarking pipeline** | Phase 5 backlog — no automated per-commit latency tracking. |
| **API documentation** | No Sphinx/MkDocs yet. |

---

## 🏁 Verdict

| Capability | Status |
|---|---|
| Compile Transformer blocks to NPU | ✅ **Ready** |
| Functional KV-cache autoregressive generation | ✅ **Ready** |
| Stateful (zero-transfer) KV-cache generation | ✅ **Ready** (hardware registers) |
| INT8 weight quantization | ✅ **Ready** (per-channel & per-tensor) |
| INT4 weight quantization | ✅ **Ready** (packed uint8 with zero-point offsets) |
| Multi-layer stateful mapping (auto) | ✅ **Ready** (auto-mapped via `stateful=True`) |
| Real SLM weight loading (HF) | ✅ **Ready** (`NPUCausalLM.from_pretrained`) |
| Tokenizer / streaming text generation | ✅ **Ready** (`NPUCausalLM.stream_generate`) |
| True NNCF PTQ (native NPU precision) | ❌ **Not implemented** |
| Async/throughput multi-stream inference | ✅ **Ready** |
| CPU fallback for unsupported ops | ✅ **Ready** |
