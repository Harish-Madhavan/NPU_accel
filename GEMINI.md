# GEMINI.md - Agentic Guidelines for Intel NPU Acceleration

Welcome! This file defines the developer rules, architecture invariants, and automated runbooks for working on `intel_npu_acceleration`.

---

## 🎯 Primary Directive: Pure Standard PyTorch

- All developer-facing APIs must conform **100% to standard PyTorch semantics**.
- Use standard `torch.compile(model, backend="npu")`, `torch.nn.*`, `torch.optim.*`, and `loss.backward()`.
- Low-level optimizations (Level Zero Turbo, QDQ fusion, zero-copy leasing, memory pool preservation) MUST be applied **automatically out of the box**.

---

## 📦 Key Directory Map

* `intel_npu_lib/csrc/`: C++ Level Zero driver backend, memory binding, and operator graphs.
* `intel_npu_lib/src/intel_npu_acceleration/`: Python package source code.
* `intel_npu_lib/src/intel_npu_acceleration/frontend/dynamo.py`: TorchDynamo backend implementation.
* `intel_npu_lib/tests/`: Pytest unit and integration test suites.
* `examples/`: End-to-end PyTorch scripts and benchmarks.

---

## ⚡ Build & Verification Commands

```powershell
# 1. Rebuild C++ Extension (after any csrc edit)
cd intel_npu_lib
python setup.py build_ext --inplace

# 2. Run Test Suite
pytest
```
* **Required Standard**: All 120+ test suites must pass with **0 warnings and 0 errors**.
