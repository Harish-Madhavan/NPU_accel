# AGENTS.md - Agentic Coding Instructions & Architecture Guide

Welcome, AI Agent! This repository implements **Intel NPU Acceleration (`intel_npu_acceleration`)**, a native PyTorch backend for Intel® Core™ Ultra Neural Processing Units (NPUs) built with oneAPI Level Zero and OpenVINO™.

Follow these guidelines when modifying, extending, or maintaining this codebase.

---

## 🏛️ Repository Architecture

```
NPU_accel/
├── AGENTS.md                  # This primary agentic guide
├── GEMINI.md                  # Mirror agent guidelines
├── README.md                  # User-facing documentation
├── SUPPORTED_OPS.md           # Operator parity matrix
├── examples/                  # Standard PyTorch example scripts
│   ├── mnist_training.py
│   ├── tiny_llama.py
│   ├── async_pipelined_inference.py
│   ├── benchmark_npu_matmul.py
│   └── profile_npu_models.py
└── intel_npu_lib/             # Core library package
    ├── setup.py               # C++ extension & entry point config
    ├── pyproject.toml         # Build dependencies and pytest config
    ├── csrc/                  # Native C++ Level Zero driver backend
    │   ├── include/
    │   │   ├── device.h       # NPUBackend singleton & Level Zero context
    │   │   └── ops.h          # C++ operator definitions (forward & backward)
    │   ├── device.cpp         # Context initialization & property probing
    │   ├── ops.cpp            # OpenVINO C++ graph nodes & zero-copy USM
    │   └── bindings.cpp       # PyBind11 Python extension bindings
    ├── src/
    │   └── intel_npu_acceleration/
    │       ├── __init__.py    # Top-level public API exports
    │       ├── device.py      # Device probing & PyTorch parity APIs
    │       ├── cache.py       # On-disk OpenVINO IR binary cache
    │       ├── optim.py       # NPUAdam & NPUSGD hardware optimizers
    │       ├── frontend/      # TorchDynamo backend & FX compiler
    │       │   ├── compiler.py
    │       │   ├── dynamo.py  # torch.compile(model, backend="npu")
    │       │   ├── partitioner.py
    │       │   └── transformations.py
    │       ├── nn/            # Drop-in PyTorch neural network modules
    │       │   ├── modules.py
    │       │   └── stateful_kv.py
    │       ├── autograd/      # PyTorch autograd Function definitions
    │       │   └── functions.py
    │       └── _functional/   # C++ dispatch and Python fallbacks
    └── tests/                 # 120+ unit and integration test suites
```

---

## 🧭 Core Principles & Constraints

1. **Pure Standard PyTorch Semantics**:
   - Never add proprietary or custom methods facing the user.
   - All accelerations and optimizations MUST be plumbed into standard PyTorch APIs (`torch.compile(model, backend="npu")`, `torch.nn.*`, `torch.optim.*`, `loss.backward()`).
   - The user/developer should not need any knowledge outside standard PyTorch.

2. **Out-of-the-Box Optimizations**:
   - Hardware Turbo frequency boost (`NPU_TURBO=YES`), memory pool preservation (`NPU_DISABLE_IDLE_MEMORY_PRUNING=YES`), aggressive QDQ fusion, and zero-copy buffer leasing must always apply **automatically by default**.

3. **Separation of Execution Targets**:
   - **Graph Mode (`torch.compile` / `compile_to_npu`)**: Fully compiles and runs the model on the NPU via Level Zero.
   - **Eager Micro-Ops**: Default to CPU fallback for tiny scalar/isolated tensor operations to avoid driver command queue thrashing and preserve float32 numerical exactness.

---

## 🛠️ Developer & Agent Workflows

### 1. Rebuilding C++ Extension
Whenever modifying any file in `intel_npu_lib/csrc/`:
```bash
cd intel_npu_lib
python setup.py build_ext --inplace
```

### 2. Running Automated Tests
Always run the full test suite before concluding work:
```bash
cd intel_npu_lib
pytest
```
* **Success Criteria**: 100% tests passing (120+ suites) with **0 errors and 0 warnings**.

### 3. Adding a New Operator
Follow the runbook in [`.agents/skills/add-npu-operator/SKILL.md`](.agents/skills/add-npu-operator/SKILL.md):
1. Declare in `csrc/include/ops.h`.
2. Implement OpenVINO graph node and data binding in `csrc/ops.cpp`.
3. Bind in `csrc/bindings.cpp`.
4. Add C++ dispatch and Python fallback in `src/intel_npu_acceleration/_functional/`.
5. Wire into `autograd/functions.py` (if gradient-aware).
6. Register in `registry.py` and `frontend/transformations.py`.
7. Add unit test in `tests/`.
