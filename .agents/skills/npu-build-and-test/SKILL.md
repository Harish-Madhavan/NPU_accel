---
name: npu-build-and-test
description: Runbook for building the C++ Level Zero extension, running the full pytest test suite, and validating Intel NPU hardware acceleration.
---

# Intel NPU Build and Test Runbook

Use this skill whenever you modify C++ source code in `csrc/`, add new operators, or need to verify codebase health.

## Step-by-Step Procedure

### 1. Build C++ Extension
From the `intel_npu_lib` directory:
```powershell
cd intel_npu_lib
python setup.py build_ext --inplace
```
- Ensure the build finishes with exit code 0.
- Verify `intel_npu_acceleration/_C.*.pyd` is produced in `src/intel_npu_acceleration/`.

### 2. Run Automated Pytest Suite
```powershell
pytest
```
- Verify 120+ test suites pass with 0 errors and 0 warnings.

### 3. Run Specific Test Suite (for targeted debugging)
```powershell
# E.g., for training & autograd:
pytest tests/test_training_autograd.py -v

# E.g., for PyTorch 2.x conformance:
pytest tests/test_pytorch_conformance.py -v
```

### 4. Check NPU Diagnostics
```powershell
python -c "import intel_npu_acceleration as npu; npu.print_info()"
```
- Verify `NPU Hardware Status: [AVAILABLE]`.
