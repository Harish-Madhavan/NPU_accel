# Testing & Verification Guidelines

## Intent
Ensure zero regression across the 120+ test suites in `intel_npu_lib/tests/`.

## Rules
1. **Always Run Pytest**:
   - Before finishing any task, run `pytest` from `intel_npu_lib/`.
   - All tests must pass with **0 failures and 0 warnings**.

2. **Clean Cache During Setup**:
   - In new test classes, call `npu.empty_cache()` or `npu.clear_graph_cache()` inside `setUp()` to avoid stale model cache artifacts across test runs.

3. **Appropriate FP16 / FP32 Tolerances**:
   - When verifying NPU execution against CPU references, use `atol=1e-3, rtol=1e-3` (or `1e-2` for multi-layer backward autograd chains) to account for FP16 systolic array floating point rounding.
