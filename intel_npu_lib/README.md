# Intel NPU Acceleration Library

High-performance, drop-in PyTorch backend for Intel® Core™ Ultra NPUs
(oneAPI Level Zero + OpenVINO™). Full documentation, operator matrix, and
examples live in the repository root:

- [`../README.md`](../README.md) — user guide and quick start
- [`../SUPPORTED_OPS.md`](../SUPPORTED_OPS.md) — operator parity matrix
- [`docs/`](./docs/getting_started.md) — environment setup guide

## Install (from this directory)

```bash
# CPU-only PyTorch keeps the download small
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openvino "numpy>=1.24"

# Windows (auto-configures MSVC): ..\build_npu.bat
python setup.py build_ext --inplace
pip install --no-build-isolation --no-deps -e .
```

Build knobs: `NPU_NO_BUILD_EXT=1` (pure-Python fallback),
`NPU_VERBOSE_BUILD=1` (OpenVINO discovery logs),
`OPENVINO_DIR` / `INTEL_OPENVINO_DIR` (custom toolkit location).

## Verify

```bash
python -m intel_npu_acceleration --info  # also installed as `intel-npu-info`
```

```python
import intel_npu_acceleration as npu
print(npu.is_available(), npu.get_device_name(), npu.device_count())
```
