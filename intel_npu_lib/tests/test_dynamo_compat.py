"""TorchDynamo compatibility: importing this package must never break tracing.

Regression coverage for two real failures found running SmolLM2-135M:
- merely importing the package broke ALL torch.compile (even backend="eager")
  because the torch.accelerator bridge pulled Dynamo into C++ stream queries
  with no npu support;
- dynamo export of models touching accelerator queries must complete.
"""

import pytest
import torch
import torch.nn as nn

import intel_npu_acceleration as npu  # noqa: F401  (import itself is the trigger)


class TestDynamoCompat:
    def test_eager_backend_unaffected_by_import(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        opt = torch.compile(model, backend="eager")
        x = torch.randn(2, 8)
        assert torch.allclose(opt(x), model(x), atol=1e-5, rtol=1e-5)

    def test_dynamo_export_completes(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
        ep = torch._dynamo.export(model)(torch.randn(2, 8))
        assert ep is not None

    def test_accelerator_available_at_runtime(self):
        # Tracing reports unavailable (Dynamo cannot represent NPU stream
        # state); runtime answers must stay truthful.
        import torch.accelerator as acc

        assert acc.is_available() == npu.is_available()

    def test_npu_backend_still_compiles(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        opt = torch.compile(model, backend="npu")
        x = torch.randn(2, 8)
        assert_close = torch.allclose(opt(x), model(x), atol=1e-2, rtol=1e-2)
        assert assert_close
