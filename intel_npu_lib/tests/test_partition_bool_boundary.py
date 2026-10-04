"""NPU boundary dtype contract: boolean outputs must survive NPU execution.

Proven failure mode (SmolLM2-135M decode): attention-mask subgraphs compiled
for the NPU materialize boolean outputs as u8 at runtime, breaking downstream
eager ops (SDPA masks). Covered by two layers:
- `_collect_outputs` restores torch.bool for model-declared boolean outputs.
- the partitioner reverts segments whose NPU binary changes output dtypes.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import intel_npu_acceleration as npu
from intel_npu_acceleration.frontend.partitioner import _outputs_match_calibration


class NeMask(nn.Module):
    def forward(self, x):
        return x.ne(0)


class TestBoolBoundaryE2E:
    def test_bool_output_preserved_through_npu(self):
        model = NeMask().eval()
        x = torch.tensor([[0, 1, -2], [3, 0, 5]])
        compiled = npu.compile_to_npu(model, x)
        out = compiled(x)
        assert out.dtype == torch.bool
        assert torch.equal(out, x.ne(0))

    def test_bool_mask_usable_as_sdpa_mask(self):
        # The original failure: an NPU-produced mask fed to eager SDPA.
        model = NeMask().eval()
        x = torch.randint(0, 2, (1, 1, 4, 4)) * 3 - 1
        compiled = npu.compile_to_npu(model, x)
        mask = compiled(x)
        assert mask.dtype == torch.bool
        q = torch.randn(1, 2, 4, 8)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
        assert out.shape == (1, 2, 4, 8)


class TestOutputsMatchCalibration:
    def _compiled(self, dtypes):
        return SimpleNamespace(
            output_info=[(True, (2,), d) for d in dtypes]
        )

    def test_match(self):
        c = self._compiled([torch.bool])
        assert _outputs_match_calibration("s", c, {"s": torch.ones(2, dtype=torch.bool)})

    def test_mismatch_rejected(self):
        c = self._compiled([torch.uint8])
        assert not _outputs_match_calibration("s", c, {"s": torch.ones(2, dtype=torch.bool)})

    def test_missing_calibration_passes(self):
        c = self._compiled([torch.uint8])
        assert _outputs_match_calibration("s", c, {})

    def test_non_tensor_outputs_ignored(self):
        c = self._compiled([torch.float32])
        assert _outputs_match_calibration("s", c, {"s": 5})

    def test_count_mismatch_passes(self):
        c = self._compiled([torch.float32, torch.float32])
        assert _outputs_match_calibration("s", c, {"s": torch.ones(2)})
