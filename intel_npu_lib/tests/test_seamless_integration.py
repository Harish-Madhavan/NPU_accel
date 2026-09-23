"""
Unit tests for out-of-the-box, seamless PyTorch integration and convenience APIs.
"""

import torch
import torch.nn as nn
from torch.testing import assert_close

import intel_npu_acceleration as npu


class TestSeamlessIntegration:
    """
    Test suite for seamless PyTorch integration, device querying, context managers,
    and nn.Module convenience extensions.
    """

    def test_device_properties_and_querying(self):
        """Verify device query functions work reliably."""
        assert npu.device_count() in [0, 1]
        assert npu.current_device() == 0
        name = npu.get_device_name()
        assert isinstance(name, str)
        props = npu.get_device_properties()
        assert "name" in props
        assert "device_id" in props
        assert "is_available" in props

    def test_empty_cache_and_synchronize(self):
        """Verify empty_cache and synchronize execute cleanly without errors."""
        npu.empty_cache()
        npu.synchronize()

    def test_turbo_context_manager(self):
        """Verify turbo context manager executes without exception."""
        with npu.turbo(True):
            x = torch.randn(4, 4)
            y = npu.relu(x)
            assert_close(y, torch.relu(x))

    def test_performance_mode_context_manager(self):
        """Verify performance_mode context manager switches hints cleanly."""
        with npu.performance_mode("THROUGHPUT"):
            x = torch.randn(4, 4)
            y = npu.gelu(x)
            assert_close(y, torch.nn.functional.gelu(x), atol=1e-2, rtol=1e-2)

    def test_accelerate_api(self):
        """Verify npu.accelerate seamlessly compiles and runs a standard PyTorch module."""
        model = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        x = torch.randn(2, 16)
        compiled_model = npu.accelerate(model, example_input=x)
        out = compiled_model(x)
        assert out.shape == (2, 2)

    def test_module_convenience_extensions(self):
        """Verify model.to_npu() and model.compile_npu() monkey patches."""
        model = nn.Linear(8, 4)
        # to_npu should return self
        res = model.to_npu()
        assert res is model

        # compile_npu should compile the module
        x = torch.randn(1, 8)
        npu_mod = model.compile_npu(example_input=x)
        out = npu_mod(x)
        assert out.shape == (1, 4)
