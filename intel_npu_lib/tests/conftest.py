"""
Shared pytest fixtures and configuration for NPU acceleration tests.
"""

import pytest
import torch

import intel_npu_acceleration as npu


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "npu: marks tests requiring NPU hardware")


@pytest.fixture(autouse=True)
def clear_npu_cache():
    """Ensure each test starts with a clean in-memory graph cache."""
    npu.clear_graph_cache()
    yield
    npu.clear_graph_cache()


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Set deterministic seed for reproducible random tensors."""
    torch.manual_seed(0)


@pytest.fixture
def skip_if_no_npu():
    if not npu.is_available():
        pytest.skip("Intel NPU not available — skipping NPU-specific test")


@pytest.fixture
def simple_addition_model():
    import torch.nn as nn

    class AdditionModel(nn.Module):
        def forward(self, x, y):
            return x + y

    return AdditionModel()


@pytest.fixture
def simple_mlp():
    import torch.nn as nn

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(32, 16)

        def forward(self, x):
            return self.fc2(self.gelu(self.fc1(x)))

    return SimpleMLP()


@pytest.fixture
def small_mlp_factory():
    """Factory for small MLP models with configurable sizes."""
    import torch.nn as nn

    def _make(in_features=16, hidden=32, out_features=8):
        return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))

    return _make


@pytest.fixture
def assert_close():
    """Fixture providing assert_close helper with NPU-tolerant atol/rtol."""
    from torch.testing import assert_close as _assert_close

    def _assert(a, b, atol=1e-2, rtol=1e-2, **kwargs):
        _assert_close(a, b, atol=atol, rtol=rtol, **kwargs)

    return _assert
