"""
Shared test helpers — model factories and assertion utilities.

Eliminates ~200 LOC duplication across test_basic, test_mlp, test_llm_ops, etc.
"""

import torch
import torch.nn as nn
from torch.testing import assert_close as _assert_close


def assert_allclose(actual, expected, atol=1e-2, rtol=1e-2, msg: str = ""):
    """NPU-tolerant assert_close with sensible defaults."""
    _assert_close(actual, expected, atol=atol, rtol=rtol, msg=msg)


# ---------------------------------------------------------------------------
# Model factories — single source of truth for duplicated test models
# ---------------------------------------------------------------------------


def make_simple_mlp(in_features=16, hidden=32, out_features=16):
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden)
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(hidden, out_features)

        def forward(self, x):
            return self.fc2(self.gelu(self.fc1(x)))

    return SimpleMLP()


def make_small_mlp(in_features=16, hidden=32, out_features=8):
    return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))


def make_conv_model(in_channels=3, out_channels=8, kernel=3):
    class SimpleCVModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, bias=False)
            nn.init.ones_(self.conv.weight)

        def forward(self, x):
            return self.conv(x)

    return SimpleCVModel()


def make_addition_model():
    class AdditionModel(nn.Module):
        def forward(self, x, y):
            return x + y

    return AdditionModel()


def make_transformer_block(hidden=64):
    """Minimal transformer block for encoder/decoder tests."""

    class RMSNorm(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(h))

        def forward(self, x):
            return torch.nn.functional.layer_norm(x, (x.shape[-1],), self.weight, None, 1e-6)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(hidden)
            self.fc1 = nn.Linear(hidden, hidden * 2)
            self.fc2 = nn.Linear(hidden * 2, hidden)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(self.ln(x))))

    return Block()
