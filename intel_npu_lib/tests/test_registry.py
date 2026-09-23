import operator

import pytest
import torch

import intel_npu_acceleration.functional as npu_func
from intel_npu_acceleration.registry import OpRegistry


@pytest.mark.parametrize(
    "fn",
    [
        # Math & elementwise
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
        torch.neg,
        torch.matmul,
        # Functional mappings
        npu_func.add,
        npu_func.linear,
        npu_func.rmsnorm,
        npu_func.update_kv_cache,
        npu_func.quantized_linear,
        # NN activations & normalizations
        torch.nn.functional.relu,
        torch.nn.functional.gelu,
        torch.nn.functional.silu,
        torch.nn.functional.softmax,
        torch.nn.functional.layer_norm,
    ],
)
def test_supported_functions(fn):
    assert OpRegistry.get_function(fn)


@pytest.mark.parametrize(
    "name",
    ["add", "matmul", "transpose", "reshape", "view", "contiguous", "clone"],
)
def test_supported_methods(name):
    assert OpRegistry.get_method(name)


def test_unsupported_method():
    assert OpRegistry.get_method("non_existent_method_xyz") is None


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.GELU,
        torch.nn.SiLU,
        torch.nn.LayerNorm,
        torch.nn.Conv2d,
        torch.nn.Embedding,
        npu_func.NPUStatefulKVCache,
    ],
)
def test_supported_modules(module):
    assert OpRegistry.get_module(module)


def test_unsupported_module():
    class UnsupportedCustomModule(torch.nn.Module):
        pass

    assert OpRegistry.get_module(UnsupportedCustomModule) is None
