import operator

import pytest
import torch

import intel_npu_acceleration.functional as npu_func
from intel_npu_acceleration.registry import OpRegistry, _canonical_op_name


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


@pytest.mark.parametrize(
    "target",
    [
        torch.add,
        torch.ops.aten.add.Tensor,
        operator.add,
        torch.nn.functional.linear,
        torch._C._nn.linear,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.t.default,
        torch.ops.aten.convolution.default,
        torch.ops.aten._softmax.default,
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.transpose.int,
        torch.ops.aten.embedding.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.clone.default,
    ],
)
def test_canonical_spellings_supported(target):
    """Dynamo aten spellings must match plain torch.* coverage."""
    assert OpRegistry.is_function_supported(target)


@pytest.mark.parametrize(
    "target",
    [
        torch.erf,
        torch.ops.aten.erf.default,
        torch.ops.aten.gather.default,
        torch.ops.aten.scatter_add.default,
        torch.ops.aten.index_put.default,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.logical_and.default,
        torch.ops.aten.topk.default,
        torch.ops.aten.sort.default,
        torch.ops.aten.randn.default,
        torch.ops.aten.nonzero.default,
        torch.ops.aten.max_pool2d_with_indices.default,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.relu_.default,
        None,
    ],
)
def test_unsupported_stays_unsupported(target):
    """In-place, RNG, scatter/gather, sorting, comparisons, and erf never route to NPU."""
    assert not OpRegistry.is_function_supported(target)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (torch.ops.aten.add.Tensor, "add"),
        (torch._C._nn.linear, "linear"),
        (torch.ops.aten._softmax.default, "softmax"),
        (torch.ops.aten.native_layer_norm.default, "layer_norm"),
        (torch.ops.aten.t.default, "transpose"),
        (torch.ops.aten.mm.default, "matmul"),
        (torch.ops.aten.sym_size.default, "size"),
        (torch.add, "add"),
        (operator.mul, "mul"),
        ("split_with_sizes", "split"),
        (None, None),
    ],
)
def test_canonical_op_name(target, expected):
    assert _canonical_op_name(target) == expected
