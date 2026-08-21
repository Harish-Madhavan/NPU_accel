import unittest
import torch
import operator
import intel_npu_acceleration.functional as npu_func
from intel_npu_acceleration.registry import OpRegistry


class TestOpRegistry(unittest.TestCase):
    def test_supported_functions(self):
        # Math & elementwise
        self.assertTrue(OpRegistry.get_function(torch.add))
        self.assertTrue(OpRegistry.get_function(torch.sub))
        self.assertTrue(OpRegistry.get_function(torch.mul))
        self.assertTrue(OpRegistry.get_function(torch.div))
        self.assertTrue(OpRegistry.get_function(torch.neg))
        self.assertTrue(OpRegistry.get_function(torch.matmul))

        # Functional mappings
        self.assertTrue(OpRegistry.get_function(npu_func.add))
        self.assertTrue(OpRegistry.get_function(npu_func.linear))
        self.assertTrue(OpRegistry.get_function(npu_func.rmsnorm))
        self.assertTrue(OpRegistry.get_function(npu_func.update_kv_cache))
        self.assertTrue(OpRegistry.get_function(npu_func.quantized_linear))

        # NN activations & normalizations
        self.assertTrue(OpRegistry.get_function(torch.nn.functional.relu))
        self.assertTrue(OpRegistry.get_function(torch.nn.functional.gelu))
        self.assertTrue(OpRegistry.get_function(torch.nn.functional.silu))
        self.assertTrue(OpRegistry.get_function(torch.nn.functional.softmax))
        self.assertTrue(OpRegistry.get_function(torch.nn.functional.layer_norm))

    def test_supported_methods(self):
        self.assertTrue(OpRegistry.get_method("add"))
        self.assertTrue(OpRegistry.get_method("matmul"))
        self.assertTrue(OpRegistry.get_method("transpose"))
        self.assertTrue(OpRegistry.get_method("reshape"))
        self.assertTrue(OpRegistry.get_method("view"))
        self.assertTrue(OpRegistry.get_method("contiguous"))
        self.assertTrue(OpRegistry.get_method("clone"))

        # Unsupported method
        self.assertIsNone(OpRegistry.get_method("non_existent_method_xyz"))

    def test_supported_modules(self):
        self.assertTrue(OpRegistry.get_module(torch.nn.Linear))
        self.assertTrue(OpRegistry.get_module(torch.nn.ReLU))
        self.assertTrue(OpRegistry.get_module(torch.nn.GELU))
        self.assertTrue(OpRegistry.get_module(torch.nn.SiLU))
        self.assertTrue(OpRegistry.get_module(torch.nn.LayerNorm))
        self.assertTrue(OpRegistry.get_module(torch.nn.Conv2d))
        self.assertTrue(OpRegistry.get_module(torch.nn.Embedding))
        self.assertTrue(OpRegistry.get_module(npu_func.NPUStatefulKVCache))

        # Unsupported custom module
        class UnsupportedCustomModule(torch.nn.Module):
            pass
        self.assertIsNone(OpRegistry.get_module(UnsupportedCustomModule))


if __name__ == "__main__":
    unittest.main()
