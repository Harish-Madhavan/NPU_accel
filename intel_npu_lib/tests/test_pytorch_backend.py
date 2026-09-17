"""
Comprehensive test suite for PyTorch Backend Conformance for Intel NPU.

Verifies that `intel_npu_acceleration` behaves as a first-class PyTorch backend:
- torch.device("npu") & torch.device("npu:0")
- torch.npu.* API surface
- torch.backends.npu.* API surface
- torch.accelerator.* universal accelerator integration (PyTorch 2.4+)
- torch.Tensor.to("npu"), torch.Tensor.npu(), torch.Tensor.is_npu, and factory functions
- torch.nn.Module.to("npu") and torch.nn.Module.npu()
- torch.autocast(device_type="npu") and torch.amp.autocast("npu")
- torch.compile(model, backend="npu") and torch.compile(model, backend="intel_npu")
"""

import unittest

import torch
import torch.nn as nn
from torch.testing import assert_close

import intel_npu_acceleration as npu


class TestPyTorchBackendConformance(unittest.TestCase):
    def setUp(self):
        npu.clear_graph_cache()
        torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # 1. Device Creation & Properties
    # -----------------------------------------------------------------------

    def test_torch_device_npu(self):
        """Verify torch.device('npu') and torch.device('npu:0') are valid devices."""
        dev = torch.device("npu")
        self.assertEqual(dev.type, "npu")

        dev0 = torch.device("npu:0")
        self.assertEqual(dev0.type, "npu")
        self.assertEqual(dev0.index, 0)

    # -----------------------------------------------------------------------
    # 2. torch.npu API surface
    # -----------------------------------------------------------------------

    def test_torch_npu_module_apis(self):
        """Verify the full torch.npu API surface mirroring torch.cuda / torch.xpu."""
        self.assertTrue(hasattr(torch, "npu"))
        self.assertTrue(torch.npu.is_available())
        self.assertGreaterEqual(torch.npu.device_count(), 1)
        self.assertEqual(torch.npu.current_device(), 0)

        # set_device
        torch.npu.set_device(0)

        # Device name and properties
        name = torch.npu.get_device_name(0)
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)

        props = torch.npu.get_device_properties(0)
        self.assertEqual(props.name, name)
        self.assertEqual(props.major, 1)
        self.assertEqual(props.minor, 0)
        self.assertGreater(props.total_memory, 0)
        self.assertGreaterEqual(props.multi_processor_count, 1)
        # Verify dict-like access for backward compatibility
        self.assertEqual(props["name"], name)

        # Capability
        cap = torch.npu.get_device_capability(0)
        self.assertEqual(cap, (1, 0))

        # Synchronize & Cache
        torch.npu.synchronize()
        torch.npu.empty_cache()

        # Memory stats
        self.assertEqual(torch.npu.memory_allocated(), 0)
        self.assertEqual(torch.npu.max_memory_allocated(), 0)
        self.assertEqual(torch.npu.memory_reserved(), 0)
        self.assertEqual(torch.npu.max_memory_reserved(), 0)
        self.assertIsInstance(torch.npu.memory_stats(), dict)
        torch.npu.reset_peak_memory_stats()

        # RNG
        torch.npu.manual_seed(42)
        torch.npu.manual_seed_all(42)
        self.assertIsInstance(torch.npu.seed(), int)
        self.assertIsInstance(torch.npu.initial_seed(), int)

        # Initialization
        self.assertTrue(torch.npu.is_initialized())
        torch.npu.init()

    def test_torch_npu_streams_and_events(self):
        """Verify Stream and Event abstractions and context managers."""
        stream = torch.npu.Stream()
        self.assertTrue(stream.query())
        stream.synchronize()

        event = torch.npu.Event(enable_timing=True)
        event.record(stream)
        self.assertTrue(event.query())
        event.synchronize()

        # Stream context
        with torch.npu.stream(stream):
            curr = torch.npu.current_stream()
            self.assertIs(curr, stream)

        # Device context
        with torch.npu.device(0):
            self.assertEqual(torch.npu.current_device(), 0)

        # default_stream and set_stream
        def_s = torch.npu.default_stream()
        self.assertIsNotNone(def_s)
        torch.npu.set_stream(def_s)

    def test_torch_npu_amp_support(self):
        """Verify AMP supported dtypes and helper queries."""
        supported = torch.npu.get_amp_supported_dtype()
        self.assertIn(torch.float16, supported)
        self.assertIn(torch.bfloat16, supported)
        self.assertTrue(torch.npu.is_fp16_supported())
        self.assertTrue(torch.npu.is_bf16_supported())
        self.assertEqual(torch.npu.get_autocast_dtype(), torch.float16)

        # torch.npu.amp submodule
        self.assertTrue(hasattr(torch.npu, "amp"))
        self.assertTrue(hasattr(torch.npu.amp, "autocast"))
        self.assertTrue(hasattr(torch.npu.amp, "GradScaler"))

    # -----------------------------------------------------------------------
    # 3. torch.backends.npu
    # -----------------------------------------------------------------------

    def test_torch_backends_npu(self):
        """Verify torch.backends.npu properties, version, and optimization flags."""
        self.assertTrue(hasattr(torch.backends, "npu"))
        self.assertTrue(torch.backends.npu.is_available())
        self.assertEqual(torch.backends.npu.version(), "0.2.0")

        self.assertFalse(torch.backends.npu.matmul.allow_tf32)
        self.assertTrue(torch.backends.npu.matmul.allow_fp16_reduced_precision_reduction)
        self.assertTrue(torch.backends.npu.matmul.allow_bf16_reduced_precision_reduction)
        self.assertFalse(torch.backends.npu.allow_tf32)

        # SDP attention flags
        self.assertTrue(torch.backends.npu.flash_sdp_enabled())
        self.assertTrue(torch.backends.npu.mem_efficient_sdp_enabled())
        self.assertTrue(torch.backends.npu.math_sdp_enabled())
        self.assertEqual(torch.backends.npu.preferred_linalg_library(), "default")

    # -----------------------------------------------------------------------
    # 4. torch.accelerator Integration (PyTorch 2.4+)
    # -----------------------------------------------------------------------

    def test_torch_accelerator(self):
        """Verify integration with PyTorch's torch.accelerator abstraction."""
        if not hasattr(torch, "accelerator"):
            self.skipTest("torch.accelerator not available in this PyTorch build")

        import torch.accelerator as acc

        self.assertTrue(acc.is_available())
        self.assertGreaterEqual(acc.device_count(), 1)
        self.assertEqual(str(acc.current_accelerator()), "npu")
        self.assertEqual(acc.current_device_index(), 0)

        # Should execute cleanly without asserts
        acc.set_device_index(0)
        acc.synchronize()
        acc.empty_cache()

        stream = acc.current_stream()
        self.assertIsNotNone(stream)

        # Memory queries
        self.assertEqual(acc.memory_allocated(), 0)
        self.assertEqual(acc.max_memory_allocated(), 0)
        self.assertEqual(acc.memory_reserved(), 0)
        self.assertEqual(acc.max_memory_reserved(), 0)
        self.assertIsInstance(acc.memory_stats(), dict)
        acc.reset_peak_memory_stats()

    # -----------------------------------------------------------------------
    # 5. Tensor Movement, Tagging & Factories
    # -----------------------------------------------------------------------

    def test_tensor_to_npu(self):
        """Verify torch.Tensor.to('npu') and movement behavior."""
        t = torch.randn(4, 4)
        self.assertFalse(t.is_npu)

        t_npu = t.to("npu")
        self.assertTrue(t_npu.is_npu)
        self.assertEqual(t_npu.shape, (4, 4))
        assert_close(t_npu, t)

        # Target device torch.device("npu:0")
        t_dev = t.to(torch.device("npu:0"))
        self.assertTrue(t_dev.is_npu)

        # Keyword device="npu"
        t_kw = t.to(device="npu")
        self.assertTrue(t_kw.is_npu)

        # Preserving is_npu across dtype conversion
        t_fp16 = t_npu.to(torch.float16)
        self.assertTrue(t_fp16.is_npu)
        self.assertEqual(t_fp16.dtype, torch.float16)

        # Clearing is_npu when explicitly moved to cpu
        t_cpu = t_npu.to("cpu")
        self.assertFalse(t_cpu.is_npu)

    def test_tensor_npu_method(self):
        """Verify torch.Tensor.npu() method."""
        t = torch.randn(3, 5)
        t_npu = t.npu()
        self.assertTrue(t_npu.is_npu)
        self.assertEqual(t_npu.shape, (3, 5))
        assert_close(t_npu, t)

    def test_tensor_factories_device_npu(self):
        """Verify factory functions accept device='npu' and tag tensor as is_npu."""
        factories = [
            (torch.empty, (3, 3)),
            (torch.zeros, (3, 3)),
            (torch.ones, (3, 3)),
            (torch.randn, (3, 3)),
            (torch.rand, (3, 3)),
            (torch.full, ((3, 3), 7.0)),
            (torch.arange, (0, 10)),
            (torch.eye, (4,)),
        ]
        for fn, args in factories:
            with self.subTest(fn=fn.__name__):
                res = fn(*args, device="npu")
                self.assertTrue(res.is_npu)

        # Like factories
        ref = torch.randn(2, 4)
        self.assertTrue(torch.empty_like(ref, device="npu").is_npu)
        self.assertTrue(torch.zeros_like(ref, device="npu").is_npu)
        self.assertTrue(torch.ones_like(ref, device="npu").is_npu)
        self.assertTrue(torch.randn_like(ref, device="npu").is_npu)

        # torch.tensor & torch.as_tensor
        t1 = torch.tensor([1.0, 2.0, 3.0], device="npu")
        self.assertTrue(t1.is_npu)
        t2 = torch.as_tensor([4.0, 5.0, 6.0], device="npu")
        self.assertTrue(t2.is_npu)

    # -----------------------------------------------------------------------
    # 6. Module Integration (.to("npu") and .npu())
    # -----------------------------------------------------------------------

    def test_module_to_npu_execution(self):
        """Verify model.to('npu') compiles and executes on NPU."""
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        x = torch.randn(2, 8, device="npu")
        compiled_model = model.to("npu")

        out = compiled_model(x)
        self.assertEqual(out.shape, (2, 4))

        # Second forward to verify cache hit
        out2 = compiled_model(x)
        assert_close(out, out2)

    def test_module_npu_method(self):
        """Verify model.npu() method."""
        model = nn.Linear(6, 3)
        compiled = model.npu()
        x = torch.randn(2, 6)
        out = compiled(x)
        self.assertEqual(out.shape, (2, 3))

    # -----------------------------------------------------------------------
    # 7. Autocast Context
    # -----------------------------------------------------------------------

    def test_autocast_npu(self):
        """Verify torch.autocast and torch.amp.autocast with npu device."""
        model = nn.Linear(8, 4).to("npu")
        x = torch.randn(2, 8, device="npu")

        with torch.autocast(device_type="npu", dtype=torch.float16):
            out = model(x)
            self.assertEqual(out.shape, (2, 4))

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            with torch.amp.autocast("npu", dtype=torch.float16):
                out2 = model(x)
                self.assertEqual(out2.shape, (2, 4))

    # -----------------------------------------------------------------------
    # 8. TorchDynamo Backend Listing & Compilation
    # -----------------------------------------------------------------------

    def test_torch_dynamo_backends_registered(self):
        """Verify npu and intel_npu are discovered by TorchDynamo."""
        backends = torch._dynamo.list_backends()
        self.assertIn("npu", backends)
        self.assertIn("intel_npu", backends)

    def test_torch_compile_direct(self):
        """Verify torch.compile(model, backend='npu') and backend='intel_npu'."""
        model = nn.Linear(4, 2)
        x = torch.randn(2, 4)

        opt_npu = torch.compile(model, backend="npu")
        out1 = opt_npu(x)
        self.assertEqual(out1.shape, (2, 2))

        opt_intel = torch.compile(model, backend="intel_npu")
        out2 = opt_intel(x)
        self.assertEqual(out2.shape, (2, 2))
        assert_close(out1, out2)


if __name__ == "__main__":
    unittest.main()
