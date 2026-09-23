import torch

import intel_npu_acceleration
from tests.helpers import assert_allclose


class TestIntelNPULib:
    def test_basic_ops(self):
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        b_safe = b + 0.1
        cases = [
            (intel_npu_acceleration.add(a, b), a + b),
            (intel_npu_acceleration.sub(a, b), a - b),
            (intel_npu_acceleration.mul(a, b), a * b),
            (intel_npu_acceleration.div(a, b_safe), a / b_safe),
        ]
        for actual, expected in cases:
            assert_allclose(actual, expected)

    def test_matmul(self):
        a = torch.randn(4, 5)
        b = torch.randn(5, 3)
        assert_allclose(intel_npu_acceleration.matmul(a, b), torch.matmul(a, b))

    def test_activations_compile(self):
        class ActivationsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()
                self.relu6 = torch.nn.ReLU6()
                self.leaky_relu = torch.nn.LeakyReLU(0.05)
                self.hardtanh = torch.nn.Hardtanh(-2.0, 2.0)

            def forward(self, x):
                s1 = self.sigmoid(x)
                s2 = torch.sigmoid(s1)
                s3 = s2.sigmoid()
                t1 = self.tanh(s3)
                t2 = torch.tanh(t1)
                t3 = t2.tanh()
                r1 = self.relu6(t3)
                r2 = torch.nn.functional.relu6(r1)
                l1 = self.leaky_relu(r2)
                l2 = torch.nn.functional.leaky_relu(l1, negative_slope=0.05)
                c1 = self.hardtanh(l2)
                c2 = torch.clamp(c1, min=-1.5, max=1.5)
                c3 = torch.nn.functional.hardtanh(c2, min_val=-1.2, max_val=1.2)
                return c3.clamp(min=-1.0, max=1.0)

        model = ActivationsModel()
        x = torch.randn(2, 4)
        compiled = intel_npu_acceleration.compile(model, x, strict=True)
        assert_allclose(compiled(x), model(x))

    def test_integration_compile(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.mul(torch.add(x, y), x)

        model = SimpleModel()
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        compiled = intel_npu_acceleration.compile(model, (x, y))
        assert_allclose(compiled(x, y), model(x, y))
        # Second run hits in-memory cache — should still be correct
        assert_allclose(compiled(x, y), model(x, y))

    def test_cache_management(self):
        size, count = intel_npu_acceleration.get_cache_size()
        assert isinstance(size, int)
        assert isinstance(count, int)
        intel_npu_acceleration.clean_old_cache(max_size_mb=10, max_files=5)
        intel_npu_acceleration.clear_cache()
        from intel_npu_acceleration.frontend.compiler import _GRAPH_CACHE

        intel_npu_acceleration.clear_graph_cache()
        assert len(_GRAPH_CACHE) == 0
        new_size, new_count = intel_npu_acceleration.get_cache_size()
        assert new_size <= size
        assert new_count <= count
        v0 = intel_npu_acceleration.get_cache_version()
        assert isinstance(v0, int)

    def test_quantize_api(self):
        class SimpleLinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)

            def forward(self, x):
                return self.fc(x)

        model = SimpleLinearModel().eval()
        from intel_npu_acceleration import quantize

        quantized = quantize(model)
        assert quantized.fc.weight.dtype == torch.int8
        assert hasattr(quantized.fc, "weight_scale")
        x = torch.randn(2, 8)
        compiled = intel_npu_acceleration.compile(quantized, x)
        w_float = quantized.fc.weight.float() * quantized.fc.weight_scale
        expected = torch.nn.functional.linear(x, w_float, quantized.fc.bias)
        assert_allclose(compiled(x), expected)

    def test_ppp_advanced(self):
        intel_npu_acceleration.clear_cache()

        class ImageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, kernel_size=1, bias=False)
                torch.nn.init.ones_(self.conv.weight)

            def forward(self, x):
                return self.conv(x)

        model = ImageModel().eval()
        x_traced = torch.randn(1, 3, 224, 224)
        x_runtime = torch.randint(0, 256, (1, 256, 256, 3), dtype=torch.uint8)
        preprocess_config = {
            "input": {
                "shape": (1, 256, 256, 3),
                "layout": "NHWC",
                "element_type": "u8",
                "model_layout": "NCHW",
                "color_format": "BGR",
                "model_color_format": "RGB",
                "resize": (224, 224),
            }
        }
        compiled = intel_npu_acceleration.compile(model, x_traced, preprocess_config=preprocess_config)
        out = compiled(x_runtime)
        assert list(out.shape) == [1, 3, 224, 224]
        assert out.dtype == torch.float32

    def test_pytorch_compile(self):
        class StandardModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y) + 1.0

        model = StandardModel()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        compiled = torch.compile(model, backend="npu")
        assert_allclose(compiled(x, y), model(x, y))
        compiled_opt = torch.compile(
            model, backend="npu", options={"clone_outputs": False, "performance_hint": "THROUGHPUT"}
        )
        assert_allclose(compiled_opt(x, y), model(x, y))
