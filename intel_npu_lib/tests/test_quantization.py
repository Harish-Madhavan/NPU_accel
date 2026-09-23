import pytest
import torch
import torch.nn as nn

from intel_npu_acceleration import compile_to_npu, quantize
from intel_npu_acceleration.functional import quantized_linear


class FunctionalQuantizedLinear(nn.Module):
    def forward(self, input, weight, scale, zero_point=None, bias=None):
        return quantized_linear(input, weight, scale, zero_point, bias)


class TestQuantization:
    def test_quantized_linear_eager(self):
        # Batch=2, InFeatures=8, OutFeatures=4
        B, C_in, C_out = 2, 8, 4
        input_t = torch.randn(B, C_in, dtype=torch.float32)

        # Quantized weight: int8 values
        weight_t = torch.randint(-100, 100, (C_out, C_in), dtype=torch.int8)

        # Per-channel scale (C_out, 1)
        scale_t = torch.rand(C_out, 1, dtype=torch.float32) * 0.1

        # Optional zero-point
        zp_t = torch.randint(-5, 5, (C_out, 1), dtype=torch.int8)

        # Optional bias
        bias_t = torch.randn(C_out, dtype=torch.float32)

        # 1. Eager mode with scales & zero-points & bias
        out_eager = quantized_linear(input_t, weight_t, scale_t, zp_t, bias_t)

        # Verify against CPU fallback calculation:
        # dequantized_w = (weight - zp) * scale
        w_dequant = (weight_t.float() - zp_t.float()) * scale_t.float()
        expected = torch.nn.functional.linear(input_t, w_dequant, bias_t)

        assert torch.allclose(out_eager, expected, atol=1e-2, rtol=1e-2)

        # 2. Eager mode without zero-points and without bias
        out_eager_simple = quantized_linear(input_t, weight_t, scale_t)
        expected_simple = torch.nn.functional.linear(
            input_t, weight_t.float() * scale_t
        )
        assert torch.allclose(out_eager_simple, expected_simple, atol=1e-2, rtol=1e-2)

    def test_quantized_linear_compiled(self):
        B, C_in, C_out = 1, 16, 8
        input_t = torch.randn(B, C_in, dtype=torch.float32)
        weight_t = torch.randint(-120, 120, (C_out, C_in), dtype=torch.int8)
        scale_t = torch.rand(C_out, 1, dtype=torch.float32) * 0.05
        zp_t = torch.randint(-3, 3, (C_out, 1), dtype=torch.int8)
        bias_t = torch.randn(C_out, dtype=torch.float32)

        model = FunctionalQuantizedLinear()
        model.eval()

        try:
            # Compile model to NPU
            npu_model = compile_to_npu(
                model, (input_t, weight_t, scale_t, zp_t, bias_t)
            )
        except Exception as e:
            pytest.fail(f"Quantized compilation failed: {e}")

        out_npu = npu_model(input_t, weight_t, scale_t, zp_t, bias_t)
        out_cpu = model(input_t, weight_t, scale_t, zp_t, bias_t)

        assert torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2)

    def test_quantized_linear_float16(self):
        # Test precision-compatibility with Float16
        B, C_in, C_out = 2, 8, 4
        input_t = torch.randn(B, C_in, dtype=torch.float16)
        weight_t = torch.randint(-50, 50, (C_out, C_in), dtype=torch.int8)
        scale_t = torch.rand(C_out, 1, dtype=torch.float16) * 0.02
        zp_t = torch.randint(-2, 2, (C_out, 1), dtype=torch.int8)
        bias_t = torch.randn(C_out, dtype=torch.float16)

        model = FunctionalQuantizedLinear()
        model.eval()

        # Eager execution
        out_eager = quantized_linear(input_t, weight_t, scale_t, zp_t, bias_t)
        assert out_eager.dtype == torch.float16

        # Compiled execution
        try:
            npu_model = compile_to_npu(
                model, (input_t, weight_t, scale_t, zp_t, bias_t)
            )
            out_npu = npu_model(input_t, weight_t, scale_t, zp_t, bias_t)
            assert torch.allclose(out_npu.float(), out_eager.float(), atol=1e-2, rtol=1e-2)
        except Exception as e:
            pytest.fail(f"Float16 quantized compilation failed: {e}")

    def test_quantized_linear_int4(self):
        # Test packed INT4 weight unpacking
        # Columns = InFeatures // 2 = 4, hidden dim = 8
        B, C_in, C_out = 2, 8, 4
        input_t = torch.randn(B, C_in, dtype=torch.float32)
        weight_t = torch.randint(0, 256, (C_out, C_in // 2), dtype=torch.uint8)
        scale_t = torch.rand(C_out, 1, dtype=torch.float32) * 0.1
        zp_t = torch.randint(0, 16, (C_out, 1), dtype=torch.int8)
        bias_t = torch.randn(C_out, dtype=torch.float32)

        model = FunctionalQuantizedLinear()
        model.eval()

        # Eager execution
        out_eager = quantized_linear(input_t, weight_t, scale_t, zp_t, bias_t)

        # Expected output via independent reference unpack (low nibble first)
        w_odd = torch.floor_divide(weight_t, 16)
        w_even = weight_t - w_odd * 16
        w_unpacked = torch.stack([w_even, w_odd], dim=-1).view(C_out, -1).float()
        w_dequant = (w_unpacked - zp_t.float()) * scale_t.float()
        expected = torch.nn.functional.linear(input_t, w_dequant, bias_t)

        assert torch.allclose(out_eager, expected, atol=1e-2, rtol=1e-2)

        # Compiled execution
        try:
            npu_model = compile_to_npu(
                model, (input_t, weight_t, scale_t, zp_t, bias_t)
            )
            out_npu = npu_model(input_t, weight_t, scale_t, zp_t, bias_t)
            assert torch.allclose(out_npu, out_eager, atol=1e-2, rtol=1e-2)
        except Exception as e:
            pytest.fail(f"INT4 quantized compilation failed: {e}")

    def test_quantize_api_int8_per_channel(self):
        model = nn.Sequential(nn.Linear(16, 8, bias=False))
        x = torch.randn(2, 16)
        orig_out = model(x)

        quantize(model, mode="int8", granularity="per-channel")
        assert model[0].weight.dtype == torch.int8
        assert model[0].weight_scale.shape == (8, 1)

        # Test compiling quantized model
        compiled_model = compile_to_npu(model, x)
        compiled_out = compiled_model(x)

        # Output should be close to unquantized output
        assert torch.allclose(compiled_out, orig_out, atol=0.2, rtol=0.2)

    def test_quantize_api_int4_packed(self):
        model = nn.Sequential(nn.Linear(16, 8, bias=False))
        x = torch.randn(2, 16)
        orig_out = model(x)

        quantize(model, mode="int4", granularity="per-channel")
        assert model[0].weight.dtype == torch.uint8
        # 16 in_features packed into 8 uint8 columns
        assert model[0].weight.shape == (8, 8)
        assert model[0].weight_scale.shape == (8, 1)
        assert model[0].weight_zero_point.shape == (8, 1)

        # Test compiling int4 model
        compiled_model = compile_to_npu(model, x)
        compiled_out = compiled_model(x)

        assert torch.allclose(compiled_out, orig_out, atol=0.3, rtol=0.3)
