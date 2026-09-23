import logging

import pytest
import torch

from intel_npu_acceleration.frontend import (
    _GRAPH_CACHE,
    compile_to_npu,
)
from tests.helpers import make_conv_model


class TestNPUHardwarePPP:
    def setup_method(self):
        # Clear cache before each test
        _GRAPH_CACHE.clear()
        self.model = make_conv_model(3, 8, 3, padding=1)
        self.model.eval()

    def test_ppp_offloading_correctness(self):
        """
        Verify that NPU Hardware PPP offloading works correctly and produces
        mathematically identical outputs (within tolerance) compared to manual CPU-side preprocessing.
        """
        # Example input shape in model layout (NCHW)
        x_example = torch.randn(1, 3, 32, 32)

        # Define PPP configuration:
        # User input tensor: NHWC layout, uint8 data, to be cast to f32,
        # followed by mean subtraction and scale division, then transposed to NCHW.
        preprocess_config = {
            "input": {
                "layout": "NHWC",
                "element_type": "u8",
                "model_layout": "NCHW",
                "mean": [127.5, 127.5, 127.5],
                "scale": [127.5, 127.5, 127.5],
            }
        }

        # Compile model with NPU Hardware PPP offloaded
        try:
            compiled_model = compile_to_npu(
                self.model,
                x_example,
                preprocess_config=preprocess_config,
            )
        except Exception as e:
            pytest.fail(f"Compilation with PPP config failed: {e}")

        # Construct actual test input: a 32x32 image with 3 channels in NHWC format, of type uint8
        # Fill with specific pattern to verify normalization and transposition
        input_tensor_u8 = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)

        # 1. Forward pass using NPU Hardware PPP (expects NHWC u8 tensor)
        out_npu = compiled_model(input_tensor_u8)

        # 2. Manual CPU Preprocessing to verify correctness
        # Cast to float32
        input_f32 = input_tensor_u8.to(torch.float32)
        # Transpose NHWC [1, 32, 32, 3] to NCHW [1, 3, 32, 32]
        input_transposed = input_f32.permute(0, 3, 1, 2)
        # Normalize: (x - mean) / scale
        mean = torch.tensor([127.5, 127.5, 127.5], dtype=torch.float32).view(1, 3, 1, 1)
        scale = torch.tensor([127.5, 127.5, 127.5], dtype=torch.float32).view(1, 3, 1, 1)
        input_normalized = (input_transposed - mean) / scale

        # Forward pass on CPU with manually preprocessed input
        with torch.no_grad():
            out_cpu = self.model(input_normalized)

        # Assert outputs are mathematically close
        assert torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2), (
            f"NPU PPP output {out_npu[0, 0, :2, :2]} not matching CPU manual PPP output {out_cpu[0, 0, :2, :2]}"
        )

    def test_cache_key_robustness(self):
        """
        Verify that compiling the same model with different preprocessing configurations
        generates unique cache keys to prevent incorrect cache hits.
        """
        x_example = torch.randn(1, 3, 32, 32)

        config_none = None
        config_a = {
            "input": {
                "layout": "NHWC",
                "element_type": "u8",
                "model_layout": "NCHW",
                "mean": [127.5, 127.5, 127.5],
                "scale": [127.5, 127.5, 127.5],
            }
        }
        config_b = {
            "input": {
                "layout": "NHWC",
                "element_type": "u8",
                "model_layout": "NCHW",
                "mean": [100.0, 100.0, 100.0],
                "scale": [50.0, 50.0, 50.0],
            }
        }

        # 1. Compile with no config
        compile_to_npu(self.model, x_example, preprocess_config=config_none)
        assert len(_GRAPH_CACHE) == 1
        key_none = list(_GRAPH_CACHE.keys())[0]

        # 2. Compile with config A
        compile_to_npu(self.model, x_example, preprocess_config=config_a)
        assert len(_GRAPH_CACHE) == 2
        key_a = list(_GRAPH_CACHE.keys())[1]
        assert key_none != key_a

        # 3. Compile with config B (different mean/scale)
        compile_to_npu(self.model, x_example, preprocess_config=config_b)
        assert len(_GRAPH_CACHE) == 3
        key_b = list(_GRAPH_CACHE.keys())[2]
        assert key_a != key_b
        assert key_none != key_b

        # 4. Compile with config A again (should trigger cache hit)
        compile_to_npu(self.model, x_example, preprocess_config=config_a)
        assert len(_GRAPH_CACHE) == 3  # Cache size should remain 3

    def test_performance_intelligence_logging(self, caplog):
        """
        Verify that the compiler emits the performance intelligence tips
        exactly when single-stream latency config is used.
        """
        x_example = torch.randn(1, 3, 32, 32)
        logger_name = "intel_npu_acceleration.frontend"

        # 1. Compile in LATENCY mode with single stream - should log the tip
        with caplog.at_level(logging.INFO, logger=logger_name):
            compile_to_npu(
                self.model,
                x_example,
                performance_hint="LATENCY",
                num_streams=1,
            )
            # Ensure the specific performance warning tip is printed
            found_tip = any(
                "[NPU Intelligence] Configuration: LATENCY mode" in record.message
                for record in caplog.records
            )
            assert found_tip, "Performance Intelligence tip was not logged!"

        caplog.clear()

        # 2. Compile in THROUGHPUT mode with multiple streams - should NOT log the tip
        with caplog.at_level(logging.INFO, logger=logger_name):
            compile_to_npu(
                self.model,
                x_example,
                performance_hint="THROUGHPUT",
                num_streams=2,
            )
            # Ensure the specific performance warning tip is NOT printed
            found_tip = any(
                "[NPU Intelligence]" in record.message for record in caplog.records
            )
            assert not found_tip, "Performance Intelligence tip was logged unexpectedly in THROUGHPUT mode!"
