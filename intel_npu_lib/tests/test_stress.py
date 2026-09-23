import gc
import os
import threading

import pytest
import torch
from torch.testing import assert_close

from intel_npu_acceleration import compile_to_npu, is_available
from tests.helpers import make_addition_model


class TestNPUStress:
    @pytest.mark.slow
    def test_inference_stability(self):
        """Stability / leak check — 500 iterations by default, 10k with NPU_STRESS_FULL=1."""
        if not is_available():
            print("WARNING: Intel NPU is not available; running stress tests in CPU fallback mode.")
        model = make_addition_model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        npu_model = compile_to_npu(model, (x, y))

        iterations = 10000 if os.environ.get("NPU_STRESS_FULL") == "1" else 500
        for i in range(iterations):
            out = npu_model(x, y)
            if i % max(1, iterations // 5) == 0:
                assert_close(out, x + y, atol=1e-3, rtol=1e-3)

    def test_context_switching(self):
        model1 = make_addition_model()

        class MultiplicationModel(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        model2 = MultiplicationModel()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        npu_model1 = compile_to_npu(model1, (x, y))
        npu_model2 = compile_to_npu(model2, (x, y))
        for _ in range(50):
            assert_close(npu_model1(x, y), x + y, atol=1e-3, rtol=1e-3)
            assert_close(npu_model2(x, y), x * y, atol=1e-3, rtol=1e-3)

    def test_multithreaded_stress(self):
        model = make_addition_model()
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        npu_model = compile_to_npu(model, (x, y))
        num_threads = 4
        iterations_per_thread = 100
        errors = []

        def worker():
            try:
                for _ in range(iterations_per_thread):
                    out = npu_model(x, y)
                    if not torch.allclose(out, x + y, atol=1e-3, rtol=1e-3):
                        errors.append("Output mismatch in thread")
            except Exception as e:
                errors.append(f"Exception in thread: {e}")

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Multi-threaded stress test failed: {errors}"

    def test_leased_buffer_pool_concurrency(self):
        model = make_addition_model()
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        npu_model = compile_to_npu(model, (x, y), num_streams=5, clone_outputs=False)
        pool = npu_model.output_pools[0]
        assert pool is not None
        assert len(pool.active_buffers) == 0

        def run_inferences():
            handles = [npu_model.infer_async(x + i, y) for i in range(5)]
            assert len(pool.active_buffers) == 5
            outputs = []
            for i, h in enumerate(handles):
                out = npu_model.wait_async(h)
                assert_close(out, x + i + y, atol=1e-3, rtol=1e-3)
                outputs.append(out)
            assert len(pool.active_buffers) == 5
            return len(pool.active_buffers)

        active_before = run_inferences()
        assert active_before == 5
        gc.collect()
        assert len(pool.active_buffers) == 0
        assert len(pool.idle_buffers) == 5
