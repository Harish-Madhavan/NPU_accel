import unittest
import torch
import torch.nn as nn
import threading
from intel_npu_acceleration import compile_to_npu, is_available


class AdditionModel(nn.Module):
    def forward(self, x, y):
        return x + y


class MultiplicationModel(nn.Module):
    def forward(self, x, y):
        return x * y


class TestNPUStress(unittest.TestCase):
    def setUp(self):
        if not is_available():
            print(
                "WARNING: Intel NPU is not available; running stress tests in CPU fallback mode."
            )

    def test_inference_stability(self):
        # 1. Compilation
        model = AdditionModel()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        npu_model = compile_to_npu(model, (x, y))

        # 2. Continuous execution loop (10,000 iterations for memory leak audit)
        print("Running 10,000 iterations for memory leak audit...")
        for i in range(10000):
            out = npu_model(x, y)
            if i % 1000 == 0:
                expected = x + y
                self.assertTrue(torch.allclose(out, expected, atol=1e-3, rtol=1e-3))
        print("Completed 10,000 iterations successfully.")

    def test_context_switching(self):
        # Compile two different models
        model1 = AdditionModel()
        model2 = MultiplicationModel()

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        npu_model1 = compile_to_npu(model1, (x, y))
        npu_model2 = compile_to_npu(model2, (x, y))

        # Toggle/interleave between them to ensure cached requests do not clash
        for _ in range(50):
            out1 = npu_model1(x, y)
            self.assertTrue(torch.allclose(out1, x + y, atol=1e-3, rtol=1e-3))

            out2 = npu_model2(x, y)
            self.assertTrue(torch.allclose(out2, x * y, atol=1e-3, rtol=1e-3))

    def test_multithreaded_stress(self):
        model = AdditionModel()
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        npu_model = compile_to_npu(model, (x, y))

        num_threads = 4
        iterations_per_thread = 100
        errors = []

        def worker():
            try:
                # Thread-local execution should reuse the thread-local InferRequest
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

        self.assertEqual(
            len(errors), 0, f"Multi-threaded stress test failed with errors: {errors}"
        )

    def test_leased_buffer_pool_concurrency(self):
        import gc
        model = AdditionModel()
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        # Compile with clone_outputs=False to test weakref buffer recycling
        npu_model = compile_to_npu(model, (x, y), num_streams=5, clone_outputs=False)

        pool = npu_model.output_pools[0]
        self.assertEqual(len(pool.active_buffers), 0)

        def run_inferences():
            # Launch 5 overlapping async inferences
            handles = []
            for i in range(5):
                h = npu_model.infer_async(x + i, y)
                handles.append(h)

            # Active buffers should now be 5
            self.assertEqual(len(pool.active_buffers), 5)

            # Wait and collect outputs
            outputs = []
            for i, h in enumerate(handles):
                out = npu_model.wait_async(h)
                self.assertTrue(torch.allclose(out, x + i + y, atol=1e-3, rtol=1e-3))
                outputs.append(out)

            # Tensors are still held in local scope `outputs`, so they must not be recycled yet
            self.assertEqual(len(pool.active_buffers), 5)
            return len(pool.active_buffers)

        # Execute in nested scope
        active_before = run_inferences()
        self.assertEqual(active_before, 5)

        # After returning, the local scope variables are completely destroyed. Force garbage collection!
        gc.collect()

        # Active buffers should drop back to 0, and all 5 should be returned to idle!
        self.assertEqual(len(pool.active_buffers), 0)
        self.assertEqual(len(pool.idle_buffers), 5)


if __name__ == "__main__":
    unittest.main()
