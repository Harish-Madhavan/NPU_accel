import torch
import unittest
import intel_npu_acceleration
from intel_npu_acceleration.frontend import (
    NPUGraphModule,
    NPUCompilationError,
)


class TestAsyncAndHybrid(unittest.TestCase):
    def setUp(self):
        self.avail = intel_npu_acceleration.is_available()
        if not self.avail:
            print("WARNING: Intel NPU not detected. Testing with CPU fallback mode.")

    def test_asynchronous_pipeline_latency(self):
        """
        Verify that start_async() / infer_async() and wait_async() work correctly,
        and yield exact correctness compared to standard CPU execution.
        """

        class SimpleModel(torch.nn.Module):
            def forward(self, x, y):
                z = torch.add(x, y)
                w = torch.mul(z, x)
                return w

        model = SimpleModel()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # Compile model with LATENCY hint
        compiled_model = intel_npu_acceleration.compile(
            model, (x, y), performance_hint="LATENCY"
        )

        # Verify it is an instance of NPUGraphModule
        self.assertTrue(isinstance(compiled_model, NPUGraphModule))

        # Start asynchronous inference
        handle = compiled_model.infer_async(x, y)
        self.assertIsInstance(handle, int)

        # Perform some concurrent dummy work on CPU
        cpu_result = torch.sin(x) * 1.5

        # Wait for the async NPU execution to complete and fetch result
        out_async = compiled_model.wait_async(handle)

        # Get expected result
        out_expected = model(x, y)

        # Check values
        self.assertTrue(torch.allclose(out_async, out_expected, rtol=1e-2, atol=1e-2))
        self.assertTrue(cpu_result is not None)

    def test_asynchronous_pipeline_throughput(self):
        """
        Verify async pipeline under THROUGHPUT mode using multiple streams.
        """

        class MatmulModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.mul(x, y)

        model = MatmulModel()
        x1 = torch.randn(3, 3)
        y1 = torch.randn(3, 3)
        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)

        # Compile with THROUGHPUT hint and 2 streams
        compiled_model = intel_npu_acceleration.compile(
            model, (x1, y1), performance_hint="THROUGHPUT", num_streams=2
        )
        self.assertEqual(compiled_model.num_streams, 2)
        self.assertEqual(len(compiled_model.infer_requests), 2)

        # Enqueue first inference on stream 0
        h1 = compiled_model.infer_async(x1, y1)
        # Enqueue second inference on stream 1
        h2 = compiled_model.infer_async(x2, y2)

        self.assertNotEqual(h1, h2)

        # Wait for results
        out1 = compiled_model.wait_async(h1)
        out2 = compiled_model.wait_async(h2)

        # Verify against expected
        self.assertTrue(torch.allclose(out1, model(x1, y1), rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(out2, model(x2, y2), rtol=1e-2, atol=1e-2))

    def test_hybrid_graph_partitioning_fallback(self):
        """
        Verify that automated hybrid graph partitioning splits the graph correctly,
        compiles supported segments for NPU, keeps unsupported operations on CPU,
        and completes without NPUCompilationError.
        """

        class HybridTestModel(torch.nn.Module):
            def forward(self, x, y):
                # 1. Supported op (torch.add)
                z = torch.add(x, y)
                # 2. Unsupported op (torch.erf - not in OpRegistry)
                s = torch.erf(z)
                # 3. Supported op (torch.mul)
                w = torch.mul(s, x)
                return w

        model = HybridTestModel()
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)

        # Verify that compiling this directly would normally fail without partitioning
        # if we forced full-NPU compilation. But with partitioning, it should succeed.
        try:
            compiled_model = intel_npu_acceleration.compile(model, (x, y))
        except NPUCompilationError as e:
            self.fail(f"Compilation failed due to NPUCompilationError: {e}")

        # The compiled model returned should be the split parent GraphModule,
        # not directly NPUGraphModule, because partitioning was triggered.
        self.assertNotIsInstance(compiled_model, NPUGraphModule)
        self.assertTrue(isinstance(compiled_model, torch.fx.GraphModule))

        # Check that we have multiple children under the split module
        children = list(compiled_model.named_children())
        self.assertTrue(
            len(children) >= 2,
            f"Expected segmented child modules, found: {len(children)}",
        )

        # Print the children structure for visibility in logs
        for name, child in children:
            print(f"Child submodule '{name}': {type(child)}")

        # Run the compiled model
        out_compiled = compiled_model(x, y)
        out_expected = model(x, y)

        # Validate numerical accuracy
        self.assertTrue(
            torch.allclose(out_compiled, out_expected, rtol=1e-2, atol=1e-2)
        )

    def test_npu_async_future_submit(self):
        """
        Verify that compiled_model.submit() returns a functional NPUAsyncFuture handle.
        """
        class FutureTestModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) * 2.0

        model = FutureTestModel()
        x = torch.randn(8, 8)
        compiled = intel_npu_acceleration.compile(model, x, num_streams=2)
        self.assertTrue(isinstance(compiled, NPUGraphModule))

        future = compiled.submit(x)
        self.assertTrue(hasattr(future, "result"))
        self.assertTrue(hasattr(future, "is_ready"))
        res = future.result()
        expected = model(x)
        self.assertTrue(torch.allclose(res.float(), expected.float(), atol=1e-2, rtol=1e-2))
        self.assertTrue(future.is_ready())

    def test_batch_infer_pipeline(self):
        """
        Verify that compiled_model.batch_infer() accurately pipelines a list of inputs.
        """
        class BatchPipelineModel(torch.nn.Module):
            def forward(self, a, b):
                return torch.matmul(a, b)

        model = BatchPipelineModel()
        a_list = [torch.randn(4, 4) for _ in range(6)]
        b_list = [torch.randn(4, 4) for _ in range(6)]
        inputs_list = [(a_list[i], b_list[i]) for i in range(6)]

        compiled = intel_npu_acceleration.compile(model, inputs_list[0], num_streams=3)
        results = compiled.batch_infer(inputs_list)

        self.assertEqual(len(results), 6)
        for i in range(6):
            expected = model(a_list[i], b_list[i])
            self.assertTrue(torch.allclose(results[i].float(), expected.float(), atol=1e-2, rtol=1e-2))

    def test_fp16_precision_optimization(self):
        """
        Verify that precision='fp16' compiles and runs successfully with OpenVINO ConvertFP32ToFP16 pass.
        """
        class FP16PrecisionModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x) + 1.0

        model = FP16PrecisionModel()
        x = torch.randn(16, 16)
        compiled = intel_npu_acceleration.compile(model, x, precision="fp16")
        res = compiled(x)
        expected = model(x)
        self.assertEqual(res.dtype, torch.float16)
        self.assertTrue(torch.allclose(res.float(), expected.float(), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
