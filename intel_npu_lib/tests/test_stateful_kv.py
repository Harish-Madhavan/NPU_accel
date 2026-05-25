import unittest
import torch
import torch.nn as nn
import numpy as np
from intel_npu_acceleration import compile_to_npu
from intel_npu_acceleration.functional import NPUStatefulKVCache
from intel_npu_acceleration.frontend import _GRAPH_CACHE

class TestStatefulKVCache(unittest.TestCase):
    def setUp(self):
        _GRAPH_CACHE.clear()

    def test_stateful_kv_correctness(self):
        """
        Verify that the NPUStatefulKVCache compiles and retains state across
        consecutive autoregressive forward steps, matching CPU eager execution.
        """
        batch_size = 1
        max_seq_len = 16
        num_heads = 2
        head_dim = 4

        # 1. Define model with NPUStatefulKVCache
        class LLMDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_cache = NPUStatefulKVCache(
                    batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float32
                )

            def forward(self, new_kv):
                # Return the updated cache state
                return self.kv_cache(new_kv)

        # 2. Compile model for NPU
        x_example = torch.randn(batch_size, 1, num_heads, head_dim)
        model_cpu = LLMDecoderLayer()
        model_cpu.eval()

        try:
            model_npu = compile_to_npu(model_cpu, x_example, strict=True)
        except Exception as e:
            self.fail(f"Compilation of stateful model failed: {e}")

        # Reset CPU and NPU states to synchronize after compilation tracing mutation
        model_npu.reset_states()
        model_cpu.kv_cache.reset()

        # 3. Perform 4 autoregressive steps of length 1 (matching standard token-by-token generation)
        np.random.seed(42)
        
        step_tensors = []
        for step in range(4):
            step_np = np.random.randn(batch_size, 1, num_heads, head_dim).astype(np.float32)
            step_tensor = torch.from_numpy(step_np)
            step_tensors.append(step_tensor)

            out_cpu = model_cpu(step_tensor)
            out_npu = model_npu(step_tensor)

            # Assert outputs match on this step
            self.assertTrue(
                torch.allclose(out_npu, out_cpu, atol=1e-3, rtol=1e-3),
                f"Mismatch at step {step}: NPU output not matching CPU eager output."
            )

        # 4. Final verification: Check that all 4 steps were sequentially appended
        final_cache_cpu = model_cpu.kv_cache.cache
        final_cache_npu = out_npu  # The last returned NPU output

        # Assert full cache matching
        self.assertTrue(torch.allclose(final_cache_npu, final_cache_cpu, atol=1e-3, rtol=1e-3))
        # Ensure first 4 slots are populated with the inputs, and the rest (4..16) are zeros
        all_steps_concat = torch.cat(step_tensors, dim=1)
        self.assertTrue(torch.allclose(final_cache_npu[:, :4], all_steps_concat, atol=1e-3, rtol=1e-3))
        self.assertEqual(torch.sum(final_cache_npu[:, 4:]), 0.0)

    def test_stateful_kv_reset(self):
        """
        Verify that calling reset_states() resets the state registers in the NPU hardware.
        """
        batch_size = 1
        max_seq_len = 8
        num_heads = 2
        head_dim = 4

        class SimpleStatefulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_cache = NPUStatefulKVCache(
                    batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float32
                )

            def forward(self, new_kv):
                return self.kv_cache(new_kv)

        model_cpu = SimpleStatefulModel()
        model_cpu.eval()
        x_example = torch.randn(batch_size, 1, num_heads, head_dim)

        model_npu = compile_to_npu(model_cpu, x_example, strict=True)

        # Write to cache (2 updates of length 1)
        step_tensor_1 = torch.randn(batch_size, 1, num_heads, head_dim)
        step_tensor_2 = torch.randn(batch_size, 1, num_heads, head_dim)
        
        model_npu(step_tensor_1)
        out_before_reset = model_npu(step_tensor_2)
        
        self.assertNotEqual(torch.sum(out_before_reset), 0.0)

        # Reset states
        model_npu.reset_states()
        model_cpu.kv_cache.reset()

        # Write again after reset: should write at index 0 rather than index 2!
        out_after_reset = model_npu(step_tensor_1)
        out_cpu_after_reset = model_cpu(step_tensor_1)

        self.assertTrue(torch.allclose(out_after_reset, out_cpu_after_reset, atol=1e-3, rtol=1e-3))
        # Verify position 0 is filled with step_tensor_1, rest is zero
        self.assertTrue(torch.allclose(out_after_reset[:, :1], step_tensor_1, atol=1e-3, rtol=1e-3))
        self.assertEqual(torch.sum(out_after_reset[:, 1:]), 0.0)

    def test_double_buffered_output_safety(self):
        """
        Verify that the round-robin double-buffered output pre-allocation guarantees
        output tensor reference safety across consecutive inference runs, even if the user
        retains a reference to the output without explicit cloning (clone_outputs=False).
        """
        class AddModel(nn.Module):
            def forward(self, x, y):
                return x + y

        model = AddModel()
        model.eval()

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        # Compile with clone_outputs=False to test raw buffer reference safety
        compiled = compile_to_npu(model, (x, y), clone_outputs=False)

        # Execution 1
        res1 = compiled(torch.tensor([[1.0, 1.0], [1.0, 1.0]]), torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
        # res1 points directly to the active double buffer (e.g. buffer 0)
        self.assertTrue(torch.allclose(res1, torch.tensor([[3.0, 3.0], [3.0, 3.0]])))

        # Execution 2 (uses other double buffer, e.g. buffer 1)
        res2 = compiled(torch.tensor([[5.0, 5.0], [5.0, 5.0]]), torch.tensor([[10.0, 10.0], [10.0, 10.0]]))
        
        # Verify res1 was NOT overwritten and still contains the values from execution 1
        self.assertTrue(torch.allclose(res1, torch.tensor([[3.0, 3.0], [3.0, 3.0]])))
        self.assertTrue(torch.allclose(res2, torch.tensor([[15.0, 15.0], [15.0, 15.0]])))

if __name__ == "__main__":
    unittest.main()
