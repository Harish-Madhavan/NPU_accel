import unittest

import numpy as np
import torch
import torch.nn as nn

from intel_npu_acceleration import compile_to_npu
from intel_npu_acceleration.frontend import _GRAPH_CACHE
from intel_npu_acceleration.functional import NPUStatefulKVCache


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

    def test_auto_stateful_mapping(self):
        """
        Verify that functional update_kv_cache is automatically mapped to stateful OpenVINO registers
        when stateful=True is passed during compilation.
        """
        batch_size = 1
        max_seq_len = 16
        num_heads = 2
        head_dim = 4

        class FunctionalLLMDecoderLayer(nn.Module):
            def forward(self, new_kv, cache, position):
                from intel_npu_acceleration.functional import update_kv_cache
                return update_kv_cache(cache, new_kv, position)

        # Traced signature: (new_kv, cache, position)
        new_kv_ex = torch.randn(batch_size, 1, num_heads, head_dim)
        cache_ex = torch.zeros(batch_size, max_seq_len, num_heads, head_dim)
        pos_ex = torch.tensor(0, dtype=torch.int64)

        model_cpu = FunctionalLLMDecoderLayer()
        model_cpu.eval()

        try:
            model_npu = compile_to_npu(model_cpu, (new_kv_ex, cache_ex, pos_ex), stateful=True)
        except Exception as e:
            self.fail(f"Compilation of auto-stateful functional model failed: {e}")

        # The auto-stateful mapping converts the model signature from 3 parameters to 2 compiled parameter inputs (new_kv, position)
        # However, the user calling interface accepts the original 3 arguments dynamically, discarding/ignoring the cache argument.
        model_npu.reset_states()

        # Let's perform 4 sequential autoregressive steps
        np.random.seed(123)
        step_tensors = []
        cache_cpu = cache_ex.clone()
        for step in range(4):
            step_np = np.random.randn(batch_size, 1, num_heads, head_dim).astype(np.float32)
            step_tensor = torch.from_numpy(step_np)
            step_tensors.append(step_tensor)

            # Eager step: updates cache_cpu in-place or returns the new copy
            out_cpu = model_cpu(step_tensor, cache_cpu, step)
            cache_cpu = out_cpu

            # NPU step: we pass step_tensor, cache_ex (as a dummy/ignored parameter to preserve signature), and step
            out_npu = model_npu(step_tensor, cache_ex, step)

            # NPU output returns the updated stateful cache! Let's assert they match!
            self.assertTrue(
                torch.allclose(out_npu[:, :step+1], out_cpu[:, :step+1], atol=1e-3, rtol=1e-3),
                f"Mismatch at step {step}: NPU auto-stateful output not matching CPU eager output."
            )


    def test_multi_layer_auto_stateful_mapping(self):
        """
        Verify that a multi-layer functional LLM (where layer-wise caches are sliced
        from a top-level placeholder kv_cache) is automatically mapped to native
        NPU state registers when stateful=True is used.
        """
        batch_size = 1
        max_seq_len = 8
        num_heads = 2
        head_dim = 4
        n_layers = 2

        class MultiLayerFunctionalLLM(nn.Module):
            def forward(self, new_kv, kv_cache, position):
                from intel_npu_acceleration.functional import update_kv_cache

                # Slicing individual caches for Layer 0 (K at 0, V at 1) and Layer 1 (K at 2, V at 3)
                ck0 = kv_cache[0]
                cv0 = kv_cache[1]
                ck1 = kv_cache[2]
                cv1 = kv_cache[3]

                new_ck0 = update_kv_cache(ck0, new_kv, position)
                new_cv0 = update_kv_cache(cv0, new_kv, position)
                new_ck1 = update_kv_cache(ck1, new_kv, position)
                new_cv1 = update_kv_cache(cv1, new_kv, position)

                # Return stacked updated caches
                return torch.stack([new_ck0, new_cv0, new_ck1, new_cv1], dim=0)

        # Pre-allocate inputs for tracing
        new_kv_ex = torch.randn(batch_size, 1, num_heads, head_dim)
        cache_ex = torch.zeros(n_layers * 2, batch_size, max_seq_len, num_heads, head_dim)
        pos_ex = torch.tensor(0, dtype=torch.int64)

        model_cpu = MultiLayerFunctionalLLM()
        model_cpu.eval()

        try:
            model_npu = compile_to_npu(model_cpu, (new_kv_ex, cache_ex, pos_ex), stateful=True)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Compilation of multi-layer auto-stateful functional LLaMA failed: {e}")

        # The auto-stateful pipeline skips compiling the top-level parent cache placeholder as a model Parameter.
        # Ensure model runs cleanly sequential step-by-step
        model_npu.reset_states()

        np.random.seed(456)
        step_tensors = []
        cache_cpu = cache_ex.clone()
        for step in range(3):
            step_np = np.random.randn(batch_size, 1, num_heads, head_dim).astype(np.float32)
            step_tensor = torch.from_numpy(step_np)
            step_tensors.append(step_tensor)

            # CPU Eager Step
            out_cpu = model_cpu(step_tensor, cache_cpu, step)
            cache_cpu = out_cpu

            # NPU Stateful Step (passes dummy/ignored cache parameter)
            out_npu = model_npu(step_tensor, cache_ex, step)

            # Assert parity on all sliced states
            self.assertTrue(
                torch.allclose(out_npu[:, :, :step+1], out_cpu[:, :, :step+1], atol=1e-3, rtol=1e-3),
                f"Mismatch at step {step} of multi-layer auto-stateful generation."
            )

        # Verify that reset_states() clears all NPU variable registers cleanly
        model_npu.reset_states()
        out_after_reset = model_npu(step_tensors[0], cache_ex, 0)
        out_cpu_after_reset = model_cpu(step_tensors[0], cache_ex, 0)
        self.assertTrue(torch.allclose(out_after_reset[:, :, :1], out_cpu_after_reset[:, :, :1], atol=1e-3, rtol=1e-3))
        self.assertEqual(torch.sum(out_after_reset[:, :, 1:]), 0.0)


if __name__ == "__main__":
    unittest.main()

