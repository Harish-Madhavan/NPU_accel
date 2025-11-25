
import torch
import torch.nn as nn
from intel_npu_acceleration.compiler import compile_to_npu
import time

class SimpleKVUpdate(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, current_key, current_value, past_key, past_value):
        # Simulate simple KV cache update
        # past_key/value has shape (batch, num_heads, seq_len, head_dim)
        # current_key/value has shape (batch, num_heads, 1, head_dim)
        
        # Concat along sequence dimension (dim=2)
        new_key = torch.cat([past_key, current_key], dim=2)
        new_value = torch.cat([past_value, current_value], dim=2)
        
        # Simulate some operation (e.g., stacking for self-attention output processing)
        # This might not be realistic for real attention, but tests `stack` op
        stacked_output = torch.stack([new_key, new_value], dim=-1) # (B, H, S, D, 2)

        return new_key, new_value, stacked_output

def test_kv_cache_ops():
    batch = 1
    num_heads = 2
    seq_len = 5
    head_dim = 4

    # Initial past state (seq_len = 5)
    past_key = torch.randn(batch, num_heads, seq_len, head_dim)
    past_value = torch.randn(batch, num_heads, seq_len, head_dim)

    # Current token (seq_len = 1)
    current_key = torch.randn(batch, num_heads, 1, head_dim)
    current_value = torch.randn(batch, num_heads, 1, head_dim)

    model = SimpleKVUpdate(head_dim)
    model.eval()

    print("Compiling KV Cache Update Model...")
    try:
        # Example inputs need to match forward signature
        npu_model = compile_to_npu(model, (current_key, current_value, past_key, past_value))
        print("Compilation successful.")
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running NPU inference...")
    start = time.time()
    npu_new_key, npu_new_value, npu_stacked_output = npu_model(current_key, current_value, past_key, past_value)
    print(f"NPU Time: {time.time() - start:.4f}s")
    
    # Run CPU
    cpu_new_key, cpu_new_value, cpu_stacked_output = model(current_key, current_value, past_key, past_value)
    
    key_match = torch.allclose(npu_new_key, cpu_new_key, atol=1e-3, rtol=1e-3)
    value_match = torch.allclose(npu_new_value, cpu_new_value, atol=1e-3, rtol=1e-3)
    stacked_match = torch.allclose(npu_stacked_output, cpu_stacked_output, atol=1e-3, rtol=1e-3)

    if key_match and value_match and stacked_match:
        print("Verification SUCCESS!")
    else:
        print("Verification FAILED!")
        if not key_match:
            print(f"Key Max Diff: {(npu_new_key - cpu_new_key).abs().max().item()}")
        if not value_match:
            print(f"Value Max Diff: {(npu_new_value - cpu_new_value).abs().max().item()}")
        if not stacked_match:
            print(f"Stacked Max Diff: {(npu_stacked_output - cpu_stacked_output).abs().max().item()}")

if __name__ == "__main__":
    test_kv_cache_ops()
