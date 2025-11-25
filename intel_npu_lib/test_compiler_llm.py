
import torch
import torch.nn as nn
from intel_npu_acceleration.compiler import compile_to_npu
import intel_npu_acceleration as npu_lib
import time

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm1 = npu_lib.NPURMSNorm(nn.modules.normalization.RMSNorm(dim))
        # Note: Using standard linear, but our compiler handles it
        self.fc1 = nn.Linear(dim, hidden_dim) 
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual # Add supported

def test_compiler_llm_block():
    dim = 32
    hidden_dim = 64
    model = TransformerBlock(dim, hidden_dim)
    model.eval()
    
    x = torch.randn(1, 10, dim)
    
    print("Compiling Transformer Block...")
    try:
        npu_model = compile_to_npu(model, x)
        print("Compilation successful.")
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running NPU inference...")
    start = time.time()
    out_npu = npu_model(x)
    print(f"NPU Time: {time.time() - start:.4f}s")
    
    out_cpu = model(x)
    
    if torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2):
        print("Verification SUCCESS!")
    else:
        print("Verification FAILED!")
        print(f"Max Diff: {(out_npu - out_cpu).abs().max().item()}")

if __name__ == "__main__":
    test_compiler_llm_block()
