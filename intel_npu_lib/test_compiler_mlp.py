
import torch
import torch.nn as nn
from intel_npu_acceleration.compiler import compile_to_npu
import time

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def test_compiler():
    model = SimpleMLP()
    model.eval()
    
    x = torch.randn(1, 16)
    
    # Compile
    print("Compiling...")
    try:
        npu_model = compile_to_npu(model, x)
        print("Compilation successful.")
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run
    print("Running NPU inference...")
    start = time.time()
    out_npu = npu_model(x)
    print(f"NPU Time: {time.time() - start:.4f}s")
    
    # Run CPU
    out_cpu = model(x)
    
    if torch.allclose(out_npu, out_cpu, atol=1e-3, rtol=1e-3):
        print("Verification SUCCESS!")
    else:
        print("Verification FAILED!")
        print(f"NPU: {out_npu[0, :5]}")
        print(f"CPU: {out_cpu[0, :5]}")

if __name__ == "__main__":
    test_compiler()
