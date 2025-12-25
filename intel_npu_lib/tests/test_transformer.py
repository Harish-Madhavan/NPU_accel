import unittest
import torch
import torch.nn as nn
from intel_npu_acceleration.frontend import compile_to_npu # Use frontend for compile_to_npu

# Define a standard PyTorch RMSNorm module (if not available in nn)
# Assuming RMSNorm exists as a standard PyTorch module or we'll define a simple one
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Use standard PyTorch modules
        self.norm1 = RMSNorm(dim) 
        self.fc1 = nn.Linear(dim, hidden_dim) 
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual

class TestTransformer(unittest.TestCase):
    def test_block(self):
        dim = 32
        hidden_dim = 64
        model = TransformerBlock(dim, hidden_dim)
        model.eval()
        
        x = torch.randn(1, 10, dim) # Example input

        try:
            # Compile the standard PyTorch model for NPU
            npu_model = compile_to_npu(model, x)
        except Exception as e:
            self.fail(f"Compilation failed: {e}")
            
        out_npu = npu_model(x)
        out_cpu = model(x)
        
        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

if __name__ == "__main__":
    unittest.main()