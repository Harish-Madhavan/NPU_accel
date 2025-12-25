
import unittest
import torch
import torch.nn as nn
import math
from intel_npu_acceleration.frontend import compile_to_npu

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, hidden_size)

    def forward(self, x, mask=None):
        # x: (B, Seq, Dim)
        residual = x
        x = self.norm1(x)
        
        B, Seq, _ = x.shape
        q = self.q_proj(x).view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot Product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
            
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, Seq, self.hidden_size)
        out = self.out_proj(out)
        
        x = residual + out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.fc2(self.act(self.fc1(x)))
        x = residual + x
        
        return x

class TestEncoderWorkload(unittest.TestCase):
    def test_encoder_block(self):
        hidden_size = 64
        num_heads = 4
        ffn_dim = 256
        model = EncoderLayer(hidden_size, num_heads, ffn_dim)
        model.eval()
        
        x = torch.randn(1, 16, hidden_size)
        mask = torch.zeros(1, 1, 16, 16)
        
        try:
            npu_model = compile_to_npu(model, (x, mask))
        except Exception as e:
            self.fail(f"Compilation failed: {e}")
            
        out_npu = npu_model(x, mask)
        out_cpu = model(x, mask)
        
        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-3, rtol=1e-3))

if __name__ == "__main__":
    unittest.main()
