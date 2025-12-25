
import unittest
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
import math
from intel_npu_acceleration.compiler import compile_to_npu
from intel_npu_acceleration.functional import update_kv_cache

@dataclass
class LlamaConfig:
    dim: int = 32
    n_layers: int = 1
    n_heads: int = 2
    n_kv_heads: int = 2
    vocab_size: int = 64
    multiple_of: int = 4
    norm_eps: float = 1e-5
    max_seq_len: int = 32
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1).type_as(x)

class Attention(nn.Module):
    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

    def forward(self, x, start_pos, freqs_cos, freqs_sin, mask, cache_k=None, cache_v=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        keys = xk
        values = xv
        
        if cache_k is not None and cache_v is not None:
            cache_k = update_kv_cache(cache_k, xk, start_pos, seqlen)
            cache_v = update_kv_cache(cache_v, xv, start_pos, seqlen)
            keys = cache_k[:, : start_pos + seqlen]
            values = cache_v[:, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), cache_k, cache_v

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class Llama(nn.Module):
    def __init__(self, params: LlamaConfig):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList([
            nn.ModuleDict({
                "attention": Attention(params),
                "feed_forward": FeedForward(params.dim, 4 * params.dim, params.multiple_of),
                "attention_norm": RMSNorm(params.dim, eps=params.norm_eps),
                "ffn_norm": RMSNorm(params.dim, eps=params.norm_eps)
            }) for _ in range(params.n_layers)
        ])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2)

    def forward(self, tokens, start_pos, kv_cache=None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        freqs_cos = self.freqs_cos.to(h.device)
        freqs_sin = self.freqs_sin.to(h.device)
        idx = torch.arange(start_pos, start_pos + seqlen, device=h.device)
        freqs_cos = freqs_cos[idx]
        freqs_sin = freqs_sin[idx]

        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        new_kvs_flat = []
        
        for i, layer in enumerate(self.layers):
            att = layer["attention"]
            ffn = layer["feed_forward"]
            norm1 = layer["attention_norm"]
            norm2 = layer["ffn_norm"]
            
            ck, cv = None, None
            if kv_cache is not None:
                 ck = kv_cache[2 * i]
                 cv = kv_cache[2 * i + 1]

            att_out, nk, nv = att(norm1(h), start_pos, freqs_cos, freqs_sin, mask, ck, cv)
            h = h + att_out
            h = h + ffn(norm2(h))
            
            if nk is not None and nv is not None:
                new_kvs_flat.extend([nk, nv])

        h = self.norm(h)
        output = self.output(h).float()
        
        new_kv_cache = None
        if len(new_kvs_flat) > 0:
            new_kv_cache = torch.stack(new_kvs_flat, dim=0)
            
        return output, new_kv_cache

class TestLlamaModel(unittest.TestCase):
    def test_llama_caching(self):
        conf = LlamaConfig()
        model = Llama(conf)
        model.eval()
        
        x = torch.randint(0, conf.vocab_size, (1, 10))
        start_pos = 0
        kv_cache = torch.zeros(conf.n_layers * 2, 1, conf.max_seq_len, conf.n_kv_heads, conf.dim // conf.n_heads)
        
        try:
            npu_model = compile_to_npu(model, (x, start_pos, kv_cache))
        except Exception as e:
            self.fail(f"Compilation failed: {e}")
            
        out_npu, cache_npu = npu_model(x, start_pos, kv_cache)
        out_cpu, cache_cpu = model(x, start_pos, kv_cache)
        
        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(cache_npu, cache_cpu, atol=1e-2, rtol=1e-2))

if __name__ == "__main__":
    unittest.main()
