
import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class LlamaConfig:
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4
    vocab_size: int = 128
    multiple_of: int = 4
    norm_eps: float = 1e-5
    max_seq_len: int = 128
    rope_theta: float = 10000.0

from intel_npu_acceleration.functional import update_kv_cache

class RMSNorm(torch.nn.Module):
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
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # Return cos and sin
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (bsz, seqlen, n_heads, head_dim)
    # cos, sin: (seqlen, head_dim/2) -> need broadcasting
    
    # Reshape x into pairs (..., head_dim/2, 2)
    # Actually, standard RoPE implementation:
    # x = [x1, x2, x3, x4]
    # rotate_half(x) = [-x2, x1, -x4, x3]
    # We need to match the precomputed freqs which correspond to pairs.
    
    # Assuming cos/sin are shaped broadcastable to x.
    # x shape: (B, S, H, D)
    # cos shape: (S, D/2) - we need to duplicate to (S, D) or apply to halves?
    # The precomputed freqs are for pairs. 
    
    # Let's use the "rotate_half" helper
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    
    # Prepare cos/sin
    # cos/sin are (S, D/2). Unsqueeze for B and H.
    # (S, D/2) -> (1, S, 1, D/2)
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    
    # Apply
    # out1 = x1 * cos - x2 * sin
    # out2 = x1 * sin + x2 * cos
    # This corresponds to rotation matrix [[cos, -sin], [sin, cos]]
    
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    return torch.cat([out1, out2], dim=-1).type_as(x)

class Attention(nn.Module):
    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: Optional[torch.Tensor], cache_k: Optional[torch.Tensor]=None, cache_v: Optional[torch.Tensor]=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply RoPE
        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        # Functional Cache Update
        keys = xk
        values = xv
        
        if cache_k is not None and cache_v is not None:
            cache_k = update_kv_cache(cache_k, xk, start_pos, seqlen)
            cache_v = update_kv_cache(cache_v, xv, start_pos, seqlen)
            
            keys = cache_k[:, : start_pos + seqlen]
            values = cache_v[:, : start_pos + seqlen]

        # Transpose for attention: (B, H, S, D)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bsz, n_local_heads, seqlen, cache_len + seqlen)
        scores = nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bsz, n_local_heads, seqlen, head_dim)
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

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LlamaConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: Optional[torch.Tensor], cache_k=None, cache_v=None):
        att_out, new_k, new_v = self.attention(self.attention_norm(x), start_pos, freqs_cos, freqs_sin, mask, cache_k, cache_v)
        h = x + att_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_k, new_v

class Llama(nn.Module):
    def __init__(self, params: LlamaConfig):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Return cos, sin
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int, kv_cache: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        freqs_cos = self.freqs_cos.to(h.device)
        freqs_sin = self.freqs_sin.to(h.device)
        
        # Use arange for slicing with dynamic shapes (FX proxies)
        idx = torch.arange(start_pos, start_pos + seqlen, device=h.device)
        freqs_cos = freqs_cos[idx]
        freqs_sin = freqs_sin[idx]

        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        new_kvs_flat = []
        for i, layer in enumerate(self.layers):
            ck, cv = None, None
            if kv_cache is not None:
                # kv_cache shape: (N_Layers * 2, B, MaxSeqLen, H, D)
                ck = kv_cache[2 * i]
                cv = kv_cache[2 * i + 1]
            
            h, nk, nv = layer(h, start_pos, freqs_cos, freqs_sin, mask, ck, cv)
            
            if nk is not None and nv is not None:
                new_kvs_flat.append(nk)
                new_kvs_flat.append(nv)

        h = self.norm(h)
        output = self.output(h).float()
        
        new_kv_cache = None
        if len(new_kvs_flat) > 0:
            new_kv_cache = torch.stack(new_kvs_flat, dim=0)
            
        return output, new_kv_cache

def test_llama_impl():
    conf = LlamaConfig()
    model = Llama(conf)
    model.eval()
    
    # Test Input
    x = torch.randint(0, conf.vocab_size, (1, 10))
    start_pos = 0
    
    # Initialize Cache (N_Layers * 2, B, MaxSeqLen, H, D)
    kv_cache = torch.zeros(conf.n_layers * 2, 1, conf.max_seq_len, conf.n_kv_heads, conf.dim // conf.n_heads)
    
    print("Running on CPU...")
    with torch.no_grad():
        logits_cpu, cache_cpu = model(x, start_pos, kv_cache)
    print(f"CPU Logits shape: {logits_cpu.shape}") 
    print(f"CPU Cache shape: {cache_cpu.shape}")

    # Compile to NPU
    try:
        import intel_npu_acceleration.compiler as npu_compiler
        import time
        
        print("\nCompiling to NPU...")
        t0 = time.time()
        # Input signature: (tokens, start_pos, kv_cache)
        npu_model = npu_compiler.compile_to_npu(model, (x, start_pos, kv_cache))
        print(f"Compilation finished in {time.time() - t0:.2f}s")
        
        print("Running on NPU...")
        t0 = time.time()
        logits_npu, cache_npu = npu_model(x, start_pos, kv_cache)
        print(f"NPU Inference time: {(time.time() - t0)*1000:.2f} ms")
        print(f"NPU Logits shape: {logits_npu.shape}")
        
        if logits_cpu.shape == logits_npu.shape:
            print("Logits Shape check PASSED")
        else:
            print(f"Logits Shape check FAILED: {logits_cpu.shape} vs {logits_npu.shape}")

        if cache_cpu.shape == cache_npu.shape:
            print("Cache Shape check PASSED")
        else:
             print(f"Cache Shape check FAILED: {cache_cpu.shape} vs {cache_npu.shape}")

    except ImportError:
        print("Intel NPU library not found. Skipping NPU test.")
    except Exception as e:
        print(f"NPU Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llama_impl()
