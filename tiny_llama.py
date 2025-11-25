
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
        
        self.cache_k = torch.zeros((1, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((1, args.max_seq_len, self.n_local_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply RoPE
        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        # Cache update - DISABLED for compilation test (FX doesn't like in-place update on proxies)
        # In a real NPU deployment, we'd pass past_k/v as inputs and return new_k/v
        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)
        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        # For this test, we assume we are processing the full sequence 'x'
        keys = xk
        values = xv

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
        return self.wo(output)

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

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cos, freqs_sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

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

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        self.freqs_cos = self.freqs_cos.to(h.device)
        self.freqs_sin = self.freqs_sin.to(h.device)
        
        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cos, freqs_sin, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

def test_llama_impl():
    conf = LlamaConfig()
    model = Llama(conf)
    model.eval()
    
    # Test Input
    x = torch.randint(0, conf.vocab_size, (1, 10))
    with torch.no_grad():
        logits = model(x, 0)
    print(f"Logits shape: {logits.shape}") # Should be (1, 10, vocab_size)

if __name__ == "__main__":
    test_llama_impl()
