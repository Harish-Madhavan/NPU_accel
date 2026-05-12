import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
import math
import intel_npu_acceleration.functional as n_f


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

    def forward(self, x):
        return n_f.rmsnorm(x, self.weight, self.eps)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # Return cos and sin
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (bsz, seqlen, n_heads, head_dim)
    # cos, sin: (seqlen, head_dim/2) -> need broadcasting

    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply RoPE
        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        # Functional Cache Update
        if cache_k is not None and cache_v is not None:
            cache_k = n_f.update_kv_cache(cache_k, xk, start_pos)
            cache_v = n_f.update_kv_cache(cache_v, xv, start_pos)

            keys = cache_k[:, : start_pos + seqlen]
            values = cache_v[:, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # Repeat KV heads to match Q heads (GQA)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention: (B, H, S, D)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Use accelerated SDPA
        output = n_f.scaled_dot_product_attention(
            xq, keys, values, attn_mask=mask, is_causal=False if mask is not None else (seqlen > 1)
        )
        
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
        # silu(w1(x)) * w3(x) is the SwiGLU pattern
        return self.w2(n_f.silu(self.w1(x)) * self.w3(x))


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

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k=None,
        cache_v=None,
    ):
        att_out, new_k, new_v = self.attention(
            self.attention_norm(x),
            start_pos,
            freqs_cos,
            freqs_sin,
            mask,
            cache_k,
            cache_v,
        )
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

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        kv_cache: Optional[torch.Tensor] = None,
    ):
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


def quantize_to_int8(model: nn.Module):
    """
    Simulate weight-only INT8 quantization by casting linear weights to int8.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Scale to [-128, 127]
                w = module.weight.data
                scale = w.abs().max() / 127.0
                quantized_w = (w / scale).round().clamp(-128, 127).to(torch.int8)
                
                # Replace weight with a non-gradient integer tensor
                # We use .data to bypass parameter checks
                module.weight.requires_grad = False
                module.weight.data = quantized_w
                
                # Store scale for later
                setattr(module, "weight_scale", scale)
    return model


def test_llama_impl():
    conf = LlamaConfig(dim=256, n_layers=4, n_heads=8)
    model = Llama(conf)
    model.eval()

    # Test Input
    x = torch.randint(0, conf.vocab_size, (1, 10))
    start_pos = 0

    # Initialize Cache (N_Layers * 2, B, MaxSeqLen, H, D)
    kv_cache = torch.zeros(
        conf.n_layers * 2,
        1,
        conf.max_seq_len,
        conf.n_kv_heads,
        conf.dim // conf.n_heads,
    )

    print("Running on CPU...")
    with torch.no_grad():
        logits_cpu, cache_cpu = model(x, start_pos, kv_cache)
    print(f"CPU Logits shape: {logits_cpu.shape}")
    print(f"CPU Cache shape: {cache_cpu.shape}")

    # Compile to NPU
    try:
        import intel_npu_acceleration as npu_compiler
        import time

        print("\nCompiling to NPU (FP16)...")
        t0 = time.time()
        npu_model_fp16 = npu_compiler.compile_to_npu(model, (x, start_pos, kv_cache))
        print(f"FP16 Compilation finished in {time.time() - t0:.2f}s")

        print("Running FP16 on NPU...")
        # Warmup
        for _ in range(5):
            _ = npu_model_fp16(x, start_pos, kv_cache)
        
        t0 = time.time()
        for _ in range(20):
            logits_npu, cache_npu = npu_model_fp16(x, start_pos, kv_cache)
        print(f"NPU FP16 Avg Inference time: {(time.time() - t0) / 20 * 1000:.2f} ms")

        # INT8 Test
        print("\nQuantizing model to INT8...")
        model_int8 = quantize_to_int8(model)
        
        print("Compiling to NPU (INT8)...")
        t0 = time.time()
        npu_model_int8 = npu_compiler.compile_to_npu(model_int8, (x, start_pos, kv_cache))
        print(f"INT8 Compilation finished in {time.time() - t0:.2f}s")

        print("Running INT8 on NPU...")
        # Warmup
        for _ in range(5):
            _ = npu_model_int8(x, start_pos, kv_cache)

        t0 = time.time()
        for _ in range(20):
            logits_int8, cache_int8 = npu_model_int8(x, start_pos, kv_cache)
        print(f"NPU INT8 Avg Inference time: {(time.time() - t0) / 20 * 1000:.2f} ms")

        if logits_cpu.shape == logits_npu.shape:
            print("\nLogits Shape check PASSED")
        else:
            print(f"\nLogits Shape check FAILED")

    except ImportError:
        print("Intel NPU library not found. Skipping NPU test.")
    except Exception as e:
        print(f"NPU Test Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_llama_impl()
