import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
import math
import time
import intel_npu_acceleration.functional as n_f
import intel_npu_acceleration as npu_compiler


@dataclass
class LlamaConfig:
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4
    vocab_size: int = 128  # ASCII Range
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
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
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

        idx = torch.arange(start_pos, start_pos + seqlen, device=h.device)
        freqs_cos = freqs_cos[idx]
        freqs_sin = freqs_sin[idx]

        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        new_kvs_flat = []
        for i, layer in enumerate(self.layers):
            ck, cv = None, None
            if kv_cache is not None:
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
                w = module.weight.data
                scale = w.abs().max() / 127.0
                quantized_w = (w / scale).round().clamp(-128, 127).to(torch.int8)
                
                module.weight.requires_grad = False
                module.weight.data = quantized_w
                setattr(module, "weight_scale", scale)
    return model


def run_text_generation(npu_model, prompt: str, max_gen_len: int, conf: LlamaConfig):
    # Convert characters to ASCII integers
    tokens = [ord(c) for c in prompt if ord(c) < conf.vocab_size]
    if not tokens:
        tokens = [32]  # space
        
    print(f"  Prompt tokens: {tokens}")
    print(f"  Generated Text : '{prompt}", end="", flush=True)

    # Initialize Cache (N_Layers * 2, B, MaxSeqLen, H, D)
    kv_cache = torch.zeros(
        conf.n_layers * 2,
        1,
        conf.max_seq_len,
        conf.n_kv_heads,
        conf.dim // conf.n_heads,
    )

    t_start = time.time()
    
    # 1. Prefill Phase (fed token-by-token to maintain static shape [1, 1])
    # Feed prompt tokens one-by-one to populate KV cache
    for start_pos, tok_id in enumerate(tokens[:-1]):
        input_t = torch.tensor([[tok_id]], dtype=torch.long)
        _, kv_cache = npu_model(input_t, start_pos, kv_cache)
        
    # The last token of the prompt will produce the first generated token
    next_token = tokens[-1]
    start_pos = len(tokens) - 1
    
    # 2. Decode Phase
    latencies = []
    for step in range(max_gen_len):
        current_pos = start_pos + step
        if current_pos >= conf.max_seq_len:
            break
            
        input_t = torch.tensor([[next_token]], dtype=torch.long)
        
        t0 = time.time()
        logits, kv_cache = npu_model(input_t, current_pos, kv_cache)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

        # Greedy decoding: pick highest logit index
        next_token = torch.argmax(logits[0, -1]).item()
        
        # Safe ASCII decode
        char = chr(next_token) if 32 <= next_token <= 126 or next_token in (10, 13) else '.'
        print(char, end="", flush=True)
        time.sleep(0.015)  # Fast typing effect

    t_total = time.time() - t_start
    avg_step = sum(latencies) / len(latencies) if latencies else 0
    print(f"'\n  [Metrics] Generation finished in {t_total:.2f}s | Avg decode step: {avg_step:.2f} ms")
    return avg_step


def main():
    print("==========================================================")
    print("      Intel NPU Tiny LLaMA Autoregressive Text Gen        ")
    print("==========================================================")

    # Initialize a tiny model
    conf = LlamaConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=4)
    model = Llama(conf)
    model.eval()

    # Define a visual seed
    torch.manual_seed(42)

    prompt = "Intel NPU:"
    max_gen_len = 30

    # We compile the graph using a static shape (1, 1)
    compile_input_token = torch.tensor([[65]], dtype=torch.long)
    compile_start_pos = 0
    compile_cache = torch.zeros(
        conf.n_layers * 2,
        1,
        conf.max_seq_len,
        conf.n_kv_heads,
        conf.dim // conf.n_heads,
    )

    # Compile FP16 model
    print("\n[Step 1] Compiling Tiny LLaMA FP16 model for NPU...")
    t0 = time.time()
    npu_model_fp16 = npu_compiler.compile_to_npu(model, (compile_input_token, compile_start_pos, compile_cache))
    print(f"Compilation finished in {time.time() - t0:.2f}s.")

    print("\n[Step 2] Executing Autoregressive Text Generation (NPU FP16):")
    fp16_latency = run_text_generation(npu_model_fp16, prompt, max_gen_len, conf)

    # Compile INT8 quantized model
    print("\n[Step 3] Simulating Weight-Only INT8 Quantization...")
    model_int8 = quantize_to_int8(model)

    print("\n[Step 4] Compiling Tiny LLaMA INT8 model for NPU...")
    t0 = time.time()
    npu_model_int8 = npu_compiler.compile_to_npu(model_int8, (compile_input_token, compile_start_pos, compile_cache))
    print(f"Compilation finished in {time.time() - t0:.2f}s.")

    print("\n[Step 5] Executing Autoregressive Text Generation (NPU INT8):")
    int8_latency = run_text_generation(npu_model_int8, prompt, max_gen_len, conf)

    # Comparison summary
    print("\n" + "=" * 60)
    print("                LLaMA GENERATION LATENCY SUMMARY                ")
    print("=" * 60)
    print(f"| Precision Target         | Avg Decode Latency (per Token)  |")
    print("-" * 60)
    print(f"| NPU FP16                 | {fp16_latency:25.2f} ms |")
    print(f"| NPU INT8 (Weight-Only)   | {int8_latency:25.2f} ms |")
    print("=" * 60)
    speedup = fp16_latency / int8_latency if int8_latency > 0 else 0
    print(f"INT8 speedup factor: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
