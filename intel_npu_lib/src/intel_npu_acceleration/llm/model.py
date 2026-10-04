"""
Hardware-Accelerated Transformer Architecture for Intel NPU.
Provides fused QKV, fused MLP, and hardware-offloaded state registers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from intel_npu_acceleration.nn import NPUStatefulKVCache


@dataclass
class NPUTransformerConfig:
    dim: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    vocab_size: int = 49152
    hidden_dim: int = 1536
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    rope_theta: float = 100000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 100000.0) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class FusedStatefulAttention(nn.Module):
    def __init__(self, args: NPUTransformerConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Fused QKV Projection: single GEMM replaces 3 separate linear passes per layer
        self.wqkv = nn.Linear(args.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Hardware-offloaded state registers residing directly in NPU SRAM / USM
        self.cache_k = NPUStatefulKVCache(1, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_v = NPUStatefulKVCache(1, args.max_seq_len, self.n_kv_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)
        xq, xk, xv = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        # Update cache in hardware without host round-trip memory copies
        keys = self.cache_k(xk)
        values = self.cache_v(xv)

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = torch.nn.functional.scaled_dot_product_attention(
            xq, keys, values, attn_mask=mask, is_causal=False
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FusedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Fused Gate-Up Projection: single GEMM replaces 2 separate linear passes
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = torch.split(gate_up, [self.hidden_dim, self.hidden_dim], dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class StatefulTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: NPUTransformerConfig):
        super().__init__()
        self.attention = FusedStatefulAttention(args)
        self.mlp = FusedMLP(args.dim, args.hidden_dim)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        att_out = self.attention(self.input_layernorm(x), freqs_cos, freqs_sin, mask)
        h = x + att_out
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class StatefulCausalLM(nn.Module):
    """Zero-transfer autoregressive causal language model for Intel NPU."""

    def __init__(self, config: NPUTransformerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([StatefulTransformerBlock(i, config) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed_tokens(tokens)
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask)
        h = self.norm(h)
        return self.lm_head(h)

    def reset_hardware_cache(self) -> None:
        """Reset internal KV-cache state registers."""
        for layer in self.layers:
            layer.attention.cache_k.reset()
            layer.attention.cache_v.reset()

    def load_huggingface_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Map and fuse weights from a standard HuggingFace Llama/SmolLM state dictionary."""
        new_sd = {
            "embed_tokens.weight": state_dict["model.embed_tokens.weight"].float(),
            "lm_head.weight": state_dict["lm_head.weight"].float(),
            "norm.weight": state_dict["model.norm.weight"].float(),
        }
        for i in range(self.config.n_layers):
            wq = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"].float()
            wk = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"].float()
            wv = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"].float()
            wqkv = torch.cat([wq, wk, wv], dim=0)

            gate = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"].float()
            up = state_dict[f"model.layers.{i}.mlp.up_proj.weight"].float()
            gate_up = torch.cat([gate, up], dim=0)

            new_sd[f"layers.{i}.attention.wqkv.weight"] = wqkv
            new_sd[f"layers.{i}.attention.wo.weight"] = state_dict[f"model.layers.{i}.self_attn.o_proj.weight"].float()
            new_sd[f"layers.{i}.mlp.gate_up_proj.weight"] = gate_up
            new_sd[f"layers.{i}.mlp.down_proj.weight"] = state_dict[f"model.layers.{i}.mlp.down_proj.weight"].float()
            new_sd[f"layers.{i}.input_layernorm.weight"] = state_dict[f"model.layers.{i}.input_layernorm.weight"].float()
            new_sd[f"layers.{i}.post_attention_layernorm.weight"] = state_dict[f"model.layers.{i}.post_attention_layernorm.weight"].float()
        self.load_state_dict(new_sd, strict=False)
