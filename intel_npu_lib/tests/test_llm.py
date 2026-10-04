"""
Unit test suite for intel_npu_acceleration.llm.
Validates transformer architecture components, zero-transfer hardware cache,
fused QKV/MLP projections, and the high-level NPUCausalLM inference engine.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from intel_npu_acceleration.llm import (
    FusedMLP,
    FusedStatefulAttention,
    NPUCausalLM,
    NPUTransformerConfig,
    RMSNorm,
    StatefulCausalLM,
    StatefulTransformerBlock,
)
from intel_npu_acceleration.llm.model import (
    apply_rotary_emb,
    precompute_freqs_cis,
    repeat_kv,
)


class DummyTokenizer:
    """Mock tokenizer for deterministic LLM engine unit tests."""

    def __init__(self):
        self.eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[10, 20, 30]], dtype=torch.long)}

    def decode(self, token_ids: list[int]) -> str:
        return f" tok_{token_ids[0]}"


def test_npu_transformer_config():
    """Verify default and custom configuration properties."""
    cfg = NPUTransformerConfig()
    assert cfg.dim == 576
    assert cfg.n_layers == 30
    assert cfg.n_heads == 9
    assert cfg.n_kv_heads == 3
    assert cfg.vocab_size == 49152
    assert cfg.hidden_dim == 1536

    custom = NPUTransformerConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=100)
    assert custom.dim == 64
    assert custom.n_layers == 2
    assert custom.n_heads == 4
    assert custom.n_kv_heads == 2


def test_rmsnorm():
    """Verify RMSNorm normalization and learnable weights."""
    norm = RMSNorm(dim=64, eps=1e-5)
    x = torch.randn(2, 5, 64)
    out = norm(x)
    assert out.shape == (2, 5, 64)
    # Variance should be normalized approximately to 1
    var = out.pow(2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_rope_and_freqs():
    """Verify RoPE table precomputation and rotary embedding application."""
    head_dim = 16
    cos, sin = precompute_freqs_cis(head_dim, end=32)
    assert cos.shape == (32, head_dim // 2)
    assert sin.shape == (32, head_dim // 2)

    x = torch.randn(1, 4, 2, head_dim)  # [bs, seqlen, n_heads, head_dim]
    rope_out = apply_rotary_emb(x, cos[:4], sin[:4])
    assert rope_out.shape == x.shape


def test_repeat_kv():
    """Verify KV head repetition for grouped-query attention."""
    # Identity when n_rep == 1
    x = torch.randn(1, 4, 3, 16)
    out1 = repeat_kv(x, n_rep=1)
    assert torch.equal(x, out1)

    # 3 heads repeated 3 times -> 9 heads
    out3 = repeat_kv(x, n_rep=3)
    assert out3.shape == (1, 4, 9, 16)


def test_fused_stateful_attention():
    """Verify FusedStatefulAttention projection and zero-transfer cache state."""
    cfg = NPUTransformerConfig(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        hidden_dim=128,
        max_seq_len=16,
    )
    attn = FusedStatefulAttention(cfg)
    cos, sin = precompute_freqs_cis(cfg.dim // cfg.n_heads, end=16)

    # Simulate sequential token generation
    mask = torch.full((1, 1, 1, 16), float("-inf"))
    mask[0, 0, 0, 0] = 0.0

    x = torch.randn(1, 1, 64)
    out1 = attn(x, cos[:1], sin[:1], mask)
    assert out1.shape == (1, 1, 64)

    # Step 2: next token
    mask[0, 0, 0, 1] = 0.0
    out2 = attn(x, cos[1:2], sin[1:2], mask)
    assert out2.shape == (1, 1, 64)

    # Reset cache
    attn.cache_k.reset()
    attn.cache_v.reset()
    assert attn.cache_k.current_pos == 0
    assert attn.cache_v.current_pos == 0


def test_fused_mlp():
    """Verify FusedMLP combined gate-up projection and down projection."""
    mlp = FusedMLP(dim=64, hidden_dim=128)
    x = torch.randn(2, 4, 64)
    out = mlp(x)
    assert out.shape == (2, 4, 64)


def test_stateful_transformer_block():
    """Verify StatefulTransformerBlock full layer forward execution."""
    cfg = NPUTransformerConfig(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        hidden_dim=128,
        max_seq_len=16,
    )
    block = StatefulTransformerBlock(layer_id=0, args=cfg)
    cos, sin = precompute_freqs_cis(cfg.dim // cfg.n_heads, end=16)

    mask = torch.zeros(1, 1, 1, 16)
    x = torch.randn(1, 1, 64)
    out = block(x, cos[:1], sin[:1], mask)
    assert out.shape == (1, 1, 64)


def test_stateful_causal_lm():
    """Verify full StatefulCausalLM model initialization, forward pass, and state reset."""
    cfg = NPUTransformerConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        hidden_dim=128,
        max_seq_len=16,
    )
    model = StatefulCausalLM(cfg).eval()
    cos, sin = precompute_freqs_cis(cfg.dim // cfg.n_heads, end=16)

    tokens = torch.tensor([[15]], dtype=torch.long)
    mask = torch.zeros(1, 1, 1, 16)

    logits = model(tokens, cos[:1], sin[:1], mask)
    assert logits.shape == (1, 1, 100)

    # Check reset hardware cache runs without error
    model.reset_hardware_cache()
    for layer in model.layers:
        assert layer.attention.cache_k.current_pos == 0
        assert layer.attention.cache_v.current_pos == 0


def test_load_huggingface_state_dict():
    """Verify mapping and fusing HuggingFace-style weights into StatefulCausalLM."""
    cfg = NPUTransformerConfig(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        hidden_dim=128,
        max_seq_len=16,
    )
    model = StatefulCausalLM(cfg)

    # Build dummy HuggingFace state dict
    head_dim = cfg.dim // cfg.n_heads
    q_dim = cfg.n_heads * head_dim
    kv_dim = cfg.n_kv_heads * head_dim

    hf_sd = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.dim),
        "lm_head.weight": torch.randn(cfg.vocab_size, cfg.dim),
        "model.norm.weight": torch.ones(cfg.dim),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(q_dim, cfg.dim),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(kv_dim, cfg.dim),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(kv_dim, cfg.dim),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(cfg.dim, q_dim),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(cfg.hidden_dim, cfg.dim),
        "model.layers.0.mlp.up_proj.weight": torch.randn(cfg.hidden_dim, cfg.dim),
        "model.layers.0.mlp.down_proj.weight": torch.randn(cfg.dim, cfg.hidden_dim),
        "model.layers.0.input_layernorm.weight": torch.ones(cfg.dim),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(cfg.dim),
    }

    model.load_huggingface_state_dict(hf_sd)

    # Check that weights were properly fused into single GEMMs
    expected_wqkv = torch.cat([
        hf_sd["model.layers.0.self_attn.q_proj.weight"],
        hf_sd["model.layers.0.self_attn.k_proj.weight"],
        hf_sd["model.layers.0.self_attn.v_proj.weight"],
    ], dim=0)
    assert torch.equal(model.layers[0].attention.wqkv.weight, expected_wqkv)

    expected_gate_up = torch.cat([
        hf_sd["model.layers.0.mlp.gate_proj.weight"],
        hf_sd["model.layers.0.mlp.up_proj.weight"],
    ], dim=0)
    assert torch.equal(model.layers[0].mlp.gate_up_proj.weight, expected_gate_up)


class MockNPUModule(nn.Module):
    """Mock NPUGraphModule for fast unit-testing NPUCausalLM generate loop."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size

    def reset_states(self):
        pass

    def forward(self, input_t, cos, sin, mask):
        # Return deterministic logits pointing to token 42, then 0 (EOS)
        logits = torch.zeros(1, 1, self.vocab_size)
        logits[0, 0, 42] = 10.0
        return logits


def test_npu_causal_lm_generate():
    """Verify NPUCausalLM streaming and complete text generation APIs."""
    cfg = NPUTransformerConfig(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        hidden_dim=128,
        max_seq_len=32,
    )
    tokenizer = DummyTokenizer()
    mock_npu = MockNPUModule(vocab_size=100)

    engine = NPUCausalLM(
        npu_model=mock_npu,  # type: ignore
        raw_model=None,
        tokenizer=tokenizer,
        config=cfg,
        model_id="mock-slm",
    )

    # Test generate
    generated = engine.generate("Hello world", max_new_tokens=3)
    assert isinstance(generated, str)
    assert len(generated) > 0
    assert "tok_42" in generated

    # Verify metrics
    metrics = engine.last_metrics
    assert "tokens_generated" in metrics
    assert "ttft_ms" in metrics
    assert "avg_decode_ms" in metrics
    assert "throughput_tok_s" in metrics
    assert metrics["tokens_generated"] == 3
