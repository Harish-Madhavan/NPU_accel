"""
High-Level Causal Language Model Inference Engine for Intel NPU.
Provides standard from_pretrained, generate, and stream_generate APIs.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

import torch

import intel_npu_acceleration as npu
from intel_npu_acceleration.llm.model import (
    NPUTransformerConfig,
    StatefulCausalLM,
    precompute_freqs_cis,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from intel_npu_acceleration.frontend.graph_module import NPUGraphModule

logger = logging.getLogger("intel_npu_acceleration.llm")


class NPUCausalLM:
    """High-performance Causal Language Model running on Intel Core Ultra NPU."""

    def __init__(
        self,
        npu_model: NPUGraphModule,
        raw_model: StatefulCausalLM | None,
        tokenizer: Any,
        config: NPUTransformerConfig,
        model_id: str = "custom",
    ):
        self.npu_model = npu_model
        self.raw_model = raw_model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id
        self.last_metrics: dict[str, float] = {}

        # Precompute static RoPE tables & slice views matching compiled graph precision
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads, config.max_seq_len * 2, config.rope_theta
        )
        input_dtype = torch.float16
        if hasattr(npu_model, "target_torch_dtypes") and len(npu_model.target_torch_dtypes) > 1:
            input_dtype = npu_model.target_torch_dtypes[1]

        self.freqs_cos = self.freqs_cos.to(input_dtype)
        self.freqs_sin = self.freqs_sin.to(input_dtype)
        self._cached_cos = [self.freqs_cos[i : i + 1] for i in range(config.max_seq_len * 2)]
        self._cached_sin = [self.freqs_sin[i : i + 1] for i in range(config.max_seq_len * 2)]

        # Pre-allocate static input buffers for zero-allocation token decode
        self._input_t = torch.zeros((1, 1), dtype=torch.long)
        mask_dtype = input_dtype
        if hasattr(npu_model, "target_torch_dtypes") and len(npu_model.target_torch_dtypes) > 3:
            mask_dtype = npu_model.target_torch_dtypes[3]
        self._mask = torch.full((1, 1, 1, config.max_seq_len), float("-inf"), dtype=mask_dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        max_seq_len: int = 256,
        precision: str = "fp16",
        quantization: str = "none",
        cache_dir: str | None = None,
        force_recompile: bool = False,
        **kwargs: Any,
    ) -> NPUCausalLM:
        """Load and instantiate a hardware-accelerated LLM for Intel NPU."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        npu.enable_turbo()
        npu.set_performance_hint("LATENCY")

        # Resolve persistent binary cache location
        resolved_cache_dir = cache_dir or npu.get_cache_dir() or os.path.abspath("npu_cache")
        os.makedirs(resolved_cache_dir, exist_ok=True)

        clean_name = re.sub(r"[^\w\-_\.]", "_", os.path.basename(model_id_or_path.rstrip("/\\")))
        normalized_name = clean_name.lower().replace("-", "_")
        q_tag = f"_{quantization}" if quantization in ("int8", "int4", "int4_sym") else ""
        candidate_blobs = [
            os.path.join(resolved_cache_dir, f"{clean_name}{q_tag}_seq{max_seq_len}_gqa_l0.blob"),
            os.path.join(resolved_cache_dir, f"{normalized_name}{q_tag}_seq{max_seq_len}_gqa_l0.blob"),
        ]
        blob_path = candidate_blobs[0]
        for c in candidate_blobs:
            if os.path.exists(c):
                blob_path = c
                break

        # Load Tokenizer
        logger.info(f"Loading tokenizer for '{model_id_or_path}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

        # Inspect HuggingFace architecture configuration
        try:
            hf_config = AutoConfig.from_pretrained(model_id_or_path, local_files_only=True)
        except Exception:
            hf_config = AutoConfig.from_pretrained(model_id_or_path)

        cfg = NPUTransformerConfig(
            dim=getattr(hf_config, "hidden_size", 576),
            n_layers=getattr(hf_config, "num_hidden_layers", 30),
            n_heads=getattr(hf_config, "num_attention_heads", 9),
            n_kv_heads=getattr(hf_config, "num_key_value_heads", 3),
            vocab_size=getattr(hf_config, "vocab_size", 49152),
            hidden_dim=getattr(hf_config, "intermediate_size", 1536),
            norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            max_seq_len=max_seq_len,
            rope_theta=getattr(hf_config, "rope_theta", 100000.0),
        )

        # 1. Fast Path: Instant Hardware Binary Reload
        if os.path.exists(blob_path) and not force_recompile:
            logger.info(f"Loading pre-compiled Level Zero hardware executable from: {blob_path}")
            t0 = time.perf_counter()
            npu_model = npu.import_compiled_model(
                blob_path,
                performance_hint="LATENCY",
                num_streams=1,
                clone_outputs=False,
            )
            logger.info(f"NPU hardware binary loaded in {time.perf_counter() - t0:.2f}s!")
            return cls(npu_model=npu_model, raw_model=None, tokenizer=tokenizer, config=cfg, model_id=model_id_or_path)

        # 2. Compile Path: Load Weights and Compile
        logger.info(f"Compiling '{model_id_or_path}' into native NPU hardware binary...")
        raw_model = StatefulCausalLM(cfg).eval()

        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_id_or_path, dtype=torch.float32, local_files_only=True)
        except Exception:
            hf_model = AutoModelForCausalLM.from_pretrained(model_id_or_path, dtype=torch.float32)

        raw_model.load_huggingface_state_dict(hf_model.state_dict())
        del hf_model
        gc.collect()

        if quantization in ("int8", "int4", "int4_sym"):
            logger.info(f"Applying {quantization.upper()} weight quantization for maximum NPU hardware efficiency...")
            if quantization == "int4_sym":
                # Keep sensitive vocab projection in INT8 while compressing transformer body to INT4
                orig_head_w = raw_model.lm_head.weight.data.clone()
                npu.nn.quantize(raw_model, mode="int4_sym")
                raw_model.lm_head.weight.data = orig_head_w
                npu.nn.quantize(raw_model.lm_head, mode="int8")
            else:
                npu.nn.quantize(raw_model, mode=quantization)

        compile_input_token = torch.tensor([[65]], dtype=torch.long)
        compile_cos = torch.zeros(1, cfg.dim // (2 * cfg.n_heads))
        compile_sin = torch.zeros(1, cfg.dim // (2 * cfg.n_heads))
        compile_mask = torch.zeros(1, 1, 1, cfg.max_seq_len)

        t_compile = time.perf_counter()
        npu_model = npu.compile_to_npu(
            raw_model,
            (compile_input_token, compile_cos, compile_sin, compile_mask),
            strict=True,
            precision=precision,
            performance_hint="LATENCY",
            clone_outputs=False,
        )
        logger.info(f"NPU compilation completed in {time.perf_counter() - t_compile:.2f}s!")

        # Export binary blob for instant future loads
        try:
            npu.export_compiled_model(npu_model, blob_path)
            logger.info(f"Saved compiled hardware binary to '{blob_path}'")
        except Exception as e:
            logger.debug(f"Failed to export binary blob: {e}")

        return cls(npu_model=npu_model, raw_model=raw_model, tokenizer=tokenizer, config=cfg, model_id=model_id_or_path)

    def reset_states(self) -> None:
        """Reset NPU hardware state registers and KV cache for a new sequence."""
        self.npu_model.reset_states()
        if self.raw_model is not None:
            self.raw_model.reset_hardware_cache()

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        repetition_penalty: float = 1.1,
    ) -> Iterator[str]:
        """Generate text in streaming mode, yielding decoded token strings in real-time."""
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        self.reset_states()

        input_t = self._input_t
        mask = self._mask
        mask.fill_(float("-inf"))

        cached_cos = self._cached_cos
        cached_sin = self._cached_sin
        config = self.config
        seen_tokens = set(tokens)

        t_start = time.perf_counter()
        decode_latencies: list[float] = []
        generated_tokens = list(tokens)

        infer_fn = getattr(self.npu_model, "infer_direct", self.npu_model)

        with torch.inference_mode():
            # Prefill Phase
            for start_pos, tok_id in enumerate(tokens[:-1]):
                input_t[0, 0] = tok_id
                cos = cached_cos[start_pos]
                sin = cached_sin[start_pos]
                mask[0, 0, 0, start_pos] = 0.0
                infer_fn(input_t, cos, sin, mask)

            t_prefill_done = time.perf_counter()
            ttft_ms = (t_prefill_done - t_start) * 1000.0

            # Decode Phase
            next_token = tokens[-1]
            start_pos = len(tokens) - 1
            use_pipelined = (
                hasattr(self.npu_model, "infer_direct_async")
                and hasattr(self.npu_model, "wait_direct")
            )

            if use_pipelined:
                # Prime the NPU pipeline with the first decode step
                curr_pos = start_pos
                input_t[0, 0] = next_token
                cos = cached_cos[curr_pos]
                sin = cached_sin[curr_pos]
                mask[0, 0, 0, curr_pos] = 0.0
                t0 = time.perf_counter()
                self.npu_model.infer_direct_async(input_t, cos, sin, mask)

                for step in range(max_new_tokens):
                    logits = self.npu_model.wait_direct()
                    decode_latencies.append((time.perf_counter() - t0) * 1000.0)

                    # Vectorized sampling
                    if temperature == 0.0 and repetition_penalty == 1.0:
                        chosen_token = int(torch.argmax(logits))
                    else:
                        flat_logits = logits.view(-1)
                        if repetition_penalty > 1.0 and seen_tokens:
                            past_list = list(seen_tokens)
                            orig_vals = flat_logits[past_list].clone()
                            flat_logits[past_list] = torch.where(
                                orig_vals < 0,
                                orig_vals * repetition_penalty,
                                orig_vals / repetition_penalty,
                            )
                            if temperature > 0.0:
                                probs = torch.softmax(flat_logits / temperature, dim=-1)
                                chosen_token = int(torch.multinomial(probs, num_samples=1))
                            else:
                                chosen_token = int(torch.argmax(flat_logits))
                            flat_logits[past_list] = orig_vals
                        else:
                            if temperature > 0.0:
                                probs = torch.softmax(flat_logits / temperature, dim=-1)
                                chosen_token = int(torch.multinomial(probs, num_samples=1))
                            else:
                                chosen_token = int(torch.argmax(flat_logits))

                    seen_tokens.add(chosen_token)
                    generated_tokens.append(chosen_token)

                    # Dispatch next token inference asynchronously into NPU hardware pipeline
                    next_step = step + 1
                    next_pos = start_pos + next_step
                    is_done = (
                        next_step >= max_new_tokens
                        or next_pos >= config.max_seq_len
                        or chosen_token == self.tokenizer.eos_token_id
                    )

                    if not is_done:
                        input_t[0, 0] = chosen_token
                        cos = cached_cos[next_pos]
                        sin = cached_sin[next_pos]
                        mask[0, 0, 0, next_pos] = 0.0
                        t0 = time.perf_counter()
                        self.npu_model.infer_direct_async(input_t, cos, sin, mask)

                    # Overlap CPU tokenizer decoding and streaming generator consumption with NPU execution!
                    tok_text = self.tokenizer.decode([chosen_token])
                    yield tok_text

                    if is_done:
                        break
            else:
                for step in range(max_new_tokens):
                    curr_pos = start_pos + step
                    if curr_pos >= config.max_seq_len:
                        break

                    input_t[0, 0] = next_token
                    cos = cached_cos[curr_pos]
                    sin = cached_sin[curr_pos]
                    mask[0, 0, 0, curr_pos] = 0.0

                    t0 = time.perf_counter()
                    logits = infer_fn(input_t, cos, sin, mask)
                    decode_latencies.append((time.perf_counter() - t0) * 1000.0)

                    # Vectorized sampling
                    if temperature == 0.0 and repetition_penalty == 1.0:
                        next_token = int(torch.argmax(logits))
                    else:
                        flat_logits = logits.view(-1)
                        if repetition_penalty > 1.0 and seen_tokens:
                            past_list = list(seen_tokens)
                            orig_vals = flat_logits[past_list].clone()
                            flat_logits[past_list] = torch.where(
                                orig_vals < 0,
                                orig_vals * repetition_penalty,
                                orig_vals / repetition_penalty,
                            )
                            if temperature > 0.0:
                                probs = torch.softmax(flat_logits / temperature, dim=-1)
                                next_token = int(torch.multinomial(probs, num_samples=1))
                            else:
                                next_token = int(torch.argmax(flat_logits))
                            flat_logits[past_list] = orig_vals
                        else:
                            if temperature > 0.0:
                                probs = torch.softmax(flat_logits / temperature, dim=-1)
                                next_token = int(torch.multinomial(probs, num_samples=1))
                            else:
                                next_token = int(torch.argmax(flat_logits))

                    seen_tokens.add(next_token)
                    generated_tokens.append(next_token)
                    tok_text = self.tokenizer.decode([next_token])
                    yield tok_text

                    if next_token == self.tokenizer.eos_token_id:
                        break

        total_time_s = time.perf_counter() - t_start
        avg_decode_ms = sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0.0
        throughput = (1000.0 / avg_decode_ms) if avg_decode_ms > 0 else 0.0

        self.last_metrics = {
            "tokens_generated": len(decode_latencies),
            "ttft_ms": ttft_ms,
            "avg_decode_ms": avg_decode_ms,
            "throughput_tok_s": throughput,
            "total_time_s": total_time_s,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text completion for a prompt, returning the full generated text."""
        result_parts = []
        for piece in self.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        ):
            result_parts.append(piece)
        return "".join(result_parts)
