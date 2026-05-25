# 💾 Stateful NPU KV-Caches & Double-Buffering Architecture

This document provides a deep dive into the **Stateful Memory API** and **Double-Buffering** architecture implemented in the Intel NPU Acceleration Library. These features are designed to maximize hardware saturation and eliminate memory bus bottlenecks during Large Language Model (LLM) inference.

---

## 🚀 1. Stateful NPU KV-Caches (`NPUStatefulKVCache`)

During traditional LLM decoding (generation phase), Key-Value (KV) caches are transferred back and forth between host (CPU) memory and the NPU on each autoregressive step. Since the NPU needs the entire history of KV states to compute self-attention, sending growing tensors over the system bus creates severe synchronization bottlenecks.

### The Stateful Solution
By leveraging OpenVINO's native **Stateful Memory API** (`ReadValue` and `Assign` operators), we allocate and persist the KV-cache tensors *directly inside the NPU's internal hardware memory registers* across the entire generation loop.

```
                  ┌──────────────────────────────────────────────┐
                  │                 Intel NPU                    │
                  │                                              │
                  │   ┌───────────────┐     ┌────────────────┐   │
  [New Token] ───┼──►│ ScatterUpdate ├────►│ Assign State   │   │
                  │   └───────▲───────┘     └────────┬───────┘   │
                  │           │                      │           │
                  │   ┌───────┴───────┐              │ (Feedback │
                  │   │  ReadValue    │◄─────────────┘  Loop)    │
                  │   └───────────────┘                          │
                  │                                              │
                  └──────────────────────────────────────────────┘
```

1. **`ReadValue`**: Retrieves the historical KV cache accumulated up to the current decoding step from NPU state registers.
2. **`ScatterUpdate`**: Appends the active token's KV projection at the current sequence index.
3. **`Assign`**: Writes the updated cache back to the NPU's state register.
4. **Zero Host Copies**: The CPU only transmits the single active token tensor (`[1, 1, num_heads, head_dim]`), dropping host-to-device memory copy overhead to **zero**!

---

## ⚙️ 2. NPU Hardware Constraints & Design Decisions

Designing stateful variables for Intel NPU hardware imposes strict constraints compared to executing on CPU/GPU:

### Constraint A: Floating-Point Only States
Intel NPU hardware state registers *natively support only floating-point element types* (`float32` and `float16`). Declaring state variables of integer type (`int32` / `int64`) results in NPU compilation errors:
```
Unsupported element type 'si32'
```

### The Solution: Float32 Position Pointer
To keep the position index pointer completely inside NPU state registers without host synchronization, we:
1. Initialize the index register as a `float32` state variable (`ov.Type.f32`).
2. Read the position index as `f32`.
3. Inside the compiler graph, dynamically cast the position index to `int32` (`ops.convert`) *only for transient shape calculations and array indices* (e.g., inside `ops.range` bounds).
4. Perform indexing updates, then convert the incremented sequence length back to `f32` to assign the new position state.
This keeps the state registers 100% floating-point compatible on the hardware while providing integer precision index calculations.

---

## 🛡️ 3. Double-Buffered Output Memory

When executing asynchronous pipelines (`infer_async` and `wait_async`) or requesting raw zero-copy views without clones (`clone_outputs=False`), memory safety becomes critical. If subsequent inference runs write into the same output memory address before the host is done processing, silent data corruption occurs.

### The Double-Buffering Solution
We implement a thread-safe **round-robin Double-Buffering** mechanism per NPU execution stream:

* **Pre-Allocation**: For each execution stream, the library pre-allocates exactly *two* output buffers.
* **Alternating Toggles**: On each forward pass or async request, the library alternates which output buffer is bound to the OpenVINO inference request:
  $$\text{Active Buffer Index} = 1 - \text{Active Buffer Index}$$
* **Reference Safety**: Even if the user does not clone the output immediately and keeps a live reference during the next forward pass, the memory remains safe because the subsequent inference executes on the *other* buffer.

---

## 🚀 4. API Usage Examples

### Autoregressive LLM Loop
```python
import torch
import torch.nn as nn
import intel_npu_acceleration as npu

class LlamaDecoderBlock(nn.Module):
    def __init__(self, batch_size, max_seq_len, num_heads, head_dim):
        super().__init__()
        # Allocate stateful cache directly on the NPU
        self.stateful_cache = npu.functional.NPUStatefulKVCache(
            batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float32
        )

    def forward(self, new_token_kv):
        return self.stateful_cache(new_token_kv)

model = LlamaDecoderBlock(batch_size=1, max_seq_len=2048, num_heads=32, head_dim=128).eval()
x_example = torch.randn(1, 1, 32, 128) # single token

# Compile to NPU with strict validation
npu_model = npu.compile(model, x_example, strict=True)

# Generate tokens
for i in range(100):
    token_kv = torch.randn(1, 1, 32, 128)
    # Stateful append directly inside NPU hardware!
    full_cache = npu_model(token_kv)

# Reset internal hardware variables back to 0 for a new sequence
npu_model.reset_states()
```
