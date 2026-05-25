# ⚡ High-Throughput Asynchronous Pipeline

This guide documents the architecture and Python APIs of the **High-Throughput Asynchronous Pipeline** in the Intel NPU Acceleration Library. It details how to leverage OpenVINO's non-blocking execution engine to overlap CPU and NPU tasks.

---

## 🏗️ Pipelined Execution Concept

During standard deep learning workflows (such as autoregressive Large Language Model generation), inference consists of a loop:
1.  **CPU Token Pre-processing**: Tokenization, embedding updates, position encoding.
2.  **Model Execution (MatMul/Attention)**: Compute-heavy mathematical layer passes.
3.  **CPU Logit Processing**: Top-K/Top-P sorting, temperature scaling, and sampling.

In a **synchronous** model, the CPU remains completely idle while waiting for the NPU to compute the matrix multiplications. Conversely, the NPU sits idle while the CPU processes logits and samples tokens. 

The **Asynchronous Pipeline** eliminates this idle time by allowing them to work concurrently:

```
Synchronous:
CPU: ──[Pre-process]───────────────► [Post-process]──► [Next Pre-process]
NPU:                 [MatMul Math]►

Asynchronous (Pipelined):
CPU: ──[Pre-process]─► [Post-process (Batch N)] ──► [Pre-process (Batch N+2)]
NPU:                 [MatMul Math (Batch N+1)] ──► [MatMul Math (Batch N+2)]
```

By overlapping CPU logit sampling for batch $N$ with parallel NPU matrix computations for batch $N+1$, you maximize the active execution cycles of both processors.

---

## 💻 Asynchronous API Reference

We expose two core methods on compiled `NPUGraphModule` objects:

### 1. `infer_async(*args) -> int`
Traces inputs, sets up zero-copy pointers, and schedules inference asynchronously on the hardware. It returns an integer `handle` (representing the queue request index) immediately.
*   **Parameters**: PyTorch tensors corresponding to model input placeholders.
*   **Returns**: `int` (Handle to be passed to `wait_async`).

### 2. `wait_async(handle: int) -> Union[torch.Tensor, Tuple[torch.Tensor]]`
Blocks the active thread **only until the specific async task finishes**, and returns the results.
*   **Parameters**: `handle` (The request index returned by `infer_async`).
*   **Returns**: The output PyTorch tensor or a tuple of tensors.

### 📝 Example Usage: Autoregressive Overlapping
```python
import torch
import intel_npu_acceleration as npu

model = MyLLM().eval()
x_input = torch.tensor([[101, 203, 405]])

# Compile with THROUGHPUT performance hint
compiled = npu.compile(model, x_input, performance_hint="THROUGHPUT", num_streams=2)

# Start first batch asynchronously
h1 = compiled.infer_async(x_input)

# CPU concurrently prepares the second batch
x_next = torch.tensor([[202, 303, 404]])
h2 = compiled.infer_async(x_next)

# Retrieve results only when required
out1 = compiled.wait_async(h1)
out2 = compiled.wait_async(h2)
```

---

## ⚙️ Performance Hints: Latency vs. Throughput

You can configure performance targets during compilation:

### 1. `performance_hint="LATENCY"` (Default)
Optimizes compilation for minimal processing delay of a single request. 
*   **Use Case**: Real-time chatbots, single-item classifications, where response time must be minimal.
*   **Hardware Allocation**: Allocates maximum compute power to a single stream.

### 2. `performance_hint="THROUGHPUT"`
Optimizes compilation for processing multiple batches in parallel.
*   **Use Case**: Batch processing, multi-user LLM serving, and pipelines processing independent streams.
*   **Streams Configuration**: Pass `num_streams=N` to configure parallel execution queues on the NPU hardware:
    ```python
    npu_model = npu.compile(model, example_input, performance_hint="THROUGHPUT", num_streams=4)
    ```

---

## 🛡️ Memory Safety & Output Cloning

When compiling models, the library binds pre-allocated memory buffers to the OpenVINO runtime via **`shared_memory=True`** to achieve maximum TFLOPS zero-copy transfers.

To prevent silent memory corruption (where subsequent inferences on the same hardware stream overwrite the memory contents of previous outputs held by the user), the `NPUGraphModule` automatically wraps output buffers in a PyTorch `.clone()` call before returning:

```python
# Safely capture outputs in a loop
results = []
for batch in batches:
    # Under-the-hood: returns a completely isolated memory block (.clone())
    results.append(npu_model(batch)) 

# RESULTS are completely safe and independent!
```
This guarantees **100% memory safety and standard copy-on-write semantics** with negligible copy overhead compared to actual NPU compute times.
