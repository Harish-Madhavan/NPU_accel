# 🔄 Automated Hybrid Graph Partitioning & CPU Fallback

This guide documents the architecture, algorithm, and configuration of the **Automated Hybrid Graph Partitioning & CPU Fallback** feature inside the Intel NPU Acceleration Library.

---

## 🏗️ Architectural Motivation

While Intel NPU hardware is optimized for high-performance tensor operations (such as linear layers, matrix multiplications, convolutions, and attention), general deep learning models frequently incorporate:
*   Dynamic slicing patterns or complex index selections.
*   Frequency domain operations (e.g., Fast Fourier Transforms `torch.fft`).
*   Custom research-level algorithms or custom C++ operators.

In standard compilers, encountering even a single unsupported operator triggers a hard `NPUCompilationError` and halts execution. 

Our **Hybrid Graph Partitioner** dynamically splits PyTorch computational graphs into NPU-compiled segments and CPU-fallback segments. This guarantees **100% out-of-the-box support** for any PyTorch module:

```
                  ┌──────────────────────────────┐
                  │       PyTorch FX Graph       │
                  └──────────────┬───────────────┘
                                 │
                     [Monotonic Partitioner]
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
  ┌──────────────────────┐               ┌──────────────────────┐
  │  Supported Segment   │               │ Unsupported Segment  │
  │ (Linear, SDPA, ReLU) │               │   (torch.fft, etc.)  │
  └──────────┬───────────┘               └───────────┬──────────┘
             ▼                                       ▼
  ┌──────────────────────┐               ┌──────────────────────┐
  │ Compiled NPU Module  │               │ PyTorch Eager (CPU)  │
  │  (Zero-Copy FP16)    │               │  (Source-of-Truth)   │
  └──────────────────────┘               └──────────────────────┘
```

---

## 🧮 Cycle-Free Monotonic Partitioner

A major challenge when partitioning graphs is preventing **dependencies cycles** between subgraphs. For example, if both the beginning and end of a model are compiled for the NPU, but the middle operator must run on the CPU:
*   A naive partitioner assigns NPU nodes to "Partition 0" and CPU nodes to "Partition 1".
*   This creates a cycle: Partition 0 depends on Partition 1, which in turn depends on Partition 0!
*   PyTorch `split_module` will crash with: `RuntimeError: cycle exists between partitions!`.

### 🛡️ The Monotonic Topological Grouping Algorithm
We solve this by pre-computing a partition mapping before calling `split_module`. We traverse the FX graph in **topological order** and assign a partition ID that is **strictly non-decreasing**:

```python
partition_map = {}
current_partition = 0
prev_is_supported = None

for node in traced.graph.nodes:
    if node.op in ["placeholder", "get_attr"]:
        partition_map[node] = 0
        continue
    if node.op == "output":
        partition_map[node] = current_partition
        continue

    is_supported = check_node_support(node)

    if prev_is_supported is None:
        prev_is_supported = is_supported
    elif prev_is_supported != is_supported:
        current_partition += 1  # Increment partition ID on transition
        prev_is_supported = is_supported

    partition_map[node] = current_partition
```

### 📈 Why this Prevents Cycles Mathematically
Edges in a Directed Acyclic Graph (DAG) can only go from a node with a lower topological index to a node with a higher topological index. 

Since our partition ID is non-decreasing along the topological sequence, any edge from node $U$ to $V$ will satisfy `partition(U) <= partition(V)`. It is mathematically impossible to have an edge from a higher partition back to a lower partition, **guaranteeing 100% cycle-free execution paths**.

---

## 🪝 Dynamic Shape Calibration Hooks

When a graph is split into separate submodules, we must propagate tensor shapes and data types across submodule boundaries. 

Instead of writing complex, error-prone analytical shape propagation algorithms, we utilize a **Dynamic Calibration Pass**:
1.  We register a standard PyTorch `register_forward_pre_hook` callback on each partitioned submodule.
2.  We perform a quick, single-pass mock execution of the parent model using your `example_input`.
3.  As the submodules execute on the CPU, the pre-hooks intercept and capture the **exact shapes and data types** of the intermediate tensors:
    ```python
    submod_inputs = {}
    def make_hook(name):
        def hook(module, inputs):
            submod_inputs[name] = inputs
            return None
        return hook
    ```
4.  Once shapes are captured, the hooks are cleanly removed, and the captured shapes are passed directly to the OpenVINO NPU compiler to build zero-copy buffers.

---

## ⚙️ Strict vs. Fallback Compilation Modes

You can customize compilation behavior through the **`strict`** flag:

### 1. `strict=False` (Default)
Enables Automated CPU Fallback. If unsupported operators are encountered, the partitioner segments the graph and executes unsupported blocks on the CPU.
*   **Best For**: General deployment, complex networks, and rapid integration.

### 2. `strict=True`
Strict NPU-Compilation Mode. The compiler will immediately raise an `NPUCompilationError` if any unsupported operator is found.
*   **Best For**: Hardware profiling, CI/CD verification tests, and strict hardware-compatibility auditing.

```python
import intel_npu_acceleration as npu

# Force strict NPU execution — raises NPUCompilationError if any CPU fallback is needed
compiled_model = npu.compile(model, x_example, strict=True)
```
