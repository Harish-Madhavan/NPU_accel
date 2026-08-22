---
name: benchmark-npu
description: Runbook for running matrix multiplication, vision, and transformer model benchmarks on Intel NPU.
---

# Intel NPU Benchmarking Runbook

Use this skill to benchmark latency, throughput, and speedup against CPU baselines.

## Benchmark Scripts

### 1. Matrix Multiplication (MatMul) Benchmark
```powershell
python examples/benchmark_npu_matmul.py --size 1024 --iters 50 --warmup 10
```
- Tests single-stream and multi-stream matrix multiplication.
- Reports GOPS throughput and speedup vs CPU.

### 2. Deep Neural Network Model Profiling
```powershell
python examples/profile_npu_models.py --size 1024 --iters 50 --warmup 10
```
- Reports histogram metrics: Mean, Min, Max, P50, P95, P99, and StdDev.

### 3. Multi-Stream Asynchronous Pipelining
```powershell
python examples/async_pipelined_inference.py --batch-count 50 --batch-size 16 --streams 4
```
- Demonstrates pipelined throughput gains across 4 hardware streams.
