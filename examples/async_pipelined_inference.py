"""
Intel NPU Asynchronous Pipelined Inference & Multi-Stream Saturation Demo.
Demonstrates:
  1. Multi-stream concurrent inference (NPUAsyncFuture & submit)
  2. Batch pipelining (batch_infer) for maximum NPU hardware saturation
  3. Automatic FP16 precision compilation pass (precision="fp16")
  4. Per-layer OpenVINO execution profiling (get_profiling_info)
"""

import os
import sys

# Ensure library is importable when run directly from repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "intel_npu_lib", "src")))

import argparse
import time

import torch

import intel_npu_acceleration as npu


class DeepMLP(torch.nn.Module):
    def __init__(self, in_features=1024, hidden_features=2048, out_features=1024):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act1 = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.act2 = torch.nn.SiLU()
        self.fc3 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        return self.fc3(h2)


def main():
    parser = argparse.ArgumentParser(description="Intel NPU Pipelined Multi-Stream Inference Demo")
    parser.add_argument("--batch-count", type=int, default=50, help="Total number of inference requests")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch dimension per request")
    parser.add_argument("--dim", type=int, default=1024, help="Feature dimension")
    parser.add_argument("--streams", type=int, default=4, help="Number of concurrent NPU hardware execution streams")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"], help="Model execution precision")
    args = parser.parse_args()

    print("=" * 72)
    print("      INTEL NPU ASYNCHRONOUS PIPELINED MULTI-STREAM INFERENCE")
    print("=" * 72)
    print(f"Device Available : {npu.is_available()}")
    print(f"Total Requests   : {args.batch_count}")
    print(f"Batch Dimension  : [{args.batch_size}, {args.dim}]")
    print(f"Hardware Streams : {args.streams}")
    print(f"Graph Precision  : {args.precision.upper()}")
    print("-" * 72)

    # 1. Initialize PyTorch model and sample inputs
    model = DeepMLP(in_features=args.dim, hidden_features=args.dim * 2, out_features=args.dim).eval()
    sample_input = torch.randn(args.batch_size, args.dim)
    input_batches = [torch.randn(args.batch_size, args.dim) for _ in range(args.batch_count)]

    # 2. Compile model for NPU with multi-stream throughput mode & FP16 optimization
    print("\n[Step 1] Compiling model for Intel NPU (Multi-Stream THROUGHPUT mode)...")
    t0 = time.perf_counter()
    npu_model = npu.compile(
        model,
        sample_input,
        performance_hint="THROUGHPUT",
        num_streams=args.streams,
        precision=args.precision,
    )
    t_comp = time.perf_counter() - t0
    print(f"Compilation finished in {t_comp:.2f}s.")

    # 3. Synchronous baseline benchmark
    print("\n[Step 2] Running Synchronous Baseline (Single Request at a Time)...")
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = npu_model(sample_input)

        t_sync_start = time.perf_counter()
        sync_results = []
        for inp in input_batches:
            sync_results.append(npu_model(inp))
        t_sync_total = time.perf_counter() - t_sync_start

    sync_throughput = (args.batch_count * args.batch_size) / t_sync_total
    sync_avg_latency = (t_sync_total / args.batch_count) * 1000.0
    print(f"  Total Time: {t_sync_total:.3f}s | Avg Latency: {sync_avg_latency:.2f} ms | Throughput: {sync_throughput:.1f} samples/s")

    # 4. Asynchronous Pipelined Batch Inference
    print(f"\n[Step 3] Running Asynchronous Pipelined Inference across {args.streams} NPU Hardware Streams...")
    with torch.no_grad():
        t_async_start = time.perf_counter()
        async_results = npu_model.batch_infer([(inp,) for inp in input_batches])
        t_async_total = time.perf_counter() - t_async_start

    async_throughput = (args.batch_count * args.batch_size) / t_async_total
    async_avg_latency = (t_async_total / args.batch_count) * 1000.0
    speedup = async_throughput / max(sync_throughput, 1e-6)

    print(f"  Total Time: {t_async_total:.3f}s | Effective Step: {async_avg_latency:.2f} ms | Throughput: {async_throughput:.1f} samples/s")
    print(f"  Pipelining Throughput Gain: {speedup:.2f}x")

    # 5. OpenVINO Layer Execution Profiling
    print("\n[Step 4] Inspecting OpenVINO Per-Layer Hardware Execution Metrics...")
    if hasattr(npu_model, "get_profiling_info"):
        profiling_data = npu_model.get_profiling_info(request_idx=0)
        print(f"  Total Executed Graph Nodes: {len(profiling_data)}")
        print("  Sample Layer Breakdown (First 5 Active Layers):")
        for layer in profiling_data[:5]:
            node_name = getattr(layer, "node_name", "N/A")
            exec_type = getattr(layer, "exec_type", "N/A")
            real_time = getattr(layer, "real_time", 0.0)
            status = getattr(layer, "status", "N/A")
            print(f"    - [{status}] {node_name:<30} | Type: {exec_type:<15} | Exec Time: {real_time / 1000.0:.3f} ms")

    # 6. Performance Summary Table
    print("\n" + "=" * 72)
    print("                 PIPELINED NPU PERFORMANCE SUMMARY")
    print("=" * 72)
    print(f"| {'Execution Mode':<28} | {'Latency / Batch':<16} | {'Throughput':<18} |")
    print("-" * 72)
    print(f"| {'Synchronous Mode':<28} | {f'{sync_avg_latency:.2f} ms':<16} | {f'{sync_throughput:.1f} smp/s':<18} |")
    print(f"| {f'Asynchronous ({args.streams} Streams)':<28} | {f'{async_avg_latency:.2f} ms':<16} | {f'{async_throughput:.1f} smp/s':<18} |")
    print("=" * 72)
    print(f"Throughput Improvement with Pipelined Streams: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
