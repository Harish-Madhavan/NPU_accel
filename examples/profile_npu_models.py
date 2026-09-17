import os
import sys

# Ensure library is importable when run directly from repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "intel_npu_lib", "src")))

import argparse
import os
import sys
import time

import numpy as np
import torch

import intel_npu_acceleration as npu


class ProfilerModel(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(size, size)
        self.act1 = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(size, size)
        self.act2 = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.fc1(x))
        return self.act2(self.fc2(h))


def profile_execution(model: torch.nn.Module, input_tensor: torch.Tensor, iters: int = 50, warmup: int = 10, mode_name: str = "NPU Model"):
    print(f"\nRunning Profiling for [{mode_name}] ({iters} iterations, {warmup} warmup runs)...")

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    return {
        "mean": mean_lat,
        "std": std_lat,
        "min": min_lat,
        "max": max_lat,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "raw": latencies,
    }


def main():
    parser = argparse.ArgumentParser(description="Intel NPU Model Profiler & Latency Histogram Suite")
    parser.add_argument("--size", type=int, default=1024, help="Matrix dimension size (NxN)")
    parser.add_argument("--iters", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export profiling metrics as JSON")
    parser.add_argument("--export-markdown", type=str, default=None, help="Path to export profiling report as Markdown")
    args = parser.parse_args()

    # Print System Diagnostic Summary
    npu.print_info()

    print(f"\nProfiling Matrix Workload Size: [{args.size} x {args.size}]")

    model = ProfilerModel(args.size).eval()
    x = torch.randn(1, args.size)

    # 1. Profile Native PyTorch CPU
    cpu_stats = profile_execution(model, x, iters=args.iters, warmup=args.warmup, mode_name="PyTorch CPU")

    # 2. Profile NPU Compiled Graph (Standard PyTorch 2.x torch.compile)
    try:
        npu_model = torch.compile(model, backend="npu")
        npu_stats = profile_execution(npu_model, x, iters=args.iters, warmup=args.warmup, mode_name="Intel NPU (torch.compile)")
    except Exception as e:
        print(f"ERROR: NPU Compilation failed: {e}")
        return

    # Compute Operations and Throughput (2 * N^2 * 2 layers ops)
    num_ops = 4 * (args.size ** 2)
    cpu_gops = (num_ops / (cpu_stats["mean"] / 1000.0)) / 1e9
    npu_gops = (num_ops / (npu_stats["mean"] / 1000.0)) / 1e9
    speedup = cpu_stats["mean"] / max(npu_stats["mean"], 1e-6)

    # Print Comparative Histogram Summary Table
    banner = "=" * 75
    print("\n" + banner)
    print("                 INTEL NPU PROFILING HISTOGRAM SUMMARY")
    print(banner)
    print(f" {'Metric':<18} | {'PyTorch CPU':<15} | {'Intel NPU':<15} | {'Speedup / Improvement':<18}")
    print("-" * 75)
    print(f" {'Mean Latency':<18} | {cpu_stats['mean']:>12.3f} ms | {npu_stats['mean']:>12.3f} ms | {speedup:>14.2f}x speedup")
    print(f" {'Min Latency':<18} | {cpu_stats['min']:>12.3f} ms | {npu_stats['min']:>12.3f} ms | -")
    print(f" {'Max Latency':<18} | {cpu_stats['max']:>12.3f} ms | {npu_stats['max']:>12.3f} ms | -")
    print(f" {'P50 (Median)':<18} | {cpu_stats['p50']:>12.3f} ms | {npu_stats['p50']:>12.3f} ms | -")
    print(f" {'P95 (95th %ile)':<18} | {cpu_stats['p95']:>12.3f} ms | {npu_stats['p95']:>12.3f} ms | -")
    print(f" {'P99 (99th %ile)':<18} | {cpu_stats['p99']:>12.3f} ms | {npu_stats['p99']:>12.3f} ms | -")
    print(f" {'Std Deviation':<18} | {cpu_stats['std']:>12.3f} ms | {npu_stats['std']:>12.3f} ms | -")
    print(f" {'Throughput':<18} | {cpu_gops:>12.2f} GOPS | {npu_gops:>12.2f} GOPS | -")
    print(banner + "\n")

    if args.export_json:
        import json
        profile_data = {
            "workload": {"model": "ProfilerModel", "size": args.size, "iters": args.iters, "warmup": args.warmup},
            "cpu": {
                "mean_ms": float(cpu_stats["mean"]),
                "min_ms": float(cpu_stats["min"]),
                "max_ms": float(cpu_stats["max"]),
                "p50_ms": float(cpu_stats["p50"]),
                "p95_ms": float(cpu_stats["p95"]),
                "p99_ms": float(cpu_stats["p99"]),
                "std_ms": float(cpu_stats["std"]),
                "gops": float(cpu_gops),
            },
            "npu": {
                "mean_ms": float(npu_stats["mean"]),
                "min_ms": float(npu_stats["min"]),
                "max_ms": float(npu_stats["max"]),
                "p50_ms": float(npu_stats["p50"]),
                "p95_ms": float(npu_stats["p95"]),
                "p99_ms": float(npu_stats["p99"]),
                "std_ms": float(npu_stats["std"]),
                "gops": float(npu_gops),
            },
            "speedup_vs_cpu": float(speedup),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.export_json)), exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
        print(f"[Export] Saved profiling metrics JSON to: {args.export_json}")

    if args.export_markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_markdown)), exist_ok=True)
        md_text = f"""# Intel NPU Model Profiling Report

- **Workload**: 2-Layer MLP with GELU + SiLU (`ProfilerModel`)
- **Dimension**: {args.size} x {args.size}
- **Benchmark Iterations**: {args.iters} (Warmup: {args.warmup})

| Metric | PyTorch CPU | Intel NPU (`torch.compile`) | Speedup / Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | {cpu_stats['mean']:.3f} ms | {npu_stats['mean']:.3f} ms | **{speedup:.2f}x speedup** |
| **Min Latency** | {cpu_stats['min']:.3f} ms | {npu_stats['min']:.3f} ms | - |
| **Max Latency** | {cpu_stats['max']:.3f} ms | {npu_stats['max']:.3f} ms | - |
| **P50 (Median)** | {cpu_stats['p50']:.3f} ms | {npu_stats['p50']:.3f} ms | - |
| **P95 (95th %ile)** | {cpu_stats['p95']:.3f} ms | {npu_stats['p95']:.3f} ms | - |
| **P99 (99th %ile)** | {cpu_stats['p99']:.3f} ms | {npu_stats['p99']:.3f} ms | - |
| **Std Deviation** | {cpu_stats['std']:.3f} ms | {npu_stats['std']:.3f} ms | - |
| **Throughput** | {cpu_gops:.2f} GOPS | **{npu_gops:.2f} GOPS** | **{speedup:.2f}x** |
"""
        with open(args.export_markdown, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[Export] Saved profiling report Markdown to: {args.export_markdown}")


if __name__ == "__main__":
    main()
