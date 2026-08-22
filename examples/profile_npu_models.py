import os
import sys

# Ensure library is importable when run directly from repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "intel_npu_lib", "src")))

import argparse
import time
import math
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


if __name__ == "__main__":
    main()
