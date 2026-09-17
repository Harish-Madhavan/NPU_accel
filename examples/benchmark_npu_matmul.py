import os
import sys

# Ensure library is importable when run directly from repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "intel_npu_lib", "src")))

import argparse
import math
import time

import torch

import intel_npu_acceleration as npu_compiler


class MatMulModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)


def run_benchmark(
    m=2048,
    n=2048,
    k=2048,
    iterations=50,
    warmup=10,
    dtype_str="float16",
    seed=42,
    num_streams=1,
    compare_cpu=True,
    cpu_iters=5,
    export_json=None,
    export_markdown=None,
):
    # Auto-tune iterations for large matrices to avoid timeout
    total_ops = 2 * m * n * k
    if total_ops > 8 * 1024**3:  # >8B ops → large
        orig_iters = iterations
        iterations = min(iterations, 5)
        warmup = min(warmup, 2)
        print(f"[Auto-tune] Large matrix {m}x{n}x{k} ({total_ops/1e9:.1f}B ops) — reducing iters {orig_iters}->{iterations}, warmup->{warmup}")

    torch.manual_seed(seed)

    print("=" * 70)
    print("              INTEL NPU MATMUL PERFORMANCE BENCHMARK             ")
    print("=" * 70)
    print(f"PyTorch Version  : {torch.__version__}")

    npu_available = npu_compiler.is_available()
    print(f"NPU Hardware     : {'Detected' if npu_available else 'Not Detected (CPU Fallback)'}")

    # Inputs
    print(f"Matrix Shape     : ({m}x{n}) x ({n}x{k}) -> ({m}x{k})")
    print(f"Data Type        : {dtype_str}")
    print(f"Iterations       : {iterations} (Warmup: {warmup})")
    print(f"Streams          : {num_streams}")
    print("-" * 70)

    # Prepare input tensors — int8 uses quantized Linear path (W8A16) for NPU DP4A
    is_int8 = dtype_str == "int8"
    if dtype_str == "float16":
        dtype = torch.float16
        a = torch.randn(m, n, dtype=dtype)
        b = torch.randn(n, k, dtype=dtype)
        model = MatMulModel()
    elif is_int8:
        # INT8 on NPU is weight-only quantized Linear (not raw matmul)
        # Use npu.quantize() to get correct QDQ graph; benchmark Linear instead of matmul
        print("INT8 mode: using weight-only quantized Linear (W8A16) for NPU DP4A path")
        import torch.nn as nn

        class QuantizedLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(n, k, bias=False)

            def forward(self, x):
                return self.fc(x)

        base = QuantizedLinearModel()
        # Quantize weight to int8 per-tensor
        try:
            from intel_npu_acceleration import quantize

            model = quantize(base)
            print(f"  Quantized weight dtype: {model.fc.weight.dtype}, scale: {float(model.fc.weight_scale):.4f}")
        except Exception as e:
            print(f"  Quantize failed ({e}), falling back to float16")
            model = base
            is_int8 = False
            dtype_str = "float16"
        a = torch.randn(m, n, dtype=torch.float16)
        b = None  # not used for Linear
    else:
        dtype = torch.float32
        a = torch.randn(m, n, dtype=dtype)
        b = torch.randn(n, k, dtype=dtype)
        model = MatMulModel()
    model.eval()

    print("Compiling model for NPU (torch.compile)...")
    t0 = time.perf_counter()
    try:
        if num_streams > 1:
            npu_model = torch.compile(
                model, backend="npu", options={"num_streams": num_streams, "performance_hint": "THROUGHPUT"}
            )
        else:
            npu_model = torch.compile(model, backend="npu")
        # Trigger compilation with a dummy run to catch MLIR errors early
        with torch.inference_mode():
            if b is None:
                _ = npu_model(a)
            else:
                _ = npu_model(a, b)
    except Exception as e:
        if is_int8:
            print(f"INT8 matmul compilation failed (expected for bare matmul, use quantized Linear): {e}")
            print("Falling back to float16 matmul for benchmark (INT8 peak requires quantized Linear API)")
            # Retry with float16
            a = torch.randn(m, n, dtype=torch.float16)
            b = torch.randn(n, k, dtype=torch.float16)
            model = MatMulModel()
            model.eval()
            try:
                npu_model = torch.compile(model, backend="npu")
                with torch.inference_mode():
                    _ = npu_model(a, b)
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return
        else:
            print(f"NPU compilation failed: {e}")
            return
    print(f"PyTorch NPU Compilation finished in {time.perf_counter() - t0:.2f}s")

    # Helper to call model with correct args for matmul vs quantized linear
    def _call_npu(*args):
        # is_int8 Linear takes single tensor, matmul takes two
        if b is None:
            return npu_model(a)
        return npu_model(a, b)

    def _call_npu_async():
        if b is None:
            return npu_model.infer_async(a)
        return npu_model.infer_async(a, b)

    # Warmup (inference_mode, perf_counter)
    print(f"Warming up NPU ({warmup} iterations)...")
    use_async = hasattr(npu_model, "infer_async") and hasattr(npu_model, "wait_async") and num_streams > 1
    with torch.inference_mode():
        if use_async:
            from collections import deque

            handles = deque(_call_npu_async() for _ in range(min(num_streams, warmup)))
            while handles:
                _ = npu_model.wait_async(handles.popleft())
        else:
            for _ in range(warmup):
                _ = _call_npu()

    print(f"Running NPU stress test ({iterations} iterations)...")
    npu_latencies: list[float] = []
    t_stress_start = time.perf_counter()
    with torch.inference_mode():
        if use_async:
            from collections import deque

            active: deque = deque()
            submitted = 0
            for _ in range(min(num_streams, iterations)):
                active.append((_call_npu_async(), time.perf_counter()))
                submitted += 1
            completed = 0
            while active:
                h, t_sub = active.popleft()
                _ = npu_model.wait_async(h)
                completed += 1
                npu_latencies.append((time.perf_counter() - t_sub) * 1000.0)
                if submitted < iterations:
                    active.append((_call_npu_async(), time.perf_counter()))
                    submitted += 1
                if iterations >= 10 and completed % (iterations // 5) == 0:
                    print(f"  Progress: {completed:3d}/{iterations:3d}")
        else:
            for i in range(iterations):
                t0 = time.perf_counter()
                _ = _call_npu()
                npu_latencies.append((time.perf_counter() - t0) * 1000.0)
                if iterations >= 10 and (i + 1) % (iterations // 5) == 0:
                    print(f"  Progress: {i + 1:3d}/{iterations:3d}")

    total_stress_duration = time.perf_counter() - t_stress_start

    # NPU statistics
    avg_npu = sum(npu_latencies) / iterations
    min_npu = min(npu_latencies)
    max_npu = max(npu_latencies)
    var_npu = sum((x - avg_npu) ** 2 for x in npu_latencies) / iterations
    std_npu = math.sqrt(var_npu)

    # Multiply-accumulate FLOPs: 2 * M * N * K
    ops = 2 * m * n * k
    total_ops = ops * iterations
    system_gops = (total_ops / total_stress_duration) / 1e9 if total_stress_duration > 0 else 0
    system_tops = (total_ops / total_stress_duration) / 1e12 if total_stress_duration > 0 else 0
    tflops_npu = (ops / (avg_npu / 1000.0)) / 1e12 if avg_npu > 0 else 0

    print("\n" + "=" * 70)
    print("                              NPU RESULTS                             ")
    print("=" * 70)
    print(f"Average Latency  : {avg_npu:.3f} ms")
    print(f"Latency Range    : {min_npu:.3f} ms - {max_npu:.3f} ms")
    print(f"Latency StdDev   : {std_npu:.3f} ms (Jitter: {std_npu / avg_npu * 100.0:.2f}%)")
    print(f"Compute Power    : {tflops_npu:.4f} TFLOPS")
    print(f"System Throughput: {system_gops:.2f} GOPS ({system_tops:.4f} TOPS)")
    print("=" * 70)

    avg_cpu = None
    min_cpu = None
    max_cpu = None
    std_cpu = None
    tflops_cpu = None
    speedup = None

    if compare_cpu and dtype_str != "int8":
        print("\nRunning CPU Comparison (PyTorch native)...")
        print("Warming up CPU...")
        with torch.inference_mode():
            torch.matmul(a, b)
        print(f"Running CPU test ({cpu_iters} iterations)...")
        cpu_latencies = []
        with torch.inference_mode():
            for _i in range(cpu_iters):
                t0 = time.perf_counter()
                torch.matmul(a, b)
                lat = (time.perf_counter() - t0) * 1000.0
                cpu_latencies.append(lat)

        avg_cpu = sum(cpu_latencies) / cpu_iters
        min_cpu = min(cpu_latencies)
        max_cpu = max(cpu_latencies)
        var_cpu = sum((x - avg_cpu) ** 2 for x in cpu_latencies) / cpu_iters
        std_cpu = math.sqrt(var_cpu)
        tflops_cpu = (ops / (avg_cpu / 1000.0)) / 1e12 if avg_cpu > 0 else 0
        speedup = avg_cpu / avg_npu if avg_npu > 0 else 0

        print("\n" + "=" * 70)
        print("                              CPU RESULTS                             ")
        print("=" * 70)
        print(f"Average Latency  : {avg_cpu:.3f} ms")
        print(f"Latency Range    : {min_cpu:.3f} ms - {max_cpu:.3f} ms")
        print(f"Latency StdDev   : {std_cpu:.3f} ms")
        print(f"Compute Power    : {tflops_cpu:.4f} TFLOPS")
        print(f"Speedup vs CPU   : {speedup:.2f}x")
        print("=" * 70 + "\n")

    results_data = {
        "matrix": {"m": m, "n": n, "k": k, "dtype": dtype_str},
        "npu": {
            "avg_latency_ms": avg_npu,
            "min_latency_ms": min_npu,
            "max_latency_ms": max_npu,
            "std_dev_ms": std_npu,
            "tflops": tflops_npu,
            "system_tops": system_tops,
        },
        "cpu": {
            "avg_latency_ms": avg_cpu,
            "min_latency_ms": min_cpu,
            "max_latency_ms": max_cpu,
            "std_dev_ms": std_cpu,
            "tflops": tflops_cpu,
        } if avg_cpu is not None else None,
        "speedup_vs_cpu": speedup,
    }

    if export_json:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(export_json)), exist_ok=True)
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)
        print(f"[Export] Saved JSON benchmark results to: {export_json}")

    if export_markdown:
        os.makedirs(os.path.dirname(os.path.abspath(export_markdown)), exist_ok=True)
        md_content = f"""# Intel NPU MatMul Benchmark Report

- **Matrix Dimensions**: ({m}x{n}) x ({n}x{k}) -> ({m}x{k})
- **Precision**: {dtype_str}
- **NPU Streams**: {num_streams}
- **Iterations**: {iterations}

| Metric | Intel NPU | PyTorch CPU | Speedup |
| :--- | :---: | :---: | :---: |
| **Average Latency** | {avg_npu:.2f} ms | {f'{avg_cpu:.2f} ms' if avg_cpu else 'N/A'} | {f'**{speedup:.2f}x**' if speedup else 'N/A'} |
| **Min Latency** | {min_npu:.2f} ms | {f'{min_cpu:.2f} ms' if min_cpu else 'N/A'} | - |
| **Max Latency** | {max_npu:.2f} ms | {f'{max_cpu:.2f} ms' if max_cpu else 'N/A'} | - |
| **Effective TFLOPS** | **{tflops_npu:.3f}** | {f'{tflops_cpu:.3f}' if tflops_cpu else 'N/A'} | {f'**{speedup:.2f}x**' if speedup else 'N/A'} |
"""
        with open(export_markdown, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[Export] Saved Markdown benchmark report to: {export_markdown}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel NPU MatMul Stress & Benchmark Test")
    parser.add_argument("--size", type=int, default=None, help="Matrix dimension (NxN). Overrides --m, --n, --k if set.")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension (Rows of A)")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension (Cols of A / Rows of B)")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension (Cols of B)")
    parser.add_argument("--iters", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "int8"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--turbo", action="store_true", help="Enable Intel Level Zero hardware Turbo boost mode")
    parser.add_argument("--sda", action="store_true", help="Enable Shared Device Address (SDA) zero-copy transfers")
    parser.add_argument("--num-streams", type=int, default=1, help="Number of pipelined execution streams")
    parser.add_argument("--compare-cpu", action="store_true", default=True, help="Run PyTorch CPU baseline comparison")
    parser.add_argument("--no-cpu", dest="compare_cpu", action="store_false", help="Disable CPU baseline comparison")
    parser.add_argument("--cpu-iters", type=int, default=5, help="Number of CPU benchmark iterations")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export benchmark metrics as JSON")
    parser.add_argument("--export-markdown", type=str, default=None, help="Path to export benchmark report as Markdown")

    args = parser.parse_args()

    # Enable global configurations
    if args.turbo:
        print("Enabling Level Zero Turbo Mode...")
        npu_compiler.enable_turbo()
    if args.sda:
        print("Enabling Level Zero Shared Device Address (SDA)...")
        npu_compiler.enable_sda()

    # Use --size if set to override m, n, k
    m_val = args.size if args.size is not None else args.m
    n_val = args.size if args.size is not None else args.n
    k_val = args.size if args.size is not None else args.k

    run_benchmark(
        m=m_val,
        n=n_val,
        k=k_val,
        iterations=args.iters,
        warmup=args.warmup,
        dtype_str=args.dtype,
        seed=args.seed,
        num_streams=args.num_streams,
        compare_cpu=args.compare_cpu,
        cpu_iters=args.cpu_iters,
        export_json=args.export_json,
        export_markdown=args.export_markdown,
    )
