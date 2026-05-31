import torch
import time
import math
import intel_npu_acceleration as npu_compiler
import argparse


class MatMulModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)


def run_benchmark(m=2048, n=2048, k=2048, iterations=50, warmup=10, dtype_str="float16", seed=42, num_streams=1):
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

    # Prepare input tensors
    if dtype_str == "float16":
        dtype = torch.float16
        a = torch.randn(m, n, dtype=dtype)
        b = torch.randn(n, k, dtype=dtype)
    elif dtype_str == "int8":
        dtype = torch.int8
        a = torch.randint(-128, 127, (m, n), dtype=dtype)
        b = torch.randint(-128, 127, (n, k), dtype=dtype)
    else:
        dtype = torch.float32
        a = torch.randn(m, n, dtype=dtype)
        b = torch.randn(n, k, dtype=dtype)

    model = MatMulModel()
    model.eval()

    # Compile to NPU
    print("Compiling model for NPU...")
    t0 = time.time()
    try:
        npu_model = npu_compiler.compile_to_npu(
            model,
            (a, b),
            num_streams=num_streams,
            performance_hint="THROUGHPUT" if num_streams > 1 else "LATENCY"
        )
    except Exception as e:
        print(f"NPU compilation failed: {e}")
        return
    print(f"Ahead-of-Time NPU Compilation finished in {time.time() - t0:.2f}s")

    # Warmup NPU
    print(f"Warming up NPU ({warmup} iterations)...")
    if hasattr(npu_model, "infer_async") and hasattr(npu_model, "wait_async") and num_streams > 1:
        from collections import deque
        active_handles = deque()
        for _ in range(min(num_streams, warmup)):
            active_handles.append(npu_model.infer_async(a, b))
        while active_handles:
            h = active_handles.popleft()
            _ = npu_model.wait_async(h)
    else:
        for _ in range(warmup):
            _ = npu_model(a, b)

    # NPU Stress iterations
    print(f"Running NPU stress test ({iterations} iterations)...")
    npu_latencies = []
    
    t_stress_start = time.time()
    if hasattr(npu_model, "infer_async") and hasattr(npu_model, "wait_async") and num_streams > 1:
        from collections import deque
        active_handles = deque()
        submitted = 0
        completed = 0
        
        # Fill the multi-stream execution pipeline
        for _ in range(min(num_streams, iterations)):
            h = npu_model.infer_async(a, b)
            active_handles.append((h, time.time()))
            submitted += 1
            
        # Interleave waits and submissions
        while active_handles:
            h, t_sub = active_handles.popleft()
            _ = npu_model.wait_async(h)
            completed += 1
            npu_latencies.append((time.time() - t_sub) * 1000.0)  # latency per request in ms
            
            if submitted < iterations:
                h_new = npu_model.infer_async(a, b)
                active_handles.append((h_new, time.time()))
                submitted += 1
                
            if iterations >= 10 and completed % (iterations // 5) == 0:
                print(f"  Progress: {completed:3d}/{iterations:3d}")
    else:
        for i in range(iterations):
            t_start = time.time()
            _ = npu_model(a, b)
            t_end = time.time()
            npu_latencies.append((t_end - t_start) * 1000.0)  # ms
            
            if iterations >= 10 and (i + 1) % (iterations // 5) == 0:
                print(f"  Progress: {i + 1:3d}/{iterations:3d}")
                
    total_stress_duration = time.time() - t_stress_start

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

    print("\n" + "=" * 70)
    print("                              NPU RESULTS                             ")
    print("=" * 70)
    print(f"Average Latency  : {avg_npu:.3f} ms")
    print(f"Latency Range    : {min_npu:.3f} ms - {max_npu:.3f} ms")
    print(f"Latency StdDev   : {std_npu:.3f} ms (Jitter: {std_npu / avg_npu * 100.0:.2f}%)")
    print(f"System Throughput: {system_gops:.2f} GOPS ({system_tops:.4f} TOPS)")
    print("=" * 70)

    # CPU Comparison (if float32 or float16)
    if dtype_str != "int8":
        print("\nRunning CPU Comparison (PyTorch FP32/FP16 native)...")
        # CPU Warmup
        print(f"Warming up CPU ({min(5, warmup)} iterations)...")
        for _ in range(min(5, warmup)):
            torch.matmul(a, b)

        # CPU Benchmark
        cpu_iters = min(10, iterations)
        print(f"Running CPU test ({cpu_iters} iterations)...")
        cpu_latencies = []
        for _ in range(cpu_iters):
            t_start = time.time()
            torch.matmul(a, b)
            t_end = time.time()
            cpu_latencies.append((t_end - t_start) * 1000.0)

        avg_cpu = sum(cpu_latencies) / cpu_iters
        min_cpu = min(cpu_latencies)
        max_cpu = max(cpu_latencies)
        var_cpu = sum((x - avg_cpu) ** 2 for x in cpu_latencies) / cpu_iters
        std_cpu = math.sqrt(var_cpu)

        # We compare CPU eager sequential throughput to NPU parallel system throughput
        total_cpu_ops = ops * cpu_iters
        total_cpu_sec = sum(cpu_latencies) / 1000.0
        cpu_system_gops = (total_cpu_ops / total_cpu_sec) / 1e9 if total_cpu_sec > 0 else 0
        speedup = system_gops / cpu_system_gops if cpu_system_gops > 0 else 0

        print("\n" + "=" * 70)
        print("                              CPU RESULTS                             ")
        print("=" * 70)
        print(f"Average Latency  : {avg_cpu:.3f} ms")
        print(f"Latency Range    : {min_cpu:.3f} ms - {max_cpu:.3f} ms")
        print(f"Latency StdDev   : {std_cpu:.3f} ms")
        print(f"Speedup vs CPU   : {speedup:.2f}x")
        print("=" * 70 + "\n")
    else:
        print("\nSkipping CPU comparison for INT8 (PyTorch CPU does not natively support direct eager i8 matmul).\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel NPU MatMul Stress & Benchmark Test")
    parser.add_argument("--size", type=int, default=None, help="Matrix dimension (NxN). Overrides --m, --n, --k if set.")
    parser.add_argument("--m", type=int, default=2048, help="Matrix M dimension (Rows of A)")
    parser.add_argument("--n", type=int, default=2048, help="Matrix N dimension (Cols of A / Rows of B)")
    parser.add_argument("--k", type=int, default=2048, help="Matrix K dimension (Cols of B)")
    parser.add_argument("--iters", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "int8"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--turbo", action="store_true", help="Enable Intel Level Zero hardware Turbo boost mode")
    parser.add_argument("--sda", action="store_true", help="Enable Shared Device Address (SDA) zero-copy transfers")
    parser.add_argument("--num-streams", type=int, default=1, help="Number of pipelined execution streams")

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
        num_streams=args.num_streams
    )
