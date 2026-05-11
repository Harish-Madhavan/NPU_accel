import torch
import time
import intel_npu_acceleration as npu_compiler
import argparse


class MatMulModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)


def run_benchmark(size=4096, iterations=50, warmup=10, dtype_str="float32"):
    print(f"Benchmarking NPU MatMul with size {size}x{size} using {dtype_str}...")

    model = MatMulModel()
    model.eval()

    # Inputs
    if dtype_str == "float16":
        dtype = torch.float16
        a = torch.randn(size, size, dtype=dtype)
        b = torch.randn(size, size, dtype=dtype)
    elif dtype_str == "int8":
        dtype = torch.int8
        a = torch.randint(-128, 127, (size, size), dtype=dtype)
        b = torch.randint(-128, 127, (size, size), dtype=dtype)
    else:
        dtype = torch.float32
        a = torch.randn(size, size, dtype=dtype)
        b = torch.randn(size, size, dtype=dtype)

    # Compile
    print("Compiling to NPU (First time)...")
    t0 = time.time()
    try:
        npu_model = npu_compiler.compile_to_npu(model, (a, b))
    except Exception as e:
        print(f"Compilation failed: {e}")
        return
    print(f"Compilation finished in {time.time() - t0:.2f}s")

    print("Compiling to NPU (Second time - Should be cached)...")
    t1 = time.time()
    npu_model = npu_compiler.compile_to_npu(model, (a, b))
    print(f"Second compilation finished in {time.time() - t1:.4f}s")

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = npu_model(a, b)

    # Benchmark
    print(f"Running stress test ({iterations} iterations)...")
    start_time = time.time()
    for i in range(iterations):
        _ = npu_model(a, b)
        # Optional: Print progress every 10%
        if iterations >= 10 and (i + 1) % (iterations // 10) == 0:
            print(f"Progress: {i + 1}/{iterations}")

    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iterations

    # OPS calculation: 2 * N^3 for matrix multiplication (NxN * NxN)
    ops = 2 * (size**3)
    flops = ops / avg_time
    gflops = flops / 1e9
    tflops = flops / 1e12

    print("\nResults:")
    print(f"Matrix Size: {size}x{size} ({dtype_str})")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Avg Latency: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {gflops:.2f} GOPS ({tflops:.4f} TOPS)")

    # CPU Comparison (Optional, for small sizes)
    if dtype_str != "int8":
        print("\nComparing with CPU (PyTorch)...")
        start_cpu = time.time()
        for _ in range(5):  # Run fewer iters for CPU
            torch.matmul(a, b)
        avg_cpu = (time.time() - start_cpu) / 5
        print(f"CPU Avg Latency: {avg_cpu * 1000:.2f} ms")
        print(f"Speedup vs CPU: {avg_cpu / avg_time:.2f}x")
    elif dtype_str == "int8":
        print("\nSkipping CPU comparison for int8.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel NPU MatMul Stress Test")
    parser.add_argument(
        "--size", type=int, default=2048, help="Matrix size (NxN). Default: 1024"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of benchmark iterations. Default: 50",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "int8"],
        help="Data type. Default: float32",
    )

    args = parser.parse_args()

    run_benchmark(size=args.size, iterations=args.iters, dtype_str=args.dtype)
