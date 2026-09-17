import os
import sys

# Ensure library is importable when run directly from repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "intel_npu_lib", "src")))

import argparse
import time

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import openvino.properties as ov_props
import openvino.properties.hint as ov_hints


def main():
    parser = argparse.ArgumentParser(description="Vanilla OpenVINO MatMul Benchmark")
    parser.add_argument("--size", type=int, default=2048, help="Matrix dimension (N x N)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type")
    parser.add_argument("--device", type=str, default="NPU", help="Target device")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num-streams", type=int, default=1, help="Number of execution streams")
    parser.add_argument("--performance-hint", type=str, default="LATENCY", choices=["LATENCY", "THROUGHPUT"], help="Performance hint")
    args = parser.parse_args()

    print("======================================================================")
    print("             VANILLA OPENVINO MATMUL PERFORMANCE BENCHMARK            ")
    print("======================================================================")
    print(f"OpenVINO Version : {ov.__version__}")
    print(f"Matrix Shape     : ({args.size}x{args.size}) x ({args.size}x{args.size}) -> ({args.size}x{args.size})")
    print(f"Data Type        : {args.dtype}")
    print(f"Target Device    : {args.device}")
    print(f"Iterations       : {args.iterations} (Warmup: {args.warmup})")
    print(f"Streams          : {args.num_streams}")
    print(f"Performance Hint : {args.performance_hint}")
    print("----------------------------------------------------------------------")

    # 1. Build the OpenVINO MatMul Graph Natively
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    ov_type = ov.Type.f16 if args.dtype == "float16" else ov.Type.f32

    param_a = ops.parameter([args.size, args.size], ov_type, name="A")
    param_b = ops.parameter([args.size, args.size], ov_type, name="B")
    matmul = ops.matmul(param_a, param_b, transpose_a=False, transpose_b=False)

    model = ov.Model([matmul.output(0)], [param_a, param_b], "Vanilla_MatMul")

    # 2. Compile Model
    core = ov.Core()
    config = {}

    try:
        hint = ov_hints.PerformanceMode.THROUGHPUT if args.performance_hint == "THROUGHPUT" else ov_hints.PerformanceMode.LATENCY
        config[ov_hints.performance_mode()] = hint
        config[ov_hints.inference_precision()] = ov_type
        if args.num_streams > 1:
            config[ov_props.streams.num()] = args.num_streams
    except Exception:
        config = {
            "PERFORMANCE_HINT": args.performance_hint,
            "INFERENCE_PRECISION_HINT": "f16" if args.dtype == "float16" else "f32"
        }
        if args.num_streams > 1:
            config["NUM_STREAMS"] = str(args.num_streams)

    print("Compiling model...")
    start_comp = time.time()
    compiled_model = core.compile_model(model, args.device, config)
    print(f"Compilation finished in {time.time() - start_comp:.2f} seconds.")

    # 3. Create Inputs & Remote/Host Tensors
    infer_requests = [compiled_model.create_infer_request() for _ in range(args.num_streams)]

    # Pre-allocate contiguous inputs to avoid overhead
    input_a = np.random.randn(args.size, args.size).astype(np_dtype)
    input_b = np.random.randn(args.size, args.size).astype(np_dtype)

    # 4. Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        req = infer_requests[0]
        req.set_tensor("A", ov.Tensor(input_a))
        req.set_tensor("B", ov.Tensor(input_b))
        req.infer()

    # 5. Benchmark Execution
    print(f"Running benchmark ({args.iterations} iterations)...")
    latencies = []

    if args.num_streams > 1:
        # Multi-stream Pipelined Asynchronous Benchmark
        start_time = time.time()

        # Schedule initial requests
        in_flight = 0
        req_idx = 0

        for _i in range(args.iterations):
            req = infer_requests[req_idx]
            req.set_tensor("A", ov.Tensor(input_a))
            req.set_tensor("B", ov.Tensor(input_b))

            req.start_async()
            in_flight += 1
            req_idx = (req_idx + 1) % args.num_streams

            if in_flight >= args.num_streams:
                # Wait for the oldest request to finish
                wait_idx = (req_idx - args.num_streams) % args.num_streams
                infer_requests[wait_idx].wait()
                in_flight -= 1

        # Wait for remaining requests
        for req in infer_requests:
            req.wait()

        total_time = time.time() - start_time
        avg_latency_ms = (total_time / args.iterations) * 1000.0
    else:
        # Single-stream Synchronous Benchmark
        for _ in range(args.iterations):
            req = infer_requests[0]
            req.set_tensor("A", ov.Tensor(input_a))
            req.set_tensor("B", ov.Tensor(input_b))

            t0 = time.time()
            req.infer()
            latencies.append((time.time() - t0) * 1000.0)

        avg_latency_ms = np.mean(latencies)

    # 6. Calculate Throughput
    # Matrix Multiplications require 2 * N^3 operations
    ops_per_matmul = 2 * (args.size ** 3)
    gops_per_matmul = ops_per_matmul / 1e9

    if args.num_streams > 1:
        system_throughput_gops = (gops_per_matmul * args.iterations) / total_time
    else:
        system_throughput_gops = gops_per_matmul / (avg_latency_ms / 1000.0)

    print("======================================================================")
    print("                               RESULTS                                ")
    print("======================================================================")
    print(f"Average Latency  : {avg_latency_ms:.3f} ms")
    if args.num_streams == 1:
        print(f"Latency Range    : {np.min(latencies):.3f} ms - {np.max(latencies):.3f} ms")
        print(f"Latency StdDev   : {np.std(latencies):.3f} ms (Jitter: {np.std(latencies)/avg_latency_ms*100.0:.2f}%)")
    else:
        print(f"Total Test Time  : {total_time:.3f} seconds")
    print(f"System Throughput: {system_throughput_gops:.2f} GOPS ({system_throughput_gops/1000.0:.4f} TOPS)")
    print("======================================================================")

if __name__ == "__main__":
    main()
