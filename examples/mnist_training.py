import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import intel_npu_acceleration as npu
import time
import argparse

# --- Model Definition ---

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Standard PyTorch design representing pure compatibility.
        # FX and Dynamo trace these layers and optimize them onto the Intel NPU.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    print(f"\n[Epoch {epoch}] Training on Hybrid NPU (Forward) + CPU (Backward)...")
    start_time = time.time()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"  Batch {batch_idx:03d}/{len(train_loader)} | "
                f"Progress: [{batch_idx * len(data):5d}/{len(train_loader.dataset):5d}] | "
                f"Loss: {loss.item():.6f}"
            )
    
    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch} finished in {elapsed:.2f}s | Avg Loss: {avg_loss:.6f}")


def run_latency_benchmark(model, npu_model, test_set, num_samples=200):
    """
    Run a fair, head-to-head latency comparison on CPU vs NPU.
    Both targets run sequentially with batch_size=1.
    """
    print(f"\nRunning Latency Benchmark (sequential execution on {num_samples} single images)...")
    
    # 1. CPU Latency Benchmark
    model.eval()
    cpu_latencies = []
    with torch.no_grad():
        for i in range(num_samples):
            img, _ = test_set[i]
            img = img.unsqueeze(0)  # Shape (1, 1, 28, 28)
            
            t0 = time.time()
            _ = model(img)
            t1 = time.time()
            if i >= 10:  # Warmup
                cpu_latencies.append((t1 - t0) * 1000.0)  # ms
                
    avg_cpu_lat = sum(cpu_latencies) / len(cpu_latencies)
    
    # 2. NPU Latency Benchmark
    npu_latencies = []
    with torch.no_grad():
        for i in range(num_samples):
            img, _ = test_set[i]
            img = img.unsqueeze(0)
            
            t0 = time.time()
            _ = npu_model(img)
            t1 = time.time()
            if i >= 10:  # Warmup
                npu_latencies.append((t1 - t0) * 1000.0)  # ms
                
    avg_npu_lat = sum(npu_latencies) / len(npu_latencies)
    
    print(f"  CPU Sequential Latency: {avg_cpu_lat:.3f} ms / image")
    print(f"  NPU Sequential Latency: {avg_npu_lat:.3f} ms / image")
    return avg_cpu_lat, avg_npu_lat


def run_throughput_benchmark(model, npu_model, test_loader_large, test_loader_single, max_batches=10):
    """
    Measure throughput in images / second.
    CPU runs with large batch size (highly parallelized).
    NPU runs with multi-stream asynchronous pipelining for maximum VPU core saturation.
    """
    print(f"\nRunning Throughput Benchmark...")
    
    # 1. CPU Throughput
    model.eval()
    cpu_images_processed = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader_large):
            if batch_idx >= max_batches:
                break
            _ = model(data)
            cpu_images_processed += data.size(0)
    cpu_duration = time.time() - t0
    cpu_throughput = cpu_images_processed / cpu_duration if cpu_duration > 0 else 0
    
    # 2. NPU Throughput
    npu_images_processed = 0
    t0 = time.time()
    num_requests_to_run = max_batches * test_loader_large.batch_size
    
    if hasattr(npu_model, "infer_async") and hasattr(npu_model, "wait_async") and getattr(npu_model, "num_streams", 1) > 1:
        from collections import deque
        active_handles = deque()
        iterator = iter(test_loader_single)
        
        with torch.no_grad():
            # Fill the multi-stream execution pipeline
            for _ in range(min(npu_model.num_streams, num_requests_to_run)):
                try:
                    data, _ = next(iterator)
                    handle = npu_model.infer_async(data)
                    active_handles.append(handle)
                    npu_images_processed += 1
                except StopIteration:
                    break
                    
            # Interleave wait and submit to keep streams saturated
            while active_handles:
                oldest_handle = active_handles.popleft()
                _ = npu_model.wait_async(oldest_handle)
                
                if npu_images_processed < num_requests_to_run:
                    try:
                        data, _ = next(iterator)
                        handle = npu_model.infer_async(data)
                        active_handles.append(handle)
                        npu_images_processed += 1
                    except StopIteration:
                        pass
    else:
        # Fallback to synchronous execution
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader_single):
                if batch_idx >= num_requests_to_run:
                    break
                _ = npu_model(data)
                npu_images_processed += 1
                
    npu_duration = time.time() - t0
    npu_throughput = npu_images_processed / npu_duration if npu_duration > 0 else 0
    
    print(f"  CPU Throughput (Batch={test_loader_large.batch_size}): {cpu_throughput:.2f} images/sec")
    print(f"  NPU Throughput (Streams={getattr(npu_model, 'num_streams', 1)}): {npu_throughput:.2f} images/sec")
    return cpu_throughput, npu_throughput


def evaluate_accuracy(model, npu_model, test_loader):
    print("\nEvaluating Accuracy...")
    
    # CPU Accuracy
    model.eval()
    cpu_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            cpu_correct += pred.eq(target.view_as(pred)).sum().item()
            
    cpu_acc = 100.0 * cpu_correct / len(test_loader.dataset)
    
    # NPU Accuracy (expects input on CPU, compiles/offloads internally)
    npu_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Evaluate using NPU compiled model
            # To evaluate correctly on batch size 1:
            for i in range(data.size(0)):
                img = data[i:i+1]
                tgt = target[i:i+1]
                output = npu_model(img)
                pred = output.argmax(dim=1, keepdim=True)
                npu_correct += pred.eq(tgt.view_as(pred)).sum().item()
                
    npu_acc = 100.0 * npu_correct / len(test_loader.dataset)
    
    print(f"  CPU Test Accuracy: {cpu_acc:.2f}%")
    print(f"  NPU Test Accuracy: {npu_acc:.2f}%")
    return cpu_acc, npu_acc


def main():
    parser = argparse.ArgumentParser(description="Intel NPU Accelerated MNIST Training & Inference")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate for Adadelta")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--compiler",
        type=str,
        default="native",
        choices=["native", "custom"],
        help="Backend compiler: native (torch.compile) or custom (npu.compile)",
    )
    parser.add_argument(
        "--performance-hint",
        type=str,
        default="LATENCY",
        choices=["LATENCY", "THROUGHPUT"],
        help="NPU performance optimization target",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=1,
        help="Number of pipelined execution streams for THROUGHPUT mode",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Enable Intel Level Zero hardware Turbo boost mode",
    )
    parser.add_argument(
        "--sda",
        action="store_true",
        help="Enable Shared Device Address (SDA) zero-copy transfers",
    )
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print("==========================================================")
    print("   Intel NPU Acceleration: MNIST Hybrid Compiler Demo   ")
    print("==========================================================")
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    print("\n[Step 1] Loading MNIST dataset...")
    train_set = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # CPU accuracy/throughput test loader
    test_loader_cpu = torch.utils.data.DataLoader(test_set, batch_size=100)
    # Sequential NPU test loader
    test_loader_npu = torch.utils.data.DataLoader(test_set, batch_size=1)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    print(f"\n[Step 2] Training model on CPU for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

    print("\n[Step 3] Compiling trained model for Intel NPU...")
    model.eval()
    example_input = torch.randn(1, 1, 28, 28)

    if not npu.is_available():
        print("\n[Warning] Intel NPU hardware not detected! Compiler will target CPU fallback.")
    else:
        print("[Info] Intel NPU detected. Performing Level Zero ahead-of-time graph compilation...")

    # Apply global configurations
    if args.turbo:
        print("[Info] Enabling Level Zero Turbo Mode...")
        npu.enable_turbo()
    if args.sda:
        print("[Info] Enabling Level Zero Shared Device Address (SDA)...")
        npu.enable_sda()

    t_comp_start = time.time()
    if args.compiler == "native":
        print("[Info] Running native PyTorch 2.x compilation backend (torch.compile)...")
        options = {
            "clone_outputs": True,
            "performance_hint": args.performance_hint,
            "num_streams": args.num_streams,
            "preprocess_config": {
                "input": {
                    "element_type": "f32"
                }
            }
        }
        npu_model = torch.compile(model, backend="npu", options=options)
        # Warm up compilation on dummy input to measure compiled graph generation time fairly
        _ = npu_model(example_input)
    else:
        print("[Info] Running custom ahead-of-time compiler (npu.compile)...")
        npu_model = npu.compile(
            model,
            example_input,
            performance_hint=args.performance_hint,
            num_streams=args.num_streams,
            preprocess_config={
                "input": {
                    "element_type": "f32"
                }
            }
        )
    t_comp_end = time.time()
    print(f"Compilation succeeded in {t_comp_end - t_comp_start:.2f}s (Cache-optimized).")

    # Run benchmarks
    cpu_lat, npu_lat = run_latency_benchmark(model, npu_model, test_set, num_samples=100)
    cpu_thr, npu_thr = run_throughput_benchmark(model, npu_model, test_loader_cpu, test_loader_npu, max_batches=5)
    
    # Evaluate accuracy on a subset to keep execution time fast (e.g. 500 images)
    subset_indices = torch.arange(500)
    test_subset = torch.utils.data.Subset(test_set, subset_indices)
    subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=50)
    cpu_acc, npu_acc = evaluate_accuracy(model, npu_model, subset_loader)

    # Print beautiful summary table
    lat_speedup = cpu_lat / npu_lat if npu_lat > 0 else 0
    thr_speedup = npu_thr / cpu_thr if cpu_thr > 0 else 0
    
    print("\n" + "=" * 70)
    print("                INTEL NPU VS CPU MNIST PERFORMANCE SUMMARY               ")
    print("=" * 70)
    print(f"| {'Benchmark Metric':<26} | {'CPU Target':<16} | {'NPU Target':<16} | {'Speedup':<8} |")
    print("-" * 70)
    print(f"| {'Latency (Batch=1)':<26} | {f'{cpu_lat:.2f} ms/img':<16} | {f'{npu_lat:.2f} ms/img':<16} | {f'{lat_speedup:.2f}x':<8} |")
    print(f"| {'Throughput (Parallel/Seq)':<26} | {f'{cpu_thr:.1f} img/s':<16} | {f'{npu_thr:.1f} img/s':<16} | {f'{thr_speedup:.2f}x':<8} |")
    print(f"| {'Model Accuracy (Subset)':<26} | {f'{cpu_acc:.2f}%':<16} | {f'{npu_acc:.2f}%':<16} | {'-':<8} |")
    print("=" * 70)
    print("Note: Latency is sequential single-image delay (lower is better).")
    print("Throughput compares parallelized CPU execution vs. sequential NPU execution.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
