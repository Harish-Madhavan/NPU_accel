"""
Common helpers for NPU examples — shared timing, seeding, and device utilities.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def timer(msg: str = "") -> Iterator[dict]:
    """Context manager for timing blocks with perf_counter."""
    data: dict = {}
    t0 = time.perf_counter()
    yield data
    data["elapsed"] = time.perf_counter() - t0
    if msg:
        print(f"{msg}: {data['elapsed']:.3f}s")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def benchmark_fn(fn, warmup: int = 3, iters: int = 10):
    """Benchmark a function with warmup and return latencies in ms."""
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies
