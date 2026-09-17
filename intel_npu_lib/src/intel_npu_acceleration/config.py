"""
Central configuration and constants for Intel NPU Acceleration Library.

All magic numbers, thresholds, and default values are centralized here
to enable consistent tuning, environment overrides, and testability.
"""

__all__ = ["NPUConfig", "config"]

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class NPUConfig:
    # Graph cache
    max_graph_cache_size: int = field(default_factory=lambda: _env_int("NPU_MAX_GRAPH_CACHE_SIZE", 100))
    cache_throttle_seconds: float = field(default_factory=lambda: _env_float("NPU_CACHE_THROTTLE_SEC", 60.0))
    disk_cache_max_size_mb: int = field(default_factory=lambda: _env_int("NPU_DISK_CACHE_MAX_MB", 2048))
    disk_cache_max_files: int = field(default_factory=lambda: _env_int("NPU_DISK_CACHE_MAX_FILES", 10000))
    disk_cache_cleanup_size_mb: int = field(default_factory=lambda: _env_int("NPU_DISK_CLEANUP_MB", 1024))
    disk_cache_cleanup_files: int = field(default_factory=lambda: _env_int("NPU_DISK_CLEANUP_FILES", 500))

    # Execution thresholds
    leased_buffer_threshold_elements: int = field(default_factory=lambda: _env_int("NPU_LEASED_BUFFER_THRESHOLD", 100_000))
    copy_threshold_elements: int = field(default_factory=lambda: _env_int("NPU_COPY_THRESHOLD", 2_000_000))

    # Dynamic bucketing
    default_bucket_sizes: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    # Compilation defaults
    default_performance_hint: str = "LATENCY"
    default_precision: str = "auto"
    default_num_streams: int = 1


# Global singleton config
config = NPUConfig()
