"""
Intel NPU Acceleration CLI Entry Point.
Allows running `python -m intel_npu_acceleration` to inspect diagnostics, manage cache, and verify runtime status.
"""

import sys
import argparse
import intel_npu_acceleration as npu
from intel_npu_acceleration.info import print_info, get_system_info


def main():
    parser = argparse.ArgumentParser(
        prog="python -m intel_npu_acceleration",
        description="Intel NPU Acceleration Command-Line Interface and Diagnostics Tool.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print comprehensive system, hardware, and cache diagnostic report (default).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear compiled NPU models from the disk cache directory.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show package and OpenVINO runtime version info.",
    )

    args = parser.parse_args()

    if args.version:
        info = get_system_info()
        print(f"intel_npu_acceleration (PyTorch: {info['pytorch_version']}, OpenVINO: {info['openvino_version']})")
        sys.exit(0)

    if args.clear_cache:
        cache_dir = npu.get_cache_dir()
        size_bytes, count = npu.get_cache_size()
        size_mb = round(size_bytes / (1024 * 1024), 2)
        npu.clear_cache()
        npu.clear_graph_cache()
        print(f"[Intel NPU] Cleared cache directory '{cache_dir}': removed {count} files ({size_mb} MB).")
        sys.exit(0)

    # Default action: print diagnostic info
    print_info()


if __name__ == "__main__":
    main()

