#!/usr/bin/env python3
"""Cross-platform automated installer for Intel NPU Acceleration Library.

Usage:
    python install.py [OPTIONS]

Options:
    --clean             Clean previous build artifacts before installing.
    --pure-python       Install in pure-Python mode without building C++ extension.
    --no-install        Build the C++ extension in-place only, skip pip install.
    --test              Run full pytest test suite after installation.
    --skip-deps         Skip checking or installing dependencies.
    --verbose           Display detailed build and compiler logs.
    --help              Display this help message.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def print_banner(msg: str) -> None:
    print("\n" + "=" * 65)
    print(f" {msg}")
    print("=" * 65)


def run_cmd(cmd: list[str] | str, verbose: bool = False, env: dict[str, str] | None = None) -> bool:
    try:
        shell = isinstance(cmd, str)
        if verbose:
            subprocess.check_call(cmd, shell=shell, env=env)
        else:
            subprocess.check_output(cmd, shell=shell, env=env, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        if not verbose and e.output:
            try:
                print(e.output.decode("utf-8", errors="replace"))
            except Exception:
                pass
        return False


def check_python_version() -> None:
    if sys.version_info < (3, 10):  # noqa: UP036
        print(f"[ERROR] Python 3.10+ is required. Detected: {sys.version.split()[0]}")
        sys.exit(1)
    print(f"[OK] Python version: {sys.version.split()[0]}")


def ensure_dependencies(skip: bool = False, verbose: bool = False) -> None:
    if skip:
        print("[INFO] Skipping dependency checks (--skip-deps).")
        return

    # Check / install PyTorch
    try:
        import torch

        print(f"[OK] PyTorch detected: {torch.__version__}")
    except ImportError:
        print("[INFO] PyTorch not detected. Installing lightweight CPU PyTorch wheel...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"]
        if not run_cmd(cmd, verbose=verbose):
            print("[ERROR] Failed to install PyTorch. Please install manually: pip install torch")
            sys.exit(1)
        print("[OK] PyTorch installed successfully.")

    # Check / install OpenVINO & NumPy
    try:
        import numpy
        import openvino

        print(f"[OK] OpenVINO detected: {openvino.__version__}")
    except ImportError:
        print("[INFO] Installing OpenVINO and NumPy dependencies...")
        cmd = [sys.executable, "-m", "pip", "install", "openvino>=2024.0.0", "numpy>=1.24"]
        if not run_cmd(cmd, verbose=verbose):
            print("[ERROR] Failed to install OpenVINO. Please install manually: pip install openvino numpy")
            sys.exit(1)
        print("[OK] OpenVINO installed successfully.")


def check_compiler(is_pure_python: bool) -> bool:
    if is_pure_python:
        print("[INFO] Pure-Python mode selected (--pure-python). Skipping C++ compiler checks.")
        return False

    if sys.platform == "win32":
        if shutil.which("cl"):
            print("[OK] Microsoft Visual C++ compiler (cl.exe) found on PATH.")
            return True
        vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        if os.path.isfile(vswhere):
            print("[OK] Visual Studio installer detected. Setup will auto-configure MSVC environment.")
            return True
        print("[WARN] No MSVC compiler detected. Falling back to pure-Python installation.")
        return False
    else:
        # Linux / Unix
        if shutil.which("g++") or shutil.which("clang++"):
            print("[OK] C++ compiler found (g++ / clang++).")
            return True
        print("[WARN] No C++ compiler (g++ / clang++) found. Falling back to pure-Python installation.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated cross-platform installer for Intel NPU Acceleration Library."
    )
    parser.add_argument("--clean", action="store_true", help="Clean build directories before compiling.")
    parser.add_argument("--pure-python", action="store_true", help="Install pure-Python fallback (no C++ build).")
    parser.add_argument("--no-install", action="store_true", help="Build extension in-place only, skip pip install.")
    parser.add_argument("--test", action="store_true", help="Run full pytest test suite after install.")
    parser.add_argument("--skip-deps", action="store_true", help="Skip checking or installing prerequisites.")
    parser.add_argument("--verbose", action="store_true", help="Display verbose compiler/pip output.")

    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(here, "intel_npu_lib")

    print_banner("Intel NPU Acceleration — Automated Setup & Installation")

    # 1. Check Python
    check_python_version()

    # 2. Check / Install dependencies
    ensure_dependencies(skip=args.skip_deps, verbose=args.verbose)

    # 3. Check C++ Compiler
    has_compiler = check_compiler(args.pure_python)
    pure_python = args.pure_python or (not has_compiler)

    env = os.environ.copy()
    if pure_python:
        env["NPU_NO_BUILD_EXT"] = "1"
        print("[INFO] Build Mode: Pure-Python (CPU fallback enabled)")
    else:
        print("[INFO] Build Mode: Native oneAPI Level Zero C++ Extension")

    # 4. Clean artifacts if requested
    if args.clean:
        print("[INFO] Cleaning build artifacts...")
        build_dir = os.path.join(lib_dir, "build")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)
        for root, _, files in os.walk(lib_dir):
            for file in files:
                if file.endswith((".pyd", ".so", ".dylib")):
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception:
                        pass
        print("[OK] Clean completed.")

    # 5. Build C++ extension inplace (if not pure-python)
    if not pure_python:
        print("[INFO] Compiling native C++ extension...")
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        try:
            cwd = lib_dir
            if args.verbose:
                subprocess.check_call(cmd, cwd=cwd, env=env)
            else:
                subprocess.check_output(cmd, cwd=cwd, env=env, stderr=subprocess.STDOUT)
            print("[OK] Native C++ extension built successfully.")
        except subprocess.CalledProcessError as e:
            print("[WARN] C++ compilation failed. Retrying with pure-Python fallback...")
            env["NPU_NO_BUILD_EXT"] = "1"
            pure_python = True

    # 6. Install package in editable mode
    if not args.no_install:
        print("[INFO] Installing 'intel_npu_acceleration' via pip...")
        cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", lib_dir]
        if not run_cmd(cmd, verbose=args.verbose, env=env):
            print("[ERROR] pip installation failed.")
            sys.exit(1)
        print("[OK] Package installed successfully in development mode.")

    # 7. Post-install Self-Test
    print_banner("Running Post-Installation Hardware Verification")
    try:
        import intel_npu_acceleration as npu

        npu.self_test(verbose=True)
    except Exception as e:
        print(f"[WARN] Self-test failed to run: {e}")

    # 8. Optional pytest suite
    if args.test:
        print_banner("Running Automated Pytest Test Suite")
        subprocess.call([sys.executable, "-m", "pytest"], cwd=lib_dir)

    print_banner("Installation & Verification Complete!")
    print(" You can now run:")
    print("   python -m intel_npu_acceleration --info       # View system & hardware status")
    print("   python -m intel_npu_acceleration --self-test   # Run hardware self-test")
    print("   python examples/smollm2.py                    # Run SmolLM2-135M on Intel NPU\n")


if __name__ == "__main__":
    main()
