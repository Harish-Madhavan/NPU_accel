#!/usr/bin/env bash
# =====================================================================
# build_npu.sh — Build and install Intel NPU Acceleration on Linux.
# Usage: ./build_npu.sh [--no-install] [--clean] [--pure-python]
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/intel_npu_lib"

NO_INSTALL=0
CLEAN=0
PURE_PYTHON=0

for arg in "$@"; do
    case "$arg" in
        --no-install) NO_INSTALL=1 ;;
        --clean) CLEAN=1 ;;
        --pure-python) PURE_PYTHON=1 ;;
        --help|-h)
            echo "Usage: $0 [--no-install] [--clean] [--pure-python]"
            exit 0
            ;;
    esac
done

echo "================================================================="
echo "   Intel NPU Acceleration — Linux Build & Installation"
echo "================================================================="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] 'python3' command not found. Install Python 3.10+ and retry."
    exit 1
fi

python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo "[ERROR] Python 3.10+ is required."
    exit 1
}

# Clean artifacts if requested
if [ "$CLEAN" -eq 1 ]; then
    echo "[INFO] Cleaning previous build artifacts..."
    rm -rf "${LIB_DIR}/build"
    find "${LIB_DIR}" -name "*.so" -delete
    echo "[OK] Clean complete."
fi

# Determine compiler and build mode
if [ "$PURE_PYTHON" -eq 1 ]; then
    echo "[INFO] Building in pure-Python mode (NPU_NO_BUILD_EXT=1)..."
    export NPU_NO_BUILD_EXT=1
else
    if command -v g++ &>/dev/null || command -v clang++ &>/dev/null; then
        echo "[OK] C++ compiler detected."
    else
        echo "[WARN] No C++ compiler (g++ / clang++) found. Falling back to pure-Python mode."
        export NPU_NO_BUILD_EXT=1
        PURE_PYTHON=1
    fi
fi

cd "${LIB_DIR}"

# Build C++ extension
if [ "$PURE_PYTHON" -ne 1 ]; then
    echo "[INFO] Building native C++ extension inplace..."
    python3 setup.py build_ext --inplace
    echo "[OK] C++ extension build complete."
fi

# Pip install
if [ "$NO_INSTALL" -ne 1 ]; then
    echo "[INFO] Installing package in development mode..."
    python3 -m pip install --no-build-isolation --no-deps -e .
    echo "[OK] Package installed."
fi

echo "================================================================="
echo " [SUCCESS] Intel NPU Acceleration successfully configured!"
echo "================================================================="
echo " Run diagnostics with:"
echo "   python3 -m intel_npu_acceleration --info"
echo "   python3 -m intel_npu_acceleration --self-test"
