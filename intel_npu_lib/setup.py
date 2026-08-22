import os
import sys
import site

# Ensure DISTUTILS_USE_SDK and C++20 flags are set on Windows for MSVC
if sys.platform == "win32":
    os.environ["DISTUTILS_USE_SDK"] = "1"
    cxxflags = os.environ.get("CXXFLAGS", "")
    if "/std:c++" not in cxxflags:
        os.environ["CXXFLAGS"] = f"{cxxflags} /std:c++20 /MP /DNOMINMAX".strip()
    cflags = os.environ.get("CFLAGS", "")
    if "/std:c++" not in cflags:
        os.environ["CFLAGS"] = f"{cflags} /std:c++20 /MP /DNOMINMAX".strip()

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


# Helper to find sources
def get_sources():
    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
    # Explicitly list sources to ensure order and inclusion
    sources = [
        os.path.join(csrc_dir, "bindings.cpp"),
        os.path.join(csrc_dir, "device.cpp"),
        os.path.join(csrc_dir, "ops.cpp"),
    ]
    return sources


def find_openvino():
    """
    Attempts to locate OpenVINO include and library paths from the python environment.
    """
    try:
        import openvino

        ov_dir = os.path.dirname(openvino.__file__)
        print(f"Found OpenVINO package at: {ov_dir}")
    except ImportError:
        print(
            "OpenVINO not found in Python environment. Attempting to find via site-packages..."
        )
        # Fallback: try to find in site-packages manually if import fails during build
        paths = site.getsitepackages() + [site.getusersitepackages()]
        ov_dir = None
        for p in paths:
            candidate = os.path.join(p, "openvino")
            if os.path.exists(candidate):
                ov_dir = candidate
                break

        if ov_dir is None:
            print(
                "WARNING: OpenVINO not found. Building without linking OpenVINO (Stub mode)."
            )
            return [], [], []

    # Define search paths relative to package root
    # Structure varies by OS and version, so we search
    include_path = None
    lib_path = None

    # Search for include directory containing 'openvino/openvino.hpp'
    for root, dirs, files in os.walk(ov_dir):
        if "include" in dirs:
            inc_candidate = os.path.join(root, "include")
            if os.path.exists(os.path.join(inc_candidate, "openvino", "openvino.hpp")):
                include_path = inc_candidate
                break

    # Search for library directory
    # Windows: look for openvino.lib
    # Linux: look for libopenvino.so
    lib_name = "openvino.lib" if os.name == "nt" else "libopenvino.so"

    for root, dirs, files in os.walk(ov_dir):
        if lib_name in files:
            lib_path = root
            break

    if include_path and lib_path:
        print(f"OpenVINO Include: {include_path}")
        print(f"OpenVINO Lib: {lib_path}")
        return [include_path], [lib_path], ["openvino"]
    else:
        print(f"WARNING: Could not locate OpenVINO headers/libs within {ov_dir}")
        return [], [], []


# Locate OpenVINO
ov_include, ov_lib_dir, ov_libs = find_openvino()

class NPUBuildExtension(BuildExtension):
    """Custom build_ext to ensure C++20 standard flags are always passed on MSVC.

    On Windows, distutils/MSVC does not honor the dict-style
    ``extra_compile_args = {"cxx": [...]}`` that PyTorch documents for
    CppExtension.  It only reads a flat ``list[str]``.  This subclass:

    1. Normalises every extension's ``extra_compile_args`` from dict → list.
    2. Injects ``/std:c++20`` (and helpers) into the compiler's own
       ``compile_options`` so the flag is present even if setuptools
       rebuilds the option list after our mutation.
    """

    def build_extensions(self):
        # ── 1. Normalise extra_compile_args on every extension ──────────
        for ext in self.extensions:
            if isinstance(ext.extra_compile_args, dict):
                # Merge all values (typically only "cxx") into a flat list
                flat = []
                for flags in ext.extra_compile_args.values():
                    flat.extend(flags)
                ext.extra_compile_args = flat
            elif ext.extra_compile_args is None:
                ext.extra_compile_args = []

            if os.name == "nt":
                for flag in ["/std:c++20", "/O2", "/MP", "/DNOMINMAX"]:
                    if flag not in ext.extra_compile_args:
                        ext.extra_compile_args.append(flag)
            else:
                for flag in ["-std=c++20", "-O3"]:
                    if flag not in ext.extra_compile_args:
                        ext.extra_compile_args.append(flag)

        # ── 2. Patch the compiler object directly (MSVC) ────────────────
        if os.name == "nt":
            # self.compiler is initialised by the time build_extensions runs
            if hasattr(self.compiler, "compile_options"):
                for flag in ["/std:c++20", "/DNOMINMAX"]:
                    if flag not in self.compiler.compile_options:
                        self.compiler.compile_options.append(flag)
            if hasattr(self.compiler, "compile_options_debug"):
                for flag in ["/std:c++20", "/DNOMINMAX"]:
                    if flag not in self.compiler.compile_options_debug:
                        self.compiler.compile_options_debug.append(flag)

        super().build_extensions()


setup(
    name="intel_npu_acceleration",
    version="0.1.0",
    description="PyTorch acceleration library for Intel NPU",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            name="intel_npu_acceleration._C",
            sources=get_sources(),
            extra_compile_args={
                "cxx": ["/std:c++20", "/O2", "/MP", "/DNOMINMAX"]
                if os.name == "nt"
                else ["-std=c++20", "-O3"]
            },
            include_dirs=ov_include,
            library_dirs=ov_lib_dir,
            libraries=ov_libs,
        )
    ],
    cmdclass={"build_ext": NPUBuildExtension.with_options(use_ninja=False)},
    install_requires=["torch", "openvino>=2024.0.0"],
    entry_points={
        "torch_dynamo_backends": [
            "npu = intel_npu_acceleration.frontend.dynamo:_compile_backend",
            "intel_npu = intel_npu_acceleration.frontend.dynamo:_compile_backend",
        ]
    },
    extras_require={
        "dev": ["pytest", "ruff"],
    },
)
