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
    """Custom build_ext to ensure C++20 standard flags are always passed on MSVC."""
    def build_extensions(self):
        if os.name == "nt" and hasattr(self.compiler, "compile_options"):
            if "/std:c++20" not in self.compiler.compile_options:
                self.compiler.compile_options.append("/std:c++20")
        for ext in self.extensions:
            if isinstance(ext.extra_compile_args, dict):
                cxx_flags = ext.extra_compile_args.get("cxx", [])
            elif isinstance(ext.extra_compile_args, list):
                cxx_flags = ext.extra_compile_args
            else:
                cxx_flags = []

            if os.name == "nt":
                for flag in ["/std:c++20", "/O2", "/MP", "/DNOMINMAX"]:
                    if flag not in cxx_flags:
                        cxx_flags.append(flag)
                ext.extra_compile_args = {"cxx": cxx_flags}
            else:
                for flag in ["-std=c++20", "-O3"]:
                    if flag not in cxx_flags:
                        cxx_flags.append(flag)
                ext.extra_compile_args = cxx_flags

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
