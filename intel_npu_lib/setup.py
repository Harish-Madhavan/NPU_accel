"""Build script for the Intel NPU C++ extension (``intel_npu_acceleration._C``).

Single-source-of-truth policy: package name, version, dependencies,
entry-points, and package layout all live in ``pyproject.toml``. This file
only defines *how* to build the native extension.

Environment knobs:
    NPU_NO_BUILD_EXT=1   Skip the C++ extension (pure-Python install).
    NPU_VERBOSE_BUILD=1  Echo OpenVINO discovery details.
    OPENVINO_DIR / INTEL_OPENVINO_DIR / OPENVINO_PACKAGE_DIR
                         Extra hints for locating OpenVINO headers/libs.
"""

import os
import site
import sys

# Ensure C++20 flags are set on Windows for MSVC
if sys.platform == "win32":
    os.environ["DISTUTILS_USE_SDK"] = "1"
    # --- MSVC cl.exe environment variables ---
    # CL:  prepended to every cl.exe invocation (read by cl.exe itself).
    # _CL_: appended to every cl.exe invocation.
    # These bypass all Python-level flag plumbing (distutils, setuptools,
    # PyTorch BuildExtension) and are the *only* reliable way to guarantee
    # that /std:c++20 reaches the compiler on Windows CI.
    _msvc_inject = "/std:c++20 /DNOMINMAX"
    for var in ("CL", "_CL_"):
        existing = os.environ.get(var, "")
        if "/std:c++" not in existing:
            os.environ[var] = f"{existing} {_msvc_inject}".strip()

from setuptools import setup


def get_sources():
    """Return the ordered list of C++ sources for the ``_C`` extension.

    Paths MUST be '/'-separated and relative to this directory: modern
    setuptools rejects absolute paths (``editable_wheel`` / ``bdist_wheel``
    fail with "setup script specifies an absolute path"), even though the
    legacy ``setup.py build_ext --inplace`` path tolerates them.
    """
    sources = [
        "csrc/bindings.cpp",
        "csrc/device.cpp",
        "csrc/ops.cpp",
    ]
    here = os.path.dirname(os.path.abspath(__file__))
    missing = [s for s in sources if not os.path.isfile(os.path.join(here, s))]
    if missing:
        raise FileNotFoundError(f"C++ sources missing under {here}: {missing}")
    return sources


def _verbose() -> bool:
    return os.environ.get("NPU_VERBOSE_BUILD", "0") == "1"


def _candidate_openvino_roots():
    """Yield OpenVINO install roots, cheapest checks first."""
    # 1. Explicit env-var hints (toolkit installs, custom sysroots).
    for var in ("OPENVINO_DIR", "INTEL_OPENVINO_DIR", "OPENVINO_PACKAGE_DIR"):
        hint = os.environ.get(var)
        if hint and os.path.isdir(hint):
            yield hint
    # 2. Imported python package (pip install openvino).
    try:
        import openvino  # noqa: PLC0415

        ov_dir = os.path.dirname(openvino.__file__)
        print(f"Found OpenVINO package at: {ov_dir}")
        yield ov_dir
    except ImportError:
        print("OpenVINO not importable; scanning site-packages as fallback...")
        try:
            paths = site.getsitepackages() + [site.getusersitepackages()]
        except Exception:
            paths = []
        for p in paths:
            candidate = os.path.join(p, "openvino")
            if os.path.isdir(candidate):
                yield candidate


def _find_in_root(root: str):
    """Look for headers/libs under known sub-layouts, then bounded walk."""
    header_rel = os.path.join("openvino", "openvino.hpp")
    lib_name = "openvino.lib" if os.name == "nt" else "libopenvino.so"

    # Known layouts first (avoids a full tree walk in the common case).
    known_inc = [
        os.path.join(root, "include"),
        os.path.join(root, "runtime", "include"),
        os.path.join(root, "share", "openvino"),
    ]
    include_path = next(
        (c for c in known_inc if os.path.isfile(os.path.join(c, header_rel))),
        None,
    )
    known_lib = [
        os.path.join(root, "libs"),
        os.path.join(root, "lib"),
        os.path.join(root, "runtime", "lib", "intel64"),
        root,
    ]
    lib_path = next(
        (c for c in known_lib if os.path.isfile(os.path.join(c, lib_name))),
        None,
    )
    if include_path and lib_path:
        return include_path, lib_path

    # Bounded fallback walk (max depth keeps `pip install` responsive).
    max_depth = 5
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        # Prune obviously irrelevant subtrees.
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "tests", "test"}]
        if include_path is None and os.path.isfile(
            os.path.join(dirpath, "include", header_rel)
            if os.path.basename(dirpath) != "include"
            else os.path.join(dirpath, header_rel)
        ):
            cand = (
                dirpath
                if os.path.basename(dirpath) == "include"
                else os.path.join(dirpath, "include")
            )
            if os.path.isfile(os.path.join(cand, header_rel)):
                include_path = cand
        if lib_path is None and lib_name in filenames:
            lib_path = dirpath
        if include_path and lib_path:
            return include_path, lib_path
    if include_path and lib_path:
        return include_path, lib_path
    return None, None


def find_openvino():
    """Locate OpenVINO include/lib dirs; return (includes, lib_dirs, libs).

    Returns empty lists (stub mode) when OpenVINO cannot be found so a
    pure-CPU fallback build can still proceed.
    """
    for root in _candidate_openvino_roots():
        inc, lib = _find_in_root(root)
        if inc and lib:
            print(f"OpenVINO Include: {inc}")
            print(f"OpenVINO Lib: {lib}")
            return [inc], [lib], ["openvino"]
        if _verbose():
            print(f"OpenVINO layout not recognised under: {root}")
    print(
        "WARNING: OpenVINO headers/libs not found. "
        "Building without linking OpenVINO (stub/CPU-fallback mode). "
        "Set OPENVINO_DIR or `pip install openvino>=2024.0.0` for full NPU support."
    )
    return [], [], []


ov_include, ov_lib_dir, ov_libs = find_openvino()


class NPUBuildExtension:
    """Lazy import wrapper so ``--help``/metadata queries don't need torch."""

    @staticmethod
    def make_cmdclass():
        from torch.utils.cpp_extension import BuildExtension  # noqa: PLC0415

        class _NPUBuildExtension(BuildExtension):
            """Ensure C++20 flags reach MSVC (flat list, not dict-style).

            On Windows, distutils/MSVC does not honor the dict-style
            ``extra_compile_args = {"cxx": [...]}`` that PyTorch documents for
            CppExtension. This subclass normalises to a flat list and patches
            the compiler object directly.
            """

            def build_extensions(self):
                for ext in self.extensions:
                    if isinstance(ext.extra_compile_args, dict):
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

                if os.name == "nt":
                    if hasattr(self.compiler, "compile_options"):
                        for flag in ["/std:c++20", "/DNOMINMAX"]:
                            if flag not in self.compiler.compile_options:
                                self.compiler.compile_options.append(flag)
                    if hasattr(self.compiler, "compile_options_debug"):
                        for flag in ["/std:c++20", "/DNOMINMAX"]:
                            if flag not in self.compiler.compile_options_debug:
                                self.compiler.compile_options_debug.append(flag)

                super().build_extensions()

        return _NPUBuildExtension.with_options(use_ninja=False)


import shutil  # noqa: E402

ext_modules = []
cmdclass = {}
skip_reason = None

if os.environ.get("NPU_NO_BUILD_EXT", "0") == "1":
    skip_reason = "NPU_NO_BUILD_EXT=1 is set"
elif os.name == "nt" and shutil.which("cl") is None:
    skip_reason = (
        "MSVC cl.exe not on PATH — install 'Desktop development with C++' "
        "or run from a VS Developer Prompt / build_npu.bat"
    )

if skip_reason is not None:
    print(f"WARNING: skipping C++ extension ({skip_reason}). Installing pure-Python fallback.")
else:
    from torch.utils.cpp_extension import CppExtension  # noqa: E402

    ext_modules = [
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
    ]
    cmdclass = {"build_ext": NPUBuildExtension.make_cmdclass()}

# NOTE: name/version/description/dependencies/packages/entry-points are all
# declared in pyproject.toml — do not duplicate them here.
setup(ext_modules=ext_modules, cmdclass=cmdclass)
