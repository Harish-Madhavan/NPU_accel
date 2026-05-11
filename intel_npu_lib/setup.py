from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import sys
import site

# Helper to find sources
def get_sources():
    csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc')
    # Explicitly list sources to ensure order and inclusion
    sources = [
        os.path.join(csrc_dir, 'bindings.cpp'),
        os.path.join(csrc_dir, 'device.cpp'),
        os.path.join(csrc_dir, 'ops.cpp'),
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
        print("OpenVINO not found in Python environment. Attempting to find via site-packages...")
        # Fallback: try to find in site-packages manually if import fails during build
        paths = site.getsitepackages() + [site.getusersitepackages()]
        ov_dir = None
        for p in paths:
            candidate = os.path.join(p, 'openvino')
            if os.path.exists(candidate):
                ov_dir = candidate
                break
        
        if ov_dir is None:
            print("WARNING: OpenVINO not found. Building without linking OpenVINO (Stub mode).")
            return [], [], []

    # Define search paths relative to package root
    # Structure varies by OS and version, so we search
    include_path = None
    lib_path = None
    
    # Search for include directory containing 'openvino/openvino.hpp'
    for root, dirs, files in os.walk(ov_dir):
        if 'include' in dirs:
            inc_candidate = os.path.join(root, 'include')
            if os.path.exists(os.path.join(inc_candidate, 'openvino', 'openvino.hpp')):
                include_path = inc_candidate
                break
    
    # Search for library directory
    # Windows: look for openvino.lib
    # Linux: look for libopenvino.so
    lib_name = 'openvino.lib' if os.name == 'nt' else 'libopenvino.so'
    
    for root, dirs, files in os.walk(ov_dir):
        if lib_name in files:
            lib_path = root
            break
    
    if include_path and lib_path:
        print(f"OpenVINO Include: {include_path}")
        print(f"OpenVINO Lib: {lib_path}")
        return [include_path], [lib_path], ['openvino']
    else:
        print(f"WARNING: Could not locate OpenVINO headers/libs within {ov_dir}")
        return [], [], []

# Locate OpenVINO
ov_include, ov_lib_dir, ov_libs = find_openvino()

setup(
    name='intel_npu_acceleration',
    version='0.1.0',
    description='PyTorch acceleration library for Intel NPU',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CppExtension(
            name='intel_npu_acceleration._C',
            sources=get_sources(),
            extra_compile_args={'cxx': ['/std:c++17'] if os.name == 'nt' else ['-std=c++17']},
            include_dirs=ov_include,
            library_dirs=ov_lib_dir,
            libraries=ov_libs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
        'openvino>=2024.0.0'
    ],
)
