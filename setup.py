"""Root setup.py forwarding to intel_npu_lib/setup.py.

Enables seamless installation (`pip install .`, `pip install -e .`) directly
from the repository root without requiring users to navigate into `intel_npu_lib/`.
"""

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(here, "intel_npu_lib")

if not os.path.isdir(lib_dir):
    raise FileNotFoundError(f"Could not find library directory: {lib_dir}")

sys.path.insert(0, lib_dir)
os.chdir(lib_dir)

setup_path = os.path.join(lib_dir, "setup.py")
with open(setup_path, "rb") as f:
    code = compile(f.read(), setup_path, "exec")
    exec(code, {"__file__": setup_path, "__name__": "__main__"})
