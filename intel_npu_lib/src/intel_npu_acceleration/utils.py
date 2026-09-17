"""
Shared utilities for Intel NPU Acceleration Library.

Centralizes common helpers to eliminate duplication across modules.
"""

__all__ = ["clean_name"]

import re
from typing import Any


def clean_name(name: str, strip_leading: bool = True) -> str:
    """Normalize operator/placeholder names for robust matching.

    Strips trailing numeric/symbol suffixes and lowercases for
    case-insensitive comparison. FX-tracer artifacts (leading ``L__``)
    are stripped only when ``strip_leading`` is True — Dynamo/JIT graphs
    use ``l_``-prefixed placeholder names where the prefix is significant.

    Example:
        'L__args_0_' -> 'args'
        'l_x_0' with strip_leading=False -> 'l_x'
    """
    if strip_leading:
        name = re.sub(r"^[lL]__?", "", name)
    return re.sub(r"[\d._]+$", "", name).lower()
