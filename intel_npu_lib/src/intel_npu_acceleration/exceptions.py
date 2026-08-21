"""
Centralized Exception Hierarchy for Intel NPU Acceleration Library.
"""


class NPUError(Exception):
    """Base exception for all Intel NPU Acceleration errors."""
    pass


class NPURuntimeError(NPUError):
    """Exception raised during NPU eager kernel execution or runtime inference."""
    pass


class NPUDeviceError(NPUError):
    """Exception raised for device probing, discovery, or initialization failures."""
    pass


class NPUCompilationError(NPUError):
    """Exception raised during graph compilation, tracing, or OpenVINO conversion."""
    pass


class NPUUnsupportedOpError(NPUError):
    """Exception raised when an operation cannot be executed or partitioned on NPU."""
    pass
