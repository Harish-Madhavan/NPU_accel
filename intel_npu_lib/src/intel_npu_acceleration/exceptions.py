"""
Centralized Exception Hierarchy for Intel NPU Acceleration Library.

This module defines standardized exceptions for device discovery, runtime execution,
graph compilation, and unsupported operation errors across the library.
"""

__all__ = [
    "NPUError",
    "NPURuntimeError",
    "NPUDeviceError",
    "NPUCompilationError",
    "NPUUnsupportedOpError",
]


class NPUError(Exception):
    """Base exception for all Intel NPU Acceleration errors.

    All custom exceptions raised by `intel_npu_acceleration` inherit from this base class.
    """

    pass


class NPURuntimeError(NPUError):
    """Exception raised during NPU eager kernel execution or runtime inference.

    Indicates an execution failure in the Level Zero driver, OpenVINO infer request,
    or C++ memory mapping.
    """
    pass


class NPUDeviceError(NPUError):
    """Exception raised for device probing, discovery, or initialization failures.

    Indicates that the Intel NPU hardware was not found, the driver is incompatible,
    or the oneAPI Level Zero context failed to initialize.
    """
    pass


class NPUCompilationError(NPUError):
    """Exception raised during graph compilation, tracing, or OpenVINO conversion.

    Indicates a failure when tracing PyTorch models with FX/Dynamo or compiling
    the resulting OpenVINO IR graph for the NPU.
    """
    pass


class NPUUnsupportedOpError(NPUError):
    """Exception raised when an operation cannot be executed or partitioned on NPU.

    Indicates an unsupported operator encountered in strict compilation mode.
    """
    pass
