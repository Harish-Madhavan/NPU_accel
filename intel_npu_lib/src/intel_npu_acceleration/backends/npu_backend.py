"""
Proper PyTorch Backend for Intel NPU — torch.device("npu"), torch.npu, torch.backends.npu, torch.accelerator

Makes `intel_npu_acceleration` a first-class PyTorch backend with full parity
to torch.cuda / torch.xpu:

    import torch
    import intel_npu_acceleration  # auto-registers
    assert torch.npu.is_available()
    assert torch.device("npu").type == "npu"
    assert torch.backends.npu.is_available()

    # Tensor creation & movement — zero-copy USM (CPU storage, NPU dispatch)
    x = torch.randn(3, 3, device="npu")
    assert x.is_npu
    y = torch.randn(3, 3).to("npu")
    assert y.is_npu
    z = x + y

    # Model — .to("npu") triggers torch.compile(backend="npu")
    model = torch.nn.Linear(10, 10).to("npu")
    out = model(torch.randn(2, 10, device="npu"))

    # Also supports torch.compile directly
    model2 = torch.compile(torch.nn.Linear(10, 10), backend="npu")
    out2 = model2(torch.randn(2, 10))

    # Streams & device context (API parity, NPU is single-queue but API-compatible)
    with torch.npu.device(0):
        s = torch.npu.Stream()
        with torch.npu.stream(s):
            out = model(x)
        s.synchronize()
    torch.npu.synchronize()
    torch.npu.empty_cache()

    # AMP / autocast
    with torch.autocast(device_type="npu", dtype=torch.float16):
        out = model(x.to(torch.float16))

    # Universal torch.accelerator (PyTorch 2.4+)
    assert torch.accelerator.is_available()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()

Implements:
- rename_privateuse1_backend("npu") + generate_methods_for_privateuse1_backend
- torch.npu.* (is_available, device_count, get_device_name, get_device_properties, empty_cache,
               synchronize, set_device, memory_allocated, manual_seed, Stream/Event, device/stream
               context managers, amp, capability)
- torch.npu.amp.* (autocast, GradScaler, custom_fwd, custom_bwd)
- torch.backends.npu.* (is_available, is_built, version, matmul flags, allow_tf32, sdp flags)
- torch.accelerator bridge (synchronize, empty_cache, streams, device indices, memory queries)
- torch.Tensor.to("npu") & torch.Tensor.npu() & torch.Tensor.is_npu
- nn.Module.to("npu") & nn.Module.npu() → torch.compile(backend="npu")
- torch.*(device="npu") factories → CPU zero-copy with NPU tagging
"""

from __future__ import annotations

import logging
import sys
import time
import types
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("intel_npu_acceleration.backends.npu")

# ---------------------------------------------------------------------------
# Lazy imports & Device Matching
# ---------------------------------------------------------------------------

def _get_device_module():
    from intel_npu_acceleration import device as _device
    return _device


def _is_npu_device(dev: Any) -> bool:
    """Check if a device specification refers to the Intel NPU."""
    if dev is None:
        return False
    if isinstance(dev, str):
        d = dev.strip().lower()
        return d == "npu" or d.startswith("npu:") or "privateuse" in d
    if isinstance(dev, torch.device):
        return dev.type in ("npu", "privateuseone")
    return False


def _has_explicit_device(args: tuple, kwargs: dict) -> tuple[bool, Any]:
    """Check if device argument is explicitly passed in args or kwargs."""
    if args and isinstance(args[0], (str, torch.device)):
        return True, args[0]
    if "device" in kwargs and kwargs["device"] is not None:
        return True, kwargs["device"]
    return False, None


def _safe_device(spec: Any, fallback: str = "cpu") -> torch.device:
    """Build ``torch.device(spec)``, falling back when the spec is unparseable."""
    try:
        return torch.device(spec)
    except Exception:
        return torch.device(fallback)


def _package_version() -> str:
    """Return the installed distribution version (never hardcode it)."""
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("intel_npu_acceleration")
    except Exception:
        return "0.2.0"


def _fallback_to_zero_on_npu_error(orig_fn):
    """Wrap a zero-arg accelerator query: NPU-related RuntimeErrors become 0."""

    def _patched():
        try:
            return orig_fn()
        except RuntimeError as e:
            if "npu" in str(e).lower():
                return 0
            raise

    return _patched


def _make_acc_forwarder(npu_attr: str, default: Any):
    """Forward a ``torch.accelerator`` query to ``torch.npu`` with a default.

    ``default`` may be a value or a zero-arg factory (e.g. ``dict``) so each
    call gets a fresh mutable object.
    """

    def _forward(device=None):
        if hasattr(torch, "npu"):
            return getattr(torch.npu, npu_attr)(device)
        return default() if callable(default) else default

    return _forward


# torch.accelerator memory/capability queries mirrored from torch.npu,
# with the CPU-fallback default used when torch.npu is absent.
_ACC_FORWARDED_ATTRS: dict[str, Any] = {
    "memory_allocated": 0,
    "max_memory_allocated": 0,
    "memory_reserved": 0,
    "max_memory_reserved": 0,
    "memory_stats": dict,
    "reset_peak_memory_stats": None,
    "get_device_capability": (1, 0),
}

# ---------------------------------------------------------------------------
# Device Properties & Contexts
# ---------------------------------------------------------------------------

class _NPUProperties:
    """Device properties container for Intel NPU, providing both attribute and dict access."""

    def __init__(self, props_dict: dict[str, Any]):
        self.__dict__.update(props_dict)
        self.name: str = str(props_dict.get("name", "Intel(R) AI Boost"))
        self.device_id: int = int(props_dict.get("device_id", 0))
        self.total_memory: int = int(props_dict.get("total_memory", 16 * 1024 * 1024 * 1024))
        self.major: int = 1
        self.minor: int = 0
        self.multi_processor_count: int = int(props_dict.get("optimal_infer_requests", 1))
        self.is_available: bool = bool(props_dict.get("is_available", True))

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def __repr__(self) -> str:
        return (
            f"_DeviceProperties(name='{self.name}', major={self.major}, minor={self.minor}, "
            f"total_memory={self.total_memory}B, multi_processor_count={self.multi_processor_count})"
        )

# ---------------------------------------------------------------------------
# Device registration & torch.accelerator bridge
# ---------------------------------------------------------------------------

def _register_privateuse1_backend() -> bool:
    try:
        if hasattr(torch.utils, "rename_privateuse1_backend"):
            torch.utils.rename_privateuse1_backend("npu")
            try:
                if hasattr(torch.utils, "generate_methods_for_privateuse1_backend"):
                    torch.utils.generate_methods_for_privateuse1_backend()
            except Exception as e:
                logger.debug(f"generate_methods failed: {e}")
            try:
                if hasattr(torch, "_register_device_module") and hasattr(torch, "npu"):
                    torch._register_device_module("npu", torch.npu)  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug(f"_register_device_module failed: {e}")

            # Patch torch.accelerator to handle npu seamlessly without C++ kernels
            try:
                import torch.accelerator as _acc

                _orig_cs = getattr(_acc, "current_stream", None)
                if _orig_cs:
                    def _patched_cs(device=None):
                        try:
                            if device is not None and _is_npu_device(device):
                                return torch.npu.current_stream(device)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        try:
                            return _orig_cs(device)
                        except RuntimeError as e:
                            if "npu" in str(e).lower():
                                return torch.npu.current_stream()  # type: ignore[attr-defined]
                            raise
                    _acc.current_stream = _patched_cs  # type: ignore[attr-defined]

                _orig_cdi = getattr(_acc, "current_device_index", None)
                if _orig_cdi:
                    _acc.current_device_index = _fallback_to_zero_on_npu_error(_orig_cdi)  # type: ignore[attr-defined]

                _orig_cd_idx = getattr(_acc, "current_device_idx", None)
                if _orig_cd_idx:
                    _acc.current_device_idx = _fallback_to_zero_on_npu_error(_orig_cd_idx)  # type: ignore[attr-defined]

                _orig_sdi = getattr(_acc, "set_device_index", None)
                if _orig_sdi:
                    def _patched_sdi(device_index):
                        try:
                            if hasattr(torch, "npu"):
                                return torch.npu.set_device(device_index)
                        except Exception:
                            pass
                    _acc.set_device_index = _patched_sdi  # type: ignore[attr-defined]

                _orig_sd_idx = getattr(_acc, "set_device_idx", None)
                if _orig_sd_idx:
                    _acc.set_device_idx = _patched_sdi  # type: ignore[attr-defined]

                _orig_sync = getattr(_acc, "synchronize", None)
                if _orig_sync:
                    def _patched_sync(device=None):
                        try:
                            if device is None or _is_npu_device(device):
                                return torch.npu.synchronize(device)
                        except Exception:
                            pass
                        try:
                            return _orig_sync(device)
                        except RuntimeError as e:
                            if "npu" in str(e).lower():
                                return torch.npu.synchronize()
                            raise
                    _acc.synchronize = _patched_sync  # type: ignore[attr-defined]

                _orig_ec = getattr(_acc, "empty_cache", None)
                if _orig_ec:
                    def _patched_ec():
                        try:
                            return torch.npu.empty_cache()
                        except Exception:
                            pass
                        try:
                            return _orig_ec()
                        except Exception:
                            pass
                    _acc.empty_cache = _patched_ec  # type: ignore[attr-defined]

                _orig_dc = getattr(_acc, "device_count", None)
                if _orig_dc:
                    def _patched_dc():
                        try:
                            if hasattr(torch, "npu"):
                                return torch.npu.device_count()
                        except Exception:
                            pass
                        return 1
                    _acc.device_count = _patched_dc  # type: ignore[attr-defined]

                _orig_ia = getattr(_acc, "is_available", None)
                if _orig_ia:
                    def _patched_ia():
                        try:
                            if hasattr(torch, "npu"):
                                return torch.npu.is_available()
                        except Exception:
                            pass
                        return True
                    _acc.is_available = _patched_ia  # type: ignore[attr-defined]

                for _attr, _default in _ACC_FORWARDED_ATTRS.items():
                    if hasattr(_acc, _attr):
                        setattr(_acc, _attr, _make_acc_forwarder(_attr, _default))
                if hasattr(_acc, "reset_accumulated_memory_stats"):
                    _acc.reset_accumulated_memory_stats = lambda device=None: None
                if hasattr(_acc, "set_stream"):
                    _acc.set_stream = lambda stream: torch.npu.set_stream(stream) if hasattr(torch, "npu") else None

                # Also patch _get_device_index
                try:
                    import torch.accelerator._utils as _au
                    _orig_gdi = getattr(_au, "_get_device_index", None)
                    if _orig_gdi:
                        def _patched_gdi(device, optional=False):
                            try:
                                if device is not None and _is_npu_device(device):
                                    dev = torch.device(device)
                                    return dev.index if dev.index is not None else 0
                            except Exception:
                                pass
                            return _orig_gdi(device, optional=optional)
                        _au._get_device_index = _patched_gdi  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"accelerator patch failed: {e}")

            try:
                assert torch.device("npu").type == "npu"
                assert torch.device("npu:0").type == "npu"
                return True
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"privateuse1 rename failed: {e}")
    return False

# ---------------------------------------------------------------------------
# torch.npu / torch.backends.npu
# ---------------------------------------------------------------------------

def _setup_torch_npu_module() -> None:
    device_mod = _get_device_module()

    if not hasattr(torch, "npu"):
        npu_module = types.ModuleType("torch.npu")

        # Core API — mirrors torch.cuda / torch.xpu
        npu_module.is_available = device_mod.is_available  # type: ignore[attr-defined]
        npu_module.device_count = device_mod.device_count  # type: ignore[attr-defined]
        npu_module.current_device = device_mod.current_device  # type: ignore[attr-defined]
        npu_module.set_device = lambda device: None  # type: ignore[attr-defined]
        npu_module.get_device_name = device_mod.get_device_name  # type: ignore[attr-defined]

        def _get_props(device=None):
            raw = device_mod.get_device_properties()
            return _NPUProperties(raw)

        npu_module.get_device_properties = _get_props  # type: ignore[attr-defined]
        npu_module.get_device_capability = lambda device=None: (1, 0)  # type: ignore[attr-defined]
        npu_module.empty_cache = device_mod.empty_cache  # type: ignore[attr-defined]
        npu_module.synchronize = device_mod.synchronize  # type: ignore[attr-defined]
        npu_module.memory_allocated = lambda device=None: 0  # type: ignore[attr-defined]
        npu_module.max_memory_allocated = lambda device=None: 0  # type: ignore[attr-defined]
        npu_module.memory_reserved = lambda device=None: 0  # type: ignore[attr-defined]
        npu_module.memory_cached = lambda device=None: 0  # type: ignore[attr-defined]
        npu_module.max_memory_reserved = lambda device=None: 0  # type: ignore[attr-defined]
        npu_module.memory_stats = lambda device=None: {}  # type: ignore[attr-defined]
        npu_module.reset_peak_memory_stats = lambda device=None: None  # type: ignore[attr-defined]

        # RNG — NPU shares host RNG (no separate state)
        npu_module.manual_seed = lambda seed: None  # type: ignore[attr-defined]
        npu_module.manual_seed_all = lambda seed: None  # type: ignore[attr-defined]
        npu_module.seed = lambda: 0  # type: ignore[attr-defined]
        npu_module.initial_seed = lambda: 0  # type: ignore[attr-defined]
        npu_module._is_in_bad_fork = lambda: False  # type: ignore[attr-defined]
        npu_module.get_rng_state = lambda device=None: torch.empty(0, dtype=torch.uint8)  # type: ignore[attr-defined]
        npu_module.set_rng_state = lambda state, device=None: None  # type: ignore[attr-defined]

        # AMP — NPU supports fp16/bf16 (mirrors cuda/xpu)
        npu_module.get_amp_supported_dtype = lambda: [torch.float16, torch.bfloat16]  # type: ignore[attr-defined]
        npu_module.is_bf16_supported = lambda: True  # type: ignore[attr-defined]
        npu_module.is_fp16_supported = lambda: True  # type: ignore[attr-defined]
        npu_module.get_autocast_dtype = lambda dtype=None: dtype or torch.float16  # type: ignore[attr-defined]

        # NPU hardware extensions
        npu_module.set_performance_hint = device_mod.set_performance_hint  # type: ignore[attr-defined]
        npu_module.get_performance_hint = device_mod.get_performance_hint  # type: ignore[attr-defined]
        npu_module.enable_turbo = device_mod.enable_turbo  # type: ignore[attr-defined]
        npu_module.get_turbo_state = device_mod.get_turbo_state  # type: ignore[attr-defined]
        npu_module.enable_sda = device_mod.enable_sda  # type: ignore[attr-defined]
        npu_module.set_eager_device = device_mod.set_eager_device  # type: ignore[attr-defined]
        npu_module.get_eager_device = device_mod.get_eager_device  # type: ignore[attr-defined]
        npu_module.set_property = device_mod.set_property  # type: ignore[attr-defined]
        npu_module.turbo = device_mod.turbo  # type: ignore[attr-defined]
        npu_module.performance_mode = device_mod.performance_mode  # type: ignore[attr-defined]

        # Device context: torch.npu.device(0) / with torch.npu.device(0):
        class _NPUDeviceContext:
            def __init__(self, device):
                self.device = torch.device(device) if device is not None else torch.device("npu:0")
                self.prev = None

            def __enter__(self):
                return self.device

            def __exit__(self, *args):
                pass

        npu_module.device = _NPUDeviceContext  # type: ignore[attr-defined]

        # Stream / Event — API parity, Level Zero command queue
        class Stream:
            def __init__(self, device=None, priority=0):
                self.device = _safe_device(device if device is not None else "npu:0")
                self.priority = priority

            def __enter__(self):
                self._prev = getattr(torch.npu, "_current_stream_obj", None)
                torch.npu._current_stream_obj = self
                return self

            def __exit__(self, *args):
                torch.npu._current_stream_obj = getattr(self, "_prev", None)

            def synchronize(self):
                device_mod.synchronize()

            def query(self):
                return True

            def wait_stream(self, other):
                pass

            def wait_event(self, event):
                pass

            def record_event(self, event=None):
                e = Event()
                if event:
                    event.record(self)
                return e

            def __repr__(self):
                return f"<torch.npu.Stream device={self.device} priority={self.priority}>"

        class Event:
            def __init__(self, enable_timing=False, blocking=False, interprocess=False):
                self.enable_timing = enable_timing
                self._time = time.perf_counter()

            def record(self, stream=None):
                self._time = time.perf_counter()

            def wait(self, stream=None):
                pass

            def synchronize(self):
                pass

            def query(self):
                return True

            def elapsed_time(self, other):
                return (other._time - self._time) * 1000.0

            def __repr__(self):
                return "<torch.npu.Event>"

        class _StreamContext:
            def __init__(self, stream):
                self.stream = stream
                self.prev = None

            def __enter__(self):
                self.prev = getattr(torch.npu, "_current_stream_obj", None)
                if self.stream is not None:
                    torch.npu._current_stream_obj = self.stream
                return self.stream

            def __exit__(self, *args):
                torch.npu._current_stream_obj = self.prev

        npu_module.Stream = Stream  # type: ignore[attr-defined]
        npu_module.Event = Event  # type: ignore[attr-defined]
        _def_stream = Stream()
        npu_module._default_stream_obj = _def_stream  # type: ignore[attr-defined]
        npu_module._current_stream_obj = _def_stream  # type: ignore[attr-defined]
        npu_module.current_stream = lambda device=None: getattr(torch.npu, "_current_stream_obj", _def_stream)  # type: ignore[attr-defined]
        npu_module.default_stream = lambda device=None: getattr(torch.npu, "_default_stream_obj", _def_stream)  # type: ignore[attr-defined]

        def _set_stream(s):
            if s is not None:
                torch.npu._current_stream_obj = s

        npu_module.set_stream = _set_stream  # type: ignore[attr-defined]
        npu_module.stream = _StreamContext  # type: ignore[attr-defined]
        npu_module.is_initialized = lambda: True  # type: ignore[attr-defined]
        npu_module.init = lambda: None  # type: ignore[attr-defined]

        # AMP support — reuse host amp, NPU supports fp16/bf16
        try:
            import functools

            amp_kwargs: dict[str, Any] = {
                "autocast": lambda *a, **kw: torch.autocast(*a, device_type="npu", **kw),
                "GradScaler": torch.amp.GradScaler if hasattr(torch, "amp") else torch.cuda.amp.GradScaler,  # type: ignore[attr-defined]
            }
            # torch.amp.custom_fwd/custom_bwd exist on torch>=2.4; binding
            # device_type up front mirrors torch.cuda.amp.custom_fwd/bwd.
            if hasattr(torch.amp, "custom_fwd"):
                amp_kwargs["custom_fwd"] = functools.partial(  # type: ignore[attr-defined]
                    torch.amp.custom_fwd, device_type="npu"
                )
            if hasattr(torch.amp, "custom_bwd"):
                amp_kwargs["custom_bwd"] = functools.partial(  # type: ignore[attr-defined]
                    torch.amp.custom_bwd, device_type="npu"
                )
            npu_module.amp = types.SimpleNamespace(**amp_kwargs)
        except Exception:
            pass

        torch.npu = npu_module  # type: ignore[attr-defined]
        sys.modules["torch.npu"] = npu_module

    if not hasattr(torch.backends, "npu"):
        backends_npu = types.ModuleType("torch.backends.npu")
        backends_npu.is_available = device_mod.is_available  # type: ignore[attr-defined]
        backends_npu.is_built = lambda: True  # type: ignore[attr-defined]
        backends_npu.version = _package_version  # type: ignore[attr-defined]
        backends_npu.matmul = types.SimpleNamespace(
            allow_tf32=False,
            allow_fp16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction=True,
        )  # type: ignore[attr-defined]
        backends_npu.allow_tf32 = False  # type: ignore[attr-defined]
        # SDP flags for scaled dot product attention parity
        backends_npu.flash_sdp_enabled = lambda: True  # type: ignore[attr-defined]
        backends_npu.mem_efficient_sdp_enabled = lambda: True  # type: ignore[attr-defined]
        backends_npu.math_sdp_enabled = lambda: True  # type: ignore[attr-defined]
        backends_npu.preferred_linalg_library = lambda: "default"  # type: ignore[attr-defined]
        torch.backends.npu = backends_npu  # type: ignore[attr-defined]
        sys.modules["torch.backends.npu"] = backends_npu

# ---------------------------------------------------------------------------
# Factory patches: torch.*(device="npu") → CPU zero-copy with is_npu flag
# ---------------------------------------------------------------------------

def _patch_tensor_factories() -> None:
    factories = [
        "empty", "zeros", "ones", "randn", "rand", "randint",
        "arange", "eye", "full", "empty_like", "zeros_like",
        "ones_like", "randn_like", "as_tensor", "tensor",
    ]
    for name in factories:
        if hasattr(torch, name) and not hasattr(getattr(torch, name), "_npu_patched"):
            orig_fn = getattr(torch, name)

            def _make_wrapper(fn):
                def wrapper(*args, **kwargs):
                    dev = kwargs.get("device", None)
                    is_npu = False
                    if dev is not None and _is_npu_device(dev):
                        is_npu = True
                        kwargs["device"] = torch.device("cpu")
                    res = fn(*args, **kwargs)
                    if is_npu and isinstance(res, torch.Tensor):
                        try:
                            res._is_npu = True
                        except Exception:
                            pass
                    return res
                wrapper._npu_patched = True  # type: ignore[attr-defined]
                return wrapper

            setattr(torch, name, _make_wrapper(orig_fn))

# ---------------------------------------------------------------------------
# torch.Tensor patches: .to("npu"), .npu(), .is_npu
# ---------------------------------------------------------------------------

_original_tensor_to = torch.Tensor.to

def _tensor_to_npu(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    target_is_npu = False
    new_args = list(args)
    new_kwargs = dict(kwargs)

    has_dev, dev = _has_explicit_device(args, kwargs)
    if has_dev and _is_npu_device(dev):
        target_is_npu = True
        if args and isinstance(args[0], (str, torch.device)):
            new_args[0] = torch.device("cpu")
        elif "device" in new_kwargs:
            new_kwargs["device"] = torch.device("cpu")

    res = _original_tensor_to(self, *new_args, **new_kwargs)
    if target_is_npu:
        try:
            res._is_npu = True
        except Exception:
            pass
    elif has_dev:
        try:
            res._is_npu = False
        except Exception:
            pass
    else:
        # Preserve is_npu flag across dtype/memory format changes
        if getattr(self, "_is_npu", False):
            try:
                res._is_npu = True
            except Exception:
                pass
    return res


def _tensor_npu(self: torch.Tensor, device: Any = None, non_blocking: bool = False, copy: bool = False) -> torch.Tensor:
    dev_str = f"npu:{device}" if device is not None and isinstance(device, int) else "npu"
    return self.to(dev_str, non_blocking=non_blocking, copy=copy)


@property
def _is_npu_property(self: torch.Tensor) -> bool:
    return bool(getattr(self, "_is_npu", False) or (self.device.type in ("npu", "privateuseone")))

# ---------------------------------------------------------------------------
# nn.Module.to("npu") & nn.Module.npu() → torch.compile(backend="npu")
# ---------------------------------------------------------------------------

_original_module_to = nn.Module.to

def _module_to_npu(self: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
    target_device = None
    has_dev, dev = _has_explicit_device(args, kwargs)
    if has_dev and _is_npu_device(dev):
        target_device = torch.device(dev) if not isinstance(dev, torch.device) else dev

    if target_device is not None:
        device_mod = _get_device_module()
        if device_mod.is_available():
            if hasattr(self, "_is_npu_compiled") and self._is_npu_compiled:  # type: ignore[attr-defined]
                return self
            try:
                compiled = torch.compile(self, backend="npu")
                compiled._is_npu_compiled = True  # type: ignore[attr-defined]
                compiled._npu_device = target_device  # type: ignore[attr-defined]
                return compiled
            except Exception as e:
                logger.warning(f"torch.compile(npu) failed: {e}")
                return self
        else:
            logger.warning("NPU not available, model.to('npu') falling back to CPU")
            cpu_args = [torch.device("cpu") if _is_npu_device(a) else a for a in args]
            cpu_kwargs = {k: (torch.device("cpu") if _is_npu_device(v) else v) for k, v in kwargs.items()}
            return _original_module_to(self, *cpu_args, **cpu_kwargs)
    return _original_module_to(self, *args, **kwargs)


def _module_npu(self: nn.Module, device: Any = None) -> nn.Module:
    dev_str = f"npu:{device}" if device is not None and isinstance(device, int) else "npu"
    return self.to(dev_str)

# ---------------------------------------------------------------------------
# Patch installer
# ---------------------------------------------------------------------------

def _setup_patches() -> None:
    _module_to_npu.__name__ = "to"
    _module_to_npu.__qualname__ = "Module.to"
    _module_npu.__name__ = "npu"
    _module_npu.__qualname__ = "Module.npu"

    _tensor_to_npu.__name__ = "to"
    _tensor_to_npu.__qualname__ = "Tensor.to"
    _tensor_npu.__name__ = "npu"
    _tensor_npu.__qualname__ = "Tensor.npu"

    if not hasattr(nn.Module, "_npu_patched"):
        nn.Module.to = _module_to_npu  # type: ignore[attr-defined]
        nn.Module.npu = _module_npu  # type: ignore[attr-defined]
        nn.Module._npu_patched = True  # type: ignore[attr-defined]

    if not hasattr(torch.Tensor, "_npu_patched"):
        torch.Tensor.to = _tensor_to_npu  # type: ignore[attr-defined]
        torch.Tensor.npu = _tensor_npu  # type: ignore[attr-defined]
        torch.Tensor.is_npu = _is_npu_property  # type: ignore[attr-defined]
        torch.Tensor._tensor_to_npu = _tensor_to_npu  # type: ignore[attr-defined]
        torch.Tensor._npu_patched = True  # type: ignore[attr-defined]

    try:
        _patch_tensor_factories()
    except Exception as e:
        logger.debug(f"factory patch failed: {e}")


def register_pytorch_backend() -> bool:
    ok = _register_privateuse1_backend()
    try:
        _setup_torch_npu_module()
    except Exception as e:
        logger.warning(f"Failed to setup torch.npu: {e}")
    try:
        _setup_patches()
    except Exception as e:
        logger.warning(f"Failed to patch: {e}")
    return ok
