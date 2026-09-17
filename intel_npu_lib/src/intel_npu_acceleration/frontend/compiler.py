"""Graph-mode compiler: TorchFX tracing -> OpenVINO IR -> NPU executables."""

__all__ = [
    "NPUTracer",
    "compile",
    "compile_to_npu",
    "export_openvino_ir",
    "clear_graph_cache",
]

import contextlib
import hashlib
import logging
import re
import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import openvino as ov
import openvino.properties as ov_props
import openvino.properties.hint as ov_hints
import torch
import torch.fx

import intel_npu_acceleration as npu_lib
from intel_npu_acceleration.config import config as npu_config
from intel_npu_acceleration.exceptions import NPUCompilationError

from .graph_module import NPUDynamicGraphModule, NPUGraphModule
from .transformations import (
    apply_pre_post_processing,
    fold_scalar_parameter_inputs,
    optimize_ov_model,
    reshape_model_inputs_to_static,
    serialize_openvino_model,
    transform_stateful_kv_cache,
)

logger = logging.getLogger(__name__)


class NPUTracer(torch.fx.Tracer):
    """Custom FX Tracer for Intel NPU."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from intel_npu_acceleration.functional import NPUStatefulKVCache, quantized_linear
        self._autowrap_function_ids.add(id(quantized_linear))
        self._stateful_kv_type = NPUStatefulKVCache

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, self._stateful_kv_type):
            return True
        if isinstance(m, torch.nn.Linear):
            weight = m._parameters.get("weight", None)
            if weight is not None and not isinstance(weight, torch.fx.Proxy):
                weight_data = getattr(weight, "data", None)
                if weight_data is not None and hasattr(weight_data, "dtype"):
                    if weight_data.dtype in [torch.int8, torch.uint8]:
                        return True
        return super().is_leaf_module(m, module_qualified_name)


# --- Global Cache for Compiled Graphs ---
_OV_CORE = None
_GRAPH_CACHE: OrderedDict = OrderedDict()
_MAX_GRAPH_CACHE_SIZE: int = npu_config.max_graph_cache_size
_GRAPH_CACHE_LOCK = threading.Lock()


def clear_graph_cache():
    """Clear all in-memory compiled model graphs from both Python and C++ backend caches."""
    global _GRAPH_CACHE
    with _GRAPH_CACHE_LOCK:
        _GRAPH_CACHE.clear()
    if npu_lib._C is not None and hasattr(npu_lib._C, "clear_cpp_model_cache"):
        try:
            npu_lib._C.clear_cpp_model_cache()
        except Exception:
            pass
    logger.info("In-memory NPU graph cache cleared.")


def _get_core():
    global _OV_CORE
    if _OV_CORE is None:
        _OV_CORE = ov.Core()

        # Disk model cache
        cache_dir = npu_lib.get_cache_dir()
        if cache_dir:
            logger.debug(f"Configuring Graph-mode cache dir: {cache_dir}")
            try:
                _OV_CORE.set_property({ov_props.cache_dir(): cache_dir})
            except Exception:
                pass
    return _OV_CORE


def _to_example_tuple(example_input: Any) -> tuple:
    """Normalize example input to a tuple."""
    if isinstance(example_input, torch.Tensor):
        return (example_input,)
    if isinstance(example_input, (tuple, list)):
        return tuple(example_input)
    return (example_input,)


def _to_converted_inputs(example_input_tuple: tuple) -> tuple:
    """Convert non-Tensor scalars to Tensors so torch.jit.trace works natively."""
    converted = []
    for val in example_input_tuple:
        if isinstance(val, torch.Tensor):
            converted.append(val)
        elif isinstance(val, bool):
            converted.append(torch.tensor(val, dtype=torch.bool))
        elif isinstance(val, int):
            converted.append(torch.tensor(val, dtype=torch.int64))
        elif isinstance(val, float):
            converted.append(torch.tensor(val, dtype=torch.float32))
        else:
            converted.append(val)
    return tuple(converted)


def _trace_model(model: torch.nn.Module) -> torch.fx.GraphModule:
    """Trace a model to a GraphModule (passthrough if already traced)."""
    if isinstance(model, torch.fx.GraphModule):
        return model
    return torch.fx.GraphModule(model, NPUTracer().trace(model))


def _sanitize_dynamo_inputs(example_input_tuple: tuple) -> tuple:
    """Sanitize SymInt / SymFloat / SymBool from PyTorch Dynamo dynamic shapes."""
    clean_inputs = []
    for val in example_input_tuple:
        if hasattr(val, "node") and not isinstance(val, torch.Tensor):
            try:
                clean_inputs.append(int(val))
            except Exception:
                try:
                    clean_inputs.append(float(val))
                except Exception:
                    clean_inputs.append(1)
        else:
            clean_inputs.append(val)
    return tuple(clean_inputs)


@contextlib.contextmanager
def _dequantized_linears(traced: torch.fx.GraphModule) -> Iterator[dict]:
    """Temporarily dequantize int8/uint8 Linear weights for OpenVINO conversion.

    Preserves per-channel scale / zero-point broadcasting semantics:
    Tensor scales are kept as Tensors (not forced to scalar).

    Yields the map of submodule -> original weight data so callers can be
    sure restoration happens even if conversion raises.
    """
    quantized_weights: dict = {}
    try:
        for submod in traced.modules():
            if isinstance(submod, torch.nn.Linear):
                weight = submod._parameters.get("weight", None)
                if weight is None or getattr(weight, "data", None) is None:
                    continue
                if weight.data.dtype == torch.int8:
                    quantized_weights[submod] = weight.data
                    scale = getattr(submod, "weight_scale", 1.0)
                    submod.weight.data = weight.data.float() * scale
                elif weight.data.dtype == torch.uint8:
                    quantized_weights[submod] = weight.data
                    scale = getattr(submod, "weight_scale", 1.0)
                    zp = getattr(submod, "weight_zero_point", 0.0)
                    w_u8 = weight.data
                    c_out = w_u8.shape[0]
                    w_odd = torch.floor_divide(w_u8, 16)
                    w_even = w_u8 - w_odd * 16
                    w_unpacked = torch.stack([w_even, w_odd], dim=-1).view(c_out, -1).float()
                    zp_val = zp.float() if isinstance(zp, torch.Tensor) else float(zp)
                    submod.weight.data = (w_unpacked - zp_val) * scale
        yield quantized_weights
    finally:
        for submod, w_data in quantized_weights.items():
            submod.weight.data = w_data


def _collect_stateful_modules(model: torch.nn.Module, traced: torch.fx.GraphModule) -> list:
    """Collect NPUStatefulKVCache submodules in topological order."""
    from intel_npu_acceleration.functional import NPUStatefulKVCache

    stateful_modules = []
    for node in traced.graph.nodes:
        if node.op == "call_module":
            submod = model
            try:
                for atom in node.target.split("."):
                    submod = getattr(submod, atom)
            except AttributeError:
                continue
            if isinstance(submod, NPUStatefulKVCache):
                stateful_modules.append(submod)
    return stateful_modules


def _convert_to_ov_model(model: torch.nn.Module, converted_inputs: tuple) -> Any:
    """Convert a PyTorch model via OpenVINO frontend.

    Suppresses the known `torch.jit.trace is deprecated` FutureWarning that
    OpenVINO emits internally during conversion.
    """
    from intel_npu_acceleration.functional import _compilation_context

    with _compilation_context():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="`torch.jit.trace` is deprecated.*",
                    category=FutureWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="`torch.jit.trace_method` is deprecated.*",
                    category=FutureWarning,
                )
                return ov.convert_model(model, example_input=converted_inputs)
        except Exception as e:
            raise NPUCompilationError(f"Failed to natively convert PyTorch model: {e}") from e


# Level Zero utilization properties shared by graph-mode compiles.
# Mirrors the C++ backend global props so both paths get identical treatment.
# Keys rejected by the driver at compile time are dropped adaptively (see
# _compile_with_ladder) and remembered for the rest of the session.
_NPU_L0_PROPS: dict[str, str] = {
    "NPU_TURBO": "YES",
    "NPU_DISABLE_IDLE_MEMORY_PRUNING": "YES",
    "NPU_RUN_INFERENCES_SEQUENTIALLY": "NO",
    "NPU_DEFER_WEIGHTS_LOAD": "YES",
    "NPU_QDQ_OPTIMIZATION": "YES",
    "NPU_QDQ_OPTIMIZATION_AGGRESSIVE": "YES",
    "NPU_COMPILATION_MODE_PARAMS": "optimization-level=2",
    "NPU_USE_SDA": "YES",
    "NPU_BACKEND_TYPE": "LEVEL_ZERO",
}

# Options the driver refused in this session ("Option 'X' is not supported").
# Populated adaptively so each rejection is only paid once per session.
_DROPPED_NPU_KEYS: set[str] = set()
_DROPPED_KEYS_LOCK = threading.Lock()

# e.g. "[ NOT_FOUND ] Option 'NPU_BACKEND_TYPE' is not supported for current configuration"
_UNSUPPORTED_OPT_RE = re.compile(r"Option '([^']+)' is not supported")


def _remember_dropped_key(error: Exception) -> str | None:
    """Record a driver-rejected config key from a compile error, if identifiable."""
    match = _UNSUPPORTED_OPT_RE.search(str(error))
    if not match:
        return None
    key = match.group(1)
    with _DROPPED_KEYS_LOCK:
        if key in _DROPPED_NPU_KEYS:
            return None  # already dropped; retrying won't help
        _DROPPED_NPU_KEYS.add(key)
    logger.warning(f"NPU driver rejected config key '{key}'; dropping it for this session.")
    return key


def _live_l0_props() -> dict[str, str]:
    """Return L0 props minus session-dropped keys."""
    with _DROPPED_KEYS_LOCK:
        dropped = set(_DROPPED_NPU_KEYS)
    if not dropped:
        return dict(_NPU_L0_PROPS)
    return {k: v for k, v in _NPU_L0_PROPS.items() if k not in dropped}


def _perf_mode(hint: str) -> Any:
    """Map a hint string to a typed PerformanceMode (CUMULATIVE supported)."""
    if hint == "THROUGHPUT":
        return ov_hints.PerformanceMode.THROUGHPUT
    if hint == "CUMULATIVE_THROUGHPUT":
        return ov_hints.PerformanceMode.CUMULATIVE_THROUGHPUT
    return ov_hints.PerformanceMode.LATENCY


def _build_npu_config(performance_hint: str, num_streams: int, precision: str = "auto") -> dict:
    """Build the OpenVINO/NPU compile config with graceful fallback.

    Respects ``precision="fp32"`` (omits the FP16 inference hint instead of
    forcing half precision) and maps ``CUMULATIVE_THROUGHPUT`` explicitly
    instead of silently degrading to LATENCY.
    """
    use_fp16 = str(precision).lower() not in ("fp32", "f32")
    streams = num_streams if performance_hint in ("THROUGHPUT", "CUMULATIVE_THROUGHPUT") else None
    l0_props = _live_l0_props()
    try:
        config = {
            ov_hints.performance_mode(): _perf_mode(performance_hint),
            ov_hints.model_priority(): ov_hints.Priority.HIGH,
            **l0_props,
        }
        if use_fp16:
            config[ov_hints.inference_precision()] = ov_props.element.f16
        if streams is not None:
            config[ov_props.streams.num()] = streams
        return config
    except Exception:
        config = {
            "PERFORMANCE_HINT": performance_hint,
            "MODEL_PRIORITY": "HIGH",
            **l0_props,
        }
        if use_fp16:
            config["INFERENCE_PRECISION_HINT"] = "f16"
        if streams is not None:
            config["NUM_STREAMS"] = str(streams)
        return config


def _minimal_npu_config(performance_hint: str, num_streams: int, precision: str = "auto") -> dict:
    """Minimal fallback config (perf mode + precision + streams only).

    Used as the middle rung of the compile fallback ladder when the full
    Level Zero property set is rejected by an older driver.
    """
    use_fp16 = str(precision).lower() not in ("fp32", "f32")
    try:
        config = {ov_hints.performance_mode(): _perf_mode(performance_hint)}
        if use_fp16:
            config[ov_hints.inference_precision()] = ov_props.element.f16
        if performance_hint in ("THROUGHPUT", "CUMULATIVE_THROUGHPUT"):
            config[ov_props.streams.num()] = num_streams
        return config
    except Exception:
        config = {"PERFORMANCE_HINT": performance_hint}
        if use_fp16:
            config["INFERENCE_PRECISION_HINT"] = "f16"
        if performance_hint in ("THROUGHPUT", "CUMULATIVE_THROUGHPUT"):
            config["NUM_STREAMS"] = str(num_streams)
        return config


def _try_compile_once(core: Any, ov_model: Any, target_device: str, config: dict) -> Any:
    """Attempt one compile, preferring the Level Zero RemoteContext on NPU."""
    if target_device == "NPU":
        try:
            # Bind directly to the Level Zero RemoteContext
            l0_context = core.get_default_context("NPU")
            return core.compile_model(ov_model, l0_context, config)
        except Exception as e_ctx:
            logger.debug(f"Level Zero RemoteContext compile fallback to target_device string: {e_ctx}")
            return core.compile_model(ov_model, target_device, config)
    return core.compile_model(ov_model, target_device, config)


def _compile_with_ladder(
    core: Any,
    ov_model: Any,
    target_device: str,
    performance_hint: str,
    num_streams: int,
    precision: str,
    config: dict,
) -> Any:
    """Compile with an adaptive fallback ladder.

    Full L0 config (minus session-dropped keys) -> drop each newly rejected
    key and retry -> minimal (perf+precision) -> bare. Keeps working on older
    drivers while preserving as much utilization tuning as the driver accepts.
    """
    if target_device != "NPU":
        return _try_compile_once(core, ov_model, target_device, config)

    attempts = len(_NPU_L0_PROPS) + 1  # bounded: one retry per droppable L0 key
    current = dict(config)
    for _ in range(attempts):
        try:
            return _try_compile_once(core, ov_model, target_device, current)
        except Exception as e:
            dropped = _remember_dropped_key(e)
            if dropped is None:
                break
            current = _build_npu_config(performance_hint, num_streams, precision)
            logger.warning(f"NPU compile rejected key '{dropped}'; retrying without it.")
    else:
        raise NPUCompilationError("NPU compile failed after exhausting adaptive config retries.")

    try:
        return _try_compile_once(
            core, ov_model, target_device, _minimal_npu_config(performance_hint, num_streams, precision)
        )
    except Exception as e2:
        logger.warning(f"NPU minimal-config compile failed ({e2}); retrying bare.")
        return _try_compile_once(core, ov_model, target_device, {})


def compile(model: torch.nn.Module, example_input: Any, *args: Any, **kwargs: Any) -> torch.nn.Module:
    """Compile a PyTorch model for hardware-accelerated execution on Intel NPU.

    Convenience wrapper around `compile_to_npu`.

    Args:
        model (torch.nn.Module): PyTorch module to compile.
        example_input (Any): Example input tensor or tuple of tensors for graph shape discovery.
        *args: Positional arguments forwarded to `compile_to_npu`.
        **kwargs: Keyword arguments forwarded to `compile_to_npu`.

    Returns:
        torch.nn.Module: An optimized NPUGraphModule executing on the Intel NPU.
    """
    return compile_to_npu(model, example_input, *args, **kwargs)


def compile_to_npu(
    model: torch.nn.Module,
    example_input: Any,
    performance_hint: str | None = None,
    num_streams: int = 1,
    strict: bool = False,
    dynamic_buckets: bool = False,
    bucket_sizes: list[int] | None = None,
    dynamic_dim: int = 1,
    clone_outputs: bool = True,
    preprocess_config: dict | None = None,
    stateful: bool = False,
    precision: str = "auto",
) -> torch.nn.Module:
    """Compile a PyTorch neural network module into a fused Intel NPU hardware executable.

    Traces the computational graph using `torch.fx`, maps supported operator subgraphs to
    OpenVINO IR, applies hardware-level constant folding and layout transformations, and compiles
    directly into the Intel Level Zero NPU runtime.

    Args:
        model (torch.nn.Module): PyTorch module or GraphModule to compile.
        example_input (Any): Sample input tensor or tuple of tensors used for graph tracing and shape discovery.
        performance_hint (str, optional): Target performance mode: 'LATENCY', 'THROUGHPUT', or 'CUMULATIVE_THROUGHPUT'. Defaults to the global hint from `npu.set_performance_hint()` ('LATENCY').
        num_streams (int, optional): Number of parallel execution streams / command queues for throughput pipelining. Defaults to 1.
        strict (bool, optional): If True, raises `NPUCompilationError` on any unsupported operation. If False, enables automatic hybrid graph partitioning with CPU fallback. Defaults to False.
        dynamic_buckets (bool, optional): Enables static bucket dispatch for variable sequence/batch lengths. Defaults to False.
        bucket_sizes (Optional[List[int]], optional): Custom bucket dimensions when `dynamic_buckets=True`. Defaults to powers of 2.
        dynamic_dim (int, optional): The tensor axis index that varies dynamically (e.g. sequence length dim=1). Defaults to 1.
        clone_outputs (bool, optional): If True, clones output tensors to decouple memory from leased backend buffers. Defaults to True.
        preprocess_config (Optional[dict], optional): Hardware Pre-Post Processing (PPP) configuration dictionary. Defaults to None.
        stateful (bool, optional): Enables stateful KV-cache register binding for autoregressive generation. Defaults to False.
        precision (str, optional): Computation precision ('auto', 'fp16', 'fp32'). Defaults to 'auto'.

    Returns:
        torch.nn.Module: An optimized `NPUGraphModule` or `NPUDynamicGraphModule`.

    Raises:
        NPUCompilationError: If graph conversion fails and `strict=True`.

    Examples:
        >>> import torch
        >>> import intel_npu_acceleration as npu
        >>> model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU())
        >>> x = torch.randn(1, 32)
        >>> npu_model = npu.compile_to_npu(model, x, performance_hint="LATENCY")
        >>> out = npu_model(x)
    """
    global _GRAPH_CACHE

    if performance_hint is None:
        from intel_npu_acceleration.device import get_performance_hint

        performance_hint = get_performance_hint()

    if dynamic_buckets and not isinstance(model, NPUDynamicGraphModule):
        return NPUDynamicGraphModule(
            model=model,
            example_input=example_input,
            performance_hint=performance_hint,
            num_streams=num_streams,
            strict=strict,
            bucket_sizes=bucket_sizes,
            dynamic_dim=dynamic_dim,
            clone_outputs=clone_outputs,
            preprocess_config=preprocess_config,
        )

    # Automatic dtype inspection from standard PyTorch model and tensors
    if precision == "auto":
        is_fp16 = False
        if isinstance(example_input, torch.Tensor) and example_input.dtype in [torch.float16, torch.bfloat16]:
            is_fp16 = True
        elif isinstance(example_input, (tuple, list)):
            if any(isinstance(x, torch.Tensor) and x.dtype in [torch.float16, torch.bfloat16] for x in example_input):
                is_fp16 = True
        if not is_fp16 and hasattr(model, "parameters"):
            try:
                for p in model.parameters():
                    if p.dtype in [torch.float16, torch.bfloat16]:
                        is_fp16 = True
                        break
            except Exception:
                pass
        if is_fp16:
            precision = "fp16"

    # Compiler Performance Intelligence Logging
    if performance_hint == "LATENCY" and num_streams == 1:
        logger.info(
            "[NPU Intelligence] Configuration: LATENCY mode with a single stream. "
            "Tip: For maximum NPU core saturation and throughput (up to 4x utilization boost), "
            "compile with performance_hint='THROUGHPUT' and num_streams=2+ to enable pipelined "
            "multi-stream parallel NPU execution!"
        )

    logger.info("Starting NPU Compilation...")
    try:
        # Trace the model into GraphModule first (solely for cache key generation and topological analysis)
        traced = _trace_model(model)

        example_input_tuple = _sanitize_dynamo_inputs(_to_example_tuple(example_input))

        # Check for unsupported nodes to enable hybrid partitioning fallback
        from intel_npu_acceleration.registry import OpRegistry
        unsupported_nodes = []
        for node in traced.graph.nodes:
            if node.op in ["placeholder", "output", "get_attr"]:
                continue
            if node.op == "call_function" and not OpRegistry.is_function_supported(node.target):
                unsupported_nodes.append(f"{node.name} (call_function: {node.target})")
            elif node.op == "call_method" and not OpRegistry.is_method_supported(node.target):
                unsupported_nodes.append(f"{node.name} (call_method: {node.target})")
            elif node.op == "call_module":
                try:
                    submod = model
                    for atom in node.target.split("."):
                        submod = getattr(submod, atom)
                except AttributeError:
                    unsupported_nodes.append(f"{node.name} (call_module: {node.target})")
                    continue
                if not OpRegistry.is_module_supported(type(submod)):
                    unsupported_nodes.append(f"{node.name} (call_module: {type(submod)})")

        if len(unsupported_nodes) > 0:
            if strict:
                raise NPUCompilationError(
                    f"Unsupported operators detected in strict compilation mode: {', '.join(unsupported_nodes)}"
                )
            logger.info(
                "Unsupported operators detected. Enabling Automated Hybrid Graph Partitioning & CPU Fallback..."
            )
            from .partitioner import partition_and_compile_hybrid
            return partition_and_compile_hybrid(
                model=model,
                traced=traced,
                example_input_tuple=example_input_tuple,
                performance_hint=performance_hint,
                num_streams=num_streams,
                dynamic_buckets=dynamic_buckets,
                bucket_sizes=bucket_sizes,
                dynamic_dim=dynamic_dim,
                clone_outputs=clone_outputs,
                preprocess_config=preprocess_config,
            )

        # Convert non-Tensor scalars to Tensors so that torch.jit.trace works natively
        example_input_tuple_converted = _to_converted_inputs(example_input_tuple)

        # 1. Generate Cache Key (include perf and precision to avoid false hits)
        graph_str = str(traced.graph)
        input_meta = []
        for t in example_input_tuple:
            if isinstance(t, torch.Tensor):
                input_meta.append((tuple(t.shape), str(t.dtype)))
            else:
                input_meta.append(type(t).__name__)
        key_raw = (
            f"{graph_str}_{input_meta}_{preprocess_config}_{stateful}_"
            f"{performance_hint}_{num_streams}_{precision}_{clone_outputs}"
        )
        key = hashlib.md5(key_raw.encode(), usedforsecurity=False).hexdigest()
        logger.debug(f"Generated Graph cache key: {key}")

        # Check Cache (thread-safe)
        all_placeholder_names = [node.name for node in traced.graph.nodes if node.op == "placeholder"]

        with _GRAPH_CACHE_LOCK:
            if key in _GRAPH_CACHE:
                logger.info("Cache hit! Using pre-compiled graph model.")
                compiled_entry = _GRAPH_CACHE.pop(key)
                _GRAPH_CACHE[key] = compiled_entry
                return NPUGraphModule(
                    compiled_entry["model"],
                    compiled_entry["input_names"],
                    performance_hint,
                    num_streams,
                    clone_outputs,
                    all_placeholder_names=all_placeholder_names,
                )

        # 2. Capture Values & Natively Convert Model using OpenVINO PyTorch Frontend
        with _dequantized_linears(traced):
            ov_model = _convert_to_ov_model(model, example_input_tuple_converted)
            ov_model = fold_scalar_parameter_inputs(
                ov_model, example_input_tuple, all_placeholder_names
            )
            reshape_model_inputs_to_static(
                ov_model, example_input_tuple, all_placeholder_names
            )

        # Find all NPUStatefulKVCache submodules in chronological/topological order
        stateful_modules = _collect_stateful_modules(model, traced)

        # 3. Post-process Stateful KV Cache nodes if requested
        ov_model = transform_stateful_kv_cache(ov_model, stateful, stateful_modules)

        # 4. OpenVINO Built-in Model Graph Optimizations (Constant Folding + Validation + Precision Conversion)
        ov_model = optimize_ov_model(ov_model, precision=precision)

        # 5. Apply PrePostProcessor (PPP) to offload layout/type transpositions and normalizations to NPU
        if preprocess_config:
            try:
                ov_model = apply_pre_post_processing(ov_model, preprocess_config)
                logger.info("Successfully offloaded Pre-Post Processing (PPP) pipeline to NPU hardware.")
            except Exception as e:
                logger.warning(f"Failed to apply PrePostProcessor: {e}. Proceeding with standard NPU graph.")

        core = _get_core()

        # Check for NPU availability
        available_devices = core.available_devices
        target_device = "NPU"
        if not any("NPU" in d for d in available_devices):
            logger.warning("Intel NPU not detected. Falling back to CPU for execution.")
            target_device = "CPU"
            config = {}
        else:
            # NPU available — Level Zero hardware acceleration & performance hints
            config = _build_npu_config(performance_hint, num_streams, precision)

        logger.info(f"Compiling model for {target_device}...")
        compiled = _compile_with_ladder(
            core, ov_model, target_device, performance_hint, num_streams, precision, config
        )

        # 5. Cache Management (thread-safe)
        input_names = [p.any_name for p in ov_model.inputs]
        with _GRAPH_CACHE_LOCK:
            if len(_GRAPH_CACHE) >= _MAX_GRAPH_CACHE_SIZE:
                _GRAPH_CACHE.popitem(last=False)
            _GRAPH_CACHE[key] = {"model": compiled, "input_names": input_names}

        # Auto-clean old disk-cache blobs off the critical path: the walk over
        # the cache dir is pure I/O, so run it in a daemon thread and return
        # the compiled module immediately.
        try:
            _max_mb = npu_config.disk_cache_cleanup_size_mb
            _max_files = npu_config.disk_cache_cleanup_files

            def _cleanup() -> None:
                try:
                    npu_lib.clean_old_cache(max_size_mb=_max_mb, max_files=_max_files)
                except Exception as e:
                    logger.debug("Disk cache cleanup failed: %s", e)

            threading.Thread(target=_cleanup, name="npu-cache-cleanup", daemon=True).start()
        except Exception as e:
            logger.debug("Disk cache cleanup scheduling failed: %s", e)

        return NPUGraphModule(
            compiled,
            input_names,
            performance_hint,
            num_streams,
            clone_outputs,
            all_placeholder_names=all_placeholder_names,
        )

    except Exception as e:
        logger.error(f"Compilation Failed: {e}")
        raise


def export_openvino_ir(
    model: torch.nn.Module,
    example_input: Any,
    output_xml_path: str,
    output_bin_path: str | None = None,
    preprocess_config: dict | None = None,
    stateful: bool = False,
) -> str:
    """Export a PyTorch model directly to OpenVINO Intermediate Representation (IR) format (.xml / .bin).

    Generates serialized OpenVINO IR files that can be visualized in Netron or deployed
    directly using the OpenVINO C++ or Python runtime engines without PyTorch dependencies.

    Args:
        model (torch.nn.Module): PyTorch module to export.
        example_input (Any): Concrete sample inputs for graph tracing.
        output_xml_path (str): Destination filesystem path for the `.xml` model topology file.
        output_bin_path (Optional[str], optional): Destination path for the `.bin` weight file.
            If omitted, defaults to the same filename with a `.bin` extension. Defaults to None.
        preprocess_config (Optional[dict], optional): Hardware Pre-Post Processing dictionary to bake into IR. Defaults to None.
        stateful (bool, optional): If True, serializes KV-cache registers as OpenVINO ReadValue/Assign nodes. Defaults to False.

    Returns:
        str: The path to the generated `.xml` topology file.

    Examples:
        >>> import torch
        >>> import intel_npu_acceleration as npu
        >>> model = torch.nn.Linear(10, 2)
        >>> xml_path = npu.export_openvino_ir(model, torch.randn(1, 10), "model.xml")
    """
    example_input_tuple = _to_example_tuple(example_input)
    converted_inputs = _to_converted_inputs(example_input_tuple)

    traced = _trace_model(model)

    all_placeholder_names = [
        node.name for node in traced.graph.nodes if node.op == "placeholder"
    ]

    # Temporarily dequantize int8/uint8 Linear weights for correct IR export
    with _dequantized_linears(traced):
        ov_model = _convert_to_ov_model(model, tuple(converted_inputs))
        ov_model = fold_scalar_parameter_inputs(
            ov_model, example_input_tuple, all_placeholder_names
        )
        reshape_model_inputs_to_static(
            ov_model, example_input_tuple, all_placeholder_names
        )

    stateful_modules = _collect_stateful_modules(model, traced)

    ov_model = transform_stateful_kv_cache(ov_model, stateful, stateful_modules)
    ov_model = optimize_ov_model(ov_model)

    if preprocess_config:
        ov_model = apply_pre_post_processing(ov_model, preprocess_config)

    serialize_openvino_model(ov_model, output_xml_path, output_bin_path)
    logger.info(f"Successfully exported OpenVINO IR model to '{output_xml_path}'")
    return output_xml_path


# --- Import TorchDynamo Backend Submodule ---
from . import dynamo  # noqa: F401, E402



