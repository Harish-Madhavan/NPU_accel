import torch
import torch.fx
import hashlib
import logging
from collections import OrderedDict
import openvino as ov
import openvino.opset13 as ops
import openvino.properties as ov_props
import openvino.properties.hint as ov_hints
from typing import Any, Optional, List
import numpy as np

import intel_npu_acceleration as npu_lib
from ..exceptions import NPUCompilationError
from .graph_module import NPUGraphModule, NPUDynamicGraphModule
from .transformations import (
    fold_scalar_parameter_inputs,
    reshape_model_inputs_to_static,
    transform_stateful_kv_cache,
    apply_pre_post_processing,
    optimize_ov_model,
    serialize_openvino_model,
)

logger = logging.getLogger(__name__)


class NPUTracer(torch.fx.Tracer):
    """Custom FX Tracer for Intel NPU."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..functional import quantized_linear, NPUStatefulKVCache
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
_GRAPH_CACHE = OrderedDict()
_MAX_GRAPH_CACHE_SIZE = 100


def clear_graph_cache():
    """Clear all in-memory compiled model graphs from both Python and C++ backend caches."""
    global _GRAPH_CACHE
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


def compile(model: torch.nn.Module, example_input: Any, *args, **kwargs) -> torch.nn.Module:
    """Compile a PyTorch model for Intel NPU."""
    return compile_to_npu(model, example_input, *args, **kwargs)


def compile_to_npu(
    model: torch.nn.Module,
    example_input: Any,
    performance_hint: str = "LATENCY",
    num_streams: int = 1,
    strict: bool = False,
    dynamic_buckets: bool = False,
    bucket_sizes: Optional[List[int]] = None,
    dynamic_dim: int = 1,
    clone_outputs: bool = True,
    preprocess_config: Optional[dict] = None,
    stateful: bool = False,
    precision: str = "auto",
) -> torch.nn.Module:
    global _GRAPH_CACHE

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
        if isinstance(model, torch.fx.GraphModule):
            traced = model
        else:
            traced = torch.fx.GraphModule(model, NPUTracer().trace(model))

        if isinstance(example_input, torch.Tensor):
            example_input_tuple = (example_input,)
        elif isinstance(example_input, (tuple, list)):
            example_input_tuple = tuple(example_input)
        else:
            example_input_tuple = (example_input,)

        # Sanitize SymInt / SymFloat / SymBool from PyTorch Dynamo dynamic shapes
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
        example_input_tuple = tuple(clean_inputs)

        # Check for unsupported nodes to enable hybrid partitioning fallback
        from intel_npu_acceleration.registry import OpRegistry
        unsupported_nodes = []
        for node in traced.graph.nodes:
            if node.op in ["placeholder", "output", "get_attr"]:
                continue
            if node.op == "call_function" and OpRegistry.get_function(node.target) is None:
                unsupported_nodes.append(f"{node.name} (call_function: {node.target})")
            elif node.op == "call_method" and OpRegistry.get_method(node.target) is None:
                unsupported_nodes.append(f"{node.name} (call_method: {node.target})")
            elif node.op == "call_module":
                submod = model
                for atom in node.target.split("."):
                    submod = getattr(submod, atom)
                if OpRegistry.get_module(type(submod)) is None:
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

        # Convert non-Tensor scalars (int/float) to Tensors so that torch.jit.trace works natively
        converted_inputs = []
        for val in example_input_tuple:
            if isinstance(val, torch.Tensor):
                converted_inputs.append(val)
            elif isinstance(val, int):
                converted_inputs.append(torch.tensor(val, dtype=torch.int64))
            elif isinstance(val, float):
                converted_inputs.append(torch.tensor(val, dtype=torch.float32))
            else:
                converted_inputs.append(val)
        example_input_tuple_converted = tuple(converted_inputs)

        # 1. Generate Cache Key (Include stateful in the cache key)
        graph_str = str(traced.graph)

        # Capture input shapes/dtypes for key
        input_meta = []
        for t in example_input_tuple:
            if isinstance(t, torch.Tensor):
                input_meta.append((tuple(t.shape), t.dtype))
            else:
                input_meta.append(type(t))

        key_raw = f"{graph_str}_{input_meta}_{preprocess_config}_{stateful}"
        key = hashlib.md5(key_raw.encode()).hexdigest()
        logger.debug(f"Generated Graph cache key: {key} from {key_raw[:100]}...")

        # Check Cache
        all_placeholder_names = [node.name for node in traced.graph.nodes if node.op == "placeholder"]

        if key in _GRAPH_CACHE:
            logger.info("Cache hit! Using pre-compiled graph model.")
            # Move to end (most recent)
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

        # Temporarily dequantize Linear weights for compilation
        quantized_weights = {}
        for name, submod in traced.named_modules():
            if isinstance(submod, torch.nn.Linear):
                weight = submod._parameters.get("weight", None)
                if weight is not None and getattr(weight, "data", None) is not None:
                    if weight.data.dtype in [torch.int8, torch.uint8]:
                        quantized_weights[submod] = weight.data
                        scale = getattr(submod, "weight_scale", 1.0)
                        submod.weight.data = weight.data.float() * scale

        try:
            # 2. Capture Values & Natively Convert Model using OpenVINO PyTorch Frontend
            import intel_npu_acceleration.functional as npu_func
            npu_func._IS_COMPILING = True
            try:
                ov_model = ov.convert_model(model, example_input=example_input_tuple_converted)
                ov_model = fold_scalar_parameter_inputs(
                    ov_model, example_input_tuple, all_placeholder_names
                )
                reshape_model_inputs_to_static(
                    ov_model, example_input_tuple, all_placeholder_names
                )
            except Exception as e:
                raise NPUCompilationError(f"Failed to natively convert PyTorch model: {e}")
            finally:
                npu_func._IS_COMPILING = False

        finally:
            # Restore original quantized weights immediately after conversion phase
            for submod, w_data in quantized_weights.items():
                submod.weight.data = w_data

        # Find all NPUStatefulKVCache submodules in chronological/topological order
        stateful_modules = []
        for node in traced.graph.nodes:
            if node.op == "call_module":
                submod = model
                for atom in node.target.split("."):
                    submod = getattr(submod, atom)
                from intel_npu_acceleration.functional import NPUStatefulKVCache
                if isinstance(submod, NPUStatefulKVCache):
                    stateful_modules.append(submod)

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
        config = {}
        if not any("NPU" in d for d in available_devices):
            logger.warning("Intel NPU not detected. Falling back to CPU for execution.")
            target_device = "CPU"
        else:
            # NPU available — per-model performance hints via typed API.
            try:
                hint = ov_hints.PerformanceMode.LATENCY
                if performance_hint == "THROUGHPUT":
                    hint = ov_hints.PerformanceMode.THROUGHPUT

                config = {
                    ov_hints.performance_mode(): hint,
                    ov_hints.inference_precision(): ov_props.element.f16,
                    "NPU_COMPILATION_MODE_PARAMS": "optimization-level=2",
                }

                # Apply streams for throughput
                if performance_hint == "THROUGHPUT":
                    config[ov_props.streams.num()] = num_streams

            except Exception:
                # Typed API unavailable (very old OV) — fall back to strings.
                config = {
                    "PERFORMANCE_HINT": performance_hint,
                    "INFERENCE_PRECISION_HINT": "f16",
                    "NPU_COMPILATION_MODE_PARAMS": "optimization-level=2",
                }

        logger.info(f"Compiling model for {target_device}...")
        try:
            compiled = core.compile_model(ov_model, target_device, config)
        except Exception as e:
            if target_device == "NPU":
                logger.warning(
                    f"NPU compile with hints failed ({e}); retrying with no config."
                )
                compiled = core.compile_model(ov_model, target_device)
            else:
                raise

        # 5. Cache Management
        input_names = [p.any_name for p in ov_model.inputs]
        if len(_GRAPH_CACHE) >= _MAX_GRAPH_CACHE_SIZE:
            _GRAPH_CACHE.popitem(last=False)  # Evict oldest

        _GRAPH_CACHE[key] = {"model": compiled, "input_names": input_names}

        # Auto-clean old disk-cache blobs to keep size under 1GB / 500 files
        try:
            npu_lib.clean_old_cache(max_size_mb=1024, max_files=500)
        except Exception as e:
            logger.debug(f"Failed to auto-clean old disk cache: {e}")

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
        raise e


def export_openvino_ir(
    model: torch.nn.Module,
    example_input: Any,
    output_xml_path: str,
    output_bin_path: Optional[str] = None,
    preprocess_config: Optional[dict] = None,
    stateful: bool = False,
) -> str:
    """
    Export a PyTorch model directly to OpenVINO Intermediate Representation (IR) format (.xml / .bin).
    Allows visualizing the compiled NPU model topology in Netron or running via OpenVINO C++ / Python engines.
    """
    if isinstance(example_input, torch.Tensor):
        example_input_tuple = (example_input,)
    else:
        example_input_tuple = example_input

    converted_inputs = []
    for val in example_input_tuple:
        if isinstance(val, torch.Tensor):
            converted_inputs.append(val)
        elif isinstance(val, int):
            converted_inputs.append(torch.tensor(val, dtype=torch.int64))
        elif isinstance(val, float):
            converted_inputs.append(torch.tensor(val, dtype=torch.float32))
        else:
            converted_inputs.append(val)

    if isinstance(model, torch.fx.GraphModule):
        traced = model
    else:
        traced = torch.fx.GraphModule(model, NPUTracer().trace(model))

    all_placeholder_names = [
        node.name for node in traced.graph.nodes if node.op == "placeholder"
    ]

    import intel_npu_acceleration.functional as npu_func

    npu_func._IS_COMPILING = True
    try:
        ov_model = ov.convert_model(model, example_input=tuple(converted_inputs))
        ov_model = fold_scalar_parameter_inputs(
            ov_model, example_input_tuple, all_placeholder_names
        )
        reshape_model_inputs_to_static(
            ov_model, example_input_tuple, all_placeholder_names
        )
    finally:
        npu_func._IS_COMPILING = False

    stateful_modules = []
    for node in traced.graph.nodes:
        if node.op == "call_module":
            submod = model
            for atom in node.target.split("."):
                submod = getattr(submod, atom)
            from intel_npu_acceleration.functional import NPUStatefulKVCache

            if isinstance(submod, NPUStatefulKVCache):
                stateful_modules.append(submod)

    ov_model = transform_stateful_kv_cache(ov_model, stateful, stateful_modules)
    ov_model = optimize_ov_model(ov_model)

    if preprocess_config:
        ov_model = apply_pre_post_processing(ov_model, preprocess_config)

    serialize_openvino_model(ov_model, output_xml_path, output_bin_path)
    logger.info(f"Successfully exported OpenVINO IR model to '{output_xml_path}'")
    return output_xml_path


# --- Import TorchDynamo Backend Submodule ---
from . import dynamo  # noqa: F401, E402



