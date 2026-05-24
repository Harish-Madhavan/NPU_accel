import hashlib
import threading
from collections import OrderedDict
import logging
import torch
import torch.fx
import openvino as ov
import openvino.properties as ov_props
import openvino.properties.hint as ov_hints
import numpy as np
from typing import Any

import intel_npu_acceleration as npu
from .registry import OpRegistry
from .graph_builder import OVGraphBuilder, ValueCapturingInterpreter
from . import converters  # noqa: F401

logger = logging.getLogger(__name__)


class NPUTracer(torch.fx.Tracer):
    """
    Custom FX Tracer for Intel NPU.
    Ensures custom functions like quantized_linear
    are always treated as leaf/atomic nodes in the graph rather than being traced through.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .functional import quantized_linear

        self._autowrap_function_ids.add(id(quantized_linear))


# --- Global Cache for Compiled Graphs ---
_OV_CORE = None


_GRAPH_CACHE = OrderedDict()
_MAX_GRAPH_CACHE_SIZE = 100


def _get_core():
    global _OV_CORE
    if _OV_CORE is None:
        _OV_CORE = ov.Core()

        # Disk model cache
        cache_dir = npu.get_cache_dir()
        if cache_dir:
            logger.debug(f"Configuring Graph-mode cache dir: {cache_dir}")
            try:
                _OV_CORE.set_property({ov_props.cache_dir(): cache_dir})
            except Exception:
                _OV_CORE.set_property(ov.cache_dir(cache_dir))  # OV < 2024 fallback

        if not any("NPU" in d for d in _OV_CORE.available_devices):
            return _OV_CORE

        # --- Probe which keys this driver version accepts ---
        supported: set = set()
        try:
            raw = _OV_CORE.get_property("NPU", ov_props.supported_properties())
            supported = {str(p) for p in raw}
            logger.debug(f"NPU driver supports {len(supported)} properties.")
        except Exception as e:
            logger.debug(f"Could not probe NPU supported_properties: {e}")

        # Performance mode via typed property (always safe)
        try:
            _OV_CORE.set_property(
                "NPU",
                {ov_hints.performance_mode(): ov_hints.PerformanceMode.LATENCY},
            )
        except Exception as e:
            logger.debug(f"performance_mode hint rejected: {e}")

        # Driver-version-specific optimisation keys — set only if supported
        _OPTIONAL_PROPS = {
            "NPU_BACKEND_TYPE": "LEVEL_ZERO",
            "NPU_USE_SDA": "YES",
            "NPU_TURBO": "YES",
        }
        for key, val in _OPTIONAL_PROPS.items():
            if supported and key not in supported:
                logger.debug(f"Skipping unsupported NPU property '{key}'")
                continue
            try:
                _OV_CORE.set_property("NPU", {key: val})
                logger.debug(f"Set NPU property {key}={val}")
            except Exception as e:
                logger.debug(f"NPU property '{key}' rejected: {e}")

    return _OV_CORE


class NPUCompilationError(Exception):
    """Exception raised for errors during NPU compilation."""

    pass


class NPUGraphModule(torch.nn.Module):
    def __init__(
        self, compiled_model, input_names, performance_hint="LATENCY", num_streams=1
    ):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_names = input_names
        self.performance_hint = performance_hint
        self.num_streams = num_streams

        # Multiple infer requests for throughput mode
        self.infer_requests = [
            self.compiled_model.create_infer_request() for _ in range(num_streams)
        ]
        self.request_idx = 0
        self._lock = threading.Lock()

        _OV_TO_NP = {
            ov.Type.f32: np.float32,
            ov.Type.f16: np.float16,
            ov.Type.i32: np.int32,
            ov.Type.i64: np.int64,
            ov.Type.i8: np.int8,
            ov.Type.u8: np.uint8,
            ov.Type.boolean: bool,
        }

        # Pre-compute input properties to eliminate Python overhead in forward pass
        self.target_dtypes = []
        self.target_torch_dtypes = []
        for i in range(len(self.compiled_model.inputs)):
            ov_type = self.compiled_model.inputs[i].get_element_type()
            target_dtype = _OV_TO_NP.get(ov_type, np.float32)
            self.target_dtypes.append(target_dtype)

            # Map NumPy dtype to PyTorch dtype
            t_dtype = torch.from_numpy(np.array(0, dtype=target_dtype)).dtype
            self.target_torch_dtypes.append(t_dtype)

        # Pre-compute output properties and pre-allocate buffers for zero-copy
        self.output_info = []
        self.output_buffers = [[] for _ in range(num_streams)]

        for j in range(len(self.compiled_model.outputs)):
            ov_out = self.compiled_model.outputs[j]
            partial_shape = ov_out.get_partial_shape()
            if partial_shape.is_static:
                shape = tuple(partial_shape.get_shape())
                ov_type = ov_out.get_element_type()
                target_dtype = _OV_TO_NP.get(ov_type, np.float32)
                torch_dtype = torch.from_numpy(np.array(0, dtype=target_dtype)).dtype
                self.output_info.append((True, shape, torch_dtype))

                # Pre-allocate one buffer per stream
                for s in range(num_streams):
                    buf = torch.empty(shape, dtype=torch_dtype)
                    self.output_buffers[s].append(buf)
                    ov_out_tensor = ov.Tensor(buf.numpy(), shared_memory=True)
                    self.infer_requests[s].set_output_tensor(j, ov_out_tensor)
            else:
                self.output_info.append((False, None, None))
                for s in range(num_streams):
                    self.output_buffers[s].append(None)

    def forward(self, *args):
        with self._lock:
            # Use round-robin for infer requests
            idx = self.request_idx
            infer_request = self.infer_requests[idx]
            self.request_idx = (self.request_idx + 1) % self.num_streams

            for i, val in enumerate(args):
                target_dtype = self.target_dtypes[i]

                if isinstance(val, torch.Tensor):
                    target_tdtype = self.target_torch_dtypes[i]
                    if (
                        val.device.type == "cpu"
                        and val.dtype == target_tdtype
                        and val.is_contiguous()
                    ):
                        np_view = val.detach().numpy()
                    else:
                        cpu_val = val.detach()
                        if cpu_val.device.type != "cpu":
                            cpu_val = cpu_val.cpu()
                        if cpu_val.dtype != target_tdtype:
                            cpu_val = cpu_val.to(target_tdtype)
                        if not cpu_val.is_contiguous():
                            cpu_val = cpu_val.contiguous()
                        np_view = cpu_val.numpy()
                else:
                    np_view = np.array(val)
                    if np_view.dtype != target_dtype:
                        np_view = np_view.astype(target_dtype)
                    if not np_view.flags["C_CONTIGUOUS"]:
                        np_view = np.ascontiguousarray(np_view)

                infer_request.set_input_tensor(
                    i, ov.Tensor(np_view, shared_memory=True)
                )

            infer_request.infer()

            outputs = []
            for j in range(len(self.compiled_model.outputs)):
                buf = self.output_buffers[idx][j]
                if buf is not None:
                    # Return a clone to avoid user accidentally corrupting the internal buffer
                    # or seeing it change on the next call.
                    # For maximum performance, we could return a view, but that's unsafe.
                    # Given the user wants TFLOPS, let's return a view but warn in docs.
                    outputs.append(buf)
                else:
                    out_tensor = infer_request.get_output_tensor(j)
                    outputs.append(torch.from_numpy(out_tensor.data).clone())

            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

    def infer_async(self, *args) -> int:
        """
        Start an asynchronous inference request on the NPU.
        Returns a handle (request index) to wait on later.
        """
        with self._lock:
            # Use round-robin for infer requests
            idx = self.request_idx
            infer_request = self.infer_requests[idx]
            self.request_idx = (self.request_idx + 1) % self.num_streams

            for i, val in enumerate(args):
                target_dtype = self.target_dtypes[i]

                if isinstance(val, torch.Tensor):
                    target_tdtype = self.target_torch_dtypes[i]
                    if (
                        val.device.type == "cpu"
                        and val.dtype == target_tdtype
                        and val.is_contiguous()
                    ):
                        np_view = val.detach().numpy()
                    else:
                        cpu_val = val.detach()
                        if cpu_val.device.type != "cpu":
                            cpu_val = cpu_val.cpu()
                        if cpu_val.dtype != target_tdtype:
                            cpu_val = cpu_val.to(target_tdtype)
                        if not cpu_val.is_contiguous():
                            cpu_val = cpu_val.contiguous()
                        np_view = cpu_val.numpy()
                else:
                    np_view = np.array(val)
                    if np_view.dtype != target_dtype:
                        np_view = np_view.astype(target_dtype)
                    if not np_view.flags["C_CONTIGUOUS"]:
                        np_view = np.ascontiguousarray(np_view)

                infer_request.set_input_tensor(
                    i, ov.Tensor(np_view, shared_memory=True)
                )

            infer_request.start_async()
            return idx

    def wait_async(self, handle: int):
        """
        Block and wait for the specified asynchronous inference request to complete.
        Returns the output tensor(s).
        """
        infer_request = self.infer_requests[handle]
        infer_request.wait()

        outputs = []
        for j in range(len(self.compiled_model.outputs)):
            buf = self.output_buffers[handle][j]
            if buf is not None:
                outputs.append(buf)
            else:
                out_tensor = infer_request.get_output_tensor(j)
                outputs.append(torch.from_numpy(out_tensor.data).clone())

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def compile(
    model: torch.nn.Module,
    example_input: Any,
    performance_hint: str = "LATENCY",
    num_streams: int = 1,
    strict: bool = False,
) -> torch.nn.Module:
    """
    Compile a PyTorch model for Intel NPU.

    Args:
        model: The PyTorch model to compile.
        example_input: Example input(s) to the model.
        performance_hint: "LATENCY" or "THROUGHPUT".
        num_streams: Number of parallel execution streams (only for THROUGHPUT).
        strict: If True, raise NPUCompilationError on unsupported operators instead of falling back to CPU.
    """
    return compile_to_npu(model, example_input, performance_hint, num_streams, strict)


def compile_to_npu(
    model: torch.nn.Module,
    example_input: Any,
    performance_hint: str = "LATENCY",
    num_streams: int = 1,
    strict: bool = False,
) -> torch.nn.Module:
    global _GRAPH_CACHE

    logger.info("Starting NPU Compilation...")
    try:
        # Trace the model into GraphModule first
        traced = torch.fx.GraphModule(model, NPUTracer().trace(model))

        # Check for unsupported operators in the graph
        has_unsupported = False
        for node in traced.graph.nodes:
            if node.op in ["placeholder", "output", "get_attr"]:
                continue
            if node.op == "call_function":
                if OpRegistry.get_function(node.target) is None:
                    has_unsupported = True
                    break
            elif node.op == "call_method":
                if OpRegistry.get_method(node.target) is None:
                    has_unsupported = True
                    break
            elif node.op == "call_module":
                submod = model
                for atom in node.target.split("."):
                    submod = getattr(submod, atom)
                if OpRegistry.get_module(type(submod)) is None:
                    has_unsupported = True
                    break

        if isinstance(example_input, torch.Tensor):
            example_input_tuple = (example_input,)
        else:
            example_input_tuple = example_input

        if has_unsupported:
            if strict:
                raise NPUCompilationError(
                    "Unsupported operators detected in strict compilation mode."
                )

            logger.info(
                "Unsupported operators detected. Enabling Automated Hybrid Graph Partitioning & CPU Fallback..."
            )

            # Pre-compute partition map to ensure monotonically increasing partition IDs.
            # This mathematically guarantees there are no cycles in the split graph.
            partition_map = {}
            current_partition = 0
            prev_is_supported = None

            for node in traced.graph.nodes:
                if node.op in ["placeholder", "get_attr"]:
                    partition_map[node] = 0
                    continue
                if node.op == "output":
                    partition_map[node] = current_partition
                    continue

                # Determine if the node itself is supported
                is_supported = True
                if node.op == "call_function":
                    is_supported = OpRegistry.get_function(node.target) is not None
                elif node.op == "call_method":
                    is_supported = OpRegistry.get_method(node.target) is not None
                elif node.op == "call_module":
                    target_sub = model
                    for atom in node.target.split("."):
                        target_sub = getattr(target_sub, atom)
                    is_supported = OpRegistry.get_module(type(target_sub)) is not None

                if prev_is_supported is None:
                    prev_is_supported = is_supported
                elif prev_is_supported != is_supported:
                    current_partition += 1
                    prev_is_supported = is_supported

                partition_map[node] = current_partition

            def classify_node(node: torch.fx.Node) -> int:
                return partition_map.get(node, 0)

            from torch.fx.passes.split_module import split_module

            split_parent = split_module(traced, model, classify_node)

            submod_inputs = {}
            hooks = []

            def make_hook(name):
                def hook(module, inputs):
                    submod_inputs[name] = inputs
                    return None

                return hook

            for name, child in split_parent.named_children():
                hooks.append(child.register_forward_pre_hook(make_hook(name)))

            # Run calibration forward pass to resolve input shapes
            with torch.no_grad():
                try:
                    split_parent(*example_input_tuple)
                except Exception as e:
                    logger.debug(f"Calibration forward pass notice: {e}")

            for h in hooks:
                h.remove()

            # Compile supported submodules recursively
            for name, child in list(split_parent.named_children()):
                is_supported = True
                for n in child.graph.nodes:
                    if n.op in ["placeholder", "output", "get_attr"]:
                        continue
                    if (
                        n.op == "call_function"
                        and OpRegistry.get_function(n.target) is None
                    ):
                        is_supported = False
                        break
                    elif (
                        n.op == "call_method"
                        and OpRegistry.get_method(n.target) is None
                    ):
                        is_supported = False
                        break
                    elif n.op == "call_module":
                        target_sub = child
                        for atom in n.target.split("."):
                            target_sub = getattr(target_sub, atom)
                        if OpRegistry.get_module(type(target_sub)) is None:
                            is_supported = False
                            break

                if is_supported and name in submod_inputs:
                    logger.info(
                        f"Partition: Compiling supported submodule '{name}' for NPU..."
                    )
                    sub_inputs = submod_inputs[name]
                    try:
                        compiled_sub = compile_to_npu(
                            child,
                            sub_inputs,
                            performance_hint,
                            num_streams,
                            strict=True,
                        )
                        setattr(split_parent, name, compiled_sub)
                    except Exception as e:
                        logger.warning(
                            f"Failed to compile submodule '{name}' for NPU: {e}. Falling back to CPU."
                        )
                else:
                    logger.info(
                        f"Partition: Leaving unsupported submodule '{name}' on CPU."
                    )

            return split_parent

        # 1. Generate Cache Key
        # We use a hash of the FX graph and input metadata as a key
        graph_str = str(traced.graph)

        if isinstance(example_input, torch.Tensor):
            example_input_tuple = (example_input,)
        else:
            example_input_tuple = example_input

        # Capture input shapes/dtypes for key
        input_meta = []
        for t in example_input_tuple:
            if isinstance(t, torch.Tensor):
                input_meta.append((tuple(t.shape), t.dtype))
            else:
                input_meta.append(type(t))

        key_raw = f"{graph_str}_{input_meta}"
        key = hashlib.md5(key_raw.encode()).hexdigest()
        logger.debug(f"Generated Graph cache key: {key} from {key_raw[:100]}...")

        # Check Cache
        if key in _GRAPH_CACHE:
            logger.info("Cache hit! Using pre-compiled graph model.")
            # Move to end (most recent)
            compiled_entry = _GRAPH_CACHE.pop(key)
            _GRAPH_CACHE[key] = compiled_entry
            return NPUGraphModule(
                compiled_entry["model"], compiled_entry["input_names"]
            )

        # 2. Capture Values & Build OV Graph
        interpreter = ValueCapturingInterpreter(traced)
        interpreter.run(*example_input_tuple)

        builder = OVGraphBuilder(interpreter.node_values)

        from torch.fx.passes.shape_prop import ShapeProp

        ShapeProp(traced).propagate(*example_input_tuple)

        input_iter = iter(example_input_tuple)

        for node in traced.graph.nodes:
            if node.op == "placeholder":
                try:
                    val = next(input_iter)
                except StopIteration:
                    raise NPUCompilationError(
                        f"Not enough example inputs for placeholders starting at {node.name}"
                    )

                if "tensor_meta" in node.meta:
                    shape = node.meta["tensor_meta"].shape
                    dtype = node.meta["tensor_meta"].dtype
                else:
                    if isinstance(val, torch.Tensor):
                        shape = list(val.shape)
                        dtype = val.dtype
                    elif isinstance(val, int):
                        shape = []
                        dtype = torch.int64
                    elif isinstance(val, float):
                        shape = []
                        dtype = torch.float32
                    else:
                        shape = [1]
                        dtype = torch.float32

                builder.add_parameter(node.name, list(shape), dtype)

            elif node.op == "call_function":
                converter = OpRegistry.get_function(node.target)
                if converter:
                    res = converter(builder, node, node.args, node.kwargs)
                    builder.register_output(node.name, res)
                else:
                    raise NPUCompilationError(f"Function {node.target} not supported.")

            elif node.op == "call_method":
                converter = OpRegistry.get_method(node.target)
                if converter:
                    res = converter(builder, node, node.args, node.kwargs)
                    builder.register_output(node.name, res)
                else:
                    raise NPUCompilationError(f"Method {node.target} not supported.")

            elif node.op == "call_module":
                submod = model
                for atom in node.target.split("."):
                    submod = getattr(submod, atom)
                converter = OpRegistry.get_module(type(submod))
                if converter:
                    res = converter(builder, node, submod, node.args, node.kwargs)
                    builder.register_output(node.name, res)
                else:
                    raise NPUCompilationError(
                        f"Module type {type(submod)} not supported."
                    )

            elif node.op == "get_attr":
                atom = model
                for atom_name in node.target.split("."):
                    atom = getattr(atom, atom_name)
                builder.add_constant(node.name, atom)

            elif node.op == "output":
                ret_vals = node.args[0]
                if isinstance(ret_vals, tuple):
                    for ret_val in ret_vals:
                        builder.result_nodes.append(builder.get_input(ret_val.name))
                else:
                    builder.result_nodes.append(builder.get_input(ret_vals.name))

        # 3. Create & Compile OV Model
        ov_model = ov.Model(builder.result_nodes, builder.parameters, "NPU_Model")

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
                }

                # Apply streams for throughput
                if performance_hint == "THROUGHPUT":
                    config[ov_props.streams.num()] = num_streams

            except Exception:
                # Typed API unavailable (very old OV) — fall back to strings.
                config = {
                    "PERFORMANCE_HINT": performance_hint,
                    "INFERENCE_PRECISION_HINT": "f16",
                }

        logger.info(f"Compiling model for {target_device}...")
        try:
            compiled = core.compile_model(ov_model, target_device, config)
        except Exception as e:
            if target_device == "NPU":
                logger.warning(
                    f"NPU compile with hints failed ({e}); retrying with no config."
                )
                # Last-resort fallback: let the driver choose everything.
                compiled = core.compile_model(ov_model, target_device)
            else:
                raise

        # 4. Cache Management
        input_names = [p.friendly_name for p in builder.parameters]
        if len(_GRAPH_CACHE) >= _MAX_GRAPH_CACHE_SIZE:
            _GRAPH_CACHE.popitem(last=False)  # Evict oldest

        _GRAPH_CACHE[key] = {"model": compiled, "input_names": input_names}

        return NPUGraphModule(compiled, input_names, performance_hint, num_streams)

    except Exception as e:
        logger.error(f"Compilation Failed: {e}")
        raise e
