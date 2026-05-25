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
from typing import Any, Optional, List

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

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        from .functional import NPUStatefulKVCache
        if isinstance(m, NPUStatefulKVCache):
            return True
        return super().is_leaf_module(m, module_qualified_name)


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
        self, compiled_model, input_names, performance_hint="LATENCY", num_streams=1, clone_outputs=True
    ):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_names = input_names
        self.performance_hint = performance_hint
        self.num_streams = num_streams
        self.clone_outputs = clone_outputs

        # Multiple infer requests for throughput mode
        self.infer_requests = [
            self.compiled_model.create_infer_request() for _ in range(num_streams)
        ]
        self.request_idx = 0
        self._lock = threading.Lock()

        # Track active inputs and their output buffer indices during async execution to prevent temporary tensors going out of scope
        self._active_inputs = [None for _ in range(num_streams)]

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

        # Pre-compute output properties and pre-allocate double buffers for zero-copy memory safety
        self.output_info = []
        self.output_buffers = [[[] for _ in range(2)] for _ in range(num_streams)]
        self.active_buffer_idx = [0 for _ in range(num_streams)]

        for j in range(len(self.compiled_model.outputs)):
            ov_out = self.compiled_model.outputs[j]
            partial_shape = ov_out.get_partial_shape()
            if partial_shape.is_static:
                shape = tuple(partial_shape.get_shape())
                ov_type = ov_out.get_element_type()
                target_dtype = _OV_TO_NP.get(ov_type, np.float32)
                torch_dtype = torch.from_numpy(np.array(0, dtype=target_dtype)).dtype
                self.output_info.append((True, shape, torch_dtype))

                # Pre-allocate two buffers per stream for safe double-buffering
                for s in range(num_streams):
                    for b in range(2):
                        buf = torch.empty(shape, dtype=torch_dtype)
                        self.output_buffers[s][b].append(buf)
            else:
                self.output_info.append((False, None, None))
                for s in range(num_streams):
                    for b in range(2):
                        self.output_buffers[s][b].append(None)

    def forward(self, *args):
        with self._lock:
            # Use round-robin for infer requests
            idx = self.request_idx
            infer_request = self.infer_requests[idx]
            self.request_idx = (self.request_idx + 1) % self.num_streams

            # Retrieve active double buffer index
            buf_idx = self.active_buffer_idx[idx]
            self.active_buffer_idx[idx] = 1 - buf_idx

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

            # Bind double buffer output tensors to the request
            for j in range(len(self.compiled_model.outputs)):
                buf = self.output_buffers[idx][buf_idx][j]
                if buf is not None:
                    ov_out_tensor = ov.Tensor(buf.numpy(), shared_memory=True)
                    infer_request.set_output_tensor(j, ov_out_tensor)

            infer_request.infer()

            outputs = []
            for j in range(len(self.compiled_model.outputs)):
                buf = self.output_buffers[idx][buf_idx][j]
                if buf is not None:
                    outputs.append(buf.clone() if self.clone_outputs else buf)
                else:
                    out_tensor = infer_request.get_output_tensor(j)
                    data_tensor = torch.from_numpy(out_tensor.data)
                    outputs.append(data_tensor.clone() if self.clone_outputs else data_tensor)

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

            # Retrieve active double buffer index
            buf_idx = self.active_buffer_idx[idx]
            self.active_buffer_idx[idx] = 1 - buf_idx

            keep_alive = []
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
                        keep_alive.append(val)
                    else:
                        cpu_val = val.detach()
                        if cpu_val.device.type != "cpu":
                            cpu_val = cpu_val.cpu()
                        if cpu_val.dtype != target_tdtype:
                            cpu_val = cpu_val.to(target_tdtype)
                        if not cpu_val.is_contiguous():
                            cpu_val = cpu_val.contiguous()
                        np_view = cpu_val.numpy()
                        keep_alive.append(cpu_val)
                else:
                    np_view = np.array(val)
                    if np_view.dtype != target_dtype:
                        np_view = np_view.astype(target_dtype)
                    if not np_view.flags["C_CONTIGUOUS"]:
                        np_view = np.ascontiguousarray(np_view)
                    keep_alive.append(np_view)

                infer_request.set_input_tensor(
                    i, ov.Tensor(np_view, shared_memory=True)
                )

            # Bind double buffer output tensors to the request
            for j in range(len(self.compiled_model.outputs)):
                buf = self.output_buffers[idx][buf_idx][j]
                if buf is not None:
                    ov_out_tensor = ov.Tensor(buf.numpy(), shared_memory=True)
                    infer_request.set_output_tensor(j, ov_out_tensor)

            self._active_inputs[idx] = (keep_alive, buf_idx)
            infer_request.start_async()
            return idx

    def wait_async(self, handle: int):
        """
        Block and wait for the specified asynchronous inference request to complete.
        Returns the output tensor(s).
        """
        infer_request = self.infer_requests[handle]
        infer_request.wait()

        # Retrieve active double buffer index for this request
        active_info = self._active_inputs[handle]
        buf_idx = active_info[1] if active_info is not None else 0

        # Clear active inputs reference to allow immediate garbage collection
        self._active_inputs[handle] = None

        outputs = []
        for j in range(len(self.compiled_model.outputs)):
            buf = self.output_buffers[handle][buf_idx][j]
            if buf is not None:
                outputs.append(buf.clone() if self.clone_outputs else buf)
            else:
                out_tensor = infer_request.get_output_tensor(j)
                data_tensor = torch.from_numpy(out_tensor.data)
                outputs.append(data_tensor.clone() if self.clone_outputs else data_tensor)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def reset_states(self):
        """Reset all internal state variables (like KV-caches) in the NPU hardware."""
        for request in self.infer_requests:
            for state in request.query_state():
                state.reset()


class NPUDynamicGraphModule(torch.nn.Module):
    """
    Runtime wrapper that matches dynamic sequence lengths to static predefined buckets,
    dynamically padding input tensors and slicing output tensors back.
    Mathematically avoids driver compilation overhead by compiling static shape graphs
    at bucket boundaries.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        example_input: Any,
        performance_hint: str = "LATENCY",
        num_streams: int = 1,
        strict: bool = False,
        bucket_sizes: Optional[List[int]] = None,
        dynamic_dim: int = 1,
        clone_outputs: bool = True,
        preprocess_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = model
        self.performance_hint = performance_hint
        self.num_streams = num_streams
        self.strict = strict
        self.dynamic_dim = dynamic_dim
        self.clone_outputs = clone_outputs
        self.preprocess_config = preprocess_config

        if bucket_sizes is None:
            # Default to power of 2 boundaries
            self.bucket_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        else:
            self.bucket_sizes = sorted(bucket_sizes)

        # Pre-process initial example input and compile
        args = (example_input,) if isinstance(example_input, torch.Tensor) else tuple(example_input)
        S = self._get_seq_len(args)
        if S is not None:
            B = self._find_bucket(S)
            padded_args = self._pad_inputs(args, S, B)
            self.initial_compiled = compile_to_npu(
                self.model,
                padded_args,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )
        else:
            self.initial_compiled = compile_to_npu(
                self.model,
                example_input,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )

    @property
    def compiled_model(self):
        return self.initial_compiled.compiled_model

    @property
    def input_names(self):
        return self.initial_compiled.input_names

    def _get_seq_len(self, args) -> Optional[int]:
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() > abs(self.dynamic_dim):
                return arg.shape[self.dynamic_dim]
        return None

    def _find_bucket(self, S: int) -> int:
        for b in self.bucket_sizes:
            if b >= S:
                return b
        import math
        return int(2 ** math.ceil(math.log2(S)))

    def _pad_inputs(self, args: tuple, S: int, B: int) -> tuple:
        if S == B:
            return args
        padded = []
        for arg in args:
            if (
                isinstance(arg, torch.Tensor)
                and arg.dim() > abs(self.dynamic_dim)
                and arg.shape[self.dynamic_dim] == S
            ):
                pad_shape = list(arg.shape)
                pad_shape[self.dynamic_dim] = B - S
                pad_tensor = torch.zeros(pad_shape, dtype=arg.dtype, device=arg.device)
                padded.append(torch.cat([arg, pad_tensor], dim=self.dynamic_dim))
            else:
                padded.append(arg)
        return tuple(padded)

    def _slice_outputs(self, outputs: Any, S: int, B: int) -> Any:
        if S == B:
            return outputs

        def slice_tensor(t):
            if (
                isinstance(t, torch.Tensor)
                and t.dim() > abs(self.dynamic_dim)
                and t.shape[self.dynamic_dim] == B
            ):
                slices = [slice(None)] * t.dim()
                slices[self.dynamic_dim] = slice(0, S)
                return t[tuple(slices)]
            return t

        if isinstance(outputs, tuple):
            return tuple(slice_tensor(t) for t in outputs)
        return slice_tensor(outputs)

    def forward(self, *args):
        args_tuple = tuple(args)
        S = self._get_seq_len(args_tuple)
        if S is not None:
            B = self._find_bucket(S)
            padded_args = self._pad_inputs(args_tuple, S, B)
            compiled_model = compile_to_npu(
                self.model,
                padded_args,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )
            outputs = compiled_model(*padded_args)
            return self._slice_outputs(outputs, S, B)
        else:
            return self.initial_compiled(*args)

    def infer_async(self, *args) -> tuple:
        args_tuple = tuple(args)
        S = self._get_seq_len(args_tuple)
        if S is not None:
            B = self._find_bucket(S)
            padded_args = self._pad_inputs(args_tuple, S, B)
            compiled_model = compile_to_npu(
                self.model,
                padded_args,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )
            req_idx = compiled_model.infer_async(*padded_args)
            return (compiled_model, req_idx, S, B)
        else:
            req_idx = self.initial_compiled.infer_async(*args)
            return (self.initial_compiled, req_idx, None, None)

    def wait_async(self, handle: tuple) -> Any:
        compiled_model, req_idx, S, B = handle
        outputs = compiled_model.wait_async(req_idx)
        if S is not None and B is not None:
            return self._slice_outputs(outputs, S, B)
        return outputs

    def reset_states(self):
        """Reset all internal state variables (like KV-caches) in the NPU hardware."""
        self.initial_compiled.reset_states()


def compile(
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
) -> torch.nn.Module:
    """
    Compile a PyTorch model for Intel NPU.

    Args:
        model: The PyTorch model to compile.
        example_input: Example input(s) to the model.
        performance_hint: "LATENCY" or "THROUGHPUT".
        num_streams: Number of parallel execution streams (only for THROUGHPUT).
        strict: If True, raise NPUCompilationError on unsupported operators instead of falling back to CPU.
        dynamic_buckets: If True, enable dynamic shape bucketing and padding.
        bucket_sizes: Predefined boundaries for sequence length buckets.
        dynamic_dim: The dimension to apply bucketing along.
        clone_outputs: If True, clones outputs. If False, returns raw buffers without clones.
        preprocess_config: Optional configuration for NPU-hardware Pre-Post Processing.
    """
    return compile_to_npu(
        model,
        example_input,
        performance_hint,
        num_streams,
        strict,
        dynamic_buckets,
        bucket_sizes,
        dynamic_dim,
        clone_outputs,
        preprocess_config,
    )


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
                            dynamic_buckets=dynamic_buckets,
                            bucket_sizes=bucket_sizes,
                            dynamic_dim=dynamic_dim,
                            clone_outputs=clone_outputs,
                            preprocess_config=preprocess_config,
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

        # Combine preprocess config into cache key to avoid cache hits mismatch
        key_raw = f"{graph_str}_{input_meta}_{preprocess_config}"
        key = hashlib.md5(key_raw.encode()).hexdigest()
        logger.debug(f"Generated Graph cache key: {key} from {key_raw[:100]}...")

        # Check Cache
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
        if hasattr(builder, "sinks") and builder.sinks:
            # Wrap all result nodes in output ports to match signature 13 of ov.Model (which takes Sequence[Output] with sinks/variables)
            results_outputs = [r.output(0) if hasattr(r, "output") else r for r in builder.result_nodes]
            ov_model = ov.Model(
                results_outputs,
                builder.sinks,
                builder.parameters,
                builder.variables,
                "NPU_Model",
            )
        else:
            ov_model = ov.Model(builder.result_nodes, builder.parameters, "NPU_Model")

        # Apply PrePostProcessor (PPP) to offload layout/type transpositions and normalizations to NPU
        if preprocess_config:
            try:
                from openvino.preprocess import PrePostProcessor
                ppp = PrePostProcessor(ov_model)

                if "input" in preprocess_config:
                    inp_cfg = preprocess_config["input"]
                    if isinstance(inp_cfg, dict) and not any(k in ["layout", "element_type", "model_layout", "mean", "scale"] for k in inp_cfg.keys()):
                        for idx, item_cfg in inp_cfg.items():
                            _configure_input_ppp(ppp.input(idx), item_cfg)
                    else:
                        _configure_input_ppp(ppp.input(0), inp_cfg)

                if "output" in preprocess_config:
                    out_cfg = preprocess_config["output"]
                    if isinstance(out_cfg, dict) and "element_type" not in out_cfg:
                        for idx, item_cfg in out_cfg.items():
                            _configure_output_ppp(ppp.output(idx), item_cfg)
                    else:
                        _configure_output_ppp(ppp.output(0), out_cfg)

                ov_model = ppp.build()
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

        return NPUGraphModule(compiled, input_names, performance_hint, num_streams, clone_outputs)

    except Exception as e:
        logger.error(f"Compilation Failed: {e}")
        raise e


def _configure_input_ppp(inp_info, cfg):
    if "layout" in cfg:
        inp_info.tensor().set_layout(ov.Layout(cfg["layout"]))
    if "element_type" in cfg:
        t = cfg["element_type"]
        if isinstance(t, str):
            t = getattr(ov.Type, t) if hasattr(ov.Type, t) else ov.Type.u8
        inp_info.tensor().set_element_type(t)
    if "model_layout" in cfg:
        inp_info.model().set_layout(ov.Layout(cfg["model_layout"]))
    if "mean" in cfg or "scale" in cfg:
        preprocess_steps = inp_info.preprocess()
        # Automatically insert element type conversion to f32 if tensor element type is an integer type,
        # since OpenVINO's mean and scale steps require floating-point input.
        t_cfg = cfg.get("element_type", ov.Type.u8)
        if isinstance(t_cfg, str):
            t_cfg = getattr(ov.Type, t_cfg) if hasattr(ov.Type, t_cfg) else ov.Type.u8
        if t_cfg in [ov.Type.u8, ov.Type.i8, ov.Type.u16, ov.Type.i16, ov.Type.i32, ov.Type.i64]:
            preprocess_steps.convert_element_type(ov.Type.f32)
        if "mean" in cfg:
            preprocess_steps.mean(cfg["mean"])
        if "scale" in cfg:
            preprocess_steps.scale(cfg["scale"])


def _configure_output_ppp(out_info, cfg):
    if "element_type" in cfg:
        t = cfg["element_type"]
        if isinstance(t, str):
            t = getattr(ov.Type, t) if hasattr(ov.Type, t) else ov.Type.f32
        out_info.tensor().set_element_type(t)
