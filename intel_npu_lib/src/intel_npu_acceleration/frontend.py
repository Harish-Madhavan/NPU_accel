import hashlib
from collections import OrderedDict
import logging
import torch
import torch.fx
import openvino as ov
import numpy as np
from typing import Any

import intel_npu_acceleration as npu
from .registry import OpRegistry
from .graph_builder import OVGraphBuilder, ValueCapturingInterpreter
from . import converters  # noqa: F401
logger = logging.getLogger(__name__)

# --- Global Cache for Compiled Graphs ---
_OV_CORE = None
_GRAPH_CACHE = OrderedDict()
_MAX_GRAPH_CACHE_SIZE = 100


def _get_core():
    global _OV_CORE
    if _OV_CORE is None:
        _OV_CORE = ov.Core()
        cache_dir = npu.get_cache_dir()
        if cache_dir:
            logger.debug(f"Configuring Graph-mode cache dir: {cache_dir}")
            _OV_CORE.set_property(ov.cache_dir(cache_dir))
    return _OV_CORE


class NPUCompilationError(Exception):
    """Exception raised for errors during NPU compilation."""

    pass


class NPUGraphModule(torch.nn.Module):
    def __init__(self, compiled_model, input_names):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_names = input_names
        self.infer_request = self.compiled_model.create_infer_request()
        
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
        for i in range(len(self.compiled_model.inputs)):
            ov_type = self.compiled_model.inputs[i].get_element_type()
            target_dtype = _OV_TO_NP.get(ov_type)
            if target_dtype is None:
                str_type = ov_type.get_type_name()
                if "i32" in str_type:
                    target_dtype = np.int32
                elif "i64" in str_type:
                    target_dtype = np.int64
                elif "f16" in str_type:
                    target_dtype = np.float16
                else:
                    target_dtype = np.float32
            self.target_dtypes.append(target_dtype)

        # Pre-compute output properties for zero-copy allocation
        self.output_info = []
        for j in range(len(self.compiled_model.outputs)):
            ov_out = self.compiled_model.outputs[j]
            partial_shape = ov_out.get_partial_shape()
            if partial_shape.is_static:
                shape = tuple(partial_shape.get_shape())
                ov_type = ov_out.get_element_type()
                target_dtype = _OV_TO_NP.get(ov_type)
                if target_dtype is None:
                    str_type = ov_type.get_type_name()
                    if "i32" in str_type:
                        target_dtype = np.int32
                    elif "i64" in str_type:
                        target_dtype = np.int64
                    elif "f16" in str_type:
                        target_dtype = np.float16
                    else:
                        target_dtype = np.float32
                torch_dtype = torch.from_numpy(np.array(0, dtype=target_dtype)).dtype
                self.output_info.append((True, shape, torch_dtype))
            else:
                self.output_info.append((False, None, None))


    def forward(self, *args):
        for i, val in enumerate(args):
            target_dtype = self.target_dtypes[i]

            if isinstance(val, torch.Tensor):
                np_view = val.detach().cpu().numpy()
            else:
                np_view = np.array(val)

            if np_view.dtype != target_dtype:
                np_view = np_view.astype(target_dtype)

            # Ensure the numpy array is C-contiguous before sharing memory with OpenVINO
            if not np_view.flags['C_CONTIGUOUS']:
                np_view = np.ascontiguousarray(np_view)

            # Using shared_memory=True avoids an extra copy into the OV Tensor
            self.infer_request.set_input_tensor(
                i, ov.Tensor(np_view, shared_memory=True)
            )

        outputs_prealloc = []
        for j in range(len(self.compiled_model.outputs)):
            is_static, shape, torch_dtype = self.output_info[j]
            
            if is_static:
                out_pt = torch.empty(shape, dtype=torch_dtype)
                ov_out_tensor = ov.Tensor(out_pt.numpy(), shared_memory=True)
                self.infer_request.set_output_tensor(j, ov_out_tensor)
                outputs_prealloc.append(out_pt)
            else:
                outputs_prealloc.append(None)

        self.infer_request.infer()

        outputs = []
        for j in range(len(self.compiled_model.outputs)):
            if outputs_prealloc[j] is not None:
                outputs.append(outputs_prealloc[j])
            else:
                out_tensor = self.infer_request.get_output_tensor(j)
                outputs.append(torch.from_numpy(out_tensor.data).clone())

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def compile_to_npu(model: torch.nn.Module, example_input: Any) -> torch.nn.Module:
    global _GRAPH_CACHE

    logger.info("Starting NPU Compilation...")
    try:
        # 1. Generate Cache Key
        # We use a hash of the FX graph and input metadata as a key
        traced = torch.fx.symbolic_trace(model)
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
            # Add performance hints for NPU if available
            config = {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "f16"
            }

        logger.info(f"Compiling model for {target_device}...")
        compiled = core.compile_model(ov_model, target_device, config)

        # 4. Cache Management
        input_names = [p.friendly_name for p in builder.parameters]
        if len(_GRAPH_CACHE) >= _MAX_GRAPH_CACHE_SIZE:
            _GRAPH_CACHE.popitem(last=False)  # Evict oldest

        _GRAPH_CACHE[key] = {"model": compiled, "input_names": input_names}

        return NPUGraphModule(compiled, input_names)

    except Exception as e:
        logger.error(f"Compilation Failed: {e}")
        raise e
