import torch
import torch.fx
import hashlib
import logging
from collections import OrderedDict
import openvino as ov
import openvino.properties as ov_props
import openvino.properties.hint as ov_hints
from typing import Any, Optional, List
import numpy as np

import intel_npu_acceleration as npu_lib
from ..registry import OpRegistry
from ..graph_builder import OVGraphBuilder, ValueCapturingInterpreter
from .graph_module import NPUGraphModule, NPUDynamicGraphModule
from .tracer import NPUTracer

logger = logging.getLogger(__name__)

# --- Global Cache for Compiled Graphs ---
_OV_CORE = None
_GRAPH_CACHE = OrderedDict()
_MAX_GRAPH_CACHE_SIZE = 100


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


class NPUCompilationError(Exception):
    """Exception raised for errors during NPU compilation."""
    pass


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
    stateful: bool = False,
) -> torch.nn.Module:
    """
    Compile a PyTorch model for Intel NPU.
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
        stateful,
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
    stateful: bool = False,
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
        if isinstance(model, torch.fx.GraphModule):
            traced = model
        else:
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

        # Identify cache placeholders used in functional update_kv_cache calls for auto-mapping
        cache_placeholders = set()
        parent_cache_placeholders = set()
        from .._functional import update_kv_cache as npu_update_kv_cache_func
        if stateful:
            import operator
            def trace_to_placeholder(n):
                curr = n
                while isinstance(curr, torch.fx.Node) and curr.op == "call_function" and curr.target in [operator.getitem, getattr]:
                    if len(curr.args) > 0:
                        curr = curr.args[0]
                    else:
                        break
                if isinstance(curr, torch.fx.Node) and curr.op == "placeholder":
                    return curr
                return None

            for node in traced.graph.nodes:
                if node.op == "call_function" and node.target == npu_update_kv_cache_func:
                    cache_node = node.args[0]
                    placeholder_node = trace_to_placeholder(cache_node)
                    if placeholder_node is not None:
                        cache_placeholders.add(cache_node.name)
                        parent_cache_placeholders.add(placeholder_node.name)

        # Temporarily dequantize Linear weights for compilation (Interpreter & ShapeProp)
        quantized_weights = {}
        for name, submod in traced.named_modules():
            if isinstance(submod, torch.nn.Linear):
                weight = submod._parameters.get("weight", None)
                if weight is not None and getattr(weight, "data", None) is not None:
                    if weight.data.dtype in [torch.int8, torch.uint8]:
                        quantized_weights[submod] = weight.data
                        submod.weight.data = weight.data.float()

        try:
            # 2. Capture Values & Build OV Graph
            interpreter = ValueCapturingInterpreter(traced)
            interpreter.run(*example_input_tuple)

            builder = OVGraphBuilder(interpreter.node_values)
            builder.stateful = stateful
            builder.cache_placeholders = cache_placeholders

            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(traced).propagate(*example_input_tuple)

            # Restore original quantized weights immediately after eager/ShapeProp phases
            for submod, w_data in quantized_weights.items():
                submod.weight.data = w_data

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

                    if stateful and node.name in parent_cache_placeholders and node.name not in cache_placeholders:
                        # Skip this top-level placeholder, as its individual sliced sub-graphs
                        # are directly backed by native stateful variables.
                        pass
                    elif stateful and node.name in cache_placeholders:
                        import openvino.opset13 as ops
                        ov_type = ov.Type.f32
                        if dtype == torch.float16:
                            ov_type = ov.Type.f16
                        elif dtype == torch.int32:
                            ov_type = ov.Type.i32
                        
                        info = ov.op.util.VariableInfo()
                        info.data_shape = ov.PartialShape(list(shape))
                        info.data_type = ov_type
                        info.variable_id = f"auto_cache_{node.name}"
                        
                        var = ov.op.util.Variable(info)
                        builder.variables.append(var)
                        
                        read_node = ops.read_value(var)
                        builder.register_output(node.name, read_node)
                    else:
                        builder.add_parameter(node.name, list(shape), dtype)

                elif node.op == "call_function":
                    if stateful and node.name in cache_placeholders:
                        import openvino.opset13 as ops
                        if "tensor_meta" in node.meta:
                            shape = node.meta["tensor_meta"].shape
                            dtype = node.meta["tensor_meta"].dtype
                        else:
                            shape = [1, 128, 4, 16]  # default fallback
                            dtype = torch.float32

                        ov_type = ov.Type.f32
                        if dtype == torch.float16:
                            ov_type = ov.Type.f16
                        elif dtype == torch.int32:
                            ov_type = ov.Type.i32
                        
                        info = ov.op.util.VariableInfo()
                        info.data_shape = ov.PartialShape(list(shape))
                        info.data_type = ov_type
                        info.variable_id = f"auto_cache_{node.name}"
                        
                        var = ov.op.util.Variable(info)
                        builder.variables.append(var)
                        
                        read_node = ops.read_value(var)
                        builder.register_output(node.name, read_node)
                    else:
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
        finally:
            # Restore original quantized weights
            for submod, w_data in quantized_weights.items():
                submod.weight.data = w_data

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

        # 4. Cache Management
        input_names = [p.friendly_name for p in builder.parameters]
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


def _configure_input_ppp(inp_info, cfg):
    if "shape" in cfg:
        inp_info.tensor().set_shape(list(cfg["shape"]))
    if "layout" in cfg:
        inp_info.tensor().set_layout(ov.Layout(cfg["layout"]))
    if "element_type" in cfg:
        t = cfg["element_type"]
        if isinstance(t, str):
            t = getattr(ov.Type, t) if hasattr(ov.Type, t) else ov.Type.u8
        inp_info.tensor().set_element_type(t)
    if "model_layout" in cfg:
        inp_info.model().set_layout(ov.Layout(cfg["model_layout"]))

    if "color_format" in cfg:
        from openvino.preprocess import ColorFormat
        cf_name = cfg["color_format"]
        cf = getattr(ColorFormat, cf_name) if hasattr(ColorFormat, cf_name) else ColorFormat.BGR
        inp_info.tensor().set_color_format(cf)

    if "model_color_format" in cfg:
        from openvino.preprocess import ColorFormat
        cf_name = cfg["model_color_format"]
        cf = getattr(ColorFormat, cf_name) if hasattr(ColorFormat, cf_name) else ColorFormat.RGB
        inp_info.preprocess().convert_color(cf)

    if "mean" in cfg or "scale" in cfg or "resize" in cfg:
        preprocess_steps = inp_info.preprocess()

        t_cfg = cfg.get("element_type", ov.Type.u8)
        if isinstance(t_cfg, str):
            t_cfg = getattr(ov.Type, t_cfg) if hasattr(ov.Type, t_cfg) else ov.Type.u8
        if t_cfg in [ov.Type.u8, ov.Type.i8, ov.Type.u16, ov.Type.i16, ov.Type.i32, ov.Type.i64]:
            preprocess_steps.convert_element_type(ov.Type.f32)

        if "resize" in cfg:
            from openvino.preprocess import ResizeAlgorithm
            h_target, w_target = cfg["resize"]
            preprocess_steps.resize(ResizeAlgorithm.RESIZE_LINEAR, h_target, w_target)

        if "mean" in cfg:
            preprocess_steps.mean(cfg["mean"])
        if "scale" in cfg:
            preprocess_steps.scale(cfg["scale"])

    if "layout" in cfg and "model_layout" in cfg:
        inp_info.preprocess().convert_layout(ov.Layout(cfg["model_layout"]))


def _configure_output_ppp(out_info, cfg):
    if "element_type" in cfg:
        t = cfg["element_type"]
        if isinstance(t, str):
            t = getattr(ov.Type, t) if hasattr(ov.Type, t) else ov.Type.f32
        out_info.tensor().set_element_type(t)


# --- TorchDynamo Backend Registration ---
try:
    from torch._dynamo import register_backend

    @register_backend
    def npu(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor], **kwargs):
        """
        Dynamo backend for compiling PyTorch models natively via torch.compile(model, backend='npu')
        """
        if "options" in kwargs:
            opts = kwargs.pop("options")
            if isinstance(opts, dict):
                kwargs.update(opts)

        compiled = compile_to_npu(gm, tuple(example_inputs), **kwargs)

        output_node = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node
                break

        if output_node is not None:
            ret_vals = output_node.args[0]
            is_tuple = isinstance(ret_vals, tuple)
            is_list = isinstance(ret_vals, list)

            if is_tuple:
                num_outputs = len(ret_vals)
                def tuple_wrapper(*args, **wrapper_kwargs):
                    res = compiled(*args, **wrapper_kwargs)
                    if num_outputs == 1:
                        if isinstance(res, tuple):
                            return res
                        return (res,)
                    return tuple(res) if isinstance(res, (list, tuple)) else (res,)
                return tuple_wrapper
            elif is_list:
                num_outputs = len(ret_vals)
                def list_wrapper(*args, **wrapper_kwargs):
                    res = compiled(*args, **wrapper_kwargs)
                    if num_outputs == 1:
                        if isinstance(res, list):
                            return res
                        return [res]
                    return list(res) if isinstance(res, (list, tuple)) else [res]
                return list_wrapper

        return compiled
except Exception:
    pass


