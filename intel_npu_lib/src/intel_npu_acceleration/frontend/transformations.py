import re
import logging
import numpy as np
import openvino as ov
import openvino.opset13 as ops
from typing import Optional, List, Set, Dict, Any

logger = logging.getLogger(__name__)


def clean_node_name(n: str) -> str:
    """Normalize and clean node names for robust placeholder-to-parameter matching."""
    n = re.sub(r"^[lL]__?", "", n)
    return re.sub(r"[\d._]+$", "", n).lower()


def fold_scalar_parameter_inputs(
    ov_model: ov.Model,
    example_input_tuple: tuple,
    all_placeholder_names: List[str],
) -> ov.Model:
    """
    Replace scalar Parameter inputs with Constant nodes to enable compile-time
    folding of dynamic shapes, slices, and scalar control signals.
    """
    placeholder_name_to_idx = {
        name.lower(): idx for idx, name in enumerate(all_placeholder_names)
    }
    cleaned_placeholder_to_idx = {
        clean_node_name(name): idx for idx, name in enumerate(all_placeholder_names)
    }

    parameters_to_keep = []
    for i, param in enumerate(ov_model.get_parameters()):
        name = param.get_friendly_name()
        name_lower = name.lower()

        idx = placeholder_name_to_idx.get(name_lower)
        if idx is None:
            idx = cleaned_placeholder_to_idx.get(clean_node_name(name), i)

        orig_val = (
            example_input_tuple[idx] if idx < len(example_input_tuple) else None
        )
        if isinstance(orig_val, (int, float)):
            orig_type = param.get_element_type()
            if orig_type == ov.Type.i64:
                const_node = ops.constant(int(orig_val), dtype=np.int64)
            elif orig_type in [ov.Type.i32, ov.Type.i8, ov.Type.u8]:
                const_node = ops.constant(int(orig_val), dtype=np.int32)
            else:
                const_node = ops.constant(float(orig_val), dtype=np.float32)
            param.output(0).replace(const_node.output(0))
            logger.debug(
                f"Replaced scalar input parameter '{name}' with Constant {orig_val}"
            )
        else:
            parameters_to_keep.append(param)

    # Reconstruct Model with filtered parameter list
    results = [
        r.output(0) if hasattr(r, "output") else r for r in ov_model.get_results()
    ]
    return ov.Model(results, parameters_to_keep, ov_model.get_friendly_name())


def reshape_model_inputs_to_static(
    ov_model: ov.Model,
    example_input_tuple: tuple,
    all_placeholder_names: List[str],
) -> None:
    """
    Reshape remaining non-scalar inputs to static shapes to satisfy NPU upper bound constraints.
    """
    new_shapes = {}
    if len(ov_model.inputs) == len(example_input_tuple):
        # Exact 1-to-1 positional correspondence
        for i, inp in enumerate(ov_model.inputs):
            name = inp.any_name
            val = example_input_tuple[i]
            if hasattr(val, "shape"):
                new_shapes[name] = list(val.shape)
            else:
                new_shapes[name] = [1]
    else:
        placeholder_name_to_idx = {
            name.lower(): idx for idx, name in enumerate(all_placeholder_names)
        }
        cleaned_placeholder_to_idx = {
            clean_node_name(name): idx for idx, name in enumerate(all_placeholder_names)
        }

        for i, inp in enumerate(ov_model.inputs):
            name = inp.any_name
            name_lower = name.lower()

            idx = placeholder_name_to_idx.get(name_lower)
            if idx is None:
                idx = cleaned_placeholder_to_idx.get(clean_node_name(name))

            if idx is not None and idx < len(example_input_tuple):
                val = example_input_tuple[idx]
                if hasattr(val, "shape"):
                    new_shapes[name] = list(val.shape)
                else:
                    new_shapes[name] = [1]
            elif i < len(example_input_tuple) and hasattr(example_input_tuple[i], "shape"):
                new_shapes[name] = list(example_input_tuple[i].shape)
            else:
                p_shape = inp.get_partial_shape()
                sh = []
                for dim in p_shape:
                    if dim.is_dynamic:
                        sh.append(1)
                    else:
                        sh.append(dim.get_length())
                new_shapes[name] = sh
    ov_model.reshape(new_shapes)


def transform_stateful_kv_cache(
    ov_model: ov.Model,
    stateful: bool,
    stateful_modules: List[Any],
) -> ov.Model:
    """
    Transform functional KV cache Scatter operations into stateful ReadValue / Assign
    variable sinks mapped directly to NPU internal memory registers.
    """
    if not stateful and len(stateful_modules) == 0:
        return ov_model

    range_nodes = []
    scatter_nodes = []
    for op in ov_model.get_ordered_ops():
        if op.get_type_name() == "Range":
            range_nodes.append(op)
        elif "Scatter" in op.get_type_name():
            scatter_nodes.append(op)

    if len(scatter_nodes) == 0:
        return ov_model

    sinks = []
    variables = []
    replaced_param_names = set()

    for idx in range(len(scatter_nodes)):
        first_input_node = scatter_nodes[idx].input_value(0).get_node()
        if first_input_node.get_type_name() == "Parameter":
            replaced_param_names.add(first_input_node.get_friendly_name())

        p_shape = scatter_nodes[idx].input_value(0).get_partial_shape()
        cache_shape = []
        for dim in p_shape:
            if dim.is_dynamic:
                cache_shape.append(1)
            else:
                cache_shape.append(dim.get_length())

        pos_shape = [1]

        # Cache variable (f32)
        info_cache = ov.op.util.VariableInfo()
        info_cache.data_shape = ov.PartialShape(cache_shape)
        info_cache.data_type = ov.Type.f32
        info_cache.variable_id = f"auto_cache_functional_{idx}_cache_state"
        var_cache = ov.op.util.Variable(info_cache)
        variables.append(var_cache)

        # Force position variable to f32 to satisfy NPU hardware constraints
        info_pos = ov.op.util.VariableInfo()
        info_pos.data_shape = ov.PartialShape(pos_shape)
        info_pos.data_type = ov.Type.f32
        info_pos.variable_id = f"auto_cache_functional_{idx}_pos_state"
        var_pos = ov.op.util.Variable(info_pos)
        variables.append(var_pos)

        read_cache = ops.read_value(var_cache)
        read_pos = ops.read_value(var_pos)

        updates_val = scatter_nodes[idx].input_value(2)
        new_kv_shape = ops.shape_of(updates_val)
        seq_len = ops.gather(
            new_kv_shape,
            ops.constant([1], dtype=np.int32),
            ops.constant([0], dtype=np.int32),
        )
        seq_len = ops.squeeze(seq_len, ops.constant([0], dtype=np.int32))

        seq_len_f32 = ops.convert(seq_len, destination_type=ov.Type.f32)
        seq_len_1d = ops.reshape(
            seq_len_f32, ops.constant([1], dtype=np.int64), special_zero=False
        )

        new_pos = ops.add(read_pos, seq_len_1d)

        # Convert read_pos to i32 only for ScatterUpdate indices
        pos_i32 = ops.convert(read_pos, destination_type=ov.Type.i32)

        # Standard ScatterUpdate: axis = 1
        axis_const = ops.constant([1], dtype=np.int32)
        updated_cache = ops.scatter_update(
            read_cache, pos_i32, updates_val, axis_const
        )

        # Replace all usages of the original ScatterElementsUpdate node with ScatterUpdate node
        scatter_nodes[idx].output(0).replace(updated_cache.output(0))

        # Assign sinks
        assign_cache = ops.assign(updated_cache, var_cache)
        sinks.append(assign_cache)

        assign_pos = ops.assign(new_pos, var_pos)
        sinks.append(assign_pos)

    # Reconstruct final stateful Model with only active remaining Parameters
    remaining_params = [
        p
        for p in ov_model.get_parameters()
        if p.get_friendly_name() not in replaced_param_names
    ]
    results_outputs = [
        r.output(0) if hasattr(r, "output") else r for r in ov_model.get_results()
    ]

    # Perform reachability analysis from outputs and sinks to clean up unused parameters
    reachable_nodes = set()
    queue = []
    for r in results_outputs:
        node = r.get_node() if hasattr(r, "get_node") else r
        if node not in reachable_nodes:
            reachable_nodes.add(node)
            queue.append(node)
    for s in sinks:
        node = s.get_node() if hasattr(s, "get_node") else s
        if node not in reachable_nodes:
            reachable_nodes.add(node)
            queue.append(node)

    while queue:
        curr = queue.pop(0)
        for inp in curr.inputs():
            src_node = inp.get_source_output().get_node()
            if src_node not in reachable_nodes:
                reachable_nodes.add(src_node)
                queue.append(src_node)

    remaining_params = [p for p in remaining_params if p in reachable_nodes]

    return ov.Model(
        results_outputs,
        sinks,
        remaining_params,
        variables,
        "Stateful_NPU_Model",
    )


def configure_input_ppp(inp_info: Any, cfg: Dict[str, Any]) -> None:
    """Configure input pre-processing steps using OpenVINO PrePostProcessor."""
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
        cf = (
            getattr(ColorFormat, cf_name)
            if hasattr(ColorFormat, cf_name)
            else ColorFormat.BGR
        )
        inp_info.tensor().set_color_format(cf)

    if "model_color_format" in cfg:
        from openvino.preprocess import ColorFormat

        cf_name = cfg["model_color_format"]
        cf = (
            getattr(ColorFormat, cf_name)
            if hasattr(ColorFormat, cf_name)
            else ColorFormat.RGB
        )
        inp_info.preprocess().convert_color(cf)

    if "mean" in cfg or "scale" in cfg or "resize" in cfg:
        preprocess_steps = inp_info.preprocess()

        t_cfg = cfg.get("element_type", ov.Type.u8)
        if isinstance(t_cfg, str):
            t_cfg = getattr(ov.Type, t_cfg) if hasattr(ov.Type, t_cfg) else ov.Type.u8
        if t_cfg in [
            ov.Type.u8,
            ov.Type.i8,
            ov.Type.u16,
            ov.Type.i16,
            ov.Type.i32,
            ov.Type.i64,
        ]:
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


def configure_output_ppp(out_info: Any, cfg: Dict[str, Any]) -> None:
    """Configure output post-processing steps using OpenVINO PrePostProcessor."""
    if "element_type" in cfg:
        t = cfg["element_type"]
        if isinstance(t, str):
            t = getattr(ov.Type, t) if hasattr(ov.Type, t) else ov.Type.f32
        out_info.tensor().set_element_type(t)


def apply_pre_post_processing(
    ov_model: ov.Model, preprocess_config: Optional[Dict[str, Any]]
) -> ov.Model:
    """
    Apply OpenVINO PrePostProcessor (PPP) pipeline to offload layout transpositions,
    precision casting, and normalizations directly into the compiled NPU graph.
    """
    if not preprocess_config:
        return ov_model

    from openvino.preprocess import PrePostProcessor

    ppp = PrePostProcessor(ov_model)

    if "input" in preprocess_config:
        inp_cfg = preprocess_config["input"]
        if isinstance(inp_cfg, dict) and not any(
            k in ["layout", "element_type", "model_layout", "mean", "scale"]
            for k in inp_cfg.keys()
        ):
            for idx, item_cfg in inp_cfg.items():
                configure_input_ppp(ppp.input(idx), item_cfg)
        else:
            configure_input_ppp(ppp.input(0), inp_cfg)

    if "output" in preprocess_config:
        out_cfg = preprocess_config["output"]
        if isinstance(out_cfg, dict) and not any(
            k in ["element_type"] for k in out_cfg.keys()
        ):
            for idx, item_cfg in out_cfg.items():
                configure_output_ppp(ppp.output(idx), item_cfg)
        else:
            configure_output_ppp(ppp.output(0), out_cfg)

    return ppp.build()


def optimize_ov_model(ov_model: ov.Model, precision: str = "auto") -> ov.Model:
    """
    Run built-in OpenVINO transformation passes (ConstantFolding, Validate, ConvertFP32ToFP16)
    to optimize the computational graph before compiling to NPU hardware.
    """
    try:
        import openvino.passes as ov_passes

        manager = ov_passes.Manager()
        if precision.lower() in ["fp16", "f16"]:
            if hasattr(ov_passes, "ConvertFP32ToFP16"):
                manager.register_pass(ov_passes.ConvertFP32ToFP16())
                logger.debug("Registered OpenVINO ConvertFP32ToFP16 precision transformation pass.")

        if hasattr(ov_passes, "ConstantFolding"):
            manager.register_pass(ov_passes.ConstantFolding())
        if hasattr(ov_passes, "Validate"):
            manager.register_pass(ov_passes.Validate())

        manager.run_passes(ov_model)
        logger.debug("Successfully executed OpenVINO Pass Manager.")
    except Exception as e:
        logger.debug(f"OpenVINO Pass Manager skipped: {e}")
    return ov_model


def serialize_openvino_model(
    ov_model: ov.Model, xml_path: str, bin_path: Optional[str] = None
) -> None:
    """
    Serialize an OpenVINO Model into Intermediate Representation (.xml and .bin).
    """
    if bin_path:
        ov.serialize(ov_model, xml_path, bin_path)
    else:
        ov.serialize(ov_model, xml_path)

