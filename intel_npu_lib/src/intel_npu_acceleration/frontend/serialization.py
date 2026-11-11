"""
Serialization and Deserialization Pipeline for Intel NPU Models.
Provides OpenVINO Intermediate Representation (IR) export and pre-compiled
Intel Level Zero hardware binary executable (.blob) export and import.
"""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING, Any

from .graph_module import NPUGraphModule

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("intel_npu_acceleration.frontend.serialization")


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
    from .compiler import _trace_and_convert_to_ov
    from .transformations import apply_pre_post_processing, serialize_openvino_model

    ov_model, _, _, _ = _trace_and_convert_to_ov(
        model, example_input, stateful=stateful, precision="auto"
    )

    if preprocess_config:
        ov_model = apply_pre_post_processing(ov_model, preprocess_config)

    serialize_openvino_model(ov_model, output_xml_path, output_bin_path)
    logger.info(f"Successfully exported OpenVINO IR model to '{output_xml_path}'")
    return output_xml_path


def export_compiled_model(npu_model: Any, output_blob_path: str) -> str:
    """Export a compiled NPUGraphModule directly to an Intel Level Zero binary executable blob (.blob).

    Saves the pre-compiled hardware executable to disk, enabling instant sub-second reloads
    via `import_compiled_model` without recurring PyTorch FX tracing or OpenVINO IR conversion overhead.

    Args:
        npu_model: The compiled `NPUGraphModule` to export.
        output_blob_path: Destination path for the `.blob` file.

    Returns:
        str: Absolute path to the saved blob file.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu_model = npu.compile_to_npu(model, x)
        >>> blob_path = npu.export_compiled_model(npu_model, "model.blob")
    """
    if not hasattr(npu_model, "compiled_model"):
        raise ValueError("Provided model is not an NPUGraphModule with a compiled_model attribute.")

    parent_dir = os.path.dirname(os.path.abspath(output_blob_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    buf = io.BytesIO()
    npu_model.compiled_model.export_model(buf)
    with open(output_blob_path, "wb") as f:
        f.write(buf.getvalue())
    logger.info(f"Successfully exported compiled NPU hardware blob to '{output_blob_path}'")
    return os.path.abspath(output_blob_path)


def import_compiled_model(
    blob_path: str,
    performance_hint: str | None = None,
    num_streams: int = 1,
    clone_outputs: bool = True,
    target_device: str = "NPU",
) -> NPUGraphModule:
    """Load a pre-compiled Intel Level Zero hardware binary blob (.blob) directly onto the NPU.

    Bypasses graph tracing, OpenVINO IR generation, and bytecode compilation entirely,
    achieving instant hardware instantiation in fractions of a second.

    Args:
        blob_path: Path to the pre-compiled `.blob` file.
        performance_hint: Target performance mode ('LATENCY', 'THROUGHPUT'). Defaults to global hint.
        num_streams: Number of parallel execution streams. Defaults to 1.
        clone_outputs: Whether to clone outputs or use leased buffers. Defaults to True.
        target_device: Execution device ('NPU'). Defaults to 'NPU'.

    Returns:
        NPUGraphModule: An executable module ready for inference on Intel NPU.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu_model = npu.import_compiled_model("model.blob")
        >>> out = npu_model(x)
    """
    if not os.path.exists(blob_path):
        raise FileNotFoundError(f"Compiled NPU blob not found: {blob_path}")

    from .compiler import _get_core, _live_l0_props

    core = _get_core()
    available_devices = core.available_devices
    if not any("NPU" in d for d in available_devices):
        target_device = "CPU"

    if performance_hint is None:
        from intel_npu_acceleration.device import get_performance_hint

        performance_hint = get_performance_hint()

    load_config = {}
    if target_device == "NPU":
        load_config = _live_l0_props()
        if performance_hint:
            load_config["PERFORMANCE_HINT"] = performance_hint
        if num_streams > 1:
            load_config["NUM_STREAMS"] = str(num_streams)

    with open(blob_path, "rb") as f:
        data = io.BytesIO(f.read())
        try:
            compiled = (
                core.import_model(data, target_device, config=load_config)
                if load_config
                else core.import_model(data, target_device)
            )
        except Exception:
            compiled = core.import_model(data, target_device)

    input_names = [p.any_name for p in compiled.inputs]
    logger.info(f"Successfully imported compiled NPU model from '{blob_path}'")
    return NPUGraphModule(
        compiled,
        input_names,
        performance_hint,
        num_streams,
        clone_outputs,
    )
