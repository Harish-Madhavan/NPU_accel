"""
TorchDynamo Backend Registration for PyTorch 2.0+ (torch.compile(model, backend='npu')).

This module registers the 'npu' and 'intel_npu' custom backend compilers in PyTorch's
TorchDynamo system, translating standard PyTorch compilation parameters (`mode`, `dynamic`,
`options`, and `torch.autocast`) into low-level Intel oneAPI Level Zero and OpenVINO runtime
optimizations.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.fx

logger = logging.getLogger("intel_npu_acceleration.frontend.dynamo")

try:
    from torch._dynamo import register_backend

    def _compile_backend(
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
        **kwargs: Any
    ) -> Callable[..., Any]:
        """TorchDynamo compiler backend entrypoint for Intel NPU compilation.

        Translates idiomatic PyTorch 2.x compilation arguments:
        - `mode="reduce-overhead"` -> LATENCY performance mode, zero-copy buffer leasing, Turbo boost.
        - `mode="max-autotune"` -> THROUGHPUT mode, 4 parallel hardware streams, FP16 precision.
        - `dynamic=True` -> Dynamic shape bucket compilation with dead placeholder symbol pruning.
        - `torch.autocast(...)` -> FP16 execution precision.

        Args:
            gm (torch.fx.GraphModule): The traced FX graph module extracted by TorchDynamo.
            example_inputs (List[torch.Tensor]): Concrete sample tensors for shape/dtype discovery.
            **kwargs: Extra configuration options passed from `torch.compile(..., options={...})`.

        Returns:
            Callable: A callable compiled module executing on the Intel NPU.
        """
        from .compiler import compile_to_npu

        # Extract standard options dict
        options = kwargs.pop("options", {}) or {}
        if isinstance(options, dict):
            kwargs.update(options)

        # Standard PyTorch mode translation
        mode = kwargs.pop("mode", None)
        if mode == "reduce-overhead":
            kwargs.setdefault("performance_hint", "LATENCY")
            kwargs.setdefault("clone_outputs", False)
        elif mode in ["max-autotune", "max-autotune-no-cudagraphs"]:
            kwargs.setdefault("performance_hint", "THROUGHPUT")
            kwargs.setdefault("num_streams", 4)
            kwargs.setdefault("precision", "fp16")

        # Standard PyTorch dynamic shapes parameter
        dynamic = kwargs.pop("dynamic", None)
        if dynamic:
            kwargs.setdefault("dynamic_buckets", True)

        # Automatic PyTorch Autocast detection
        if torch.is_autocast_enabled() or (
            hasattr(torch, "is_autocast_cpu_enabled") and torch.is_autocast_cpu_enabled()
        ):
            kwargs.setdefault("precision", "fp16")

        # Prune unused placeholders (e.g. Dynamo SymInt symbols with 0 users in dynamic shapes)
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        used_mask = []
        filtered_inputs = []
        for node, val in zip(placeholders, example_inputs, strict=False):
            if len(node.users) == 0:
                gm.graph.erase_node(node)
                used_mask.append(False)
            else:
                filtered_inputs.append(val)
                used_mask.append(True)
        if not all(used_mask):
            gm.recompile()
            example_inputs = filtered_inputs

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

                def tuple_wrapper(*args: Any, **wrapper_kwargs: Any) -> tuple:
                    res = compiled(*args, **wrapper_kwargs)
                    if num_outputs == 1:
                        if isinstance(res, tuple):
                            return res
                        return (res,)
                    return tuple(res) if isinstance(res, (list, tuple)) else (res,)

                return tuple_wrapper
            elif is_list:
                num_outputs = len(ret_vals)

                def list_wrapper(*args: Any, **wrapper_kwargs: Any) -> list:
                    res = compiled(*args, **wrapper_kwargs)
                    if num_outputs == 1:
                        if isinstance(res, list):
                            return res
                        return [res]
                    return list(res) if isinstance(res, (list, tuple)) else [res]

                return list_wrapper

        return compiled

    register_backend(name="npu", compiler_fn=_compile_backend)
    register_backend(name="intel_npu", compiler_fn=_compile_backend)
    logger.debug("Successfully registered 'npu' and 'intel_npu' TorchDynamo backends.")

except Exception as e:
    logger.debug(f"TorchDynamo NPU backend registration skipped: {e}")
