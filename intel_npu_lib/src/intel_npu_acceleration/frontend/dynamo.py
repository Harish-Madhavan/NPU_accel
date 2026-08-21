"""
TorchDynamo Backend Registration for PyTorch 2.0 (torch.compile(model, backend='npu')).
Seamlessly plugs Intel NPU hardware acceleration into standard PyTorch compilation APIs.
"""

import logging
from typing import List
import torch
import torch.fx

logger = logging.getLogger("intel_npu_acceleration.frontend.dynamo")

try:
    from torch._dynamo import register_backend

    def _compile_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
        """
        Dynamo backend for compiling PyTorch models natively via torch.compile(model, backend='npu').
        Plumbs standard PyTorch parameters (mode, dynamic, options, autocast) into NPU hardware optimizations.
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
        if torch.is_autocast_enabled() or (hasattr(torch, "is_autocast_cpu_enabled") and torch.is_autocast_cpu_enabled()):
            kwargs.setdefault("precision", "fp16")

        # Prune unused placeholders (e.g. Dynamo SymInt symbols with 0 users in dynamic shapes)
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        used_mask = []
        filtered_inputs = []
        for node, val in zip(placeholders, example_inputs):
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

    # Register multiple convenient backend names for seamless PyTorch 2.x integration
    register_backend(compiler_fn=_compile_backend, name="npu")
    register_backend(compiler_fn=_compile_backend, name="intel_npu")
    npu = _compile_backend

except Exception as e:
    logger.debug(f"TorchDynamo backend registration not active: {e}")
