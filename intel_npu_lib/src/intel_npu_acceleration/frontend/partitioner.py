from typing import Optional, List, Dict, Any
import torch
import torch.fx
import logging
from ..registry import OpRegistry

logger = logging.getLogger(__name__)


def _is_node_supported(node: torch.fx.Node, root_model: torch.nn.Module) -> bool:
    """Check if an FX graph node is supported by the OpRegistry."""
    if node.op == "call_function":
        return OpRegistry.get_function(node.target) is not None
    if node.op == "call_method":
        return OpRegistry.get_method(node.target) is not None
    if node.op == "call_module":
        target_sub = root_model
        for atom in str(node.target).split("."):
            target_sub = getattr(target_sub, atom)
        return OpRegistry.get_module(type(target_sub)) is not None
    return True


def partition_and_compile_hybrid(
    model: torch.nn.Module,
    traced: torch.fx.GraphModule,
    example_input_tuple: tuple,
    performance_hint: str,
    num_streams: int,
    dynamic_buckets: bool,
    bucket_sizes: Optional[List[int]],
    dynamic_dim: int,
    clone_outputs: bool,
    preprocess_config: Optional[dict],
) -> torch.nn.Module:
    """
    Automated Hybrid Graph Partitioning & CPU Fallback.
    Partitions the graph into supported NPU subgraphs and unsupported CPU subgraphs,
    recursively compiling the supported parts for the NPU.
    """
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

        is_supported = _is_node_supported(node, model)

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

    from .compiler import compile_to_npu  # Delay-import to avoid circular dependency

    # Compile supported submodules recursively
    for name, child in list(split_parent.named_children()):
        is_supported = True

        # Check if the child module is a GraphModule before checking graph.nodes
        if not hasattr(child, "graph"):
            is_supported = OpRegistry.get_module(type(child)) is not None
        else:
            for n in child.graph.nodes:
                if n.op not in ["placeholder", "output", "get_attr"]:
                    if not _is_node_supported(n, child):
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

    def reset_states():
        for child in split_parent.children():
            if hasattr(child, "reset_states"):
                child.reset_states()

    def infer_async(*args):
        return split_parent(*args)

    def wait_async(handle):
        return handle

    split_parent.reset_states = reset_states
    split_parent.infer_async = infer_async
    split_parent.wait_async = wait_async

    return split_parent
