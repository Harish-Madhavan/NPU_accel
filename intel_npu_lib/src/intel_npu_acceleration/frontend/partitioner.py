import torch
import torch.fx
import logging
from ..registry import OpRegistry

logger = logging.getLogger(__name__)


def partition_and_compile_hybrid(
    model: torch.nn.Module,
    traced: torch.fx.GraphModule,
    example_input_tuple: tuple,
    performance_hint: str,
    num_streams: int,
    dynamic_buckets: bool,
    bucket_sizes: list,
    dynamic_dim: int,
    clone_outputs: bool,
    preprocess_config: dict,
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

    from .compiler import compile_to_npu  # Delay-import to avoid circular dependency

    # Compile supported submodules recursively
    for name, child in list(split_parent.named_children()):
        is_supported = True
        
        # Check if the child module is a GraphModule before checking graph.nodes
        if not hasattr(child, "graph"):
            # This is an original leaf module copied directly to the split parent.
            # Check if it has a registered converter.
            if OpRegistry.get_module(type(child)) is not None:
                is_supported = True
            else:
                is_supported = False
        else:
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
