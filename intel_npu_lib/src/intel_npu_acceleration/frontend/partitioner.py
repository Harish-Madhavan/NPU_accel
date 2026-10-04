import logging
from typing import Any

import torch
import torch.fx

from intel_npu_acceleration.registry import OpRegistry

logger = logging.getLogger(__name__)


def _is_node_supported(node: torch.fx.Node, root_model: torch.nn.Module) -> bool:
    """Check if an FX graph node is supported by the OpRegistry."""
    if node.op == "call_function":
        return OpRegistry.is_function_supported(node.target)
    if node.op == "call_method":
        return OpRegistry.is_method_supported(node.target)
    if node.op == "call_module":
        target_sub = root_model
        try:
            for atom in str(node.target).split("."):
                target_sub = getattr(target_sub, atom)
        except AttributeError:
            return False
        return OpRegistry.get_module(type(target_sub)) is not None
    return True


# Dynamo-injected boilerplate with no numerical effect on inference.
# A single such node (e.g. _set_grad_enabled) poisons its entire segment to
# CPU, so they are pruned before partitioning. Both return None / have no
# data users by construction; only erase when that holds.
_DYNAMO_NOOP_TARGETS = frozenset({
    torch._C._set_grad_enabled,
    torch._C._log_api_usage_once,
})


def _prune_dynamo_noops(traced: torch.fx.GraphModule) -> None:
    """Erase side-effect-free dynamo boilerplate nodes from a traced graph."""
    pruned = False
    for node in list(traced.graph.nodes):
        if (
            node.op == "call_function"
            and node.target in _DYNAMO_NOOP_TARGETS
            and len(node.users) == 0
        ):
            traced.graph.erase_node(node)
            pruned = True
    if pruned:
        traced.graph.lint()
        traced.recompile()


def _outputs_match_calibration(name: str, compiled_sub: Any, submod_outputs: dict) -> bool:
    """Verify an NPU-compiled segment preserves its eager output dtypes.

    The NPU boundary does not preserve every dtype losslessly (proven case:
    boolean mask outputs materialize as u8, breaking downstream eager ops
    such as SDPA masks). The calibration forward pass records ground-truth
    outputs; any dtype divergence rejects the NPU binary for that segment.
    Missing calibration data conservatively passes (legacy behavior).
    """
    if name not in submod_outputs:
        return True
    ref = submod_outputs[name]
    ref_list = list(ref) if isinstance(ref, (tuple, list)) else [ref]
    infos = getattr(compiled_sub, "output_info", None)
    if not infos or len(infos) != len(ref_list):
        return True
    for (_, _, dtype), t in zip(infos, ref_list, strict=False):
        if isinstance(t, torch.Tensor) and dtype is not None and dtype != t.dtype:
            return False
    return True


def _calibrate_children(parent: Any, example_inputs: tuple) -> tuple[dict, dict]:
    """Run a parent eagerly, recording per-child inputs and outputs.

    Inputs resolve submodule shapes for compilation; outputs are the
    ground truth for the NPU boundary dtype contract check.
    """
    submod_inputs: dict = {}
    submod_outputs: dict = {}
    hooks = []

    def make_pre_hook(name):
        def hook(module, inputs):
            submod_inputs[name] = inputs
            return None

        return hook

    def make_post_hook(name):
        def hook(module, inputs, output):
            submod_outputs[name] = output
            return None

        return hook

    for name, child in parent.named_children():
        hooks.append(child.register_forward_pre_hook(make_pre_hook(name)))
        hooks.append(child.register_forward_hook(make_post_hook(name)))

    with torch.no_grad():
        try:
            parent(*example_inputs)
        except Exception as e:
            logger.debug(f"Calibration forward pass notice: {e}")
        finally:
            for h in hooks:
                h.remove()
    return submod_inputs, submod_outputs


def _compute_node_count(child: Any) -> int:
    graph = getattr(child, "graph", None)
    if graph is None:
        return 0
    return sum(1 for n in graph.nodes if n.op not in ("placeholder", "output", "get_attr"))


def _is_flat_child(child: Any) -> bool:
    """True when a child can be re-split without external scope.

    Only flat dynamo-style graphs (no call_module/get_attr references)
    qualify; anything else falls back to CPU on failure as before.
    """
    graph = getattr(child, "graph", None)
    if graph is None:
        return False
    return all(n.op != "call_module" and n.op != "get_attr" for n in graph.nodes)


# Bounds for adaptive re-splitting: an oversized/fragile segment that fails
# NPU compilation is bisected (bounded depth) so it degrades to compilable
# chunks instead of falling back to CPU wholesale.
_MIN_SPLIT_NODES = 150
_MAX_SPLIT_DEPTH = 3


def _compile_segment(
    parent: Any,
    name: str,
    child: Any,
    submod_inputs: dict,
    submod_outputs: dict,
    ctx: dict,
    depth: int,
) -> None:
    """Compile one segment for NPU, mutating the parent in place.

    On NPU compile failure, flat oversized segments are bisected and each
    half is compiled recursively; anything else falls back to CPU.
    """
    from .compiler import compile_to_npu  # Delay-import to avoid circular dependency

    is_supported = True
    if not hasattr(child, "graph"):
        is_supported = OpRegistry.is_module_supported(type(child))
    elif _compute_node_count(child) == 0:
        return  # wiring-only segment; nothing to compile
    else:
        for n in child.graph.nodes:
            if n.op not in ["placeholder", "output", "get_attr"]:
                if not _is_node_supported(n, child):
                    is_supported = False
                    break

    if not is_supported or name not in submod_inputs:
        logger.info(f"Partition: Leaving unsupported submodule '{name}' on CPU.")
        return

    logger.info(f"Partition: Compiling supported submodule '{name}' for NPU...")
    try:
        compiled_sub = compile_to_npu(
            child,
            submod_inputs[name],
            ctx["performance_hint"],
            ctx["num_streams"],
            strict=True,
            dynamic_buckets=ctx["dynamic_buckets"],
            bucket_sizes=ctx["bucket_sizes"],
            dynamic_dim=ctx["dynamic_dim"],
            clone_outputs=ctx["clone_outputs"],
            preprocess_config=ctx["preprocess_config"],
        )
    except Exception as e:
        _split_and_retry(parent, name, child, submod_inputs[name], ctx, depth, str(e))
        return
    if not _outputs_match_calibration(name, compiled_sub, submod_outputs):
        logger.warning(
            f"Partition: NPU binary for submodule '{name}' changes output "
            "dtypes vs eager (e.g. bool materialized as u8 at the NPU "
            "boundary). Falling back to CPU."
        )
        return
    setattr(parent, name, compiled_sub)


def _split_and_retry(
    parent: Any,
    name: str,
    child: Any,
    sub_inputs: tuple,
    ctx: dict,
    depth: int,
    reason: str,
) -> None:
    """Bisect a failed oversized segment and compile each half recursively."""
    from torch.fx.passes.split_module import split_module

    n = _compute_node_count(child)
    if n <= _MIN_SPLIT_NODES or depth >= _MAX_SPLIT_DEPTH or not _is_flat_child(child):
        logger.warning(
            f"Failed to compile submodule '{name}' for NPU: {reason}. Falling back to CPU."
        )
        return
    logger.warning(
        f"NPU compile failed for '{name}' ({n} nodes): {reason}. "
        f"Retrying as halves (depth {depth + 1})."
    )
    try:
        compute = [
            nd for nd in child.graph.nodes if nd.op not in ("placeholder", "output", "get_attr")
        ]
        first_half = set(compute[: len(compute) // 2])

        def classify(nd: torch.fx.Node) -> int:
            if nd.op in ("placeholder", "get_attr"):
                return 0
            if nd.op == "output":
                return 1
            return 0 if nd in first_half else 1

        halves = split_module(child, child, classify)
        h_inputs, h_outputs = _calibrate_children(halves, sub_inputs)
        for hname, hchild in list(halves.named_children()):
            _compile_segment(halves, hname, hchild, h_inputs, h_outputs, ctx, depth + 1)
        setattr(parent, name, halves)
    except Exception as e:
        logger.debug(f"Segment split retry failed for '{name}': {e}. Falling back to CPU.")


def partition_and_compile_hybrid(
    model: torch.nn.Module,
    traced: torch.fx.GraphModule,
    example_input_tuple: tuple,
    performance_hint: str,
    num_streams: int,
    dynamic_buckets: bool,
    bucket_sizes: list[int] | None,
    dynamic_dim: int,
    clone_outputs: bool,
    preprocess_config: dict | None,
) -> torch.nn.Module:
    """
    Automated Hybrid Graph Partitioning & CPU Fallback.
    Partitions the graph into supported NPU subgraphs and unsupported CPU subgraphs,
    recursively compiling the supported parts for the NPU.
    """
    # Drop dynamo boilerplate first: a single no-op node would otherwise
    # poison its whole segment to CPU.
    _prune_dynamo_noops(traced)

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

    submod_inputs, submod_outputs = _calibrate_children(split_parent, example_input_tuple)

    ctx = {
        "performance_hint": performance_hint,
        "num_streams": num_streams,
        "dynamic_buckets": dynamic_buckets,
        "bucket_sizes": bucket_sizes,
        "dynamic_dim": dynamic_dim,
        "clone_outputs": clone_outputs,
        "preprocess_config": preprocess_config,
    }

    # Compile supported submodules (recursively splitting on failure)
    for name, child in list(split_parent.named_children()):
        _compile_segment(split_parent, name, child, submod_inputs, submod_outputs, ctx, depth=0)

    from .graph_module import NPUAsyncFuture, NPUGraphModule

    npu_subs = [name for name, c in split_parent.named_children() if isinstance(c, NPUGraphModule)]
    cpu_subs = [name for name, c in split_parent.named_children() if not isinstance(c, NPUGraphModule)]

    def reset_states():
        for child in split_parent.children():
            if hasattr(child, "reset_states"):
                child.reset_states()

    def infer_async(*args):
        return split_parent(*args)

    def wait_async(handle):
        return handle

    def submit(*args, **kwargs):
        res = split_parent(*args, **kwargs)
        fut = NPUAsyncFuture(split_parent, 0)
        fut._result = res
        fut._completed = True
        return fut

    def batch_infer(inputs_list):
        return [split_parent(*(inp if isinstance(inp, tuple) else (inp,))) for inp in inputs_list]

    split_parent.reset_states = reset_states
    split_parent.infer_async = infer_async
    split_parent.wait_async = wait_async
    split_parent.submit = submit
    split_parent.batch_infer = batch_infer
    split_parent.npu_submodules = npu_subs
    split_parent.cpu_submodules = cpu_subs
    split_parent.performance_hint = performance_hint
    split_parent.num_streams = num_streams
    split_parent.clone_outputs = clone_outputs

    return split_parent
