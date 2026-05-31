import torch
import torch.fx


class NPUTracer(torch.fx.Tracer):
    """
    Custom FX Tracer for Intel NPU.
    Ensures custom functions like quantized_linear
    are always treated as leaf/atomic nodes in the graph rather than being traced through.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..functional import quantized_linear

        self._autowrap_function_ids.add(id(quantized_linear))

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        from ..functional import NPUStatefulKVCache
        if isinstance(m, NPUStatefulKVCache):
            return True
        if isinstance(m, torch.nn.Linear):
            weight = m._parameters.get("weight", None)
            if weight is not None and not isinstance(weight, torch.fx.Proxy):
                weight_data = getattr(weight, "data", None)
                if weight_data is not None and hasattr(weight_data, "dtype"):
                    if weight_data.dtype in [torch.int8, torch.uint8]:
                        return True
        return super().is_leaf_module(m, module_qualified_name)
