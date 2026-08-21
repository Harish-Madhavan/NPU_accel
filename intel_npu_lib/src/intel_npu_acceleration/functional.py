import torch.fx
from . import _functional as F_base
from . import autograd as F_auto

# --- Combined API (Autograd-aware) ---

# Ops with full autograd (forward on NPU, backward computed on CPU)
add = F_auto.add
sub = F_auto.sub
mul = F_auto.mul
matmul = F_auto.matmul
linear = F_auto.linear
relu = F_auto.relu
gelu = F_auto.gelu  # was F_base — now gradient-aware
silu = F_auto.silu  # was F_base — now gradient-aware
rmsnorm = F_auto.rmsnorm  # was F_base — now gradient-aware
softmax = F_auto.softmax
conv2d = F_auto.conv2d  # was F_base — now gradient-aware (groups OK)
layer_norm = F_auto.layer_norm
hardsigmoid = F_auto.hardsigmoid
hardswish = F_auto.hardswish
div = F_auto.div
neg = F_auto.neg
transpose = F_auto.transpose
reshape = F_auto.reshape
cat = F_auto.cat
stack = F_auto.stack
mean = F_auto.mean
embedding = F_auto.embedding
scaled_dot_product_attention = F_auto.scaled_dot_product_attention

# Ops without explicit autograd support (NPU dispatch, no gradient tape)
squeeze = F_base.squeeze
unsqueeze = F_base.unsqueeze
index_select = F_base.index_select
zeros = F_base.zeros
ones = F_base.ones
full = F_base.full
max_pool2d = F_base.max_pool2d
update_kv_cache = F_base.update_kv_cache
quantized_linear = F_base.quantized_linear
identity = F_base.identity
dropout = F_base.dropout

torch.fx.wrap(quantized_linear)
torch.fx.wrap(update_kv_cache)


# Compilation flag to ensure functional execution during native OpenVINO tracing
_IS_COMPILING = False

# Re-export NPUStatefulKVCache for backwards compatibility with functional namespace
from .nn.stateful_kv import NPUStatefulKVCache  # noqa: F401, E402

