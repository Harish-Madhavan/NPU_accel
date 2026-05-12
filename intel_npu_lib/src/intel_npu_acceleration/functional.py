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

# Ops without explicit autograd support (NPU dispatch, no gradient tape)
div = F_base.div
neg = F_base.neg
transpose = F_base.transpose
reshape = F_base.reshape
max_pool2d = F_base.max_pool2d
update_kv_cache = F_base.update_kv_cache
identity = F_base.identity
dropout = F_base.dropout
layer_norm = F_base.layer_norm
