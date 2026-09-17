"""
NPU Graph Execution — compiled model wrapper with zero-copy and async support.

Optimized hot paths: static dtype maps, O(1) placeholder lookup, deduplicated
input/output binding.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

import numpy as np
import openvino as ov
import torch

from intel_npu_acceleration.config import config as npu_config
from intel_npu_acceleration.utils import clean_name


def _clean_name(n: str) -> str:
    # Dynamo/JIT placeholder names (e.g. 'l_x') keep their leading prefix —
    # stripping it causes false cleaned-name collisions across arguments.
    return clean_name(n, strip_leading=False)

# ---------------------------------------------------------------------------
# Static dtype maps — avoid per-init torch.from_numpy allocations
# ---------------------------------------------------------------------------

_OV_TO_NP: dict[Any, Any] = {
    ov.Type.f32: np.float32,
    ov.Type.f16: np.float16,
    ov.Type.i32: np.int32,
    ov.Type.i64: np.int64,
    ov.Type.i8: np.int8,
    ov.Type.u8: np.uint8,
    ov.Type.boolean: bool,
}
_NP_TO_TORCH: dict[Any, torch.dtype] = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.int8: torch.int8,
    np.uint8: torch.uint8,
    bool: torch.bool,
}

# Also support ov.Type keys directly
_OV_TO_TORCH: dict[Any, torch.dtype] = {
    ov.Type.f32: torch.float32,
    ov.Type.f16: torch.float16,
    ov.Type.i32: torch.int32,
    ov.Type.i64: torch.int64,
    ov.Type.i8: torch.int8,
    ov.Type.u8: torch.uint8,
    ov.Type.boolean: torch.bool,
}


class NPUAsyncFuture:
    """Future handle for asynchronous NPU execution."""

    def __init__(self, module: NPUGraphModule, handle: int):
        self._module = module
        self._handle = handle
        self._result: Any = None
        self._completed = False

    def wait(self) -> None:
        if not self._completed:
            self._result = self._module.wait_async(self._handle)
            self._completed = True

    def result(self) -> Any:
        self.wait()
        return self._result

    def is_ready(self) -> bool:
        """Non-blocking poll — never consumes output buffers."""
        if self._completed:
            return True
        req = self._module.infer_requests[self._handle]
        if hasattr(req, "wait_for"):
            try:
                return bool(req.wait_for(0))
            except Exception:
                pass
        return self._completed


class LeasedBufferPool:
    """Pool of pre-allocated tensors with weakref-based auto-return."""

    __slots__ = ("shape", "dtype", "idle_buffers", "_active", "_lock")

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype
        self.idle_buffers: list[torch.Tensor] = []
        self._active: set[int] = set()
        self._lock = threading.Lock()

    @property
    def active_buffers(self) -> set[int]:
        return self._active

    @active_buffers.setter
    def active_buffers(self, value: set[int]) -> None:
        self._active = value

    def lease(self) -> torch.Tensor:
        with self._lock:
            buf = self.idle_buffers.pop() if self.idle_buffers else torch.empty(self.shape, dtype=self.dtype)
            self._active.add(id(buf))
            return buf

    def release(self, buf_id: int, buf: torch.Tensor) -> None:
        with self._lock:
            if buf_id in self._active:
                self._active.remove(buf_id)
                self.idle_buffers.append(buf)


class NPUGraphModule(torch.nn.Module):
    """Wrapper around an OpenVINO compiled model with NPU-specific optimizations."""

    def __init__(
        self,
        compiled_model: Any,
        input_names: list[str],
        performance_hint: str = "LATENCY",
        num_streams: int = 1,
        clone_outputs: bool = True,
        all_placeholder_names: list[str] | None = None,
    ):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_names = input_names
        self.performance_hint = performance_hint
        self.num_streams = num_streams
        self.clone_outputs = clone_outputs
        self.arg_indices = self._build_arg_indices(input_names, all_placeholder_names)
        self.cpu_device = torch.device("cpu")
        self.infer_requests = [self.compiled_model.create_infer_request() for _ in range(num_streams)]
        self.request_idx = 0
        self._lock = threading.Lock()
        # Per-request locks: the global lock only protects round-robin index
        # assignment; each InferRequest has its own lock so multi-stream
        # forwards run truly in parallel instead of serializing on one lock.
        self._request_locks = [threading.Lock() for _ in range(num_streams)]
        self._active_inputs: list[tuple[list[Any], list[torch.Tensor | None] | None] | None] = [
            None for _ in range(num_streams)
        ]
        self._init_io_metadata()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_arg_indices(
        input_names: list[str], all_placeholder_names: list[str] | None
    ) -> list[int]:
        if all_placeholder_names is None:
            return list(range(len(input_names)))
        exact = {p.lower(): idx for idx, p in enumerate(all_placeholder_names)}
        # O(1) lookup instead of O(n) list.index scan.
        # Preserve first occurrence for duplicates.
        cleaned_first: dict[str, int] = {}
        for idx, p in enumerate(all_placeholder_names):
            k = _clean_name(p)
            if k not in cleaned_first:
                cleaned_first[k] = idx
        arg_indices: list[int] = []
        for name in input_names:
            low = name.lower()
            if low in exact:
                arg_indices.append(exact[low])
            else:
                cleaned = _clean_name(name)
                if cleaned in cleaned_first:
                    arg_indices.append(cleaned_first[cleaned])
                else:
                    arg_indices.append(len(arg_indices))
        return arg_indices

    def _init_io_metadata(self) -> None:
        self.target_dtypes: list[Any] = []
        self.target_torch_dtypes: list[torch.dtype] = []
        self.input_static: list[bool] = []
        self.input_sizes: list[int] = []
        for i in range(len(self.compiled_model.inputs)):
            ov_in = self.compiled_model.inputs[i]
            ov_type = ov_in.get_element_type()
            target_dtype = _OV_TO_NP.get(ov_type, np.float32)
            self.target_dtypes.append(target_dtype)
            self.target_torch_dtypes.append(_OV_TO_TORCH.get(ov_type, torch.float32))
            p_shape = ov_in.get_partial_shape()
            is_static = bool(p_shape.is_static)
            self.input_static.append(is_static)
            self.input_sizes.append(int(np.prod(list(p_shape.get_shape()))) if is_static else 0)

        self.input_tensors: list[list[ov.Tensor]] = [
            [req.get_input_tensor(i) for i in range(len(self.compiled_model.inputs))]
            for req in self.infer_requests
        ]

        self.output_info: list[tuple[bool, tuple[int, ...] | None, torch.dtype | None]] = []
        self.output_pools: list[LeasedBufferPool | None] = []
        for j in range(len(self.compiled_model.outputs)):
            ov_out = self.compiled_model.outputs[j]
            p_shape = ov_out.get_partial_shape()
            if p_shape.is_static:
                shape = tuple(p_shape.get_shape())
                ov_type = ov_out.get_element_type()
                target_dtype = _OV_TO_NP.get(ov_type, np.float32)
                torch_dtype = _OV_TO_TORCH.get(ov_type, torch.float32)
                self.output_info.append((True, shape, torch_dtype))
                num_elements = int(np.prod(shape))
                if num_elements > npu_config.leased_buffer_threshold_elements or not self.clone_outputs:
                    self.output_pools.append(LeasedBufferPool(shape, torch_dtype))
                else:
                    self.output_pools.append(None)
            else:
                self.output_info.append((False, None, None))
                self.output_pools.append(None)

        self.has_output_pools = any(p is not None for p in self.output_pools)
        self.num_outputs = len(self.output_info)
        # Cache per-request output tensor wrappers (static outputs only).
        # Dynamic outputs are looked up fresh per forward since their
        # descriptors can change shape between inferences.
        self.output_tensors: list[list[Any]] = [
            [req.get_output_tensor(j) for j in range(len(self.compiled_model.outputs))]
            for req in self.infer_requests
        ]
        self.input_meta = list(
            zip(
                self.arg_indices,
                self.target_dtypes,
                self.target_torch_dtypes,
                self.input_static,
                self.input_sizes, strict=False,
            )
        )

    # ------------------------------------------------------------------
    # Hot-path helpers — deduplicated between sync and async
    # ------------------------------------------------------------------

    def _get_numpy_view(
        self, val: Any, target_dtype: Any, target_tdtype: torch.dtype
    ) -> tuple[np.ndarray, Any]:
        cpu_device = self.cpu_device
        if isinstance(val, torch.Tensor):
            if val.device == cpu_device and val.dtype == target_tdtype and val.is_contiguous():
                return val.detach().numpy(), val
            cpu_val = val.detach()
            if cpu_val.device != cpu_device:
                cpu_val = cpu_val.cpu()
            if cpu_val.dtype != target_tdtype:
                cpu_val = cpu_val.to(target_tdtype)
            if not cpu_val.is_contiguous():
                cpu_val = cpu_val.contiguous()
            return cpu_val.numpy(), cpu_val
        np_view = np.asarray(val)
        if np_view.dtype != target_dtype:
            np_view = np_view.astype(target_dtype, copy=False)
        if not np_view.flags["C_CONTIGUOUS"]:
            np_view = np.ascontiguousarray(np_view)
        return np_view, np_view

    def _next_request(self) -> tuple[int, Any]:
        with self._lock:
            idx = self.request_idx
            req = self.infer_requests[idx]
            if self.num_streams > 1:
                self.request_idx = (self.request_idx + 1) % self.num_streams
            return idx, req

    def _bind_inputs(
        self,
        infer_request: Any,
        idx: int,
        args: tuple[Any, ...],
        keep_alive_out: list[Any] | None = None,
    ) -> None:
        input_meta = self.input_meta
        input_tensors = self.input_tensors[idx]
        copy_threshold = npu_config.copy_threshold_elements
        get_view = self._get_numpy_view
        for i, (arg_idx, target_dtype, target_tdtype, is_static, target_size) in enumerate(input_meta):
            val = args[arg_idx]
            np_view, keep_obj = get_view(val, target_dtype, target_tdtype)
            if keep_alive_out is not None:
                keep_alive_out.append(keep_obj)
            if is_static and target_size < copy_threshold:
                ov_tensor = input_tensors[i]
                if np_view.shape != ov_tensor.shape:
                    if np_view.ndim == 0 and len(ov_tensor.shape) == 1:
                        np_view = np_view.reshape(1)
                    elif np_view.ndim == 1 and np_view.shape[0] == 1 and len(ov_tensor.shape) == 0:
                        np_view = np_view.reshape(())
                np.copyto(ov_tensor.data, np_view)
            else:
                try:
                    infer_request.set_input_tensor(i, ov.Tensor(np_view, shared_memory=True))
                except RuntimeError:
                    expected_shape = list(self.compiled_model.inputs[i].get_partial_shape().get_shape())
                    if np_view.ndim == 0 and expected_shape == [1]:
                        np_view = np_view.reshape(1)
                    elif np_view.ndim == 1 and np_view.shape[0] == 1 and expected_shape == []:
                        np_view = np_view.reshape(())
                    infer_request.set_input_tensor(i, ov.Tensor(np_view, shared_memory=True))

    def _lease_outputs(self, infer_request: Any) -> list[torch.Tensor | None] | None:
        if not self.has_output_pools:
            return None
        leased: list[torch.Tensor | None] = []
        for j, pool in enumerate(self.output_pools):
            if pool is not None:
                buf = pool.lease()
                leased.append(buf)
                infer_request.set_output_tensor(j, ov.Tensor(buf.numpy(), shared_memory=True))
            else:
                leased.append(None)
        return leased

    def _collect_outputs(
        self, infer_request: Any, leased_bufs: list[torch.Tensor | None] | None,
        request_idx: int = 0,
    ) -> list[torch.Tensor]:
        clone_outputs = self.clone_outputs
        output_pools = self.output_pools
        output_info = self.output_info
        output_tensors = self.output_tensors[request_idx] if request_idx < len(self.output_tensors) else []
        from_numpy = torch.from_numpy
        outputs: list[torch.Tensor] = []
        if leased_bufs is not None:
            for j, buf in enumerate(leased_bufs):
                if buf is not None:
                    pool = output_pools[j]
                    assert pool is not None
                    if clone_outputs:
                        outputs.append(buf.clone())
                        pool.release(id(buf), buf)
                    else:
                        view = buf.detach()
                        weakref.finalize(view, pool.release, id(buf), buf)
                        outputs.append(view)
                elif output_info[j][0] and j < len(output_tensors):
                    t = from_numpy(output_tensors[j].data)
                    outputs.append(t.clone() if clone_outputs else t)
                else:
                    out = infer_request.get_output_tensor(j)
                    t = from_numpy(out.data)
                    outputs.append(t.clone() if clone_outputs else t)
        else:
            for j in range(self.num_outputs):
                if output_info[j][0] and j < len(output_tensors):
                    t = from_numpy(output_tensors[j].data)
                    outputs.append(t.clone() if clone_outputs else t)
                else:
                    out = infer_request.get_output_tensor(j)
                    t = from_numpy(out.data)
                    outputs.append(t.clone() if clone_outputs else t)
        return outputs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, *args: Any) -> Any:
        idx, req = self._next_request()
        with self._request_locks[idx]:
            self._bind_inputs(req, idx, args)
            leased = self._lease_outputs(req)
            req.infer()
            outputs = self._collect_outputs(req, leased, idx)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def infer_async(self, *args: Any) -> int:
        """Start an asynchronous inference request. Returns request index handle."""
        idx, req = self._next_request()
        with self._request_locks[idx]:
            keep_alive: list[Any] = []
            self._bind_inputs(req, idx, args, keep_alive_out=keep_alive)
            leased = self._lease_outputs(req)
            self._active_inputs[idx] = (keep_alive, leased)
            req.start_async()
            return idx

    def wait_async(self, handle: int) -> Any:
        req = self.infer_requests[handle]
        with self._request_locks[handle]:
            req.wait()
            active = self._active_inputs[handle]
            leased = active[1] if active is not None else None
            self._active_inputs[handle] = None
            outputs = self._collect_outputs(req, leased, handle)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def submit(self, *args: Any) -> NPUAsyncFuture:
        return NPUAsyncFuture(self, self.infer_async(*args))

    def batch_infer(self, inputs_list: list[tuple[Any, ...]]) -> list[Any]:
        """Pipelined batch inference across all streams."""
        if not inputs_list:
            return []
        results: list[Any] = [None] * len(inputs_list)
        active: list[tuple[int, NPUAsyncFuture]] = []
        for i, inp_args in enumerate(inputs_list):
            if not isinstance(inp_args, tuple):
                inp_args = (inp_args,)  # type: ignore[assignment]
            while len(active) >= self.num_streams:
                idx, fut = active.pop(0)
                results[idx] = fut.result()
            active.append((i, self.submit(*inp_args)))
        for idx, fut in active:
            results[idx] = fut.result()
        return results

    def reset_states(self) -> None:
        for req in self.infer_requests:
            for state in req.query_state():
                state.reset()

    def get_profiling_info(self, request_idx: int = 0) -> list[Any]:
        if 0 <= request_idx < len(self.infer_requests):
            try:
                return self.infer_requests[request_idx].get_profiling_info()
            except Exception as e:
                return [{"error": str(e)}]
        return []


class NPUDynamicGraphModule(torch.nn.Module):
    """Bucket-based dynamic sequence length wrapper."""

    def __init__(
        self,
        model: torch.nn.Module,
        example_input: Any,
        performance_hint: str | None = None,
        num_streams: int = 1,
        strict: bool = False,
        bucket_sizes: list[int] | None = None,
        dynamic_dim: int = 1,
        clone_outputs: bool = True,
        preprocess_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model = model
        if performance_hint is None:
            from intel_npu_acceleration.device import get_performance_hint

            performance_hint = get_performance_hint()
        self.performance_hint = performance_hint
        self.num_streams = num_streams
        self.strict = strict
        self.dynamic_dim = dynamic_dim
        self.clone_outputs = clone_outputs
        self.preprocess_config = preprocess_config
        self.bucket_sizes = sorted(bucket_sizes) if bucket_sizes is not None else list(npu_config.default_bucket_sizes)
        self._compiled_buckets: dict[int, NPUGraphModule] = {}
        args = (example_input,) if isinstance(example_input, torch.Tensor) else tuple(example_input)
        seq_len = self._get_seq_len(args)
        from .compiler import compile_to_npu

        if seq_len is not None:
            bucket = self._find_bucket(seq_len)
            padded = self._pad_inputs(args, seq_len, bucket)
            self.initial_compiled = compile_to_npu(
                self.model,
                padded,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )
            self._compiled_buckets[bucket] = self.initial_compiled
        else:
            self.initial_compiled = compile_to_npu(
                self.model,
                example_input,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )

    @property
    def compiled_model(self) -> Any:
        return self.initial_compiled.compiled_model

    @property
    def input_names(self) -> list[str]:
        return self.initial_compiled.input_names

    def _get_seq_len(self, args: tuple[Any, ...]) -> int | None:
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() > abs(self.dynamic_dim):
                return int(arg.shape[self.dynamic_dim])
        return None

    def _find_bucket(self, seq_len: int) -> int:
        import bisect
        import math

        idx = bisect.bisect_left(self.bucket_sizes, seq_len)
        if idx < len(self.bucket_sizes):
            return self.bucket_sizes[idx]
        return int(2 ** math.ceil(math.log2(seq_len)))

    def _pad_inputs(self, args: tuple[Any, ...], seq_len: int, bucket: int) -> tuple[Any, ...]:
        if seq_len == bucket:
            return args
        padded: list[Any] = []
        for arg in args:
            if (
                isinstance(arg, torch.Tensor)
                and arg.dim() > abs(self.dynamic_dim)
                and arg.shape[self.dynamic_dim] == seq_len
            ):
                pad_shape = list(arg.shape)
                pad_shape[self.dynamic_dim] = bucket - seq_len
                pad_tensor = torch.zeros(pad_shape, dtype=arg.dtype, device=arg.device)
                padded.append(torch.cat([arg, pad_tensor], dim=self.dynamic_dim))
            else:
                padded.append(arg)
        return tuple(padded)

    def _slice_outputs(self, outputs: Any, seq_len: int, bucket: int) -> Any:
        if seq_len == bucket:
            return outputs

        def _slice(t: Any) -> Any:
            if (
                isinstance(t, torch.Tensor)
                and t.dim() > abs(self.dynamic_dim)
                and t.shape[self.dynamic_dim] == bucket
            ):
                slices = [slice(None)] * t.dim()
                slices[self.dynamic_dim] = slice(0, seq_len)
                return t[tuple(slices)]
            return t

        if isinstance(outputs, tuple):
            return tuple(_slice(t) for t in outputs)
        return _slice(outputs)

    def _get_or_compile_bucket(self, seq_len: int, args_tuple: tuple[Any, ...]) -> tuple[int, NPUGraphModule]:
        bucket = self._find_bucket(seq_len)
        if bucket not in self._compiled_buckets:
            padded = self._pad_inputs(args_tuple, seq_len, bucket)
            from .compiler import compile_to_npu

            self._compiled_buckets[bucket] = compile_to_npu(
                self.model,
                padded,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )
        return bucket, self._compiled_buckets[bucket]

    def forward(self, *args: Any) -> Any:
        args_tuple = tuple(args)
        seq_len = self._get_seq_len(args_tuple)
        if seq_len is not None:
            bucket, compiled = self._get_or_compile_bucket(seq_len, args_tuple)
            padded = self._pad_inputs(args_tuple, seq_len, bucket)
            return self._slice_outputs(compiled(*padded), seq_len, bucket)
        return self.initial_compiled(*args)

    def infer_async(self, *args: Any) -> tuple[Any, int, int | None, int | None]:
        args_tuple = tuple(args)
        seq_len = self._get_seq_len(args_tuple)
        if seq_len is not None:
            bucket, compiled = self._get_or_compile_bucket(seq_len, args_tuple)
            padded = self._pad_inputs(args_tuple, seq_len, bucket)
            return (compiled, compiled.infer_async(*padded), seq_len, bucket)
        return (self.initial_compiled, self.initial_compiled.infer_async(*args), None, None)

    def wait_async(self, handle: tuple[Any, int, int | None, int | None]) -> Any:
        compiled, req_idx, seq_len, bucket = handle
        outputs = compiled.wait_async(req_idx)
        if seq_len is not None and bucket is not None:
            return self._slice_outputs(outputs, seq_len, bucket)
        return outputs

    def reset_states(self) -> None:
        self.initial_compiled.reset_states()
        for compiled in self._compiled_buckets.values():
            compiled.reset_states()
