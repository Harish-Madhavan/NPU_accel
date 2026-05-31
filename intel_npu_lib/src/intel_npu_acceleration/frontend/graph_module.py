import torch
import openvino as ov
import numpy as np
import threading
import weakref
from typing import Any, Optional, List


class LeasedBufferPool:
    """
    Manages a pool of pre-allocated PyTorch tensors of a specific static shape and dtype.
    Allows leasing a buffer and automatically releasing it back to the pool once the user's
    tensor is garbage-collected using weakref.finalize.
    """
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.idle_buffers = []
        self.active_buffers = set()
        self._lock = threading.Lock()

    def lease(self) -> torch.Tensor:
        with self._lock:
            if self.idle_buffers:
                buf = self.idle_buffers.pop()
            else:
                buf = torch.empty(self.shape, dtype=self.dtype)
            self.active_buffers.add(id(buf))
            return buf

    def release(self, buf_id: int, buf: torch.Tensor):
        with self._lock:
            if buf_id in self.active_buffers:
                self.active_buffers.remove(buf_id)
                self.idle_buffers.append(buf)


class NPUGraphModule(torch.nn.Module):
    def __init__(
        self,
        compiled_model,
        input_names,
        performance_hint="LATENCY",
        num_streams=1,
        clone_outputs=True,
        all_placeholder_names=None,
    ):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_names = input_names
        self.performance_hint = performance_hint
        self.num_streams = num_streams
        self.clone_outputs = clone_outputs

        # Build mapping from compiled input index to original args index
        self.arg_indices = []
        if all_placeholder_names is not None:
            for name in input_names:
                if name in all_placeholder_names:
                    self.arg_indices.append(all_placeholder_names.index(name))
                else:
                    self.arg_indices.append(len(self.arg_indices))
        else:
            self.arg_indices = list(range(len(input_names)))


        self.cpu_device = torch.device("cpu")

        # Multiple infer requests for throughput mode
        self.infer_requests = [
            self.compiled_model.create_infer_request() for _ in range(num_streams)
        ]
        self.request_idx = 0
        self._lock = threading.Lock()

        # Track active inputs and leased output buffers during async execution to prevent temporary tensors going out of scope
        self._active_inputs = [None for _ in range(num_streams)]

        _OV_TO_NP = {
            ov.Type.f32: np.float32,
            ov.Type.f16: np.float16,
            ov.Type.i32: np.int32,
            ov.Type.i64: np.int64,
            ov.Type.i8: np.int8,
            ov.Type.u8: np.uint8,
            ov.Type.boolean: bool,
        }

        # Pre-compute input properties to eliminate Python overhead in forward pass
        self.target_dtypes = []
        self.target_torch_dtypes = []
        self.input_static = []
        self.input_sizes = []
        for i in range(len(self.compiled_model.inputs)):
            ov_in = self.compiled_model.inputs[i]
            ov_type = ov_in.get_element_type()
            target_dtype = _OV_TO_NP.get(ov_type, np.float32)
            self.target_dtypes.append(target_dtype)

            # Map NumPy dtype to PyTorch dtype
            t_dtype = torch.from_numpy(np.array(0, dtype=target_dtype)).dtype
            self.target_torch_dtypes.append(t_dtype)

            # Pre-compute shape static status and total size to avoid runtime checks
            p_shape = ov_in.get_partial_shape()
            is_static = p_shape.is_static
            self.input_static.append(is_static)
            if is_static:
                self.input_sizes.append(int(np.prod(list(p_shape.get_shape()))))
            else:
                self.input_sizes.append(0)

        # Pre-allocate input tensor references per request to support direct memory copy fast-path
        self.input_tensors = []
        for request in self.infer_requests:
            req_tensors = []
            for i in range(len(self.compiled_model.inputs)):
                req_tensors.append(request.get_input_tensor(i))
            self.input_tensors.append(req_tensors)

        # Pre-compute output properties and pre-allocate leased buffer pools for zero-copy memory safety
        self.output_info = []
        self.output_pools = []

        for j in range(len(self.compiled_model.outputs)):
            ov_out = self.compiled_model.outputs[j]
            partial_shape = ov_out.get_partial_shape()
            if partial_shape.is_static:
                shape = tuple(partial_shape.get_shape())
                ov_type = ov_out.get_element_type()
                target_dtype = _OV_TO_NP.get(ov_type, np.float32)
                torch_dtype = torch.from_numpy(np.array(0, dtype=target_dtype)).dtype
                self.output_info.append((True, shape, torch_dtype))
                
                # Zero-copy lease optimization threshold: only lease for large output tensors or when clone_outputs is disabled
                num_elements = int(np.prod(shape))
                if num_elements > 100000 or not self.clone_outputs:
                    self.output_pools.append(LeasedBufferPool(shape, torch_dtype))
                else:
                    self.output_pools.append(None)
            else:
                self.output_info.append((False, None, None))
                self.output_pools.append(None)
        self.has_output_pools = any(pool is not None for pool in self.output_pools)
        self.num_outputs = len(self.output_info)
        self.input_meta = list(zip(
            self.arg_indices,
            self.target_dtypes,
            self.target_torch_dtypes,
            self.input_static,
            self.input_sizes
        ))

    def forward(self, *args):
        with self._lock:
            # Use round-robin for infer requests
            idx = self.request_idx
            infer_request = self.infer_requests[idx]
            if self.num_streams > 1:
                self.request_idx = (self.request_idx + 1) % self.num_streams

            for i, (arg_idx, target_dtype, target_tdtype, is_static, target_size) in enumerate(self.input_meta):
                val = args[arg_idx]

                # Fast Path: pre-allocated tensor copying for small/medium static inputs
                if is_static and target_size < 2000000:
                    ov_tensor = self.input_tensors[idx][i]
                    if isinstance(val, torch.Tensor):
                        if (
                            val.device == self.cpu_device
                            and val.dtype == target_tdtype
                            and val.is_contiguous()
                        ):
                            np_view = val.detach().numpy()
                        else:
                            cpu_val = val.detach()
                            if cpu_val.device != self.cpu_device:
                                cpu_val = cpu_val.cpu()
                            if cpu_val.dtype != target_tdtype:
                                cpu_val = cpu_val.to(target_tdtype)
                            if not cpu_val.is_contiguous():
                                cpu_val = cpu_val.contiguous()
                            np_view = cpu_val.numpy()
                    else:
                        np_view = np.array(val)
                        if np_view.dtype != target_dtype:
                            np_view = np_view.astype(target_dtype)
                        if not np_view.flags["C_CONTIGUOUS"]:
                            np_view = np.ascontiguousarray(np_view)

                    if np_view.shape != ov_tensor.shape:
                        if np_view.ndim == 0 and len(ov_tensor.shape) == 1:
                            np_view = np_view.reshape(1)
                        elif np_view.ndim == 1 and np_view.shape[0] == 1 and len(ov_tensor.shape) == 0:
                            np_view = np_view.reshape(())

                    np.copyto(ov_tensor.data, np_view)
                else:
                    # Fallback Path: zero-copy pointer binding for large/dynamic inputs
                    if isinstance(val, torch.Tensor):
                        if (
                            val.device == self.cpu_device
                            and val.dtype == target_tdtype
                            and val.is_contiguous()
                        ):
                            np_view = val.detach().numpy()
                        else:
                            cpu_val = val.detach()
                            if cpu_val.device != self.cpu_device:
                                cpu_val = cpu_val.cpu()
                            if cpu_val.dtype != target_tdtype:
                                cpu_val = cpu_val.to(target_tdtype)
                            if not cpu_val.is_contiguous():
                                cpu_val = cpu_val.contiguous()
                            np_view = cpu_val.numpy()
                    else:
                        np_view = np.array(val)
                        if np_view.dtype != target_dtype:
                            np_view = np_view.astype(target_dtype)
                        if not np_view.flags["C_CONTIGUOUS"]:
                            np_view = np.ascontiguousarray(np_view)

                    try:
                        infer_request.set_input_tensor(
                            i, ov.Tensor(np_view, shared_memory=True)
                        )
                    except RuntimeError:
                        # Align 0-D scalar and 1-D [1] shapes dynamically on demand
                        expected_shape = list(self.compiled_model.inputs[i].get_partial_shape().get_shape())
                        if np_view.ndim == 0 and expected_shape == [1]:
                            np_view = np_view.reshape(1)
                        elif np_view.ndim == 1 and np_view.shape[0] == 1 and expected_shape == []:
                            np_view = np_view.reshape(())
                        infer_request.set_input_tensor(
                            i, ov.Tensor(np_view, shared_memory=True)
                        )

            # Lease output buffers (if pre-allocated pool exists)
            if self.has_output_pools:
                leased_bufs = []
                for j, pool in enumerate(self.output_pools):
                    if pool is not None:
                        buf = pool.lease()
                        leased_bufs.append(buf)
                    else:
                        leased_bufs.append(None)

                # Bind leased output tensors to the request (if leased)
                for j, buf in enumerate(leased_bufs):
                    if buf is not None:
                        ov_out_tensor = ov.Tensor(buf.numpy(), shared_memory=True)
                        infer_request.set_output_tensor(j, ov_out_tensor)
            else:
                leased_bufs = None

            infer_request.infer()

            outputs = []
            if leased_bufs is not None:
                for j, buf in enumerate(leased_bufs):
                    if buf is not None:
                        pool = self.output_pools[j]
                        if self.clone_outputs:
                            outputs.append(buf.clone())
                            pool.release(id(buf), buf)
                        else:
                            view = buf.detach()
                            weakref.finalize(view, pool.release, id(buf), buf)
                            outputs.append(view)
                    else:
                        out_tensor = infer_request.get_output_tensor(j)
                        data_tensor = torch.from_numpy(out_tensor.data)
                        outputs.append(data_tensor.clone() if self.clone_outputs else data_tensor)
            else:
                for j in range(self.num_outputs):
                    out_tensor = infer_request.get_output_tensor(j)
                    data_tensor = torch.from_numpy(out_tensor.data)
                    outputs.append(data_tensor.clone() if self.clone_outputs else data_tensor)

            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

    def infer_async(self, *args) -> int:
        """
        Start an asynchronous inference request on the NPU.
        Returns a handle (request index) to wait on later.
        """
        with self._lock:
            # Use round-robin for infer requests
            idx = self.request_idx
            infer_request = self.infer_requests[idx]
            if self.num_streams > 1:
                self.request_idx = (self.request_idx + 1) % self.num_streams

            keep_alive = []
            for i, (arg_idx, target_dtype, target_tdtype, is_static, target_size) in enumerate(self.input_meta):
                val = args[arg_idx]

                # Fast Path: pre-allocated tensor copying for small/medium static inputs
                if is_static and target_size < 2000000:
                    ov_tensor = self.input_tensors[idx][i]
                    if isinstance(val, torch.Tensor):
                        if (
                            val.device == self.cpu_device
                            and val.dtype == target_tdtype
                            and val.is_contiguous()
                        ):
                            np_view = val.detach().numpy()
                            keep_alive.append(val)
                        else:
                            cpu_val = val.detach()
                            if cpu_val.device != self.cpu_device:
                                cpu_val = cpu_val.cpu()
                            if cpu_val.dtype != target_tdtype:
                                cpu_val = cpu_val.to(target_tdtype)
                            if not cpu_val.is_contiguous():
                                cpu_val = cpu_val.contiguous()
                            np_view = cpu_val.numpy()
                            keep_alive.append(cpu_val)
                    else:
                        np_view = np.array(val)
                        if np_view.dtype != target_dtype:
                            np_view = np_view.astype(target_dtype)
                        if not np_view.flags["C_CONTIGUOUS"]:
                            np_view = np.ascontiguousarray(np_view)
                        keep_alive.append(np_view)

                    if np_view.shape != ov_tensor.shape:
                        if np_view.ndim == 0 and len(ov_tensor.shape) == 1:
                            np_view = np_view.reshape(1)
                        elif np_view.ndim == 1 and np_view.shape[0] == 1 and len(ov_tensor.shape) == 0:
                            np_view = np_view.reshape(())

                    np.copyto(ov_tensor.data, np_view)
                else:
                    # Fallback Path: zero-copy pointer binding for large/dynamic inputs
                    if isinstance(val, torch.Tensor):
                        if (
                            val.device == self.cpu_device
                            and val.dtype == target_tdtype
                            and val.is_contiguous()
                        ):
                            np_view = val.detach().numpy()
                            keep_alive.append(val)
                        else:
                            cpu_val = val.detach()
                            if cpu_val.device != self.cpu_device:
                                cpu_val = cpu_val.cpu()
                            if cpu_val.dtype != target_tdtype:
                                cpu_val = cpu_val.to(target_tdtype)
                            if not cpu_val.is_contiguous():
                                cpu_val = cpu_val.contiguous()
                            np_view = cpu_val.numpy()
                            keep_alive.append(cpu_val)
                    else:
                        np_view = np.array(val)
                        if np_view.dtype != target_dtype:
                            np_view = np_view.astype(target_dtype)
                        if not np_view.flags["C_CONTIGUOUS"]:
                            np_view = np.ascontiguousarray(np_view)
                        keep_alive.append(np_view)

                    try:
                        infer_request.set_input_tensor(
                            i, ov.Tensor(np_view, shared_memory=True)
                        )
                    except RuntimeError:
                        # Align 0-D scalar and 1-D [1] shapes dynamically on demand
                        expected_shape = list(self.compiled_model.inputs[i].get_partial_shape().get_shape())
                        if np_view.ndim == 0 and expected_shape == [1]:
                            np_view = np_view.reshape(1)
                        elif np_view.ndim == 1 and np_view.shape[0] == 1 and expected_shape == []:
                            np_view = np_view.reshape(())
                        infer_request.set_input_tensor(
                            i, ov.Tensor(np_view, shared_memory=True)
                        )

            # Lease output buffers for this async request (if pre-allocated pool exists)
            if self.has_output_pools:
                leased_bufs = []
                for j, pool in enumerate(self.output_pools):
                    if pool is not None:
                        buf = pool.lease()
                        leased_bufs.append(buf)
                    else:
                        leased_bufs.append(None)

                # Bind leased output tensors to the request (if leased)
                for j, buf in enumerate(leased_bufs):
                    if buf is not None:
                        ov_out_tensor = ov.Tensor(buf.numpy(), shared_memory=True)
                        infer_request.set_output_tensor(j, ov_out_tensor)
            else:
                leased_bufs = None

            self._active_inputs[idx] = (keep_alive, leased_bufs)
            infer_request.start_async()
            return idx

    def wait_async(self, handle: int):
        """
        Block and wait for the specified asynchronous inference request to complete.
        Returns the output tensor(s).
        """
        infer_request = self.infer_requests[handle]
        infer_request.wait()

        # Retrieve leased buffers for this request
        active_info = self._active_inputs[handle]
        leased_bufs = active_info[1] if active_info is not None else None

        # Clear active inputs reference to allow immediate garbage collection
        self._active_inputs[handle] = None

        outputs = []
        if leased_bufs is not None:
            for j, buf in enumerate(leased_bufs):
                if buf is not None:
                    pool = self.output_pools[j]
                    if self.clone_outputs:
                        outputs.append(buf.clone())
                        pool.release(id(buf), buf)
                    else:
                        view = buf.detach()
                        weakref.finalize(view, pool.release, id(buf), buf)
                        outputs.append(view)
                else:
                    out_tensor = infer_request.get_output_tensor(j)
                    data_tensor = torch.from_numpy(out_tensor.data)
                    outputs.append(data_tensor.clone() if self.clone_outputs else data_tensor)
        else:
            for j in range(self.num_outputs):
                out_tensor = infer_request.get_output_tensor(j)
                data_tensor = torch.from_numpy(out_tensor.data)
                outputs.append(data_tensor.clone() if self.clone_outputs else data_tensor)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def reset_states(self):
        """Reset all internal state variables (like KV-caches) in the NPU hardware."""
        for request in self.infer_requests:
            for state in request.query_state():
                state.reset()


class NPUDynamicGraphModule(torch.nn.Module):
    """
    Runtime wrapper that matches dynamic sequence lengths to static predefined buckets,
    dynamically padding input tensors and slicing output tensors back.
    Mathematically avoids driver compilation overhead by compiling static shape graphs
    at bucket boundaries.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        example_input: Any,
        performance_hint: str = "LATENCY",
        num_streams: int = 1,
        strict: bool = False,
        bucket_sizes: Optional[List[int]] = None,
        dynamic_dim: int = 1,
        clone_outputs: bool = True,
        preprocess_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = model
        self.performance_hint = performance_hint
        self.num_streams = num_streams
        self.strict = strict
        self.dynamic_dim = dynamic_dim
        self.clone_outputs = clone_outputs
        self.preprocess_config = preprocess_config

        if bucket_sizes is None:
            # Default to power of 2 boundaries
            self.bucket_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        else:
            self.bucket_sizes = sorted(bucket_sizes)

        self._compiled_buckets = {}

        # Pre-process initial example input and compile
        args = (example_input,) if isinstance(example_input, torch.Tensor) else tuple(example_input)
        S = self._get_seq_len(args)
        from .compiler import compile_to_npu  # Delay-import to avoid circular dependency
        if S is not None:
            B = self._find_bucket(S)
            padded_args = self._pad_inputs(args, S, B)
            self.initial_compiled = compile_to_npu(
                self.model,
                padded_args,
                self.performance_hint,
                self.num_streams,
                self.strict,
                dynamic_buckets=False,
                clone_outputs=self.clone_outputs,
                preprocess_config=self.preprocess_config,
            )
            self._compiled_buckets[B] = self.initial_compiled
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
    def compiled_model(self):
        return self.initial_compiled.compiled_model

    @property
    def input_names(self):
        return self.initial_compiled.input_names

    def _get_seq_len(self, args) -> Optional[int]:
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() > abs(self.dynamic_dim):
                return arg.shape[self.dynamic_dim]
        return None

    def _find_bucket(self, S: int) -> int:
        import bisect
        idx = bisect.bisect_left(self.bucket_sizes, S)
        if idx < len(self.bucket_sizes):
            return self.bucket_sizes[idx]
        import math
        return int(2 ** math.ceil(math.log2(S)))

    def _pad_inputs(self, args: tuple, S: int, B: int) -> tuple:
        if S == B:
            return args
        padded = []
        for arg in args:
            if (
                isinstance(arg, torch.Tensor)
                and arg.dim() > abs(self.dynamic_dim)
                and arg.shape[self.dynamic_dim] == S
            ):
                pad_shape = list(arg.shape)
                pad_shape[self.dynamic_dim] = B - S
                pad_tensor = torch.zeros(pad_shape, dtype=arg.dtype, device=arg.device)
                padded.append(torch.cat([arg, pad_tensor], dim=self.dynamic_dim))
            else:
                padded.append(arg)
        return tuple(padded)

    def _slice_outputs(self, outputs: Any, S: int, B: int) -> Any:
        if S == B:
            return outputs

        def slice_tensor(t):
            if (
                isinstance(t, torch.Tensor)
                and t.dim() > abs(self.dynamic_dim)
                and t.shape[self.dynamic_dim] == B
            ):
                slices = [slice(None)] * t.dim()
                slices[self.dynamic_dim] = slice(0, S)
                return t[tuple(slices)]
            return t

        if isinstance(outputs, tuple):
            return tuple(slice_tensor(t) for t in outputs)
        return slice_tensor(outputs)

    def forward(self, *args):
        args_tuple = tuple(args)
        S = self._get_seq_len(args_tuple)
        if S is not None:
            B = self._find_bucket(S)
            if B not in self._compiled_buckets:
                padded_args = self._pad_inputs(args_tuple, S, B)
                from .compiler import compile_to_npu  # Delay-import to avoid circular dependency
                self._compiled_buckets[B] = compile_to_npu(
                    self.model,
                    padded_args,
                    self.performance_hint,
                    self.num_streams,
                    self.strict,
                    dynamic_buckets=False,
                    clone_outputs=self.clone_outputs,
                    preprocess_config=self.preprocess_config,
                )
            compiled_model = self._compiled_buckets[B]
            padded_args = self._pad_inputs(args_tuple, S, B)
            outputs = compiled_model(*padded_args)
            return self._slice_outputs(outputs, S, B)
        else:
            return self.initial_compiled(*args)

    def infer_async(self, *args) -> tuple:
        args_tuple = tuple(args)
        S = self._get_seq_len(args_tuple)
        if S is not None:
            B = self._find_bucket(S)
            if B not in self._compiled_buckets:
                padded_args = self._pad_inputs(args_tuple, S, B)
                from .compiler import compile_to_npu  # Delay-import to avoid circular dependency
                self._compiled_buckets[B] = compile_to_npu(
                    self.model,
                    padded_args,
                    self.performance_hint,
                    self.num_streams,
                    self.strict,
                    dynamic_buckets=False,
                    clone_outputs=self.clone_outputs,
                    preprocess_config=self.preprocess_config,
                )
            compiled_model = self._compiled_buckets[B]
            padded_args = self._pad_inputs(args_tuple, S, B)
            req_idx = compiled_model.infer_async(*padded_args)
            return (compiled_model, req_idx, S, B)
        else:
            req_idx = self.initial_compiled.infer_async(*args)
            return (self.initial_compiled, req_idx, None, None)

    def wait_async(self, handle: tuple) -> Any:
        compiled_model, req_idx, S, B = handle
        outputs = compiled_model.wait_async(req_idx)
        if S is not None and B is not None:
            return self._slice_outputs(outputs, S, B)
        return outputs

    def reset_states(self):
        """Reset all internal state variables (like KV-caches) in the NPU hardware."""
        self.initial_compiled.reset_states()
        for compiled in self._compiled_buckets.values():
            compiled.reset_states()
