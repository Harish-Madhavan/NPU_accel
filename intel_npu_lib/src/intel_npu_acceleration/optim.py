"""
Intel NPU Accelerated Optimizers (Level Zero Hardware Optimizer Steps).

This module provides hardware-accelerated optimizer implementations where parameter
and gradient momentum updates execute directly on the Intel NPU via fused Level Zero kernels.
"""

from collections.abc import Callable

import torch

from . import _functional as F_npu

__all__ = ["NPUAdam", "NPUSGD", "clip_grad_norm_"]


def _run_closure(closure: Callable[[], float] | None) -> float | None:
    """Evaluate the optimizer closure with grads enabled, if one was given."""
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()


class NPUAdam(torch.optim.Optimizer):
    """Adam optimizer with fused Level Zero NPU kernel step execution.

    Implements the Adam algorithm with hardware-accelerated parameter updates.
    The first moment vector ($m_t$), second moment vector ($v_t$), bias corrections,
    weight decay, and parameter updates are fused into a single hardware execution step
    executed directly on the Intel NPU.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): Coefficients for computing running averages
            of gradient and its square. Defaults to (0.9, 0.999).
        eps (float, optional): Term added to denominator for numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay coefficient (L2 penalty). Defaults to 0.0.

    Examples:
        >>> import torch
        >>> from intel_npu_acceleration.optim import NPUAdam
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = NPUAdam(model.parameters(), lr=1e-3)
        >>> output = model(torch.randn(4, 10))
        >>> loss = output.sum()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step on the NPU hardware.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[float]: The loss value returned by the closure, if provided.
        """
        loss = _run_closure(closure)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                new_p, new_exp_avg, new_exp_avg_sq = F_npu.adam_step(
                    p.data, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step
                )
                p.data.copy_(new_p)
                exp_avg.copy_(new_exp_avg)
                exp_avg_sq.copy_(new_exp_avg_sq)

        return loss


class NPUSGD(torch.optim.Optimizer):
    """SGD optimizer with fused Level Zero NPU momentum updates.

    Implements Stochastic Gradient Descent (with optional momentum, weight decay,
    and Nesterov acceleration) accelerated directly on the Intel NPU.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        momentum (float, optional): Momentum factor. Defaults to 0.0.
        dampening (float, optional): Dampening for momentum. Defaults to 0.0.
        weight_decay (float, optional): Weight decay coefficient (L2 penalty). Defaults to 0.0.
        nesterov (bool, optional): Enables Nesterov momentum. Defaults to False.

    Examples:
        >>> import torch
        >>> from intel_npu_acceleration.optim import NPUSGD
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = NPUSGD(model.parameters(), lr=1e-2, momentum=0.9)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single SGD optimization step on the NPU hardware.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[float]: The loss value returned by the closure, if provided.
        """
        loss = _run_closure(closure)

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                has_buf = "momentum_buffer" in state
                buf = state.get("momentum_buffer", torch.empty(0, dtype=p.dtype))

                new_p, new_buf = F_npu.sgd_step(
                    p.data, grad, buf, lr, momentum, weight_decay, dampening, nesterov, has_buf
                )
                p.data.copy_(new_p)
                if momentum != 0:
                    state["momentum_buffer"] = new_buf

        return loss


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clips gradient norm of an iterable of parameters.

    Delegates to the PyTorch standard implementation so NPU training clips
    bit-identically to CPU/CUDA training.

    Args:
        parameters: Iterable of Tensors or a single Tensor that will have gradients normalized.
        max_norm (float): Max norm of the gradients.
        norm_type (float, optional): Type of the used p-norm. Defaults to 2.0.
        error_if_nonfinite (bool, optional): If True, raises RuntimeError on non-finite norm. Defaults to False.

    Returns:
        torch.Tensor: Total norm of the parameter gradients.
    """
    return torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, norm_type, error_if_nonfinite
    )

