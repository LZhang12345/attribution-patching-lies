"""Directional Taylor estimates without constructing a Hessian.

``metric_fn(a)`` reruns the downstream computation with one component's
activation set to ``a``. The direction is always corrupt minus clean, and
the expansion starts at clean. The return value estimates patched minus clean
metric. Summing scores for isolated components is a different operation from
patching all of those components together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

Metric = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class HVPResult:
    """The first-order term, leading correction, and their sum."""

    atp: Tensor
    correction: Tensor
    estimate: Tensor

    @property
    def exposure(self) -> Tensor | None:
        """|quadratic| / |AtP|, undefined at an exactly zero AtP score.

        This is a curvature diagnostic, not a calibrated probability or a
        certificate that higher-order terms are small. Near-zero nonzero
        scores should be interpreted in the metric's native units.
        """
        if self.atp.item() == 0:
            return None
        return self.correction.abs() / self.atp.abs()


def _endpoints(clean: Tensor, corrupt: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(clean, Tensor) or not isinstance(corrupt, Tensor):
        raise TypeError("clean and corrupt must be torch tensors")
    if clean.shape != corrupt.shape or clean.numel() == 0:
        raise ValueError("endpoints must have the same nonempty shape")
    if clean.dtype != corrupt.dtype or clean.device != corrupt.device:
        raise ValueError("endpoints must share dtype and device")
    if clean.dtype not in (torch.float32, torch.float64):
        raise ValueError("use float32 or float64 for directional derivatives")
    if not torch.isfinite(clean).all() or not torch.isfinite(corrupt).all():
        raise ValueError("endpoints must be finite")
    return clean.detach(), (corrupt - clean).detach()


def _value(metric_fn: Metric, activation: Tensor) -> Tensor:
    value = metric_fn(activation)
    if not isinstance(value, Tensor) or value.ndim != 0 or not value.is_floating_point():
        raise ValueError("metric_fn must return a floating-point scalar tensor")
    if not torch.isfinite(value):
        raise ValueError("metric_fn returned a nonfinite value")
    return value


def _terms(metric_fn: Metric, base: Tensor, delta: Tensor, second_order: bool):
    with torch.enable_grad():
        point = base.detach().clone().requires_grad_(True)
        value = _value(metric_fn, point)
        zero = torch.zeros((), dtype=point.dtype, device=point.device)
        if not value.requires_grad:
            return zero, zero
        gradient = torch.autograd.grad(value, point, create_graph=second_order, allow_unused=True)[
            0
        ]
        if gradient is None:
            return zero, zero
        first = (gradient * delta).sum()
        correction = zero
        if second_order and first.requires_grad:
            product = torch.autograd.grad(first, point, allow_unused=True)[0]
            if product is not None:
                correction = 0.5 * (product * delta).sum()
        if not torch.isfinite(first) or not torch.isfinite(correction):
            raise ValueError("directional derivatives are nonfinite")
        return first.detach(), correction.detach()


def activation_patching(metric_fn: Metric, clean: Tensor, corrupt: Tensor) -> Tensor:
    """Direct endpoint change M(corrupt) - M(clean), using two forward calls."""
    base, _ = _endpoints(clean, corrupt)
    with torch.no_grad():
        return (_value(metric_fn, corrupt.detach()) - _value(metric_fn, base)).detach()


def attribution_patching(metric_fn: Metric, clean: Tensor, corrupt: Tensor) -> Tensor:
    """First-order estimate grad M(clean) dot (corrupt - clean)."""
    base, delta = _endpoints(clean, corrupt)
    return _terms(metric_fn, base, delta, second_order=False)[0]


def hvp(metric_fn: Metric, clean: Tensor, corrupt: Tensor) -> HVPResult:
    """Add 1/2 delta^T H(clean) delta using a Hessian-vector product.

    The quadratic term is computed by autograd. The endpoint estimate still
    has higher-order Taylor error; no smoothness constant is needed to run it.
    """
    base, delta = _endpoints(clean, corrupt)
    first, correction = _terms(metric_fn, base, delta, second_order=True)
    return HVPResult(first, correction, first + correction)


def _steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")


def multi_step_hvp(metric_fn: Metric, clean: Tensor, corrupt: Tensor, steps: int = 5) -> Tensor:
    """Sum second-order Taylor changes on equally sized path segments.

    Each segment starts at clean + k*delta/K and moves by delta/K. The
    quadratic contribution therefore scales by 1/K^2 on each segment.
    With a valid pairwise Hessian-Lipschitz condition on the path, the total
    remainder is bounded by L3*||delta||^3/(6*K^2).
    """
    _steps(steps)
    base, delta = _endpoints(clean, corrupt)
    segment = delta / steps
    total = torch.zeros((), dtype=base.dtype, device=base.device)
    for k in range(steps):
        first, correction = _terms(
            metric_fn, base + (k / steps) * delta, segment, second_order=True
        )
        total = total + first + correction
    return total.detach()
