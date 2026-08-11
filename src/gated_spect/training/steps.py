"""One-batch training steps with no hidden target or filename access."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..losses import poisson_deviance, weighted_poisson_nll
from ..data.torch_dataset import model_batch


@dataclass(frozen=True)
class PretrainingLoss:
    total: Tensor
    masked: Tensor
    visible: Tensor
    deviance: Tensor
    masked_deviance: Tensor
    health: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class DenoisingLoss:
    total: Tensor
    deviance: Tensor
    health: dict[str, Tensor] = field(default_factory=dict)


def _rate_health(
    rate: Tensor,
    target_weight: Tensor,
    masked_target: Tensor | None = None,
    visible_target: Tensor | None = None,
    target_counts: Tensor | None = None,
    target_exposure_s: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return detached scalar diagnostics for prediction and target coverage."""

    # Accumulate diagnostics in float32; half-precision reductions can overflow
    # for high-count Dataset 2 projections even when the model loss is finite.
    values = rate.detach().float()
    health = {
        "rate_mean": values.mean(),
        "rate_std": values.std(unbiased=False),
        "rate_min": values.amin(),
        "rate_max": values.amax(),
        "valid_fraction": target_weight.detach().float().mean(),
    }
    active = target_weight.detach().to(dtype=torch.bool)[..., None, None]
    if masked_target is not None:
        active_pixels = active.expand_as(masked_target)
        denominator = active_pixels.sum().clamp_min(1).to(dtype=torch.float32)
        health["masked_fraction"] = (
            masked_target.detach().to(dtype=torch.bool) & active_pixels
        ).sum() / denominator
    if visible_target is not None:
        active_pixels = active.expand_as(visible_target)
        denominator = active_pixels.sum().clamp_min(1).to(dtype=torch.float32)
        health["visible_fraction"] = (
            visible_target.detach().to(dtype=torch.bool) & active_pixels
        ).sum() / denominator
    if target_counts is not None and target_exposure_s is not None:
        weights = target_weight.detach().to(dtype=values.dtype)[..., None, None]
        predicted = values * target_exposure_s.detach().to(dtype=values.dtype)[..., None, None]
        target = target_counts.detach().to(dtype=values.dtype)
        predicted_total = (predicted * weights).flatten(1).sum(dim=1)
        target_total = (target * weights).flatten(1).sum(dim=1)
        bias = (predicted_total - target_total) / target_total.clamp_min(1.0)
        health["total_count_bias_fraction"] = bias.mean()
        health["absolute_total_count_bias_fraction"] = bias.abs().mean()
    return health


def _target_mean(rate: Tensor, target_exposure_s: Tensor) -> Tensor:
    if rate.ndim != 5 or target_exposure_s.shape != rate.shape[:3]:
        raise ValueError("rate must be [B,G,V,H,W] and exposure must be [B,G,V]")
    return target_exposure_s.to(dtype=rate.dtype)[..., None, None] * rate


def pretraining_objective(
    rate: Tensor,
    target_counts: Tensor,
    target_exposure_s: Tensor,
    target_weight: Tensor,
    masked_target: Tensor,
    visible_target: Tensor,
) -> PretrainingLoss:
    """Compute masked + 0.25 visible Poisson objectives and deviance."""

    if target_counts.shape != rate.shape or masked_target.shape != rate.shape:
        raise ValueError("target tensors must match model rate shape [B,G,V,H,W]")
    if visible_target.shape != rate.shape:
        raise ValueError("visible_target must match model rate shape")
    mean = _target_mean(rate, target_exposure_s)
    masked = weighted_poisson_nll(
        mean,
        target_counts.to(dtype=rate.dtype),
        target_weight,
        bin_weight=masked_target,
    )
    visible = weighted_poisson_nll(
        mean,
        target_counts.to(dtype=rate.dtype),
        target_weight,
        bin_weight=visible_target,
    )
    selected = (masked_target.to(dtype=rate.dtype) + visible_target.to(dtype=rate.dtype)).clamp_max(1.0)
    masked_deviance = weighted_poisson_nll(
        poisson_deviance(mean, target_counts.to(dtype=rate.dtype), reduction="none"),
        torch.zeros_like(mean),
        target_weight,
        bin_weight=masked_target,
    )
    # Keep the union metric for diagnostics, while checkpoint selection uses
    # the literal masked-target metric required by the training plan.
    deviance = weighted_poisson_nll(
        poisson_deviance(mean, target_counts.to(dtype=rate.dtype), reduction="none"),
        torch.zeros_like(mean),
        target_weight,
        bin_weight=selected,
    )
    return PretrainingLoss(
        total=masked + 0.25 * visible,
        masked=masked,
        visible=visible,
        deviance=deviance,
        masked_deviance=masked_deviance,
        health=_rate_health(
            rate,
            target_weight,
            masked_target,
            visible_target,
            target_counts,
            target_exposure_s,
        ),
    )


def denoising_objective(
    rate: Tensor,
    target_counts: Tensor,
    target_exposure_s: Tensor,
    target_weight: Tensor,
) -> DenoisingLoss:
    """Default supervised denoising objective: weighted target Poisson NLL."""

    mean = _target_mean(rate, target_exposure_s)
    total = weighted_poisson_nll(mean, target_counts.to(dtype=rate.dtype), target_weight)
    deviance = weighted_poisson_nll(
        poisson_deviance(mean, target_counts.to(dtype=rate.dtype), reduction="none"),
        torch.zeros_like(mean),
        target_weight,
    )
    return DenoisingLoss(
        total=total,
        deviance=deviance,
        health=_rate_health(rate, target_weight, target_counts=target_counts, target_exposure_s=target_exposure_s),
    )


def pretraining_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> PretrainingLoss:
    """Run one optional optimizer step; targets never enter model inputs."""

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = model(**model_batch(batch))
    loss = pretraining_objective(
        output.rate,
        batch["target_counts"],
        batch["target_exposure_s"],
        batch["target_weight"],
        batch["masked_target"],
        batch["visible_target"],
    )
    if optimizer is not None:
        loss.total.backward()
        optimizer.step()
    return loss


def denoising_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> DenoisingLoss:
    """Run one optional supervised denoising optimizer step."""

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = model(
        **model_batch(batch, include_counts_raw=bool(getattr(model, "baseline_residual", False)))
    )
    loss = denoising_objective(
        output.rate,
        batch["target_counts"],
        batch["target_exposure_s"],
        batch["target_weight"],
    )
    if optimizer is not None:
        loss.total.backward()
        optimizer.step()
    return loss
