"""Integrated two-view self-supervised training step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..data.torch_dataset import model_batch
from ..losses import (
    EMATeacher,
    GlobalConsistencyHead,
    hidden_gate_consistency_loss,
)
from .steps import PretrainingLoss, pretraining_objective


@dataclass(frozen=True)
class TwoViewPretrainingLoss:
    total: Tensor
    masked: Tensor
    visible: Tensor
    deviance: Tensor
    latent: Tensor
    gate: Tensor
    variance: Tensor
    queue_values: Tensor | None = None
    health: dict[str, Tensor] = field(default_factory=dict)


def _representation_health(projected: Tensor, queue: Any) -> dict[str, Tensor]:
    """Compute covariance-spectrum and correlation collapse diagnostics."""

    queue_values = queue.values().detach().to(device=projected.device, dtype=projected.dtype)
    values = torch.cat((queue_values, projected.detach()), dim=0).float()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("representation values must have shape [N,D] with N > 0")
    component_std = values.std(dim=0, unbiased=False)
    component_variance = component_std.square()
    centered = values - values.mean(dim=0, keepdim=True)
    if values.shape[0] > 1:
        singular_values = torch.linalg.svdvals(centered)
        eigenvalues = singular_values.square()
        spectrum_total = eigenvalues.sum()
        probability = eigenvalues / spectrum_total.clamp_min(1.0e-12)
        effective_rank = torch.exp(
            -(probability.clamp_min(1.0e-12) * probability.clamp_min(1.0e-12).log()).sum()
        )
        participation_rank = spectrum_total.square() / eigenvalues.square().sum().clamp_min(1.0e-12)
        rank_denominator = float(min(values.shape[1], max(values.shape[0] - 1, 1)))
        rank_fraction = effective_rank / rank_denominator
        covariance = centered.transpose(0, 1).matmul(centered) / float(values.shape[0] - 1)
        standard_deviation = covariance.diagonal().clamp_min(1.0e-12).sqrt()
        correlation = covariance / (standard_deviation[:, None] * standard_deviation[None, :]).clamp_min(1.0e-12)
        off_diagonal = ~torch.eye(values.shape[1], dtype=torch.bool, device=values.device)
        absolute_off_diagonal = correlation.abs()[off_diagonal]
        if absolute_off_diagonal.numel() == 0:
            absolute_off_diagonal = values.new_zeros((1,))
    else:
        effective_rank = values.new_zeros(())
        participation_rank = values.new_zeros(())
        rank_fraction = values.new_zeros(())
        absolute_off_diagonal = values.new_zeros((1,))
    norms = values.norm(dim=1)
    return {
        "representation_mean": values.mean(),
        "representation_std": values.std(unbiased=False),
        "representation_norm_mean": norms.mean(),
        "representation_norm_std": norms.std(unbiased=False),
        "representation_component_std_mean": component_std.mean(),
        "representation_component_std_min": component_std.amin(),
        "representation_component_std_max": component_std.amax(),
        "representation_effective_rank": effective_rank,
        "representation_participation_rank": participation_rank,
        "representation_effective_rank_fraction": rank_fraction,
        "mean_absolute_offdiagonal_correlation": absolute_off_diagonal.mean(),
        # Quantile does not support half/bfloat16 on all CUDA versions.  The
        # diagnostic is intentionally non-training state, so promote it to
        # float32 before the reduction while retaining the model's device.
        "p95_absolute_offdiagonal_correlation": torch.quantile(
            absolute_off_diagonal.float(), 0.95
        ),
        "max_absolute_offdiagonal_correlation": absolute_off_diagonal.amax(),
        "minimum_component_std": component_std.amin(),
        "median_component_std": component_std.median(),
        "maximum_component_std": component_std.amax(),
        "queue_size": projected.new_tensor(float(queue.num_entries)),
        "queue_ready": projected.new_tensor(float(queue.num_entries >= 512)),
    }


def _student_masked_gate(batch: Mapping[str, Any]) -> Tensor:
    visibility = batch["visibility"]
    if visibility.ndim != 5:
        raise ValueError("visibility must have shape [B,G,V,H,W]")
    # A gate is considered hidden only when every detector bin in every view is
    # invisible.  Partial view/patch masks do not create a gate-consistency target.
    return ~visibility.any(dim=(2, 3, 4))


def optimizer_step_succeeded(
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None,
    autocast_enabled: bool,
) -> bool:
    """Step an optimizer and report whether AMP skipped it due to overflow."""

    if scaler is not None and autocast_enabled:
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        return float(scaler.get_scale()) >= old_scale
    optimizer.step()
    return True


def two_view_pretraining_step(
    model: nn.Module,
    ema_teacher: EMATeacher,
    consistency_head: GlobalConsistencyHead,
    batch: Mapping[str, Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> TwoViewPretrainingLoss:
    """Train two masked student views and match hidden gates to an EMA teacher.

    The queue and EMA teacher are updated only after a successful optimizer step.
    ``batch`` must contain ``student_a``, ``student_b``, and an unmasked
    ``teacher`` view produced by :class:`TwoViewPretrainingDataset`.
    """

    required = ("student_a", "student_b", "teacher")
    if any(key not in batch for key in required):
        raise ValueError("two-view batch must contain student_a, student_b, and teacher")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    device = next(model.parameters()).device
    autocast_enabled = bool(amp and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        student_a = model.backbone(**model_batch(batch["student_a"]))
        student_b = model.backbone(**model_batch(batch["student_b"]))
        teacher = ema_teacher(**model_batch(batch["teacher"]))
        rate_a, _ = model.decoder(student_a)
        rate_b, _ = model.decoder(student_b)
        loss_a = pretraining_objective(
            rate_a,
            batch["student_a"]["target_counts"],
            batch["student_a"]["target_exposure_s"],
            batch["student_a"]["target_weight"],
            batch["student_a"]["masked_target"],
            batch["student_a"]["visible_target"],
        )
        loss_b = pretraining_objective(
            rate_b,
            batch["student_b"]["target_counts"],
            batch["student_b"]["target_exposure_s"],
            batch["student_b"]["target_weight"],
            batch["student_b"]["masked_target"],
            batch["student_b"]["visible_target"],
        )
        global_loss = consistency_head(
            student_a.global_representation,
            student_b.global_representation,
        )
        gate_loss = hidden_gate_consistency_loss(
            student_a.bottleneck,
            teacher.bottleneck,
            batch["student_a"]["valid_frames"],
            batch["teacher"]["valid_frames"],
            batch["teacher"]["relative_exposure_q"],
            _student_masked_gate(batch["student_a"]),
        )
        masked = 0.5 * (loss_a.masked + loss_b.masked)
        visible = 0.5 * (loss_a.visible + loss_b.visible)
        deviance = 0.5 * (loss_a.deviance + loss_b.deviance)
        total = masked + 0.25 * visible + 0.05 * global_loss.total + 0.05 * gate_loss
        health = {
            key: 0.5 * (loss_a.health[key] + loss_b.health[key])
            for key in loss_a.health.keys()
            if key in loss_b.health
        }
        health.update(_representation_health(torch.cat((global_loss.projected_a, global_loss.projected_b), dim=0), consistency_head.queue))
        health["covariance_offdiagonal_loss"] = global_loss.covariance.detach()
        health["covariance_ready"] = global_loss.covariance_ready.detach()
        clip_values = [
            value
            for value in (
                getattr(student_a, "exposure_clip_fraction", None),
                getattr(student_b, "exposure_clip_fraction", None),
            )
            if value is not None
        ]
        if clip_values:
            health["exposure_conditioning_clip_fraction"] = torch.stack(clip_values).mean().detach()
        health["latent_similarity"] = 1.0 - global_loss.cosine.detach()
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite two-view self-supervised loss")
    if optimizer is not None:
        successful_step = True
        if scaler is not None and autocast_enabled:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(consistency_head.parameters()), 1.0)
            successful_step = optimizer_step_succeeded(
                optimizer, scaler=scaler, autocast_enabled=autocast_enabled
            )
        else:
            total.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(consistency_head.parameters()), 1.0)
            successful_step = optimizer_step_succeeded(
                optimizer, scaler=None, autocast_enabled=False
            )
        if successful_step:
            ema_teacher.update(model.backbone)
            consistency_head.queue.enqueue(
                torch.cat((global_loss.projected_a.detach(), global_loss.projected_b.detach()), dim=0)
            )
        health["optimizer_step_succeeded"] = total.new_tensor(float(successful_step))
    return TwoViewPretrainingLoss(
        total=total,
        masked=masked,
        visible=visible,
        deviance=deviance,
        latent=global_loss.cosine,
        gate=gate_loss,
        variance=global_loss.variance,
        queue_values=torch.cat(
            (global_loss.projected_a.detach(), global_loss.projected_b.detach()), dim=0
        ),
        health=health,
    )
