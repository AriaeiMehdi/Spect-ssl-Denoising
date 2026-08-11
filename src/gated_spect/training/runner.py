"""Minimal reproducible training/evaluation loops and atomic checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import tempfile
import time
import math
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .consistency_steps import (
    TwoViewPretrainingLoss,
    optimizer_step_succeeded,
    two_view_pretraining_step,
)
from .steps import DenoisingLoss, PretrainingLoss, denoising_step, pretraining_step
from ..data.torch_dataset import model_batch
from ..losses import poisson_deviance
from ..versioning import (
    IncompatibleArtifactError,
    artifact_version_metadata,
    require_artifact_versions,
)


TaskName = Literal["pretraining", "denoising"]
ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    deviance: float
    batches: int
    latent: float = 0.0
    gate: float = 0.0
    variance: float = 0.0
    health: dict[str, float] = field(default_factory=dict)
    masked: float = 0.0
    visible: float = 0.0
    masked_deviance: float | None = None


def pretraining_checkpoint_is_eligible(
    validation: EpochMetrics,
    *,
    scalar_baseline_deviance: float,
    minimum_representation_std: float = 0.01,
) -> bool:
    """Apply scientific validation guardrails before selecting a checkpoint."""

    validation_deviance = (
        validation.masked_deviance
        if validation.masked_deviance is not None
        else validation.deviance
    )
    values = (validation.loss, validation_deviance, scalar_baseline_deviance)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if validation_deviance >= scalar_baseline_deviance:
        return False
    representation_std = validation.health.get("representation_std")
    if representation_std is None or not math.isfinite(float(representation_std)):
        return False
    return float(representation_std) >= float(minimum_representation_std)


def update_post_warmup_patience(
    epochs_without_improvement: int,
    *,
    epoch_number: int,
    warmup_epochs: int,
    improved: bool,
) -> int:
    """Update early-stop patience, counting only epochs after warm-up."""

    if improved or epoch_number <= warmup_epochs:
        return 0
    return int(epochs_without_improvement) + 1


def parameter_summary(*modules: nn.Module) -> dict[str, int]:
    """Return parameter counts for one or more modules.

    Parameters shared by multiple module arguments are counted once by object
    identity.  This keeps the reported trainable count correct when a caller
    passes a wrapper and one of its submodules together.
    """

    seen: set[int] = set()
    total = 0
    trainable = 0
    buffers = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            total += int(parameter.numel())
            if parameter.requires_grad:
                trainable += int(parameter.numel())
        buffers += sum(int(buffer.numel()) for buffer in module.buffers())
    return {"total": total, "trainable": trainable, "frozen": total - trainable, "buffers": buffers}


def _loader_batch_count(loader: Iterable[Mapping[str, Any]], max_batches: int | None) -> int | None:
    try:
        count = int(len(loader))  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return max_batches
    return min(count, max_batches) if max_batches is not None else count


def _emit_progress(callback: ProgressCallback | None, payload: Mapping[str, Any]) -> None:
    if callback is not None:
        callback(payload)


def _health_values(health: Mapping[str, Tensor]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in health.items():
        numeric = float(value.detach().float().cpu())
        if math.isfinite(numeric):
            values[key] = numeric
    return values


def _health_update(sums: dict[str, float], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        sums[key] = sums.get(key, 0.0) + float(value)


def _health_average(sums: Mapping[str, float], count: int) -> dict[str, float]:
    if count < 1:
        return {}
    return {key: value / count for key, value in sums.items()}


def _record_gradient_norm(measured: Tensor, values: list[float]) -> float | None:
    numeric = float(measured.detach().float().cpu())
    if math.isfinite(numeric):
        values.append(numeric)
        return numeric
    return None


def _move_batch(
    batch: Mapping[str, Any], device: torch.device, *, non_blocking: bool = False
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device=device, non_blocking=non_blocking)
        elif isinstance(value, Mapping):
            moved[key] = _move_batch(value, device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _compute_loss(model: nn.Module, batch: Mapping[str, Any], task: TaskName) -> PretrainingLoss | DenoisingLoss:
    if task == "pretraining":
        return pretraining_step(model, batch)
    if task == "denoising":
        return denoising_step(model, batch)
    raise ValueError(f"unsupported task {task!r}")


def train_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    task: TaskName,
    device: torch.device | str,
    gradient_clip_norm: float = 1.0,
    max_batches: int | None = None,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    log_interval: int | None = None,
    epoch: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EpochMetrics:
    """Train one epoch with explicit gradient clipping and finite checks."""

    device = torch.device(device)
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    loss_total = 0.0
    deviance_total = 0.0
    masked_deviance_total = 0.0
    has_masked_deviance = False
    masked_total = 0.0
    visible_total = 0.0
    batches = 0
    health_sums: dict[str, float] = {}
    optimizer_steps = 0
    gradient_norms: list[float] = []
    last_gradient_norm: float | None = None
    started = time.perf_counter()
    total_batches = _loader_batch_count(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        moved = _move_batch(batch, device, non_blocking=device.type == "cuda")
        autocast_enabled = bool(amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            loss = _compute_loss(model, moved, task)
        if not torch.isfinite(loss.total):
            raise FloatingPointError(f"non-finite {task} loss at batch {batches}")
        scaled_loss = loss.total / accumulation_steps
        if scaler is not None and autocast_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        pending += 1
        should_step = pending >= accumulation_steps
        is_last = max_batches is not None and batches + 1 >= max_batches
        if should_step or is_last:
            if scaler is not None and autocast_enabled:
                scaler.unscale_(optimizer)
            if pending < accumulation_steps:
                correction = accumulation_steps / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if gradient_clip_norm > 0:
                measured_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                last_gradient_norm = _record_gradient_norm(measured_norm, gradient_norms)
            if scaler is not None and autocast_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            optimizer_steps += 1
        loss_total += float(loss.total.detach().cpu())
        deviance_total += float(loss.deviance.detach().cpu())
        if hasattr(loss, "masked_deviance"):
            has_masked_deviance = True
            masked_deviance_total += float(loss.masked_deviance.detach().cpu())
        if hasattr(loss, "masked"):
            masked_total += float(loss.masked.detach().cpu())
            visible_total += float(loss.visible.detach().cpu())
        batches += 1
        batch_health = _health_values(loss.health)
        _health_update(health_sums, batch_health)
        if log_interval and (batches == 1 or batches % log_interval == 0):
            elapsed = time.perf_counter() - started
            _emit_progress(
                progress_callback,
                {
                    "event": "batch",
                    "phase": "train",
                    "epoch": epoch,
                    "batch": batches,
                    "batches_total": total_batches,
                    "loss": float(loss.total.detach().cpu()),
                    "deviance": float(loss.deviance.detach().cpu()),
                    "masked": (
                        float(loss.masked.detach().cpu()) if hasattr(loss, "masked") else None
                    ),
                    "visible": (
                        float(loss.visible.detach().cpu()) if hasattr(loss, "visible") else None
                    ),
                    "elapsed_seconds": elapsed,
                    "batches_per_second": batches / max(elapsed, 1e-9),
                    "optimizer_steps": optimizer_steps,
                    "gradient_norm": last_gradient_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "amp_scale": (
                        float(scaler.get_scale())
                        if scaler is not None and autocast_enabled
                        else None
                    ),
                    **batch_health,
                },
            )
    if pending:
        if scaler is not None and amp and device.type == "cuda":
            scaler.unscale_(optimizer)
        correction = accumulation_steps / pending
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        if gradient_clip_norm > 0:
            measured_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            last_gradient_norm = _record_gradient_norm(measured_norm, gradient_norms)
        if scaler is not None and amp and device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    if batches == 0:
        raise ValueError("training loader produced no batches")
    phase_health = _health_average(health_sums, batches)
    if gradient_norms:
        phase_health["gradient_norm_mean"] = sum(gradient_norms) / len(gradient_norms)
        phase_health["gradient_norm_max"] = max(gradient_norms)
    phase_health["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    if scaler is not None and amp and device.type == "cuda":
        phase_health["amp_scale"] = float(scaler.get_scale())
    metrics = EpochMetrics(
        loss_total / batches,
        deviance_total / batches,
        batches,
        health=phase_health,
        masked=masked_total / batches if masked_total else 0.0,
        visible=visible_total / batches if visible_total else 0.0,
        masked_deviance=(masked_deviance_total / batches if has_masked_deviance else None),
    )
    _emit_progress(
        progress_callback,
        {
            "event": "phase_end",
            "phase": "train",
            "epoch": epoch,
            "batches": batches,
            "loss": metrics.loss,
            "deviance": metrics.deviance,
            "masked_deviance": metrics.masked_deviance,
            "masked": metrics.masked,
            "visible": metrics.visible,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_steps": optimizer_steps,
            **phase_health,
        },
    )
    return metrics


@torch.no_grad()
def _masked_projection_summary(
    prediction: Tensor,
    target: Tensor,
    deviance: Tensor,
    mask: Tensor,
    selected: Tensor,
) -> dict[str, float]:
    """Summarize detector bins without treating them as independent patients."""

    active = mask & selected
    count = int(active.sum().item())
    selected_count = int(selected.sum().item())
    if count == 0:
        return {
            "mean_prediction": 0.0,
            "mean_target": 0.0,
            "bias": 0.0,
            "mae": 0.0,
            "deviance": 0.0,
            "cv": 0.0,
            "bin_count": 0.0,
            "coverage_fraction": 0.0,
        }
    predicted = prediction[active].float()
    observed = target[active].float()
    mean_prediction = float(predicted.mean().cpu())
    mean_target = float(observed.mean().cpu())
    return {
        "mean_prediction": mean_prediction,
        "mean_target": mean_target,
        "bias": (mean_prediction - mean_target) / max(abs(mean_target), 1.0e-8),
        "mae": float((predicted - observed).abs().mean().cpu()),
        "deviance": float(deviance[active].float().mean().cpu()),
        "cv": float((predicted.std(unbiased=False) / max(abs(mean_prediction), 1.0e-8)).cpu()),
        "bin_count": float(count),
        "coverage_fraction": float(count / max(selected_count, 1)),
    }


def default_calibration_ranges() -> tuple[tuple[str, float, float | None], ...]:
    return (
        ("0", 0.0, 0.0),
        ("1", 1.0, 1.0),
        ("2", 2.0, 2.0),
        ("3-5", 3.0, 5.0),
        ("6-10", 6.0, 10.0),
        ("11-20", 11.0, 20.0),
        (">20", 20.0, None),
    )


def parse_calibration_ranges(spec: str) -> tuple[tuple[str, float, float | None], ...]:
    """Parse labels such as ``0,1,2,3-5,6-10,11-20,>20``."""

    ranges: list[tuple[str, float, float | None]] = []
    for raw in (item.strip() for item in spec.split(",")):
        if not raw:
            continue
        if raw.startswith(">"):
            low = float(raw[1:])
            ranges.append((raw, low, None))
        elif "-" in raw:
            left, right = raw.split("-", 1)
            ranges.append((raw, float(left), float(right)))
        else:
            value = float(raw)
            ranges.append((raw, value, value))
    if not ranges:
        raise ValueError("calibration range specification is empty")
    return tuple(ranges)


def _range_mask(
    values: Tensor, low: float, high: float | None, *, exclusive_low: bool = False
) -> Tensor:
    if high is None:
        return values > low if exclusive_low else values >= low
    return (values >= low) & (values <= high)


@torch.no_grad()
def evaluate_denoising_patients(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    max_batches: int | None = None,
    amp: bool = False,
    low_signal_threshold: float = 5.0,
    calibration_ranges: tuple[tuple[str, float, float | None], ...] | None = None,
) -> list[dict[str, Any]]:
    """Return validation deviance/bias rows retaining patient identity.

    Epoch reducers intentionally average within a batch and discard identity.
    This helper uses the same selected-bin weighting but emits pair-level rows
    so downstream reports can average by patient and bootstrap patients.  The
    caller must provide a validation-only loader; this function never touches
    a locked test split.
    """

    device = torch.device(device)
    model.eval()
    if low_signal_threshold < 0:
        raise ValueError("low_signal_threshold must be nonnegative")
    calibration_ranges = calibration_ranges or default_calibration_ranges()
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        moved = _move_batch(batch, device, non_blocking=device.type == "cuda")
        autocast_enabled = bool(amp and device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            output = model(
                **model_batch(
                    moved,
                    include_counts_raw=bool(getattr(model, "baseline_residual", False)),
                )
            )
        rate = output.rate.float()
        target_counts = moved["target_counts"].float()
        target_exposure = moved["target_exposure_s"].float()
        target_weight = moved["target_weight"].float()
        input_exposure = moved["input_exposure_s"].float()
        counts_raw = moved.get("counts_raw")
        if counts_raw is None:
            counts_raw = moved["counts_input"].squeeze(3) * input_exposure[..., None, None]
        counts_raw = counts_raw.float()
        mean = target_exposure[..., None, None] * rate
        weights = target_weight[..., None, None].expand_as(mean)
        naive_rate = counts_raw / input_exposure[..., None, None].clamp_min(1.0e-8)
        naive_mean = target_exposure[..., None, None] * naive_rate
        model_values = poisson_deviance(mean, target_counts, reduction="none")
        naive_values = poisson_deviance(naive_mean, target_counts, reduction="none")
        selected = weights > 0
        denominator = weights.flatten(1).sum(dim=1).clamp_min(1.0)
        model_deviance = (model_values * weights).flatten(1).sum(dim=1) / denominator
        naive_deviance = (naive_values * weights).flatten(1).sum(dim=1) / denominator
        predicted_total = (mean * weights).flatten(1).sum(dim=1)
        naive_total = (naive_mean * weights).flatten(1).sum(dim=1)
        target_total = (target_counts * weights).flatten(1).sum(dim=1)
        model_bias = (predicted_total - target_total) / target_total.clamp_min(1.0)
        naive_bias = (naive_total - target_total) / target_total.clamp_min(1.0)
        metadata = batch.get("meta", [])
        for row_index in range(rate.shape[0]):
            # Compute all expanded diagnostics after selecting one patient/pair
            # from the batch.  Aggregating these tensors before this loop would
            # leak one patient's counts into every other patient's row whenever
            # ``batch_size > 1``.
            row_mean = mean[row_index]
            row_target = target_counts[row_index]
            row_model_values = model_values[row_index]
            row_counts_raw = counts_raw[row_index]
            row_selected = selected[row_index]
            per_view_target = (row_target * row_selected).sum(dim=(0, 2, 3))
            per_view_prediction = (row_mean * row_selected).sum(dim=(0, 2, 3))
            per_view_bias = (per_view_prediction - per_view_target) / per_view_target.clamp_min(1.0e-8)
            valid_views = per_view_target > 0
            per_view_abs = per_view_bias.abs()[valid_views]
            if per_view_abs.numel():
                per_view_metrics = {
                    "mean_absolute_bias": float(per_view_abs.mean().cpu()),
                    "p95_absolute_bias": float(torch.quantile(per_view_abs.float(), 0.95).cpu()),
                    "maximum_absolute_bias": float(per_view_abs.max().cpu()),
                    "view_count": float(per_view_abs.numel()),
                }
            else:
                per_view_metrics = {
                    "mean_absolute_bias": 0.0,
                    "p95_absolute_bias": 0.0,
                    "maximum_absolute_bias": 0.0,
                    "view_count": 0.0,
                }
            zero_input = _masked_projection_summary(
                row_mean, row_target, row_model_values, row_counts_raw == 0, row_selected
            )
            low_count_metrics = {
                "0": _masked_projection_summary(row_mean, row_target, row_model_values, row_counts_raw == 0, row_selected),
                "1": _masked_projection_summary(row_mean, row_target, row_model_values, row_counts_raw == 1, row_selected),
                "2": _masked_projection_summary(row_mean, row_target, row_model_values, row_counts_raw == 2, row_selected),
                "3-5": _masked_projection_summary(
                    row_mean, row_target, row_model_values, (row_counts_raw >= 3) & (row_counts_raw <= 5), row_selected
                ),
                ">5": _masked_projection_summary(row_mean, row_target, row_model_values, row_counts_raw > 5, row_selected),
            }
            calibration = {
                label: _masked_projection_summary(
                    row_mean,
                    row_target,
                    row_model_values,
                    _range_mask(row_counts_raw, low, high, exclusive_low=label.startswith(">")),
                    row_selected,
                )
                for label, low, high in calibration_ranges
            }
            low_signal = _masked_projection_summary(
                row_mean,
                row_target,
                row_model_values,
                row_target <= low_signal_threshold,
                row_selected,
            )
            item_meta = metadata[row_index] if row_index < len(metadata) else {}
            if not isinstance(item_meta, Mapping):
                item_meta = {}
            rows.append(
                {
                    "batch": batch_index,
                    "patient_key": str(item_meta.get("patient_key", "")),
                    "pair_id": str(item_meta.get("pair_id", "")),
                    "model_deviance": float(model_deviance[row_index].cpu()),
                    "naive_deviance": float(naive_deviance[row_index].cpu()),
                    "model_total_count_bias_fraction": float(model_bias[row_index].cpu()),
                    "naive_total_count_bias_fraction": float(naive_bias[row_index].cpu()),
                    "target_total_counts": float(target_total[row_index].cpu()),
                    "selected_bins": float(denominator[row_index].cpu()),
                    "per_view_bias": {
                        key: float(value)
                        for key, value in {
                            "mean_absolute_bias": per_view_metrics["mean_absolute_bias"],
                            "p95_absolute_bias": per_view_metrics["p95_absolute_bias"],
                            "maximum_absolute_bias": per_view_metrics["maximum_absolute_bias"],
                            "view_count": per_view_metrics["view_count"],
                        }.items()
                    },
                    "zero_input_bin_metrics": {
                        key: float(value) for key, value in zero_input.items()
                    },
                    "low_count_strata": low_count_metrics,
                    "projection_low_signal_threshold": float(low_signal_threshold),
                    "projection_low_signal_mask": low_signal,
                    "calibration_table": calibration,
                }
            )
    if not rows:
        raise ValueError("patient evaluation produced no rows")
    return rows


def _finalize_two_view_window(
    model: nn.Module,
    consistency_head: Any,
    ema_teacher: Any,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None,
    autocast_enabled: bool,
    accumulation_steps: int,
    pending: int,
    queued_values: list[Tensor],
) -> tuple[bool, Tensor]:
    """Finalize one gradient-accumulation window exactly once.

    The caller owns ``pending`` and resets it after this helper returns.  The
    representation list is cleared here on both success and failure so a
    later optimizer window can never re-enqueue historical values.
    """

    if pending < 1:
        raise ValueError("cannot finalize an empty optimizer window")
    if scaler is not None and autocast_enabled:
        scaler.unscale_(optimizer)
    if pending < accumulation_steps:
        correction = accumulation_steps / pending
        for parameter in list(model.parameters()) + list(consistency_head.parameters()):
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
    parameters = list(model.parameters()) + list(consistency_head.parameters())
    measured_norm = nn.utils.clip_grad_norm_(parameters, 1.0)
    successful_step = optimizer_step_succeeded(
        optimizer,
        scaler=scaler if autocast_enabled else None,
        autocast_enabled=autocast_enabled,
    )
    optimizer.zero_grad(set_to_none=True)
    if successful_step:
        ema_teacher.update(model.backbone)
        if queued_values:
            consistency_head.queue.enqueue(torch.cat(queued_values, dim=0))
    queued_values.clear()
    return successful_step, measured_norm


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    task: TaskName,
    device: torch.device | str,
    max_batches: int | None = None,
    amp: bool = False,
    log_interval: int | None = None,
    epoch: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EpochMetrics:
    """Evaluate one epoch without updating weights."""

    device = torch.device(device)
    model.eval()
    loss_total = 0.0
    deviance_total = 0.0
    masked_deviance_total = 0.0
    has_masked_deviance = False
    masked_total = 0.0
    visible_total = 0.0
    batches = 0
    health_sums: dict[str, float] = {}
    started = time.perf_counter()
    total_batches = _loader_batch_count(loader, max_batches)
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        autocast_enabled = bool(amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            loss = _compute_loss(model, _move_batch(batch, device), task)
        if not torch.isfinite(loss.total):
            raise FloatingPointError(f"non-finite {task} validation loss at batch {batches}")
        loss_total += float(loss.total.cpu())
        deviance_total += float(loss.deviance.cpu())
        if hasattr(loss, "masked_deviance"):
            has_masked_deviance = True
            masked_deviance_total += float(loss.masked_deviance.cpu())
        if hasattr(loss, "masked"):
            masked_total += float(loss.masked.cpu())
            visible_total += float(loss.visible.cpu())
        batches += 1
        batch_health = _health_values(loss.health)
        _health_update(health_sums, batch_health)
        if log_interval and (batches == 1 or batches % log_interval == 0):
            elapsed = time.perf_counter() - started
            _emit_progress(
                progress_callback,
                {
                    "event": "batch",
                    "phase": "validation",
                    "epoch": epoch,
                    "batch": batches,
                    "batches_total": total_batches,
                    "loss": float(loss.total.cpu()),
                    "deviance": float(loss.deviance.cpu()),
                    "masked": float(loss.masked.cpu()) if hasattr(loss, "masked") else None,
                    "visible": float(loss.visible.cpu()) if hasattr(loss, "visible") else None,
                    "elapsed_seconds": elapsed,
                    "batches_per_second": batches / max(elapsed, 1e-9),
                    **batch_health,
                },
            )
    if batches == 0:
        raise ValueError("evaluation loader produced no batches")
    phase_health = _health_average(health_sums, batches)
    metrics = EpochMetrics(
        loss_total / batches,
        deviance_total / batches,
        batches,
        health=phase_health,
        masked=masked_total / batches if masked_total else 0.0,
        visible=visible_total / batches if visible_total else 0.0,
        masked_deviance=(masked_deviance_total / batches if has_masked_deviance else None),
    )
    _emit_progress(
        progress_callback,
        {
            "event": "phase_end",
            "phase": "validation",
            "epoch": epoch,
            "batches": batches,
            "loss": metrics.loss,
            "deviance": metrics.deviance,
            "masked_deviance": metrics.masked_deviance,
            "masked": metrics.masked,
            "visible": metrics.visible,
            "elapsed_seconds": time.perf_counter() - started,
            **phase_health,
        },
    )
    return metrics


@torch.no_grad()
def evaluate_scalar_rate_baseline(
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    """Evaluate one exposure-corrected scalar rate per acquisition.

    The comparator uses the actually observed (thinned and visible) input
    counts, not the original full acquisition and not a separate rate for each
    detector bin.  This is the M3 baseline specified by the training plan.
    """

    def _view_deviance(batch: Mapping[str, Any]) -> Tensor:
        """Return one patient-balanced baseline deviance per view.

        The validation objective is evaluated independently for student A and
        student B and then averaged.  A teacher view is deliberately not used
        as a substitute: it has a different thinning and no target mask.
        Keeping the scalar-rate comparator on the same view and masked-bin
        contract makes the checkpoint guardrail scientifically matched.
        """

        moved = _move_batch(batch, device, non_blocking=device.type == "cuda")
        input_counts_raw = moved.get("input_counts_raw")
        if input_counts_raw is None:
            # Backward-compatible fallback for hand-built evaluation batches;
            # real pretraining batches always carry input_counts_raw.
            input_counts_raw = moved.get("counts_raw")
        if input_counts_raw is None:
            raise ValueError("scalar-rate baseline requires input_counts_raw in validation batches")
        input_exposure = moved["input_exposure_s"].float().clamp_min(1.0e-8)
        target_exposure = moved["target_exposure_s"].float()
        target_counts = moved["target_counts"].float()
        frame_weight = moved["target_weight"].float()
        target_weight = frame_weight[..., None, None]
        visibility = moved.get("visibility")
        if visibility is None:
            visibility = torch.ones_like(input_counts_raw, dtype=torch.bool)
        valid_input = visibility & (target_weight > 0)
        observed_counts = input_counts_raw.float()
        observed_exposure = input_exposure[..., None, None]
        numerator = (observed_counts * valid_input).flatten(1).sum(dim=1)
        denominator = (observed_exposure * valid_input).flatten(1).sum(dim=1).clamp_min(1.0e-8)
        scalar_rate = numerator / denominator
        mean = (
            target_exposure[..., None, None] * scalar_rate[:, None, None, None, None]
        ).expand_as(target_counts)
        values = poisson_deviance(mean, target_counts, reduction="none")
        masked_target = moved.get("masked_target")
        visible_target = moved.get("visible_target")
        if masked_target is not None:
            # M3 checkpoint selection is the literal masked-target deviance.
            # The visible 5% objective remains part of training but is not
            # allowed to alter the scientific baseline gate.
            selected = masked_target.to(dtype=torch.bool)
        else:
            # Hand-built legacy evaluation batches have no target masks.  In
            # that case every frame-weighted bin is the selected target.
            selected = torch.ones_like(values, dtype=torch.bool)
        expanded_weight = target_weight.expand_as(values) * selected.to(dtype=target_weight.dtype)
        denominator = expanded_weight.flatten(1).sum(dim=1).clamp_min(1.0)
        return (values * expanded_weight).flatten(1).sum(dim=1).div(denominator)

    device = torch.device(device)
    total = 0.0
    batches = 0
    for index, raw_batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        if "student_a" in raw_batch and "student_b" in raw_batch:
            # Match two_view_pretraining_step: average the two independently
            # thinned/masked student objectives, never the unmasked teacher.
            views = (raw_batch["student_a"], raw_batch["student_b"])
        elif "teacher" in raw_batch:
            views = (raw_batch["teacher"],)
        else:
            views = (raw_batch,)
        view_values = [_view_deviance(view) for view in views]
        total += float(torch.stack([value.mean() for value in view_values]).mean().cpu())
        batches += 1
    if batches == 0:
        raise ValueError("scalar-rate baseline loader produced no batches")
    result = total / batches
    if not math.isfinite(result):
        raise FloatingPointError("scalar-rate baseline produced a non-finite deviance")
    return result


def _two_view_metrics(losses: list[TwoViewPretrainingLoss]) -> EpochMetrics:
    if not losses:
        raise ValueError("two-view loader produced no batches")
    count = float(len(losses))
    health_sums: dict[str, float] = {}
    for item in losses:
        _health_update(health_sums, _health_values(item.health))
    return EpochMetrics(
        loss=sum(float(item.total.detach().cpu()) for item in losses) / count,
        deviance=sum(float(item.deviance.detach().cpu()) for item in losses) / count,
        batches=len(losses),
        latent=sum(float(item.latent.detach().cpu()) for item in losses) / count,
        gate=sum(float(item.gate.detach().cpu()) for item in losses) / count,
        variance=sum(float(item.variance.detach().cpu()) for item in losses) / count,
        health=_health_average(health_sums, len(losses)),
        masked=sum(float(item.masked.detach().cpu()) for item in losses) / count,
        visible=sum(float(item.visible.detach().cpu()) for item in losses) / count,
        masked_deviance=(
            sum(
                float(
                    (item.masked_deviance if item.masked_deviance is not None else item.deviance)
                    .detach()
                    .cpu()
                )
                for item in losses
            )
            / count
        ),
    )


def train_two_view_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    ema_teacher: nn.Module,
    consistency_head: nn.Module,
    *,
    device: torch.device | str,
    max_batches: int | None = None,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    log_interval: int | None = None,
    epoch: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EpochMetrics:
    """Train the two-view masked/latent/gate objective for one epoch."""

    device = torch.device(device)
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    consistency_head.train()
    metric_values: list[tuple[float, float, float, float, float, float, float, float]] = []
    health_sums: dict[str, float] = {}
    gradient_norms: list[float] = []
    last_gradient_norm: float | None = None
    optimizer_steps = 0
    optimizer_step_skips = 0
    started = time.perf_counter()
    total_batches = _loader_batch_count(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    queued_values: list[Tensor] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        moved = _move_batch(batch, device, non_blocking=device.type == "cuda")
        loss = two_view_pretraining_step(
            model,
            ema_teacher,
            consistency_head,
            moved,
            optimizer=None,
            amp=amp,
        )
        scaled_loss = loss.total / accumulation_steps
        autocast_enabled = bool(amp and device.type == "cuda")
        if scaler is not None and autocast_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        pending += 1
        if loss.queue_values is not None:
            queued_values.append(loss.queue_values)
        is_last = max_batches is not None and index + 1 >= max_batches
        if pending >= accumulation_steps or is_last:
            successful_step, measured_norm = _finalize_two_view_window(
                model,
                consistency_head,
                ema_teacher,
                optimizer,
                scaler=scaler,
                autocast_enabled=autocast_enabled,
                accumulation_steps=accumulation_steps,
                pending=pending,
                queued_values=queued_values,
            )
            last_gradient_norm = _record_gradient_norm(measured_norm, gradient_norms)
            if successful_step:
                optimizer_steps += 1
            else:
                optimizer_step_skips += 1
            pending = 0
        metric_values.append(
            (
                float(loss.total.detach().cpu()),
                float(loss.deviance.detach().cpu()),
                float(loss.latent.detach().cpu()),
                float(loss.gate.detach().cpu()),
                float(loss.variance.detach().cpu()),
                float(loss.masked.detach().cpu()),
                float(loss.visible.detach().cpu()),
                float(
                    (loss.masked_deviance if loss.masked_deviance is not None else loss.deviance)
                    .detach()
                    .cpu()
                ),
            )
        )
        batch_health = _health_values(loss.health)
        _health_update(health_sums, batch_health)
        if log_interval and (index == 0 or (index + 1) % log_interval == 0):
            elapsed = time.perf_counter() - started
            _emit_progress(
                progress_callback,
                {
                    "event": "batch",
                    "phase": "train",
                    "epoch": epoch,
                    "batch": index + 1,
                    "batches_total": total_batches,
                    "loss": float(loss.total.detach().cpu()),
                    "deviance": float(loss.deviance.detach().cpu()),
                    "masked": float(loss.masked.detach().cpu()),
                    "visible": float(loss.visible.detach().cpu()),
                    "latent": float(loss.latent.detach().cpu()),
                    "gate": float(loss.gate.detach().cpu()),
                    "variance": float(loss.variance.detach().cpu()),
                    "elapsed_seconds": elapsed,
                    "batches_per_second": (index + 1) / max(elapsed, 1e-9),
                    "optimizer_steps": optimizer_steps,
                    "gradient_norm": last_gradient_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "amp_scale": (
                        float(scaler.get_scale())
                        if scaler is not None and autocast_enabled
                        else None
                    ),
                    **batch_health,
                },
            )
    if pending:
        autocast_enabled = bool(amp and device.type == "cuda")
        successful_step, measured_norm = _finalize_two_view_window(
            model,
            consistency_head,
            ema_teacher,
            optimizer,
            scaler=scaler,
            autocast_enabled=autocast_enabled,
            accumulation_steps=accumulation_steps,
            pending=pending,
            queued_values=queued_values,
        )
        last_gradient_norm = _record_gradient_norm(measured_norm, gradient_norms)
        if successful_step:
            optimizer_steps += 1
        else:
            optimizer_step_skips += 1
    if not metric_values:
        raise ValueError("two-view loader produced no batches")
    count = float(len(metric_values))
    phase_health = _health_average(health_sums, len(metric_values))
    if gradient_norms:
        phase_health["gradient_norm_mean"] = sum(gradient_norms) / len(gradient_norms)
        phase_health["gradient_norm_max"] = max(gradient_norms)
    phase_health["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    phase_health["optimizer_steps"] = float(optimizer_steps)
    phase_health["optimizer_step_skips"] = float(optimizer_step_skips)
    if scaler is not None and amp and device.type == "cuda":
        phase_health["amp_scale"] = float(scaler.get_scale())
    metrics = EpochMetrics(
        sum(row[0] for row in metric_values) / count,
        sum(row[1] for row in metric_values) / count,
        len(metric_values),
        sum(row[2] for row in metric_values) / count,
        sum(row[3] for row in metric_values) / count,
        sum(row[4] for row in metric_values) / count,
        health=phase_health,
        masked=sum(row[5] for row in metric_values) / count,
        visible=sum(row[6] for row in metric_values) / count,
        masked_deviance=sum(row[7] for row in metric_values) / count,
    )
    _emit_progress(
        progress_callback,
        {
            "event": "phase_end",
            "phase": "train",
            "epoch": epoch,
            "batches": metrics.batches,
            "loss": metrics.loss,
            "deviance": metrics.deviance,
            "masked_deviance": metrics.masked_deviance,
            "masked": metrics.masked,
            "visible": metrics.visible,
            "latent": metrics.latent,
            "gate": metrics.gate,
            "variance": metrics.variance,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_steps": optimizer_steps,
            **phase_health,
        },
    )
    return metrics


@torch.no_grad()
def evaluate_two_view_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    ema_teacher: nn.Module,
    consistency_head: nn.Module,
    *,
    device: torch.device | str,
    max_batches: int | None = None,
    amp: bool = False,
    log_interval: int | None = None,
    epoch: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EpochMetrics:
    """Evaluate two-view objectives without changing the EMA or queue."""

    device = torch.device(device)
    model.eval()
    consistency_head.eval()
    losses: list[TwoViewPretrainingLoss] = []
    started = time.perf_counter()
    total_batches = _loader_batch_count(loader, max_batches)
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        losses.append(
            two_view_pretraining_step(
                model,
                ema_teacher,
                consistency_head,
                _move_batch(batch, device, non_blocking=device.type == "cuda"),
                optimizer=None,
                amp=amp,
            )
        )
        if log_interval and (index == 0 or (index + 1) % log_interval == 0):
            loss = losses[-1]
            batch_health = _health_values(loss.health)
            elapsed = time.perf_counter() - started
            _emit_progress(
                progress_callback,
                {
                    "event": "batch",
                    "phase": "validation",
                    "epoch": epoch,
                    "batch": index + 1,
                    "batches_total": total_batches,
                    "loss": float(loss.total.detach().cpu()),
                    "deviance": float(loss.deviance.detach().cpu()),
                    "latent": float(loss.latent.detach().cpu()),
                    "gate": float(loss.gate.detach().cpu()),
                    "variance": float(loss.variance.detach().cpu()),
                    "elapsed_seconds": elapsed,
                    "batches_per_second": (index + 1) / max(elapsed, 1e-9),
                    **batch_health,
                },
            )
    metrics = _two_view_metrics(losses)
    _emit_progress(
        progress_callback,
        {
            "event": "phase_end",
            "phase": "validation",
            "epoch": epoch,
            "batches": metrics.batches,
            "loss": metrics.loss,
            "deviance": metrics.deviance,
            "latent": metrics.latent,
            "gate": metrics.gate,
            "variance": metrics.variance,
            "elapsed_seconds": time.perf_counter() - started,
            **metrics.health,
        },
    )
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save model/optimizer state and reproducibility metadata."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing checkpoint {destination}; "
            "choose a fresh run path or pass overwrite=True explicitly"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {**artifact_version_metadata(), **dict(metadata or {})}
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metrics": dict(metrics),
        "metadata": checkpoint_metadata,
        "extra_state": dict(extra_state or {}),
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def capture_rng_state() -> dict[str, Any]:
    """Capture all process RNG streams needed for an exact training resume."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch_state = state["torch"]
        if torch.is_tensor(torch_state):
            torch_state = torch_state.detach().cpu().to(dtype=torch.uint8)
        torch.random.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            value.detach().cpu().to(dtype=torch.uint8) if torch.is_tensor(value) else value
            for value in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def validate_resume_state(
    extra_state: Mapping[str, Any], *, require_cuda: bool = False
) -> None:
    """Fail closed when a latest checkpoint lacks exact-resume state."""

    required = ("scheduler", "scaler", "rng_state", "training_state", "loader_generators")
    if not all(isinstance(extra_state.get(key), Mapping) for key in required):
        raise IncompatibleArtifactError(
            "checkpoint lacks complete scheduler/scaler/RNG/training/DataLoader state"
        )
    rng_state = extra_state["rng_state"]
    if not {"python", "numpy", "torch"}.issubset(rng_state):
        raise IncompatibleArtifactError("checkpoint has incomplete process RNG state")
    if require_cuda and "cuda" not in rng_state:
        raise IncompatibleArtifactError("checkpoint lacks CUDA RNG state")
    loader_states = extra_state["loader_generators"]
    if not torch.is_tensor(loader_states.get("train")) or not torch.is_tensor(
        loader_states.get("validation")
    ):
        raise IncompatibleArtifactError("checkpoint has incomplete DataLoader generator state")
    training_state = extra_state["training_state"]
    if not {"best_deviance", "best_epoch", "epochs_without_improvement"}.issubset(
        training_state
    ):
        raise IncompatibleArtifactError("checkpoint has incomplete training state")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    extra_modules: Mapping[str, nn.Module] | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint and return its epoch/metrics/metadata."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    require_artifact_versions(payload.get("metadata", {}), artifact_name="checkpoint")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if not isinstance(payload.get("optimizer"), Mapping):
            raise IncompatibleArtifactError(
                "checkpoint is missing required optimizer state for resume"
            )
        optimizer.load_state_dict(payload["optimizer"])
    extra_state = payload.get("extra_state", {})
    if not isinstance(extra_state, Mapping):
        raise IncompatibleArtifactError("checkpoint extra_state is not a mapping")
    for name, module in (extra_modules or {}).items():
        if name not in extra_state or not isinstance(extra_state[name], Mapping):
            raise IncompatibleArtifactError(
                f"checkpoint is missing required resumable module state: {name}"
            )
        module.load_state_dict(extra_state[name])
    return {
        "epoch": int(payload["epoch"]),
        "metrics": dict(payload.get("metrics", {})),
        "metadata": dict(payload.get("metadata", {})),
        "extra_state": dict(extra_state),
    }
