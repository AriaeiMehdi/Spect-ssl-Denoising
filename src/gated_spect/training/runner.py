"""Minimal reproducible training/evaluation loops and atomic checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import time
import math
from typing import Any, Callable, Iterable, Literal, Mapping

import torch
from torch import Tensor, nn

from .consistency_steps import TwoViewPretrainingLoss, two_view_pretraining_step
from .steps import DenoisingLoss, PretrainingLoss, denoising_step, pretraining_step


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
            "masked": metrics.masked,
            "visible": metrics.visible,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_steps": optimizer_steps,
            **phase_health,
        },
    )
    return metrics


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
            "masked": metrics.masked,
            "visible": metrics.visible,
            "elapsed_seconds": time.perf_counter() - started,
            **phase_health,
        },
    )
    return metrics


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
    metric_values: list[tuple[float, float, float, float, float, float, float, int]] = []
    health_sums: dict[str, float] = {}
    gradient_norms: list[float] = []
    last_gradient_norm: float | None = None
    optimizer_steps = 0
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
            if scaler is not None and autocast_enabled:
                scaler.unscale_(optimizer)
            if pending < accumulation_steps:
                correction = accumulation_steps / pending
                for parameter in list(model.parameters()) + list(consistency_head.parameters()):
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            parameters = list(model.parameters()) + list(consistency_head.parameters())
            measured_norm = nn.utils.clip_grad_norm_(parameters, 1.0)
            last_gradient_norm = _record_gradient_norm(measured_norm, gradient_norms)
            if scaler is not None and autocast_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema_teacher.update(model.backbone)
            if queued_values:
                consistency_head.queue.enqueue(torch.cat(queued_values, dim=0))
            queued_values.clear()
            pending = 0
            optimizer_steps += 1
        metric_values.append(
            (
                float(loss.total.detach().cpu()),
                float(loss.deviance.detach().cpu()),
                float(loss.latent.detach().cpu()),
                float(loss.gate.detach().cpu()),
                float(loss.variance.detach().cpu()),
                float(loss.masked.detach().cpu()),
                float(loss.visible.detach().cpu()),
                1,
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
        if scaler is not None and autocast_enabled:
            scaler.unscale_(optimizer)
        correction = accumulation_steps / pending
        for parameter in list(model.parameters()) + list(consistency_head.parameters()):
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        parameters = list(model.parameters()) + list(consistency_head.parameters())
        measured_norm = nn.utils.clip_grad_norm_(parameters, 1.0)
        last_gradient_norm = _record_gradient_norm(measured_norm, gradient_norms)
        if scaler is not None and autocast_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema_teacher.update(model.backbone)
        if queued_values:
            consistency_head.queue.enqueue(torch.cat(queued_values, dim=0))
        optimizer_steps += 1
    if not metric_values:
        raise ValueError("two-view loader produced no batches")
    count = float(len(metric_values))
    phase_health = _health_average(health_sums, len(metric_values))
    if gradient_norms:
        phase_health["gradient_norm_mean"] = sum(gradient_norms) / len(gradient_norms)
        phase_health["gradient_norm_max"] = max(gradient_norms)
    phase_health["learning_rate"] = float(optimizer.param_groups[0]["lr"])
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
) -> Path:
    """Atomically save model/optimizer state and reproducibility metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metrics": dict(metrics),
        "metadata": dict(metadata or {}),
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
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    for name, module in (extra_modules or {}).items():
        if name in payload.get("extra_state", {}):
            module.load_state_dict(payload["extra_state"][name])
    return {
        "epoch": int(payload["epoch"]),
        "metrics": dict(payload.get("metrics", {})),
        "metadata": dict(payload.get("metadata", {})),
    }
