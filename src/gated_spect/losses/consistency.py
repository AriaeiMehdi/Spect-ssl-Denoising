"""SimSiam-style global and hidden-gate consistency objectives."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNormMLP(nn.Module):
    """Two-layer LayerNorm MLP; BatchNorm is intentionally not used."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def stop_gradient_cosine_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Cosine distance with the target branch detached."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have matching shapes [N,C]")
    return 1.0 - F.cosine_similarity(prediction, target.detach(), dim=-1).mean()


class RepresentationQueue(nn.Module):
    """FIFO queue of detached projected representations for variance regularization."""

    def __init__(self, dimension: int = 256, capacity: int = 127) -> None:
        super().__init__()
        if dimension < 1 or capacity < 1:
            raise ValueError("queue dimension and capacity must be positive")
        self.dimension = int(dimension)
        self.capacity = int(capacity)
        self.register_buffer("storage", torch.zeros(capacity, dimension))
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("size", torch.zeros((), dtype=torch.long))

    @property
    def num_entries(self) -> int:
        return int(self.size.item())

    def values(self) -> Tensor:
        return self.storage[: self.num_entries]

    @torch.no_grad()
    def enqueue(self, values: Tensor) -> None:
        values = values.detach()
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"values must have shape [N,{self.dimension}]")
        if values.numel() == 0:
            return
        if values.shape[0] >= self.capacity:
            values = values[-self.capacity :]
        count = values.shape[0]
        start = int(self.pointer.item())
        indices = (torch.arange(count, device=self.storage.device) + start) % self.capacity
        self.storage[indices] = values.to(device=self.storage.device, dtype=self.storage.dtype)
        self.pointer[...] = (start + count) % self.capacity
        self.size[...] = min(self.capacity, self.num_entries + count)


def variance_floor_loss(
    projected: Tensor,
    queue: RepresentationQueue,
    *,
    minimum_entries: int = 31,
    target_std: float = 0.10,
) -> Tensor:
    """Penalize collapsed projected dimensions after the queue warms up."""

    if projected.ndim != 2 or projected.shape[1] != queue.dimension:
        raise ValueError("projected must have shape [N,queue.dimension]")
    if queue.num_entries < minimum_entries:
        return projected.new_zeros(())
    combined = torch.cat((queue.values().to(projected), projected), dim=0)
    standard_deviation = combined.std(dim=0, unbiased=False)
    return F.relu(float(target_std) - standard_deviation).mean()


@dataclass(frozen=True)
class GlobalConsistencyLoss:
    total: Tensor
    cosine: Tensor
    variance: Tensor
    projected_a: Tensor
    projected_b: Tensor


class GlobalConsistencyHead(nn.Module):
    """Projection/prediction heads and queue-backed SimSiam objective."""

    def __init__(self, dimension: int = 256) -> None:
        super().__init__()
        self.projector = LayerNormMLP(dimension, dimension, dimension)
        self.predictor = LayerNormMLP(dimension, dimension, dimension)
        self.queue = RepresentationQueue(dimension=dimension, capacity=127)

    def forward(self, representation_a: Tensor, representation_b: Tensor) -> GlobalConsistencyLoss:
        projected_a = self.projector(representation_a)
        projected_b = self.projector(representation_b)
        predicted_a = self.predictor(projected_a)
        predicted_b = self.predictor(projected_b)
        cosine = 0.5 * (
            stop_gradient_cosine_loss(predicted_a, projected_b)
            + stop_gradient_cosine_loss(predicted_b, projected_a)
        )
        variance = variance_floor_loss(torch.cat((projected_a, projected_b), dim=0), self.queue)
        return GlobalConsistencyLoss(
            total=cosine + variance,
            cosine=cosine,
            variance=variance,
            projected_a=projected_a,
            projected_b=projected_b,
        )


def pool_gate_bottleneck(bottleneck: Tensor, valid_frames: Tensor) -> Tensor:
    """Pool bottleneck tokens to ``[B,G,C]`` using valid-view masks."""

    if bottleneck.ndim != 6 or valid_frames.shape != bottleneck.shape[:3]:
        raise ValueError("bottleneck must be [B,G,V,C,h,w] and mask [B,G,V]")
    frame_features = bottleneck.mean(dim=(-2, -1))
    weights = valid_frames.to(dtype=bottleneck.dtype)[..., None]
    denominator = weights.sum(dim=2).clamp_min(1.0)
    return (frame_features * weights).sum(dim=2) / denominator


def hidden_gate_consistency_loss(
    student_bottleneck: Tensor,
    teacher_bottleneck: Tensor,
    student_valid_frames: Tensor,
    teacher_valid_frames: Tensor,
    relative_exposure_q: Tensor,
    hidden_gate_mask: Tensor,
) -> Tensor:
    """Match hidden student gates to unmasked teacher gates above q=0.20."""

    if teacher_bottleneck.shape != student_bottleneck.shape:
        raise ValueError("student and teacher bottlenecks must have identical shapes")
    if student_valid_frames.shape != teacher_valid_frames.shape:
        raise ValueError("student and teacher masks must have identical shapes")
    if relative_exposure_q.shape != student_valid_frames.shape:
        raise ValueError("relative_exposure_q must have shape [B,G,V]")
    if hidden_gate_mask.shape != student_valid_frames.shape[:2]:
        raise ValueError("hidden_gate_mask must have shape [B,G]")
    student = pool_gate_bottleneck(student_bottleneck, student_valid_frames)
    teacher = pool_gate_bottleneck(teacher_bottleneck, teacher_valid_frames)
    q_values = relative_exposure_q.masked_fill(~teacher_valid_frames, float("nan"))
    gate_q = torch.nanmedian(q_values, dim=2).values
    gate_q = torch.nan_to_num(gate_q, nan=0.0)
    eligible = hidden_gate_mask & (gate_q >= 0.20) & teacher_valid_frames.any(dim=2)
    if not torch.any(eligible):
        return student.new_zeros(())
    return 1.0 - F.cosine_similarity(student[eligible], teacher[eligible].detach(), dim=-1).mean()


class EMATeacher(nn.Module):
    """Frozen exponential-moving-average copy updated only after optimizer steps."""

    def __init__(self, student: nn.Module, decay: float = 0.996) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = float(decay)
        self.teacher = deepcopy(student).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        teacher_parameters = dict(self.teacher.named_parameters())
        student_parameters = dict(student.named_parameters())
        if teacher_parameters.keys() != student_parameters.keys():
            raise ValueError("EMA student and teacher parameter structures differ")
        for name, teacher_parameter in teacher_parameters.items():
            teacher_parameter.mul_(self.decay).add_(student_parameters[name], alpha=1.0 - self.decay)
        teacher_buffers = dict(self.teacher.named_buffers())
        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in teacher_buffers.items():
            if name in student_buffers:
                teacher_buffer.copy_(student_buffers[name])

    def forward(self, *args: Tensor, **kwargs: Tensor) -> Any:
        with torch.no_grad():
            return self.teacher(*args, **kwargs)
