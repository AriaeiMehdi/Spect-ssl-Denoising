"""Poisson objectives, deterministic binomial thinning, and masking rules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import torch
from torch import Tensor


def count_split_seed(
    global_seed: int,
    epoch: int,
    sample_id: str,
    augmentation_stream: str = "default",
) -> int:
    """Derive a worker-independent per-acquisition/per-epoch RNG seed.

    The stream identifies the scientific view (for example ``student_a``,
    ``student_b`` or ``teacher``).  DataLoader worker assignment is
    deliberately absent from this contract.
    """

    payload = f"{int(global_seed)}|{int(epoch)}|{sample_id}|{str(augmentation_stream)}".encode("utf-8")
    # Torch generators accept signed 64-bit seeds.  Keep a nonzero value so a
    # malformed all-zero digest cannot accidentally select the default stream.
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)
    return seed or 1


def binomial_count_split(
    counts: Tensor,
    p: float,
    *,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Split integer counts into exact binomial complements.

    The calculation is performed in float64 because clinical bins can exceed
    float32's exact-integer range.  ``a`` and ``b`` are returned as int64 and
    satisfy ``a + b == counts`` elementwise.
    """

    if not torch.is_floating_point(counts) and not counts.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("counts must be an integer tensor")
    if torch.any(counts < 0) or torch.any(counts != counts.to(dtype=torch.int64)):
        raise ValueError("counts must be nonnegative integer-valued")
    if not 0.0 < float(p) < 1.0:
        raise ValueError("p must lie strictly between zero and one")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=counts.device)
        generator.manual_seed(int(seed))
    total = counts.to(dtype=torch.float64)
    probability = torch.full_like(total, float(p))
    first = torch.binomial(total, probability, generator=generator).to(dtype=torch.int64)
    second = counts.to(dtype=torch.int64) - first
    if torch.any(first < 0) or torch.any(second < 0) or torch.any(first + second != counts):
        raise RuntimeError("binomial count split violated conservation")
    return first, second


def poisson_nll(mean: Tensor, counts: Tensor, *, reduction: str = "mean") -> Tensor:
    """Constant-free Poisson negative log likelihood."""

    if mean.shape != counts.shape:
        raise ValueError("mean and counts must have identical shapes")
    value = mean.clamp_min(1.0e-8) - counts * torch.log(mean.clamp_min(1.0e-8))
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    raise ValueError("reduction must be none, mean, or sum")


def poisson_deviance(mean: Tensor, counts: Tensor, *, reduction: str = "mean") -> Tensor:
    """Poisson deviance with the ``y=0`` logarithmic term defined as zero."""

    if mean.shape != counts.shape:
        raise ValueError("mean and counts must have identical shapes")
    safe_mean = mean.clamp_min(1.0e-8)
    log_term = torch.where(counts > 0, counts * torch.log(counts.clamp_min(1.0e-8) / safe_mean), 0.0)
    value = 2.0 * (safe_mean - counts + log_term)
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    raise ValueError("reduction must be none, mean, or sum")


def weighted_poisson_nll(
    mean: Tensor,
    counts: Tensor,
    frame_weight: Tensor,
    *,
    bin_weight: Tensor | None = None,
) -> Tensor:
    """Patient-balanced weighted Poisson NLL.

    ``frame_weight`` is ``[B,G,V]`` and is applied to every detector bin.  An
    optional ``bin_weight`` with shape ``[B,G,V,H,W]`` supports masked-bin
    objectives.  The reduction is a mean of per-patient weighted means, so a
    high-count patient cannot dominate by having more eligible bins.
    """

    if mean.shape != counts.shape or mean.ndim != 5:
        raise ValueError("mean and counts must have shape [B,G,V,H,W]")
    if tuple(frame_weight.shape) != tuple(mean.shape[:3]):
        raise ValueError("frame_weight must have shape [B,G,V]")
    # Frame weights apply to every detector bin.  Expand them across H/W so
    # the denominator counts selected bins rather than only selected frames.
    # Without this expansion, an unmasked [64, 64] detector loss is inflated
    # by 4096 and produces incorrectly scaled denoising gradients.
    # Keep the reduction in float32: a 32-view 64x64 acquisition has 131,072
    # selected bins, which exceeds float16's finite range for the denominator.
    weights = frame_weight.to(dtype=torch.float32)[..., None, None].expand_as(mean)
    if bin_weight is not None:
        if bin_weight.shape != mean.shape:
            raise ValueError("bin_weight must match mean/counts shape")
        weights = weights * bin_weight.to(dtype=torch.float32)
    values = poisson_nll(mean.to(dtype=torch.float32), counts.to(dtype=torch.float32), reduction="none")
    numerator = (values * weights).flatten(1).sum(dim=1)
    denominator = weights.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).mean()


@dataclass(frozen=True)
class MaskPlan:
    """Visibility and target masks for one self-supervised acquisition."""

    task: Literal["whole_views", "angular_block", "detector_patches", "complete_gate"]
    visibility: Tensor  # [G,V,H,W], false means artificially hidden or invalid
    masked_target: Tensor  # [G,V,H,W]
    visible_target: Tensor  # [G,V,H,W], random visible-bin subset


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def make_mask_plan(
    valid_frames: Tensor,
    relative_exposure_q: Tensor,
    *,
    height: int,
    width: int,
    seed: int,
) -> MaskPlan:
    """Sample one contract-defined primary mask and a 5% visible-bin subset."""

    if valid_frames.ndim != 2 or relative_exposure_q.shape != valid_frames.shape:
        raise ValueError("valid_frames and relative_exposure_q must have shape [G,V]")
    g, v = valid_frames.shape
    if (height < 8 or width < 8 or height % 8 or width % 8):
        raise ValueError("detector dimensions must be positive multiples of eight")
    device = valid_frames.device
    generator = _generator(seed, device)
    visibility = valid_frames.to(dtype=torch.bool)[:, :, None, None].expand(g, v, height, width).clone()
    valid_bins = visibility.clone()
    task_draw = float(torch.rand((), generator=generator, device=device))
    if task_draw < 0.50:
        task: Literal["whole_views", "angular_block", "detector_patches", "complete_gate"] = "whole_views"
        count = min(3, max(v - 1, 0))
        if count:
            selected = torch.randperm(v, generator=generator, device=device)[:count]
            visibility[:, selected] = False
    elif task_draw < 0.75 and v >= 3:
        task = "angular_block"
        start = int(torch.randint(0, v - 1, (), generator=generator, device=device))
        visibility[:, start : start + 2] = False
    elif task_draw < 0.90 and height >= 8 and width >= 8:
        task = "detector_patches"
        tiles_h, tiles_w = height // 8, width // 8
        tile_count = max(1, int(round(0.25 * tiles_h * tiles_w)))
        selected = torch.randperm(tiles_h * tiles_w, generator=generator, device=device)[:tile_count]
        for tile in selected.tolist():
            top = (tile // tiles_w) * 8
            left = (tile % tiles_w) * 8
            visibility[:, :, top : top + 8, left : left + 8] = False
    else:
        task = "complete_gate"
        eligible_gates = torch.where(
            (relative_exposure_q.masked_fill(~valid_frames, 0.0).median(dim=1).values >= 0.20)
            & valid_frames.any(dim=1)
        )[0]
        if eligible_gates.numel():
            gate = eligible_gates[
                torch.randint(eligible_gates.numel(), (), generator=generator, device=device)
            ]
            visibility[gate] = False

    masked_target = valid_bins & ~visibility
    eligible_visible = visibility & valid_bins
    visible_target = torch.zeros_like(eligible_visible)
    visible_indices = torch.where(eligible_visible.flatten())[0]
    if visible_indices.numel():
        count = max(1, int(round(0.05 * visible_indices.numel())))
        selected = visible_indices[
            torch.randperm(visible_indices.numel(), generator=generator, device=device)[:count]
        ]
        visible_target.view(-1)[selected] = True
    return MaskPlan(task, visibility, masked_target, visible_target)
