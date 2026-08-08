"""Exposure-aware MLEM and OSEM for the reference projector."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from .projector import Geometry, IdealParallelProjector


def subset_view_indices(num_views: int, num_subsets: int | None = None) -> tuple[Tensor, ...]:
    """Return deterministic interleaved view subsets ``v mod K``."""

    if num_views < 1:
        raise ValueError("num_views must be positive")
    subsets = min(4, num_views) if num_subsets is None else int(num_subsets)
    if subsets < 1 or subsets > num_views:
        raise ValueError("num_subsets must be in [1,num_views]")
    result = tuple(torch.arange(index, num_views, subsets, dtype=torch.long) for index in range(subsets))
    if any(index.numel() == 0 for index in result):
        raise ValueError("every OSEM subset must be nonempty")
    return result


def poisson_nll(mean: Tensor, counts: Tensor, *, reduction: str = "mean") -> Tensor:
    """Constant-free Poisson NLL ``mean-count*log(mean)``."""

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


def _fov_mask(geometry: Geometry, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    z, y, x = geometry.volume_shape
    yy = torch.arange(y, device=device, dtype=dtype) - (y - 1) / 2.0
    xx = torch.arange(x, device=device, dtype=dtype) - (x - 1) / 2.0
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    radius = min(y, x) / 2.0
    disk = ((xx / radius) ** 2 + (yy / radius) ** 2 <= 1.0).to(dtype)
    return disk.expand(z, y, x)


def _initial_volume(
    projector: IdealParallelProjector,
    counts: Tensor,
    exposure_s: Tensor,
    scatter_rate: Tensor | None,
    fov_mask: Tensor | None,
) -> Tensor:
    b, g, _, _, _ = counts.shape
    z, y, x = projector.geometry.volume_shape
    mask = _fov_mask(projector.geometry, device=counts.device, dtype=counts.dtype)
    if fov_mask is not None:
        if tuple(fov_mask.shape) != (z, y, x):
            raise ValueError("fov_mask must match volume_shape")
        mask = mask * fov_mask.to(device=counts.device, dtype=counts.dtype)
    unit = mask.expand(b, g, z, y, x)
    unit_projection = projector.forward(unit)
    background = torch.zeros_like(counts) if scatter_rate is None else scatter_rate
    if background.shape != counts.shape:
        raise ValueError("scatter_rate must match counts shape")
    signal_total = (counts - exposure_s[..., None, None] * background).sum(dim=(2, 3, 4))
    primary_total = signal_total.clamp_min(1.0e-8)
    unit_total = (exposure_s[..., None, None] * unit_projection).sum(dim=(2, 3, 4))
    scale = primary_total / unit_total.clamp_min(1.0e-8)
    return unit * scale[..., None, None, None]


def _validate_inputs(projector: IdealParallelProjector, counts: Tensor, exposure_s: Tensor) -> None:
    if counts.ndim != 5:
        raise ValueError("counts must have shape [B,G,V,H,W]")
    if exposure_s.shape != counts.shape[:3]:
        raise ValueError("exposure_s must have shape [B,G,V]")
    expected = (
        projector.geometry.num_views,
        projector.geometry.detector_height,
        projector.geometry.detector_width,
    )
    if tuple(counts.shape[2:]) != expected:
        raise ValueError(f"counts detector shape must be {expected}")
    if torch.any(counts < 0) or torch.any(exposure_s < 0):
        raise ValueError("counts and exposure_s must be nonnegative")


def _subset_mask(
    num_views: int,
    subset: Iterable[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    mask = torch.zeros(num_views, device=device, dtype=dtype)
    indices = torch.as_tensor(tuple(int(x) for x in subset), device=device, dtype=torch.long)
    mask[indices] = 1
    return mask.reshape(1, 1, num_views, 1, 1)


def _update(
    projector: IdealParallelProjector,
    volume: Tensor,
    counts: Tensor,
    exposure_s: Tensor,
    *,
    scatter_rate: Tensor | None,
    view_mask: Tensor | None,
    sensitivity_cache: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    prediction = projector.forward(volume)
    background = torch.zeros_like(prediction) if scatter_rate is None else scatter_rate
    if background.shape != prediction.shape:
        raise ValueError("scatter_rate must match projected counts shape")
    if view_mask is None:
        selected = torch.ones_like(prediction)
    else:
        selected = view_mask.to(device=prediction.device, dtype=prediction.dtype)
    exposure = exposure_s[..., None, None] * selected
    y = counts * selected
    mu = exposure * (prediction + background)
    ratio = exposure * y / mu.clamp_min(1.0e-8)
    numerator = projector.adjoint(ratio)
    if sensitivity_cache is None:
        sensitivity_cache = projector.adjoint(exposure)
    updated = volume * numerator / sensitivity_cache.clamp_min(1.0e-8)
    return updated.clamp_min(1.0e-8), sensitivity_cache


def mlem_reconstruct(
    projector: IdealParallelProjector,
    counts: Tensor,
    exposure_s: Tensor,
    *,
    iterations: int = 10,
    scatter_rate: Tensor | None = None,
    initial_volume: Tensor | None = None,
    fov_mask: Tensor | None = None,
) -> Tensor:
    """Run full-data MLEM iterations and return a positive volume."""

    _validate_inputs(projector, counts, exposure_s)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    volume = (
        _initial_volume(projector, counts, exposure_s, scatter_rate, fov_mask)
        if initial_volume is None
        else initial_volume.to(device=counts.device, dtype=counts.dtype).clamp_min(1.0e-8)
    )
    if tuple(volume.shape) != (counts.shape[0], counts.shape[1], *projector.geometry.volume_shape):
        raise ValueError("initial_volume has the wrong shape")
    sensitivity: Tensor | None = None
    for _ in range(iterations):
        volume, sensitivity = _update(
            projector,
            volume,
            counts,
            exposure_s,
            scatter_rate=scatter_rate,
            view_mask=None,
            sensitivity_cache=sensitivity,
        )
    return volume


def osem_reconstruct(
    projector: IdealParallelProjector,
    counts: Tensor,
    exposure_s: Tensor,
    *,
    iterations: int = 8,
    num_subsets: int | None = None,
    scatter_rate: Tensor | None = None,
    initial_volume: Tensor | None = None,
    fov_mask: Tensor | None = None,
) -> Tensor:
    """Run exposure-aware ordered-subsets EM with interleaved views."""

    _validate_inputs(projector, counts, exposure_s)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    volume = (
        _initial_volume(projector, counts, exposure_s, scatter_rate, fov_mask)
        if initial_volume is None
        else initial_volume.to(device=counts.device, dtype=counts.dtype).clamp_min(1.0e-8)
    )
    if tuple(volume.shape) != (counts.shape[0], counts.shape[1], *projector.geometry.volume_shape):
        raise ValueError("initial_volume has the wrong shape")
    subsets = subset_view_indices(projector.geometry.num_views, num_subsets)
    sensitivity: dict[tuple[int, ...], Tensor] = {}
    for subset in subsets:
        mask = _subset_mask(
            projector.geometry.num_views,
            subset.tolist(),
            device=counts.device,
            dtype=counts.dtype,
        )
        exposure = (exposure_s[..., None, None] * mask).expand_as(counts)
        sensitivity[tuple(subset.tolist())] = projector.adjoint(exposure)
    for _ in range(iterations):
        for subset in subsets:
            mask = _subset_mask(
                projector.geometry.num_views,
                subset.tolist(),
                device=counts.device,
                dtype=counts.dtype,
            )
            volume, _ = _update(
                projector,
                volume,
                counts,
                exposure_s,
                scatter_rate=scatter_rate,
                view_mask=mask,
                sensitivity_cache=sensitivity[tuple(subset.tolist())],
            )
    return volume
