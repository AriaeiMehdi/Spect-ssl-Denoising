"""Projection-conditioned unrolled OSEM reconstruction.

This module is intentionally separate from the projection denoiser.  It uses
the original low-count projections in every stage and a fixed projector; a
real-data training command should only enable it after the projector geometry
gate has been signed off.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..physics import Geometry, IdealParallelProjector, osem_reconstruct
from .backbone import ProjectionDenoiser


@dataclass(frozen=True)
class UnrolledOutput:
    """Final volume and all differentiable stage outputs."""

    volume: Tensor  # [B,G,Z,Y,X]
    stages: tuple[Tensor, ...]


class _VolumeResidualBlock(nn.Module):
    def __init__(self, channels: int, *, groups: int = 8) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("volume channels must be divisible by GroupNorm groups")
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class _TemporalVolumeAttention(nn.Module):
    def __init__(self, width: int = 96, heads: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _safe_mask(mask: Tensor) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        empty = mask.all(dim=1)
        if torch.any(empty):
            mask = mask.clone()
            mask[empty, 0] = False
        return mask

    def forward(self, x: Tensor, valid_gates: Tensor) -> Tensor:
        # x: [B,G,96,Z,Y,X] -> sequences over G at each spatial voxel.
        b, g, c, z, y, x_width = x.shape
        sequence = x.permute(0, 3, 4, 5, 1, 2).reshape(b * z * y * x_width, g, c)
        padding = (~valid_gates)[:, None, :].expand(b, z * y * x_width, g).reshape(-1, g)
        valid = valid_gates[:, None, :].expand(b, z * y * x_width, g).reshape(-1, g)
        normalized = self.norm1(sequence)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=self._safe_mask(padding),
            need_weights=False,
        )
        sequence = sequence + self.dropout(attended)
        sequence = sequence + self.mlp(self.norm2(sequence))
        sequence = sequence * valid.unsqueeze(-1).to(dtype=sequence.dtype)
        return sequence.reshape(b, z, y, x_width, g, c).permute(0, 4, 5, 1, 2, 3)


class VolumeRegularizer(nn.Module):
    """Shared 3-D U-Net regularizer with bottleneck temporal mixing."""

    def __init__(self, input_channels: int = 11) -> None:
        super().__init__()
        if input_channels != 11:
            raise ValueError("the contract-defined regularizer input has exactly 11 channels")
        self.in_conv = nn.Conv3d(input_channels, 24, 3, padding=1)
        self.blocks24 = nn.Sequential(_VolumeResidualBlock(24), _VolumeResidualBlock(24))
        self.down48 = nn.Conv3d(24, 48, 3, stride=2, padding=1)
        self.blocks48 = nn.Sequential(_VolumeResidualBlock(48), _VolumeResidualBlock(48))
        self.down96 = nn.Conv3d(48, 96, 3, stride=2, padding=1)
        self.blocks96 = nn.Sequential(_VolumeResidualBlock(96), _VolumeResidualBlock(96))
        self.temporal = _TemporalVolumeAttention()
        self.up48 = nn.Conv3d(96 + 48, 48, 3, padding=1)
        self.decode48 = nn.Sequential(_VolumeResidualBlock(48), _VolumeResidualBlock(48))
        self.up24 = nn.Conv3d(48 + 24, 24, 3, padding=1)
        self.decode24 = nn.Sequential(_VolumeResidualBlock(24), _VolumeResidualBlock(24))
        self.output = nn.Conv3d(24, 1, 1)

    def forward(self, features: Tensor, valid_gates: Tensor) -> Tensor:
        if features.ndim != 6 or features.shape[2] != 11:
            raise ValueError("regularizer features must have shape [B,G,11,Z,Y,X]")
        b, g, _, z, y, x = features.shape
        if min(z, y, x) < 4 or any(size % 4 for size in (z, y, x)):
            raise ValueError("volume dimensions must be at least four and divisible by four")
        flat = features.reshape(b * g, 11, z, y, x)
        skip24 = self.blocks24(self.in_conv(flat))
        skip48 = self.blocks48(self.down48(skip24))
        bottleneck = self.blocks96(self.down96(skip48))
        bottleneck = bottleneck.reshape(b, g, 96, *bottleneck.shape[-3:])
        bottleneck = self.temporal(bottleneck, valid_gates)
        x_bottleneck = bottleneck.reshape(b * g, 96, *bottleneck.shape[-3:])

        x_bottleneck = F.interpolate(x_bottleneck, size=skip48.shape[-3:], mode="trilinear", align_corners=False)
        x_bottleneck = self.decode48(self.up48(torch.cat((x_bottleneck, skip48), dim=1)))
        x_bottleneck = F.interpolate(x_bottleneck, size=skip24.shape[-3:], mode="trilinear", align_corners=False)
        x_bottleneck = self.decode24(self.up24(torch.cat((x_bottleneck, skip24), dim=1)))
        return self.output(x_bottleneck).reshape(b, g, z, y, x)


def _lift_projection_features(
    projector: IdealParallelProjector,
    features: Tensor,
    exposure_s: Tensor,
    sensitivity: Tensor,
) -> Tensor:
    """Lift [B,G,V,C,H,W] detector features into [B,G,C,Z,Y,X]."""

    if features.ndim != 6 or features.shape[3] != 8:
        raise ValueError("projection features must have shape [B,G,V,8,H,W]")
    b, g, v, channels, h, w = features.shape
    lifted: list[Tensor] = []
    exposure = exposure_s[..., None, None]
    for channel in range(channels):
        detector_feature = features[:, :, :, channel]
        weighted = exposure * detector_feature
        lifted.append(projector.adjoint(weighted) / sensitivity.clamp_min(1.0e-8))
    return torch.stack(lifted, dim=2)


class UnrolledOSEM(nn.Module):
    """Five-stage projection-conditioned OSEM with shared 3-D regularizer."""

    def __init__(
        self,
        projector: IdealParallelProjector,
        projection_model: ProjectionDenoiser,
        *,
        stages: int = 5,
        multiplicative_limit: float = 0.5,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be positive")
        if multiplicative_limit <= 0:
            raise ValueError("multiplicative_limit must be positive")
        self.projector = projector
        self.projection_model = projection_model
        self.stages = int(stages)
        self.multiplicative_limit = float(multiplicative_limit)
        self.regularizer = VolumeRegularizer(input_channels=11)

    def _sensitivity(self, exposure_s: Tensor, reference: Tensor) -> Tensor:
        exposure = exposure_s[..., None, None].expand_as(reference)
        return self.projector.adjoint(exposure).clamp_min(1.0e-8)

    def forward(
        self,
        *,
        projection_batch: dict[str, Tensor],
        low_counts: Tensor,
        low_exposure_s: Tensor,
        valid_frames: Tensor,
    ) -> UnrolledOutput:
        if low_counts.ndim != 5:
            raise ValueError("low_counts must have shape [B,G,V,H,W]")
        if low_exposure_s.shape != low_counts.shape[:3] or valid_frames.shape != low_counts.shape[:3]:
            raise ValueError("low exposure and valid_frames must have shape [B,G,V]")
        geometry = self.projector.geometry
        if tuple(low_counts.shape[2:]) != (
            geometry.num_views,
            geometry.detector_height,
            geometry.detector_width,
        ):
            raise ValueError("low_counts shape does not match projector geometry")
        denoised = self.projection_model(**projection_batch)
        if denoised.conditioning is None:
            raise ValueError("projection model must expose eight conditioning channels")

        # One full conventional OSEM sweep is the differentiable initialization.
        x = osem_reconstruct(
            self.projector,
            low_counts,
            low_exposure_s,
            iterations=1,
            num_subsets=min(4, geometry.num_views),
        )
        sensitivity = self._sensitivity(low_exposure_s, low_counts)
        lifted = _lift_projection_features(
            self.projector,
            denoised.conditioning,
            low_exposure_s,
            sensitivity,
        )
        valid_gates = valid_frames.any(dim=2)
        outputs: list[Tensor] = []
        for _ in range(self.stages):
            u = osem_reconstruct(
                self.projector,
                low_counts,
                low_exposure_s,
                iterations=1,
                num_subsets=min(4, geometry.num_views),
                initial_volume=x,
            )
            predicted = self.projector.forward(u)
            mean = low_exposure_s[..., None, None] * predicted
            residual = (low_counts - mean) / torch.sqrt(mean.clamp_min(0.0) + 1.0)
            residual_backprojection = self.projector.adjoint(
                low_exposure_s[..., None, None] * residual
            ) / torch.sqrt(sensitivity)
            log_volume = torch.log(u.clamp_min(1.0e-8))
            log_sensitivity = torch.log(sensitivity)
            log_sensitivity = (log_sensitivity - log_sensitivity.mean(dim=(2, 3, 4), keepdim=True)) / (
                log_sensitivity.std(dim=(2, 3, 4), keepdim=True).clamp_min(1.0e-6)
            )
            normalized_residual = residual_backprojection / residual_backprojection.std(
                dim=(2, 3, 4), keepdim=True
            ).clamp_min(1.0e-6)
            regularizer_input = torch.cat(
                (
                    log_volume[:, :, None],
                    log_sensitivity[:, :, None],
                    normalized_residual[:, :, None],
                    lifted,
                ),
                dim=2,
            )
            delta = self.regularizer(regularizer_input, valid_gates)
            x = (u * torch.exp(self.multiplicative_limit * torch.tanh(delta))).clamp_min(1.0e-8)
            outputs.append(x)
        return UnrolledOutput(volume=x, stages=tuple(outputs))


def denoise_then_osem(
    projector: IdealParallelProjector,
    denoiser: ProjectionDenoiser,
    *,
    projection_batch: dict[str, Tensor],
    requested_exposure_s: Tensor,
    iterations: int = 8,
    num_subsets: int | None = None,
) -> Tensor:
    """Required baseline: model-mean projections followed by conventional OSEM."""

    output = denoiser(**projection_batch)
    if requested_exposure_s.shape != output.rate.shape[:3]:
        raise ValueError("requested_exposure_s must have shape [B,G,V]")
    expected_counts = requested_exposure_s[..., None, None] * output.rate
    return osem_reconstruct(
        projector,
        expected_counts,
        requested_exposure_s,
        iterations=iterations,
        num_subsets=num_subsets,
    )
