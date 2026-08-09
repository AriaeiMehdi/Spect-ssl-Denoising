"""Contract-first gated projection backbone and rate decoder.

The model keeps the acquisition axes explicit.  A frame encoder is shared over
all ``G x V`` detector frames, followed by alternating angular and temporal
attention at the 1/8-resolution bottleneck.  Padding is represented by masks;
no padded frame is allowed to contribute to the global representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


_EPS: Final[float] = 1.0e-6
_EXPOSURE_CLIP: Final[float] = 8.0


def standardized_log_exposure(
    exposure_s: Tensor,
    mean: float,
    std: float,
) -> Tensor:
    """Return the bounded network conditioning transform for exposure.

    Likelihood exposure tensors remain untouched; only neural conditioning is
    standardized and clipped to the documented ``[-8, 8]`` range.
    """

    if std <= 0:
        raise ValueError("exposure standard deviation must be positive")
    value = (torch.log(exposure_s.to(dtype=torch.float32).clamp_min(_EPS)) - float(mean)) / float(std)
    return value.clamp(min=-_EXPOSURE_CLIP, max=_EXPOSURE_CLIP)


def pool_global_bottleneck(bottleneck_tokens: Tensor, valid_frames: Tensor) -> Tensor:
    """Mean-pool valid spatial tokens and frames to ``[B, C]``."""

    if bottleneck_tokens.ndim != 6 or valid_frames.shape != bottleneck_tokens.shape[:3]:
        raise ValueError("bottleneck_tokens must be [B,G,V,C,H,W] with [B,G,V] validity")
    frame_features = bottleneck_tokens.mean(dim=(-2, -1))
    frame_weights = valid_frames.to(dtype=bottleneck_tokens.dtype)[..., None]
    denominator = frame_weights.sum(dim=(1, 2)).clamp_min(1.0)
    return (frame_features * frame_weights).sum(dim=(1, 2)) / denominator


@dataclass(frozen=True)
class BackboneOutput:
    """Features returned by :class:`ProjectionBackbone`."""

    bottleneck: Tensor  # [B,G,V,96,H/8,W/8]
    skip16: Tensor  # [B,G,V,64,H/4,W/4]
    skip32: Tensor  # [B,G,V,48,H/2,W/2]
    skip64: Tensor  # [B,G,V,32,H,W]
    global_representation: Tensor  # [B,256]
    exposure_clip_fraction: Tensor | None = None


@dataclass(frozen=True)
class ProjectionOutput:
    """Denoiser output in rate and optional conditioning-feature form."""

    rate: Tensor  # [B,G,V,H,W], counts per second
    conditioning: Tensor | None  # [B,G,V,8,H,W]
    global_representation: Tensor  # [B,256]
    exposure_clip_fraction: Tensor | None = None


class ResidualBlock(nn.Module):
    """Pre-activation residual block used by the frame and projection decoders."""

    def __init__(self, channels: int, *, groups: int = 8) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by the GroupNorm group count")
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class _TokenBlock(nn.Module):
    """Pre-norm transformer block with a safe key-padding mask."""

    def __init__(self, width: int = 96, heads: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.drop2 = nn.Dropout(dropout)

    @staticmethod
    def _safe_padding_mask(mask: Tensor) -> Tensor:
        """Ensure MultiheadAttention never receives an all-masked sequence."""

        mask = mask.to(dtype=torch.bool)
        all_padding = mask.all(dim=1)
        if torch.any(all_padding):
            mask = mask.clone()
            mask[all_padding, 0] = False
        return mask

    def forward(self, x: Tensor, padding_mask: Tensor, valid_tokens: Tensor) -> Tensor:
        # A valid mask is applied after attention as well as being used for keys.
        # This prevents the unmasked fallback token in an entirely empty sequence
        # from leaking into the representation.
        safe_mask = self._safe_padding_mask(padding_mask)
        y = self.norm1(x)
        y, _ = self.attention(y, y, y, key_padding_mask=safe_mask, need_weights=False)
        x = x + self.drop1(y)
        x = x + self.drop2(self.feed_forward(self.norm2(x)))
        return x * valid_tokens.unsqueeze(-1).to(dtype=x.dtype)


class ProjectionBackbone(nn.Module):
    """Shared frame encoder plus alternating angular/temporal bottleneck mixing.

    ``counts_input`` is the standardized log1p-count channel described by the
    data contract.  Exposure is transformed here from seconds to standardized
    log exposure using the supplied statistics.  The model intentionally does
    not infer a duration from a filename or from a scalar batch value.
    """

    width: Final[int] = 96

    def __init__(
        self,
        *,
        log_exposure_mean: float = 0.0,
        log_exposure_std: float = 1.0,
        radius_mean_mm: float = 0.0,
        radius_std_mm: float = 1.0,
        scanner_vocab_size: int = 1,
        collimator_vocab_size: int = 1,
        stress_rest_vocab_size: int = 1,
        attention_dropout: float = 0.10,
        frame_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if log_exposure_std <= 0 or radius_std_mm <= 0:
            raise ValueError("normalization standard deviations must be positive")
        if min(scanner_vocab_size, collimator_vocab_size, stress_rest_vocab_size) < 1:
            raise ValueError("vocabulary sizes must be at least one (index 0 is unknown)")
        if frame_chunk_size is not None and frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive when supplied")

        self.log_exposure_mean = float(log_exposure_mean)
        self.log_exposure_std = float(log_exposure_std)
        self.radius_mean_mm = float(radius_mean_mm)
        self.radius_std_mm = float(radius_std_mm)
        self.frame_chunk_size = int(frame_chunk_size) if frame_chunk_size is not None else None

        # Three explicit channels: standardized log1p counts, standardized log
        # exposure, and visibility.  Valid-frame masking is applied before the
        # shared encoder so padded frames cannot create features.
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.frame32 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.down48 = nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.frame48 = nn.Sequential(ResidualBlock(48), ResidualBlock(48))
        self.down64 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.frame64 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down96 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.frame96 = nn.Sequential(ResidualBlock(96), ResidualBlock(96))

        # Seven continuous metadata values: sin/cos angle, standardized radius,
        # sin/cos gate phase, standardized log exposure, and split fraction.
        self.continuous_metadata = nn.Sequential(
            nn.Linear(7, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )

        self.scanner_embedding = (
            nn.Embedding(scanner_vocab_size, 8) if scanner_vocab_size > 1 else None
        )
        self.collimator_embedding = (
            nn.Embedding(collimator_vocab_size, 8) if collimator_vocab_size > 1 else None
        )
        self.stress_rest_embedding = (
            nn.Embedding(stress_rest_vocab_size, 8) if stress_rest_vocab_size > 1 else None
        )
        enabled_categories = sum(
            x is not None
            for x in (
                self.scanner_embedding,
                self.collimator_embedding,
                self.stress_rest_embedding,
            )
        )
        self.category_projection = (
            nn.Linear(8 * enabled_categories, self.width) if enabled_categories else None
        )

        self.angular1 = _TokenBlock(self.width, dropout=attention_dropout)
        self.temporal1 = _TokenBlock(self.width, dropout=attention_dropout)
        self.angular2 = _TokenBlock(self.width, dropout=attention_dropout)
        self.temporal2 = _TokenBlock(self.width, dropout=attention_dropout)

        self.global_norm = nn.LayerNorm(self.width)
        self.global_projection = nn.Sequential(
            nn.Linear(self.width, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def _encode_frame_chunk(self, flat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode a flattened frame chunk, returning all U-Net skip levels."""

        skip64 = self.frame32(self.stem(flat))
        skip32 = self.frame48(self.down48(skip64))
        skip16 = self.frame64(self.down64(skip32))
        bottleneck = self.frame96(self.down96(skip16))
        return skip64, skip32, skip16, bottleneck

    def _encode_frames(self, flat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode frames directly or with checkpointed chunks to save GPU memory."""

        chunk_size = self.frame_chunk_size
        if chunk_size is None or chunk_size >= flat.shape[0]:
            return self._encode_frame_chunk(flat)
        chunks: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for start in range(0, flat.shape[0], chunk_size):
            chunk = flat[start : start + chunk_size]
            if self.training and torch.is_grad_enabled():
                chunks.append(
                    checkpoint(self._encode_frame_chunk, chunk, use_reentrant=False)
                )
            else:
                chunks.append(self._encode_frame_chunk(chunk))
        return tuple(torch.cat(levels, dim=0) for levels in zip(*chunks))  # type: ignore[return-value]

    @staticmethod
    def _check_shapes(
        counts_input: Tensor,
        input_exposure_s: Tensor,
        visibility: Tensor,
        valid_frames: Tensor,
        angles_rad: Tensor,
        radii_mm: Tensor,
        gate_phase_rad: Tensor,
        split_fraction: Tensor,
    ) -> tuple[int, int, int, int, int]:
        if counts_input.ndim != 6 or counts_input.shape[3] != 1:
            raise ValueError("counts_input must have shape [B,G,V,1,H,W]")
        b, g, v, _, h, w = counts_input.shape
        if h < 8 or w < 8 or h % 8 or w % 8:
            raise ValueError("detector H and W must be divisible by eight")
        expected = {
            "input_exposure_s": (b, g, v),
            "visibility": (b, g, v, h, w),
            "valid_frames": (b, g, v),
            "angles_rad": (b, v),
            "radii_mm": (b, v),
            "gate_phase_rad": (b, g),
        }
        values = {
            "input_exposure_s": input_exposure_s,
            "visibility": visibility,
            "valid_frames": valid_frames,
            "angles_rad": angles_rad,
            "radii_mm": radii_mm,
            "gate_phase_rad": gate_phase_rad,
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(values[name].shape)}")
        if split_fraction.ndim == 1:
            if split_fraction.shape[0] != b:
                raise ValueError("split_fraction must have batch dimension B")
        elif tuple(split_fraction.shape) != (b, 1):
            raise ValueError("split_fraction must have shape [B] or [B,1]")
        return b, g, v, h, w

    def _category_features(
        self,
        scanner_index: Tensor | None,
        collimator_index: Tensor | None,
        stress_rest_index: Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        if self.category_projection is None:
            return None
        embeddings: list[Tensor] = []
        specs = (
            (self.scanner_embedding, scanner_index),
            (self.collimator_embedding, collimator_index),
            (self.stress_rest_embedding, stress_rest_index),
        )
        for embedding, index in specs:
            if embedding is None:
                continue
            if index is None:
                index = torch.zeros(batch_size, dtype=torch.long, device=device)
            index = index.to(device=device, dtype=torch.long).reshape(batch_size)
            if torch.any(index < 0) or torch.any(index >= embedding.num_embeddings):
                raise ValueError("categorical index is outside its training vocabulary")
            embeddings.append(embedding(index))
        return self.category_projection(torch.cat(embeddings, dim=-1))

    def _metadata(
        self,
        input_exposure_s: Tensor,
        angles_rad: Tensor,
        radii_mm: Tensor,
        gate_phase_rad: Tensor,
        split_fraction: Tensor,
        scanner_index: Tensor | None,
        collimator_index: Tensor | None,
        stress_rest_index: Tensor | None,
    ) -> Tensor:
        b, g, v = input_exposure_s.shape
        angle = angles_rad[:, None, :].expand(b, g, v)
        radius = torch.nan_to_num(radii_mm, nan=self.radius_mean_mm)
        radius = ((radius - self.radius_mean_mm) / self.radius_std_mm)[:, None, :].expand(b, g, v)
        phase = gate_phase_rad[:, :, None].expand(b, g, v)
        log_exposure = standardized_log_exposure(
            input_exposure_s, self.log_exposure_mean, self.log_exposure_std
        )
        fraction = split_fraction.reshape(b, 1, 1).expand(b, g, v)
        continuous = torch.stack(
            (
                torch.sin(angle),
                torch.cos(angle),
                radius,
                torch.sin(phase),
                torch.cos(phase),
                log_exposure,
                fraction,
            ),
            dim=-1,
        )
        output = self.continuous_metadata(continuous)
        category = self._category_features(
            scanner_index,
            collimator_index,
            stress_rest_index,
            b,
            input_exposure_s.device,
        )
        if category is not None:
            output = output + category[:, None, None, :]
        return output

    def _mix_angular(self, x: Tensor, block: _TokenBlock, valid_frames: Tensor) -> Tensor:
        b, g, v, c, h, w = x.shape
        sequence = x.permute(0, 1, 4, 5, 2, 3).reshape(b * g * h * w, v, c)
        valid_views = valid_frames.any(dim=1)
        padding = (~valid_views)[:, None, :].expand(b, g * h * w, v).reshape(-1, v)
        valid = valid_views[:, None, :].expand(b, g * h * w, v).reshape(-1, v)
        sequence = block(sequence, padding, valid)
        return sequence.reshape(b, g, h, w, v, c).permute(0, 1, 4, 5, 2, 3)

    def _mix_temporal(self, x: Tensor, block: _TokenBlock, valid_frames: Tensor) -> Tensor:
        b, g, v, c, h, w = x.shape
        sequence = x.permute(0, 2, 4, 5, 1, 3).reshape(b * v * h * w, g, c)
        valid_gates = valid_frames.any(dim=2)
        padding = (~valid_gates)[:, None, :].expand(b, v * h * w, g).reshape(-1, g)
        valid = valid_gates[:, None, :].expand(b, v * h * w, g).reshape(-1, g)
        sequence = block(sequence, padding, valid)
        return sequence.reshape(b, v, h, w, g, c).permute(0, 4, 1, 5, 2, 3)

    def forward(
        self,
        counts_input: Tensor,
        input_exposure_s: Tensor,
        visibility: Tensor,
        valid_frames: Tensor,
        angles_rad: Tensor,
        radii_mm: Tensor,
        gate_phase_rad: Tensor,
        split_fraction: Tensor,
        scanner_index: Tensor | None = None,
        collimator_index: Tensor | None = None,
        stress_rest_index: Tensor | None = None,
    ) -> BackboneOutput:
        b, g, v, h, w = self._check_shapes(
            counts_input,
            input_exposure_s,
            visibility,
            valid_frames,
            angles_rad,
            radii_mm,
            gate_phase_rad,
            split_fraction,
        )
        counts_input = counts_input.to(dtype=torch.float32)
        exposure = input_exposure_s.to(dtype=counts_input.dtype)
        valid = valid_frames.to(dtype=counts_input.dtype)
        visible = visibility.to(dtype=counts_input.dtype)
        raw_standardized_log_exposure = (
            torch.log(exposure.clamp_min(_EPS)) - self.log_exposure_mean
        ) / self.log_exposure_std
        log_exposure = standardized_log_exposure(
            exposure, self.log_exposure_mean, self.log_exposure_std
        )
        exposure_clip_fraction = (raw_standardized_log_exposure.abs() > _EXPOSURE_CLIP).to(
            dtype=counts_input.dtype
        ).mean()
        frame_input = torch.cat(
            (
                counts_input,
                log_exposure[:, :, :, None, None, None].expand(b, g, v, 1, h, w),
                visible[:, :, :, None, :, :],
            ),
            dim=3,
        )
        frame_input = frame_input * valid[:, :, :, None, None, None]
        flat = frame_input.reshape(b * g * v, 3, h, w)
        skip64, skip32, skip16, bottleneck = self._encode_frames(flat)

        def unflatten(x: Tensor) -> Tensor:
            return x.reshape(b, g, v, x.shape[1], x.shape[2], x.shape[3])

        skip64_tokens = unflatten(skip64)
        skip32_tokens = unflatten(skip32)
        skip16_tokens = unflatten(skip16)
        bottleneck_tokens = unflatten(bottleneck)
        metadata = self._metadata(
            exposure,
            angles_rad,
            radii_mm,
            gate_phase_rad,
            split_fraction,
            scanner_index,
            collimator_index,
            stress_rest_index,
        )
        bottleneck_tokens = bottleneck_tokens + metadata[..., None, None]
        bottleneck_tokens = self._mix_angular(bottleneck_tokens, self.angular1, valid_frames)
        bottleneck_tokens = self._mix_temporal(bottleneck_tokens, self.temporal1, valid_frames)
        bottleneck_tokens = self._mix_angular(bottleneck_tokens, self.angular2, valid_frames)
        bottleneck_tokens = self._mix_temporal(bottleneck_tokens, self.temporal2, valid_frames)

        pooled = pool_global_bottleneck(bottleneck_tokens, valid_frames)
        global_representation = self.global_projection(self.global_norm(pooled))
        return BackboneOutput(
            bottleneck=bottleneck_tokens,
            skip16=skip16_tokens,
            skip32=skip32_tokens,
            skip64=skip64_tokens,
            global_representation=global_representation,
            exposure_clip_fraction=exposure_clip_fraction,
        )


class ProjectionRateDecoder(nn.Module):
    """U-Net-style per-frame decoder producing a positive count-rate map."""

    def __init__(
        self,
        *,
        conditioning_channels: int = 8,
        frame_chunk_size: int | None = None,
        return_rate_logits: bool = False,
    ) -> None:
        super().__init__()
        if conditioning_channels < 0:
            raise ValueError("conditioning_channels must be nonnegative")
        if frame_chunk_size is not None and frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive when supplied")
        self.frame_chunk_size = int(frame_chunk_size) if frame_chunk_size is not None else None
        self.return_rate_logits = bool(return_rate_logits)
        self.up16 = nn.Conv2d(96 + 64, 64, 3, padding=1)
        self.blocks16 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.up32 = nn.Conv2d(64 + 48, 48, 3, padding=1)
        self.blocks32 = nn.Sequential(ResidualBlock(48), ResidualBlock(48))
        self.up64 = nn.Conv2d(48 + 32, 32, 3, padding=1)
        self.blocks64 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.rate_head = nn.Conv2d(32, 1, 1)
        self.conditioning_head = (
            nn.Conv2d(32, conditioning_channels, 1) if conditioning_channels else None
        )

    def _decode_frame_chunk(
        self,
        bottleneck: Tensor,
        skip16: Tensor,
        skip32: Tensor,
        skip64: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        x = bottleneck
        x = F.interpolate(x, size=skip16.shape[-2:], mode="bilinear", align_corners=False)
        x = self.blocks16(self.up16(torch.cat((x, skip16), dim=1)))

        x = F.interpolate(x, size=skip32.shape[-2:], mode="bilinear", align_corners=False)
        x = self.blocks32(self.up32(torch.cat((x, skip32), dim=1)))

        x = F.interpolate(x, size=skip64.shape[-2:], mode="bilinear", align_corners=False)
        x = self.blocks64(self.up64(torch.cat((x, skip64), dim=1)))

        rate_logits = self.rate_head(x)
        rate = rate_logits if self.return_rate_logits else F.softplus(rate_logits) + _EPS
        conditioning = self.conditioning_head(x) if self.conditioning_head is not None else None
        return rate, conditioning

    def forward(self, features: BackboneOutput) -> tuple[Tensor, Tensor | None]:
        b, g, v, _, _, _ = features.bottleneck.shape
        total_frames = b * g * v
        bottleneck = features.bottleneck.reshape(total_frames, 96, *features.bottleneck.shape[-2:])
        skip16 = features.skip16.reshape(total_frames, 64, *features.skip16.shape[-2:])
        skip32 = features.skip32.reshape(total_frames, 48, *features.skip32.shape[-2:])
        skip64 = features.skip64.reshape(total_frames, 32, *features.skip64.shape[-2:])
        chunk_size = self.frame_chunk_size
        if chunk_size is None or chunk_size >= total_frames:
            rate, conditioning = self._decode_frame_chunk(bottleneck, skip16, skip32, skip64)
        else:
            outputs: list[tuple[Tensor, Tensor | None]] = []
            for start in range(0, total_frames, chunk_size):
                tensors = (
                    bottleneck[start : start + chunk_size],
                    skip16[start : start + chunk_size],
                    skip32[start : start + chunk_size],
                    skip64[start : start + chunk_size],
                )
                if self.training and torch.is_grad_enabled():
                    outputs.append(checkpoint(self._decode_frame_chunk, *tensors, use_reentrant=False))
                else:
                    outputs.append(self._decode_frame_chunk(*tensors))
            rate = torch.cat([item[0] for item in outputs], dim=0)
            conditioning = (
                torch.cat([item[1] for item in outputs if item[1] is not None], dim=0)
                if self.conditioning_head is not None
                else None
            )
        rate = rate.reshape(b, g, v, 1, *rate.shape[-2:]).squeeze(3)
        if conditioning is not None:
            conditioning = conditioning.reshape(
                b, g, v, conditioning.shape[1], *conditioning.shape[-2:]
            )
        return rate, conditioning


class ProjectionDenoiser(nn.Module):
    """Backbone plus rate decoder with the public inference contract."""

    def __init__(
        self,
        backbone: ProjectionBackbone | None = None,
        *,
        conditioning_channels: int = 8,
        frame_chunk_size: int | None = None,
        baseline_residual: bool = False,
        baseline_pseudocount: float = 0.5,
    ) -> None:
        super().__init__()
        if baseline_pseudocount <= 0:
            raise ValueError("baseline_pseudocount must be positive")
        self.backbone = backbone or ProjectionBackbone()
        self.baseline_residual = bool(baseline_residual)
        self.baseline_pseudocount = float(baseline_pseudocount)
        decoder_chunk_size = (
            frame_chunk_size
            if frame_chunk_size is not None
            else getattr(self.backbone, "frame_chunk_size", None)
        )
        self.decoder = ProjectionRateDecoder(
            conditioning_channels=conditioning_channels,
            frame_chunk_size=decoder_chunk_size,
            return_rate_logits=self.baseline_residual,
        )

    def forward(self, counts_raw: Tensor | None = None, **batch: Tensor) -> ProjectionOutput:
        features = self.backbone(**batch)
        rate_or_logits, conditioning = self.decoder(features)
        if self.baseline_residual:
            if counts_raw is None:
                raise ValueError("baseline_residual denoiser requires counts_raw")
            exposure = batch["input_exposure_s"].to(dtype=torch.float32)
            baseline_rate = (
                counts_raw.to(dtype=torch.float32) + self.baseline_pseudocount
            ) / exposure[..., None, None].clamp_min(_EPS)
            correction = torch.exp(rate_or_logits.clamp(min=-4.0, max=4.0))
            rate = baseline_rate * correction
        else:
            rate = rate_or_logits
        return ProjectionOutput(
            rate=rate,
            conditioning=conditioning,
            global_representation=features.global_representation,
            exposure_clip_fraction=features.exposure_clip_fraction,
        )
