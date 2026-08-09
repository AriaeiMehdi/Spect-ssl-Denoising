"""Scientific compatibility checks for persisted model checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..versioning import (
    EXPOSURE_SEMANTICS_VERSION,
    IncompatibleArtifactError,
    SCIENTIFIC_SEMANTICS_VERSION,
    require_artifact_versions,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(module: Any) -> str:
    """Hash a module state deterministically, including names and tensor bytes."""

    digest = hashlib.sha256()
    state = module.state_dict() if hasattr(module, "state_dict") else module
    if not isinstance(state, Mapping):
        raise TypeError("state must be a module or mapping of tensors")
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach"):
            raise TypeError(f"state entry {name!r} is not tensor-like")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


_MIGRATION_HASH_FIELDS = (
    "legacy_checkpoint_sha256",
    "old_dataset1_manifest_sha256",
    "new_dataset1_manifest_sha256",
    "old_dataset2_manifest_sha256",
    "new_dataset2_manifest_sha256",
    "old_splits_sha256",
    "new_splits_sha256",
    "old_normalization_sha256",
    "new_normalization_sha256",
)


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    normalization_sha256: str | None = None,
    manifest_sha256: str | None = None,
    splits_sha256: str | None = None,
    pairs_sha256: str | None = None,
    allow_legacy_backbone_transfer: bool = False,
    migration_report: str | Path | None = None,
    legacy_checkpoint_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Reject stale or physically ambiguous checkpoints before model loading.

    A version-1 checkpoint can only be used for an explicitly requested
    backbone transfer, and only when a migration report proves the exposure
    standardized inputs are compatible.  Full rate-model loading never uses
    this exception.
    """

    try:
        require_artifact_versions(metadata, artifact_name="checkpoint")
    except IncompatibleArtifactError:
        if not allow_legacy_backbone_transfer:
            raise
        if migration_report is None:
            raise IncompatibleArtifactError(
                "legacy checkpoint requires an explicit migration report; "
                "do not silently reinterpret version-1 rates"
            )
        report_path = Path(migration_report)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("migration_schema_version", 0)) != 2:
            raise IncompatibleArtifactError(
                "legacy migration report is missing migration_schema_version=2; "
                "regenerate the exact-artifact report"
            )
        missing = [
            field
            for field in _MIGRATION_HASH_FIELDS
            if not isinstance(report.get(field), str) or not report[field]
        ]
        if missing or not bool(report.get("artifact_hashes_complete", False)):
            raise IncompatibleArtifactError(
                "legacy migration report lacks complete exact-artifact hashes: "
                + ", ".join(missing or ["artifact_hashes_complete"])
            )
        if legacy_checkpoint_sha256 is None:
            raise IncompatibleArtifactError(
                "legacy transfer requires the SHA-256 of the exact checkpoint bytes"
            )
        if legacy_checkpoint_sha256 != report["legacy_checkpoint_sha256"]:
            raise IncompatibleArtifactError(
                "legacy checkpoint SHA-256 does not match the migration report"
            )
        if not bool(report.get("legacy_backbone_compatibility")):
            raise IncompatibleArtifactError(
                "legacy backbone transfer is incompatible according to the "
                "v1-to-v2 exposure migration report; clean v2 pretraining is required"
            )
        if normalization_sha256 is not None and report["new_normalization_sha256"] != normalization_sha256:
            raise IncompatibleArtifactError(
                "runtime normalization checksum does not match the migration report"
            )
        if splits_sha256 is not None and report["new_splits_sha256"] != splits_sha256:
            raise IncompatibleArtifactError(
                "runtime split checksum does not match the migration report"
            )
        if manifest_sha256 is not None:
            expected_manifest = report.get("new_dataset2_pairs_sha256") or report.get(
                "new_dataset2_manifest_sha256"
            )
            if expected_manifest != manifest_sha256:
                raise IncompatibleArtifactError(
                    "runtime manifest checksum does not match the migration report"
                )
        if pairs_sha256 is not None:
            expected_pairs = report.get("new_dataset2_pairs_sha256")
            if not isinstance(expected_pairs, str) or not expected_pairs:
                raise IncompatibleArtifactError(
                    "legacy migration report lacks the active Dataset2 pair-manifest checksum"
                )
            if expected_pairs != pairs_sha256:
                raise IncompatibleArtifactError(
                    "runtime pair-manifest checksum does not match the migration report"
                )
        return report

    if normalization_sha256 is not None:
        actual = metadata.get("normalization_sha256")
        if actual != normalization_sha256:
            raise IncompatibleArtifactError(
                "checkpoint normalization checksum does not match runtime normalization"
            )
    if manifest_sha256 is not None and metadata.get("manifest_sha256") != manifest_sha256:
        raise IncompatibleArtifactError("checkpoint manifest checksum does not match runtime manifest")
    if splits_sha256 is not None and metadata.get("splits_sha256") != splits_sha256:
        raise IncompatibleArtifactError("checkpoint split checksum does not match runtime splits")
    if pairs_sha256 is not None and metadata.get("pairs_sha256") != pairs_sha256:
        raise IncompatibleArtifactError("checkpoint pair-manifest checksum does not match runtime pairs")
    return None


__all__ = ["sha256_file", "state_dict_sha256", "validate_checkpoint_metadata"]
