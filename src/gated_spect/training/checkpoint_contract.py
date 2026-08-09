"""Scientific compatibility checks for persisted model checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

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


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    normalization_sha256: str | None = None,
    manifest_sha256: str | None = None,
    splits_sha256: str | None = None,
    pairs_sha256: str | None = None,
    allow_legacy_backbone_transfer: bool = False,
    migration_report: str | Path | None = None,
) -> None:
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
        report = json.loads(Path(migration_report).read_text(encoding="utf-8"))
        if not bool(report.get("legacy_backbone_compatibility")):
            raise IncompatibleArtifactError(
                "legacy backbone transfer is incompatible according to the "
                "v1-to-v2 exposure migration report; clean v2 pretraining is required"
            )
        return

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


__all__ = ["sha256_file", "validate_checkpoint_metadata"]
