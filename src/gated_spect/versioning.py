"""Scientific and artifact compatibility versions.

These versions are deliberately separate from the Python package version.  A
scientific-semantics change can invalidate an otherwise internally consistent
cache or checkpoint, so every persisted artifact records the relevant values.
"""

from __future__ import annotations

from typing import Any, Mapping

SCIENTIFIC_SEMANTICS_VERSION = 2
EXPOSURE_SEMANTICS_VERSION = 2
CANONICAL_CACHE_SCHEMA_VERSION = 2
AUGMENTATION_SEMANTICS_VERSION = 2


class IncompatibleArtifactError(RuntimeError):
    """Raised when a persisted artifact cannot be used under current semantics."""


def artifact_version_metadata() -> dict[str, int]:
    """Return the version fields shared by manifests and checkpoints."""

    return {
        "scientific_semantics_version": SCIENTIFIC_SEMANTICS_VERSION,
        "exposure_semantics_version": EXPOSURE_SEMANTICS_VERSION,
        "augmentation_semantics_version": AUGMENTATION_SEMANTICS_VERSION,
    }


def require_artifact_versions(
    metadata: Mapping[str, Any],
    *,
    artifact_name: str,
    require_augmentation: bool = False,
) -> None:
    """Reject missing or stale persisted artifact version fields."""

    expected = artifact_version_metadata()
    if not require_augmentation:
        expected.pop("augmentation_semantics_version")
    missing = [key for key in expected if key not in metadata]
    incompatible = [
        f"{key}={metadata.get(key)!r} (expected {value})"
        for key, value in expected.items()
        if key in metadata and int(metadata[key]) != value
    ]
    if missing or incompatible:
        detail = ", ".join(missing + incompatible)
        raise IncompatibleArtifactError(
            f"{artifact_name} is incompatible with scientific semantics v2: {detail}; "
            "preserve it as a legacy artifact and regenerate a version-2 artifact"
        )
