"""Scientific and artifact compatibility versions.

These versions are deliberately separate from the Python package version.  A
scientific-semantics change can invalidate an otherwise internally consistent
cache or checkpoint, so every persisted artifact records the relevant values.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

SCIENTIFIC_SEMANTICS_VERSION = 2
EXPOSURE_SEMANTICS_VERSION = 2
CANONICAL_CACHE_SCHEMA_VERSION = 3
AUGMENTATION_SEMANTICS_VERSION = 2


class IncompatibleArtifactError(RuntimeError):
    """Raised when a persisted artifact cannot be used under current semantics."""


class ArtifactCollisionError(RuntimeError):
    """Raised when a create-only artifact path contains different bytes."""


def sha256_file(path: str | Path) -> str:
    """Return the exact SHA-256 digest of a persisted artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_create_only(path: str | Path, data: bytes) -> Path:
    """Atomically create an artifact, reusing only identical existing bytes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mktemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        temporary.write_bytes(data)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != data:
                raise ArtifactCollisionError(
                    f"artifact already exists with different bytes: {destination}"
                )
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def write_text_create_only(path: str | Path, text: str) -> Path:
    """Create UTF-8 text without overwriting a different artifact."""

    return write_bytes_create_only(path, text.encode("utf-8"))


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


def validate_normalization_binding(
    normalization_path: str | Path,
    *,
    manifest_path: str | Path,
    splits_path: str | Path,
) -> dict[str, str]:
    """Require normalization statistics to bind to the exact runtime artifacts.

    A version match alone is insufficient: two independently generated v2
    manifests or split files can still describe different populations. The
    normalization writer records exact byte hashes, so training must compare
    those hashes before constructing a dataset.
    """

    path = Path(normalization_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_artifact_versions(payload, artifact_name=f"normalization {path}")
    expected_manifest = payload.get("manifest_sha256")
    expected_splits = payload.get("splits_sha256")
    if not expected_manifest or not expected_splits:
        raise IncompatibleArtifactError(
            f"normalization {path} lacks manifest_sha256/splits_sha256 bindings; "
            "regenerate the normalization artifact"
        )
    actual_manifest = sha256_file(manifest_path)
    actual_splits = sha256_file(splits_path)
    mismatches: list[str] = []
    if expected_manifest != actual_manifest:
        mismatches.append(
            f"manifest_sha256={expected_manifest!r} (runtime {actual_manifest!r})"
        )
    if expected_splits != actual_splits:
        mismatches.append(f"splits_sha256={expected_splits!r} (runtime {actual_splits!r})")
    if mismatches:
        raise IncompatibleArtifactError(
            f"normalization {path} is not bound to the supplied training artifacts: "
            + "; ".join(mismatches)
        )
    return {
        "normalization_sha256": sha256_file(path),
        "manifest_sha256": actual_manifest,
        "splits_sha256": actual_splits,
    }


def validate_manifest_source_bindings(
    root: str | Path,
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    eligible_only: bool = True,
) -> int:
    """Verify that manifest rows still point to the exact source DICOM bytes.

    A manifest checksum binds the row metadata, but it cannot detect a source
    image being replaced in-place.  Training therefore verifies every source
    selected by the locked split before constructing a dataset.  Legacy rows
    without ``source_sha256`` fail closed instead of being silently trusted.
    """

    root_path = Path(root).resolve()
    checked = 0
    for row in rows:
        if eligible_only and not bool(row.get("training_eligible", False)):
            continue
        relative = str(row.get("relative_path", ""))
        expected = row.get("source_sha256")
        if not relative or not isinstance(expected, str) or not expected:
            raise IncompatibleArtifactError(
                "selected manifest row lacks source_sha256; regenerate the manifest "
                "before training"
            )
        source = (root_path / relative).resolve()
        try:
            source.relative_to(root_path)
        except ValueError as exc:
            raise IncompatibleArtifactError(
                f"manifest source path escapes dataset root: {relative!r}"
            ) from exc
        if not source.is_file():
            raise IncompatibleArtifactError(f"manifest source file is missing: {source}")
        actual = sha256_file(source)
        if actual != expected:
            raise IncompatibleArtifactError(
                f"source checksum mismatch for {source}: manifest={expected!r} runtime={actual!r}"
            )
        checked += 1
    return checked


def validate_pair_source_bindings(
    root: str | Path,
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    accepted_only: bool = True,
) -> int:
    """Verify low/high DICOM provenance recorded in accepted pair rows."""

    root_path = Path(root).resolve()
    checked = 0
    for row in rows:
        if accepted_only and not bool(row.get("accepted", False)):
            continue
        for prefix in ("low", "high"):
            relative = str(row.get(f"{prefix}_relative_path", ""))
            expected = row.get(f"{prefix}_source_sha256")
            if not relative or not isinstance(expected, str) or not expected:
                raise IncompatibleArtifactError(
                    f"accepted pair row lacks {prefix}_source_sha256; regenerate the pair manifest"
                )
            source = (root_path / relative).resolve()
            try:
                source.relative_to(root_path)
            except ValueError as exc:
                raise IncompatibleArtifactError(
                    f"pair source path escapes dataset root: {relative!r}"
                ) from exc
            if not source.is_file():
                raise IncompatibleArtifactError(f"pair source file is missing: {source}")
            actual = sha256_file(source)
            if actual != expected:
                raise IncompatibleArtifactError(
                    f"pair source checksum mismatch for {source}: "
                    f"manifest={expected!r} runtime={actual!r}"
                )
            checked += 1
    return checked


def validate_resume_contract(
    stored: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    """Reject scientifically meaningful training-setting changes on resume."""

    mismatches = [
        key for key, value in runtime.items() if stored.get(key) != value
    ]
    if mismatches:
        raise IncompatibleArtifactError(
            "resume training contract differs for: " + ", ".join(sorted(mismatches))
        )
