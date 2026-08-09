"""Small command-line entry points for M0/M1 data work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pydicom
import scipy
import torch

from .data.audit import audit_root
from .data.dicom_reader import read_nm_dicom
from .data.inventory import inventory_headers, write_jsonl
from .data.manifest import (
    build_acquisition_manifest,
    read_jsonl,
    write_locked_splits,
    write_manifest,
)
from .data.pair_manifest import audit_duration_pairs, write_pair_manifest
from .data.splits import SplitConfig
from .types import ReaderConfig
from .versioning import artifact_version_metadata, require_artifact_versions
from .training.checkpoint_contract import (
    sha256_file,
    state_dict_sha256,
    validate_checkpoint_metadata,
)


def environment_report() -> dict[str, Any]:
    """Return the runtime facts that are saved beside each experiment."""

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pydicom": pydicom.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _write_or_print(payload: Any, output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _require_versioned_rows(rows: list[Mapping[str, Any]], artifact_name: str) -> None:
    """Reject legacy manifests before they can enter a version-2 workflow."""

    if not rows:
        raise SystemExit(f"{artifact_name} is empty")
    for row in rows:
        require_artifact_versions(row, artifact_name=artifact_name)


def _require_versioned_splits(payload: Mapping[str, Any]) -> None:
    require_artifact_versions(payload, artifact_name="locked split manifest", require_augmentation=True)


def _reader_config(args: argparse.Namespace) -> ReaderConfig:
    key_text = os.environ.get(args.hash_key_env)
    if not key_text and not args.allow_unhashed:
        raise SystemExit(
            f"{args.hash_key_env} is not set; use a secret HMAC key or "
            "--allow-unhashed only for synthetic fixtures"
        )
    spacing = None
    if args.spacing_row is not None and args.spacing_col is not None:
        spacing = (args.spacing_row, args.spacing_col)
    return ReaderConfig(
        hash_key=key_text.encode("utf-8") if key_text else None,
        allow_unhashed_for_test=args.allow_unhashed,
        default_start_angle_deg=args.start_angle,
        default_angular_step_deg=args.angular_step,
        default_rotation_direction=args.rotation_direction,
        default_view_dwell_s=args.view_dwell_s,
        default_detector_spacing_mm=spacing,
        default_radius_mm=args.radius_mm,
        institution_id=args.institution_id,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gated-spect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("environment-report")
    env_parser.add_argument("--output")

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--root", action="append", required=True)
    inventory_parser.add_argument("--dataset-id", choices=("dataset1", "dataset2"), required=True)
    inventory_parser.add_argument("--output", required=True)
    inventory_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    inventory_parser.add_argument("--institution-id", default="unknown")
    inventory_parser.add_argument("--allow-unhashed", action="store_true")

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--root", required=True)
    audit_parser.add_argument("--dataset-id", choices=("dataset1", "dataset2"), required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    audit_parser.add_argument("--institution-id", default="unknown")
    audit_parser.add_argument("--allow-unhashed", action="store_true")

    pair_parser = subparsers.add_parser("pair-audit")
    pair_parser.add_argument("--root", required=True)
    pair_parser.add_argument("--output", required=True)
    pair_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    pair_parser.add_argument("--institution-id", default="unknown")
    pair_parser.add_argument("--radius-tolerance-mm", type=float, default=2.0)
    pair_parser.add_argument("--allow-unhashed", action="store_true")

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--root", required=True)
    manifest_parser.add_argument("--dataset-id", choices=("dataset1", "dataset2"), required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    manifest_parser.add_argument("--institution-id", default="unknown")
    manifest_parser.add_argument("--allow-unhashed", action="store_true")

    splits_parser = subparsers.add_parser("lock-splits")
    splits_parser.add_argument("--dataset1-manifest", required=True)
    splits_parser.add_argument("--dataset2-manifest", required=True)
    splits_parser.add_argument("--pair-manifest", required=True)
    splits_parser.add_argument("--output", required=True)
    splits_parser.add_argument("--seed", type=int, default=20260806)

    normalization_parser = subparsers.add_parser("normalization")
    normalization_parser.add_argument("--root", required=True)
    normalization_parser.add_argument("--manifest", required=True)
    normalization_parser.add_argument("--splits", required=True)
    normalization_parser.add_argument("--output", required=True)
    normalization_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    normalization_parser.add_argument("--institution-id", default="unknown")
    normalization_parser.add_argument("--allow-unhashed", action="store_true")

    cache_parser = subparsers.add_parser(
        "cache",
        help="build a versioned canonical NumPy cache from a parsed acquisition manifest",
    )
    cache_parser.add_argument("--root", required=True)
    cache_parser.add_argument("--manifest", required=True)
    cache_parser.add_argument("--output-dir", required=True)
    cache_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    cache_parser.add_argument("--institution-id", default="unknown")
    cache_parser.add_argument("--allow-unhashed", action="store_true")

    migration_parser = subparsers.add_parser(
        "exposure-migration-audit",
        help="compare legacy v1 and corrected v2 exposure semantics from raw DICOM",
    )
    migration_parser.add_argument("--dataset1-root", required=True)
    migration_parser.add_argument("--dataset2-root", required=True)
    migration_parser.add_argument("--dataset1-manifest", required=True)
    migration_parser.add_argument("--dataset2-manifest", required=True)
    migration_parser.add_argument("--new-dataset1-manifest", required=True)
    migration_parser.add_argument("--new-dataset2-manifest", required=True)
    migration_parser.add_argument(
        "--new-dataset2-pairs",
        help=(
            "optional exact Dataset2 pair-manifest bytes; required when the "
            "report will authorize legacy denoising transfer"
        ),
    )
    migration_parser.add_argument("--old-splits", required=True)
    migration_parser.add_argument("--new-splits", required=True)
    migration_parser.add_argument(
        "--legacy-checkpoint",
        required=True,
        help="exact legacy checkpoint bytes evaluated by this migration report",
    )
    migration_parser.add_argument("--old-normalization", required=True)
    migration_parser.add_argument("--new-normalization", required=True)
    migration_parser.add_argument("--output-json", required=True)
    migration_parser.add_argument("--output-csv", required=True)
    migration_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    migration_parser.add_argument("--institution-id", default="unknown")
    migration_parser.add_argument("--allow-unhashed", action="store_true")

    pretrain_parser = subparsers.add_parser("pretrain")
    pretrain_parser.add_argument("--root", required=True)
    pretrain_parser.add_argument("--manifest", required=True)
    pretrain_parser.add_argument("--splits", required=True)
    pretrain_parser.add_argument("--normalization", required=True)
    pretrain_parser.add_argument("--checkpoint", required=True)
    pretrain_parser.add_argument("--metrics-output")
    pretrain_parser.add_argument(
        "--log-output",
        help="JSONL event log; defaults to a sibling .events.jsonl file beside metrics-output",
    )
    pretrain_parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="print and persist a batch event every N batches (0 disables batch events)",
    )
    pretrain_parser.add_argument("--epochs", type=int, default=250)
    pretrain_parser.add_argument("--batch-size", type=int, default=1)
    pretrain_parser.add_argument(
        "--frame-chunk-size",
        type=int,
        default=None,
        help="checkpoint the frame encoder in chunks of this many projections to reduce GPU memory",
    )
    pretrain_parser.add_argument("--accumulation-steps", type=int, default=8)
    pretrain_parser.add_argument("--num-workers", type=int, default=0)
    pretrain_parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    pretrain_parser.add_argument("--weight-decay", type=float, default=0.05)
    pretrain_parser.add_argument("--device", default=None)
    pretrain_parser.add_argument("--max-batches", type=int, default=None)
    pretrain_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    pretrain_parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=65536.0,
        help="initial AMP loss scale for pretraining",
    )
    pretrain_parser.add_argument("--two-view", action=argparse.BooleanOptionalAction, default=True)
    pretrain_parser.add_argument("--representation-queue-capacity", type=int, default=2048)
    pretrain_parser.add_argument("--covariance-loss-weight", type=float, default=0.0)
    pretrain_parser.add_argument(
        "--covariance-min-samples",
        type=int,
        default=512,
        help="minimum queue-plus-current samples before covariance regularization is active",
    )
    pretrain_parser.add_argument("--seed", type=int, default=20260806)
    pretrain_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    pretrain_parser.add_argument("--institution-id", default="unknown")
    pretrain_parser.add_argument("--allow-unhashed", action="store_true")

    denoise_parser = subparsers.add_parser("denoise")
    denoise_parser.add_argument("--root", required=True)
    denoise_parser.add_argument("--pairs", required=True)
    denoise_parser.add_argument("--splits", required=True)
    denoise_parser.add_argument("--normalization", required=True)
    denoise_parser.add_argument("--pretrained-checkpoint", required=True)
    denoise_parser.add_argument(
        "--scratch",
        action="store_true",
        help=(
            "initialize the same architecture randomly while retaining the "
            "pretrained checkpoint metadata vocabularies"
        ),
    )
    denoise_parser.add_argument(
        "--legacy-backbone-transfer",
        action="store_true",
        help="allow a version-1 backbone only when a compatible migration report is supplied",
    )
    denoise_parser.add_argument("--exposure-migration-report")
    denoise_parser.add_argument("--checkpoint", required=True)
    denoise_parser.add_argument("--metrics-output")
    denoise_parser.add_argument("--log-output")
    denoise_parser.add_argument(
        "--patient-metrics-output",
        help="append validation pair metrics for patient-level experiment analysis",
    )
    denoise_parser.add_argument("--log-interval", type=int, default=10)
    denoise_parser.add_argument("--epochs", type=int, default=80)
    denoise_parser.add_argument("--freeze-epochs", type=int, default=8)
    denoise_parser.add_argument(
        "--comparison-protocol",
        choices=("full_finetune", "partial_freeze_transfer"),
        default=None,
        help="protocol label recorded in Dataset2 checkpoints; inferred from freeze_epochs when omitted",
    )
    denoise_parser.add_argument("--batch-size", type=int, default=1)
    denoise_parser.add_argument("--accumulation-steps", type=int, default=8)
    denoise_parser.add_argument("--num-workers", type=int, default=0)
    denoise_parser.add_argument("--backbone-learning-rate", type=float, default=1.0e-5)
    denoise_parser.add_argument("--decoder-learning-rate", type=float, default=1.0e-4)
    denoise_parser.add_argument("--weight-decay", type=float, default=0.01)
    denoise_parser.add_argument("--warmup-epochs", type=int, default=2)
    denoise_parser.add_argument("--early-stop-patience", type=int, default=15)
    denoise_parser.add_argument("--device", default=None)
    denoise_parser.add_argument("--max-batches", type=int, default=None)
    denoise_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    denoise_parser.add_argument(
        "--baseline-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="learn a bounded correction around the exposure-scaled low-count rate",
    )
    denoise_parser.add_argument(
        "--baseline-pseudocount",
        type=float,
        default=0.5,
        help="positive baseline pseudocount alpha used by the residual-rate target",
    )
    denoise_parser.add_argument(
        "--low-signal-threshold",
        type=float,
        default=5.0,
        help="projection-domain target-count threshold for low-signal metrics",
    )
    denoise_parser.add_argument(
        "--calibration-ranges",
        default="0,1,2,3-5,6-10,11-20,>20",
        help="comma-separated raw-count calibration bins",
    )
    denoise_parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=1.0,
        help="initial AMP loss scale for Dataset 2 denoising",
    )
    denoise_parser.add_argument("--frame-chunk-size", type=int, default=None)
    denoise_parser.add_argument("--seed", type=int, default=20260806)
    denoise_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    denoise_parser.add_argument("--institution-id", default="unknown")
    denoise_parser.add_argument("--allow-unhashed", action="store_true")

    visualize_parser = subparsers.add_parser(
        "visualize-denoise",
        help="save a common-scale 10s/denoised/30s detector-slice comparison",
    )
    visualize_parser.add_argument("--root", required=True)
    visualize_parser.add_argument("--pairs", required=True)
    visualize_parser.add_argument("--normalization", required=True)
    visualize_parser.add_argument("--pretrained-checkpoint", required=True)
    visualize_parser.add_argument("--checkpoint", required=True)
    visualize_parser.add_argument("--exposure-migration-report")
    visualize_parser.add_argument("--legacy-backbone-transfer", action="store_true")
    visualize_parser.add_argument("--output", required=True)
    visualize_parser.add_argument("--pair-index", type=int, default=0)
    visualize_parser.add_argument("--pair-id")
    visualize_parser.add_argument("--patient-key")
    visualize_parser.add_argument("--gate-index", type=int, default=0)
    visualize_parser.add_argument("--view-index", type=int, default=0)
    visualize_parser.add_argument("--scale", choices=("log1p", "linear"), default="log1p")
    visualize_parser.add_argument("--percentile", type=float, default=99.5)
    visualize_parser.add_argument("--dpi", type=int, default=150)
    visualize_parser.add_argument("--device", default=None)
    visualize_parser.add_argument("--frame-chunk-size", type=int, default=None)
    visualize_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    visualize_parser.add_argument(
        "--baseline-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="interpret the checkpoint as a baseline-residual denoiser",
    )
    visualize_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    visualize_parser.add_argument("--institution-id", default="unknown")
    visualize_parser.add_argument("--allow-unhashed", action="store_true")

    export_parser = subparsers.add_parser(
        "export-denoise-dicom",
        help="export validation projection stacks as derived multi-frame NM DICOM",
    )
    export_parser.add_argument("--root", required=True)
    export_parser.add_argument("--pairs", required=True)
    export_parser.add_argument("--splits", required=True)
    export_parser.add_argument("--normalization", required=True)
    export_parser.add_argument("--pretrained-checkpoint", required=True)
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--exposure-migration-report")
    export_parser.add_argument("--legacy-backbone-transfer", action="store_true")
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="number of validation pairs to export in manifest order",
    )
    export_parser.add_argument("--device", default=None)
    export_parser.add_argument("--frame-chunk-size", type=int, default=None)
    export_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    export_parser.add_argument(
        "--baseline-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="interpret the checkpoint as a baseline-residual denoiser",
    )
    export_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    export_parser.add_argument("--institution-id", default="unknown")
    export_parser.add_argument("--allow-unhashed", action="store_true")

    validate_parser = subparsers.add_parser("validate-dicom")
    validate_parser.add_argument("path")
    validate_parser.add_argument("--output")
    validate_parser.add_argument("--hash-key-env", default="SPECT_HASH_KEY")
    validate_parser.add_argument("--institution-id", default="unknown")
    validate_parser.add_argument("--allow-unhashed", action="store_true")
    validate_parser.add_argument("--start-angle", type=float)
    validate_parser.add_argument("--angular-step", type=float)
    validate_parser.add_argument("--rotation-direction")
    validate_parser.add_argument("--view-dwell-s", type=float)
    validate_parser.add_argument("--spacing-row", type=float)
    validate_parser.add_argument("--spacing-col", type=float)
    validate_parser.add_argument("--radius-mm", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "environment-report":
        _write_or_print(environment_report(), args.output)
        return 0
    if args.command == "inventory":
        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        rows = inventory_headers(args.root, dataset_id=args.dataset_id, config=config)
        write_jsonl(rows, args.output)
        print(json.dumps({"rows": len(rows), "output": str(Path(args.output).resolve())}))
        return 0
    if args.command == "audit":
        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        report = audit_root(args.root, dataset_id=args.dataset_id, config=config)
        _write_or_print(report, args.output)
        print(json.dumps({"output": str(Path(args.output).resolve())}))
        return 0
    if args.command == "pair-audit":
        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        rows = audit_duration_pairs(
            args.root,
            config=config,
            radius_tolerance_mm=args.radius_tolerance_mm,
        )
        write_pair_manifest(rows, args.output)
        print(json.dumps({"rows": len(rows), "output": str(Path(args.output).resolve())}))
        return 0
    if args.command == "manifest":
        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        rows = build_acquisition_manifest(args.root, dataset_id=args.dataset_id, config=config)
        write_manifest(rows, args.output)
        print(json.dumps({"rows": len(rows), "output": str(Path(args.output).resolve())}))
        return 0
    if args.command == "lock-splits":
        config = SplitConfig(seed=args.seed)
        dataset1_rows = read_jsonl(args.dataset1_manifest)
        dataset2_rows = read_jsonl(args.dataset2_manifest)
        pair_rows = read_jsonl(args.pair_manifest)
        _require_versioned_rows(dataset1_rows, "Dataset1 acquisition manifest")
        _require_versioned_rows(dataset2_rows, "Dataset2 acquisition manifest")
        _require_versioned_rows(pair_rows, "Dataset2 pair manifest")
        output = write_locked_splits(
            dataset1_rows,
            dataset2_rows,
            pair_rows,
            output=args.output,
            config=config,
        )
        print(json.dumps({"output": str(output.resolve())}))
        return 0
    if args.command == "normalization":
        from .data.torch_dataset import compute_normalization_stats

        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        manifest_rows = read_jsonl(args.manifest)
        _require_versioned_rows(manifest_rows, "acquisition manifest")
        split_payload = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        _require_versioned_splits(split_payload)
        stats, counts = compute_normalization_stats(
            args.root,
            args.manifest,
            config=config,
            patient_splits=split_payload.get("dataset1", {}),
        )

        def checksum(path: str) -> str:
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        payload = {
            **artifact_version_metadata(),
            "dataset_id": "dataset1",
            "split": "pretrain",
            "stats": stats.to_dict(),
            "element_counts": counts,
            "manifest_sha256": checksum(args.manifest),
            "splits_sha256": checksum(args.splits),
        }
        _write_or_print(payload, args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()), "files": counts["files"]}))
        return 0
    if args.command == "cache":
        from .data.cache import write_sample_cache

        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        rows = read_jsonl(args.manifest)
        _require_versioned_rows(rows, "acquisition manifest")
        root = Path(args.root).resolve()
        output_dir = Path(args.output_dir).resolve()
        written = 0
        skipped = 0
        for row in rows:
            if row.get("parse_status") != "parsed" or not row.get("training_eligible", False):
                skipped += 1
                continue
            sample = read_nm_dicom(root / str(row["relative_path"]), config)
            write_sample_cache(sample, output_dir)
            written += 1
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "written": written,
                    "skipped": skipped,
                    "scientific_semantics_version": 2,
                }
            )
        )
        return 0
    if args.command == "exposure-migration-audit":
        from .data.migration import build_exposure_migration_report, write_exposure_migration_report

        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        old_dataset1 = read_jsonl(args.dataset1_manifest)
        old_dataset2 = read_jsonl(args.dataset2_manifest)
        new_manifests: dict[str, list[dict[str, Any]]] = {}
        new_manifests["dataset1"] = read_jsonl(args.new_dataset1_manifest)
        _require_versioned_rows(new_manifests["dataset1"], "new Dataset1 acquisition manifest")
        new_manifests["dataset2"] = read_jsonl(args.new_dataset2_manifest)
        _require_versioned_rows(new_manifests["dataset2"], "new Dataset2 acquisition manifest")
        new_splits_payload = json.loads(Path(args.new_splits).read_text(encoding="utf-8"))
        _require_versioned_splits(new_splits_payload)
        old_splits_payload = json.loads(Path(args.old_splits).read_text(encoding="utf-8"))
        report = build_exposure_migration_report(
            (
                ("dataset1", args.dataset1_root, old_dataset1),
                ("dataset2", args.dataset2_root, old_dataset2),
            ),
            config=config,
            old_normalization=args.old_normalization,
            new_normalization=args.new_normalization,
            new_manifests=new_manifests,
            old_splits=old_splits_payload,
            new_splits=new_splits_payload,
            legacy_checkpoint=args.legacy_checkpoint,
            old_dataset1_manifest=args.dataset1_manifest,
            new_dataset1_manifest=args.new_dataset1_manifest,
            old_dataset2_manifest=args.dataset2_manifest,
            new_dataset2_manifest=args.new_dataset2_manifest,
            new_dataset2_pairs_manifest=args.new_dataset2_pairs,
            old_splits_path=args.old_splits,
            new_splits_path=args.new_splits,
        )
        write_exposure_migration_report(report, args.output_json, args.output_csv)
        print(
            json.dumps(
                {
                    "output_json": str(Path(args.output_json).resolve()),
                    "output_csv": str(Path(args.output_csv).resolve()),
                    "legacy_backbone_compatibility": report["legacy_backbone_compatibility"],
                    "mean_abs_delta_z": report["mean_abs_delta_z"],
                    "p99_abs_delta_z": report["p99_abs_delta_z"],
                }
            )
        )
        return 0
    if args.command == "pretrain":
        from torch.utils.data import DataLoader

        from .data.torch_dataset import (
            AcquisitionDataset,
            NormalizationStats,
            PretrainingDataset,
            TwoViewPretrainingDataset,
            collate_acquisitions,
            collate_two_views,
        )
        from .models import ProjectionBackbone, ProjectionDenoiser
        from .losses import EMATeacher, GlobalConsistencyHead
        from .training import (
            evaluate_epoch,
            evaluate_two_view_epoch,
            save_checkpoint,
            parameter_summary,
            train_epoch,
            train_two_view_epoch,
        )

        if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
            raise SystemExit("--epochs, --batch-size, and --accumulation-steps must be positive")
        if args.frame_chunk_size is not None and args.frame_chunk_size < 1:
            raise SystemExit("--frame-chunk-size must be positive when supplied")
        if args.num_workers < 0:
            raise SystemExit("--num-workers cannot be negative")
        if args.max_batches is not None and args.max_batches < 1:
            raise SystemExit("--max-batches must be positive when supplied")
        if args.log_interval < 0:
            raise SystemExit("--log-interval cannot be negative")
        if args.amp_init_scale <= 0:
            raise SystemExit("--amp-init-scale must be positive")
        if args.representation_queue_capacity < 1:
            raise SystemExit("--representation-queue-capacity must be positive")
        if args.covariance_loss_weight < 0:
            raise SystemExit("--covariance-loss-weight cannot be negative")
        if args.covariance_min_samples < 2:
            raise SystemExit("--covariance-min-samples must be at least 2")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        normalization = NormalizationStats.from_json(args.normalization)
        split_payload = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        _require_versioned_splits(split_payload)
        all_records = read_jsonl(args.manifest)
        _require_versioned_rows(all_records, "Dataset1 acquisition manifest")
        split_map = split_payload.get("dataset1", {})
        train_records = [
            row
            for row in all_records
            if bool(row.get("training_eligible"))
            and split_map.get(str(row.get("patient_key"))) == "pretrain"
        ]
        validation_records = [
            row
            for row in all_records
            if bool(row.get("training_eligible"))
            and split_map.get(str(row.get("patient_key"))) == "validation"
        ]
        if not train_records or not validation_records:
            raise SystemExit("locked splits produced an empty pretraining or validation set")
        train_acquisitions = AcquisitionDataset(
            args.root,
            train_records,
            config=config,
            normalization=normalization,
        )
        validation_acquisitions = AcquisitionDataset(
            args.root,
            validation_records,
            config=config,
            normalization=normalization,
        )
        validation_acquisitions.scanner_vocab = train_acquisitions.scanner_vocab
        validation_acquisitions.collimator_vocab = train_acquisitions.collimator_vocab
        validation_acquisitions.stress_vocab = train_acquisitions.stress_vocab
        dataset_class = TwoViewPretrainingDataset if args.two_view else PretrainingDataset
        train_data = dataset_class(train_acquisitions, global_seed=args.seed)
        validation_data = dataset_class(validation_acquisitions, global_seed=args.seed)
        collate_function = collate_two_views if args.two_view else collate_acquisitions
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            # The dataset carries epoch-dependent augmentation state.  Worker
            # processes must be recreated so every epoch receives the new
            # deterministic seed contract.
            persistent_workers=False,
            **({"prefetch_factor": 2} if args.num_workers > 0 else {}),
            collate_fn=collate_function,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            **({"prefetch_factor": 2} if args.num_workers > 0 else {}),
            collate_fn=collate_function,
        )
        backbone = ProjectionBackbone(
            log_exposure_mean=normalization.log_exposure_mean,
            log_exposure_std=normalization.log_exposure_std,
            radius_mean_mm=normalization.radius_mean_mm,
            radius_std_mm=normalization.radius_std_mm,
            scanner_vocab_size=len(train_acquisitions.scanner_vocab) + 1,
            collimator_vocab_size=len(train_acquisitions.collimator_vocab) + 1,
            stress_rest_vocab_size=len(train_acquisitions.stress_vocab) + 1,
            frame_chunk_size=args.frame_chunk_size,
        )
        model = ProjectionDenoiser(backbone=backbone).to(device)
        ema_teacher = EMATeacher(model.backbone).to(device) if args.two_view else None
        consistency_head = (
            GlobalConsistencyHead(
                queue_capacity=args.representation_queue_capacity,
                covariance_loss_weight=args.covariance_loss_weight,
                covariance_min_samples=args.covariance_min_samples,
            ).to(device)
            if args.two_view
            else None
        )
        trainable_parameters = list(model.parameters())
        if consistency_head is not None:
            trainable_parameters.extend(consistency_head.parameters())
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(args.amp and device.type == "cuda"),
            init_scale=args.amp_init_scale,
        )
        metrics_path = Path(args.metrics_output) if args.metrics_output else Path(args.checkpoint).with_name("metrics.jsonl")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = (
            Path(args.log_output)
            if args.log_output
            else metrics_path.with_name(f"{metrics_path.stem}.events.jsonl")
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.resolve() == metrics_path.resolve():
            raise SystemExit("--log-output and --metrics-output must be different files")
        run_started = time.perf_counter()

        def terminal_event_text(payload: Mapping[str, Any]) -> str:
            event = str(payload.get("event", "event"))
            if event == "run_start":
                return (
                    f"[run] model={payload['model']} "
                    f"trainable={int(payload['trainable_parameter_count']):,} "
                    f"total={int(payload['total_parameter_count']):,} "
                    f"train_acquisitions={payload['train_acquisitions']} "
                    f"validation_acquisitions={payload['validation_acquisitions']} "
                    f"device={payload['device']} amp={payload['amp']} "
                    f"two_view={payload['two_view']} "
                    f"effective_batch={payload['effective_batch_size']}"
                )
            if event == "batch":
                total = payload.get("batches_total")
                progress = f"{payload['batch']}/{total}" if total is not None else str(payload["batch"])
                suffix = (
                    f" latent={float(payload['latent']):.4f}"
                    f" gate={float(payload['gate']):.4f}"
                    if "latent" in payload
                    else ""
                )
                variance = (
                    f" variance={float(payload['variance']):.4f}"
                    if "variance" in payload
                    else ""
                )
                loss_parts = "".join(
                    (
                        f" masked_loss={float(payload['masked']):.6f}"
                        if payload.get("masked") is not None
                        else "",
                        f" visible_loss={float(payload['visible']):.6f}"
                        if payload.get("visible") is not None
                        else "",
                    )
                )
                health = "".join(
                    (
                        f" mask={float(payload['masked_fraction']):.3f}"
                        if "masked_fraction" in payload
                        else "",
                        f" rate={float(payload['rate_mean']):.4g}+/-{float(payload['rate_std']):.3g}"
                        if "rate_mean" in payload and "rate_std" in payload
                        else "",
                        f" rep_std={float(payload['representation_std']):.4g}"
                        if "representation_std" in payload
                        else "",
                        f" rank={float(payload['representation_effective_rank']):.1f}"
                        if "representation_effective_rank" in payload
                        else "",
                        f" pr={float(payload['representation_participation_rank']):.1f}"
                        if "representation_participation_rank" in payload
                        else "",
                        f" corr95={float(payload['p95_absolute_offdiagonal_correlation']):.3f}"
                        if "p95_absolute_offdiagonal_correlation" in payload
                        else "",
                        f" expclip={float(payload['exposure_conditioning_clip_fraction']):.3f}"
                        if "exposure_conditioning_clip_fraction" in payload
                        else "",
                        f" grad={float(payload['gradient_norm']):.3g}"
                        if payload.get("gradient_norm") is not None
                        else "",
                    )
                )
                return (
                    f"[{payload['phase']}] epoch={int(payload['epoch']) + 1}/{args.epochs} "
                    f"batch={progress} loss={float(payload['loss']):.6f} "
                    f"deviance={float(payload['deviance']):.6f}{loss_parts}{suffix}{variance}{health} "
                    f"rate={float(payload['batches_per_second']):.3f} batches/s "
                    f"elapsed={float(payload['elapsed_seconds']):.1f}s"
                )
            if event == "phase_end":
                health = "".join(
                    (
                        f" rep_std={float(payload['representation_std']):.4g}"
                        if "representation_std" in payload
                        else "",
                        f" rank={float(payload['representation_effective_rank']):.1f}"
                        if "representation_effective_rank" in payload
                        else "",
                        f" grad={float(payload['gradient_norm_mean']):.3g}"
                        if "gradient_norm_mean" in payload
                        else "",
                    )
                )
                return (
                    f"[{payload['phase']}] epoch={int(payload['epoch']) + 1} complete "
                    f"batches={payload['batches']} loss={float(payload['loss']):.6f} "
                    f"deviance={float(payload['deviance']):.6f} "
                    f"time={float(payload['elapsed_seconds']):.1f}s{health}"
                )
            if event == "epoch_end":
                return (
                    f"[epoch] {int(payload['epoch']) + 1}/{args.epochs} "
                    f"train_loss={float(payload['train_loss']):.6f} "
                    f"validation_loss={float(payload['validation_loss']):.6f} "
                    f"validation_deviance={float(payload['validation_deviance']):.6f} "
                    f"time={float(payload['epoch_seconds']):.1f}s"
                )
            if event == "run_end":
                return (
                    f"[run] complete best_validation_deviance="
                    f"{float(payload['best_validation_deviance']):.6f}"
                )
            return f"[{event}]"

        def emit_event(event: Mapping[str, Any]) -> None:
            payload = dict(event)
            payload.setdefault("run_elapsed_seconds", time.perf_counter() - run_started)
            line = json.dumps(payload, sort_keys=True, allow_nan=False)
            print(terminal_event_text(payload), flush=True)
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

        train_parameter_counts = parameter_summary(
            model, *(tuple([consistency_head]) if consistency_head is not None else tuple())
        )
        emit_event(
            {
                "event": "run_start",
                "model": type(model).__name__,
                "model_parameters": parameter_summary(model),
                "consistency_head_parameters": (
                    parameter_summary(consistency_head) if consistency_head is not None else None
                ),
                "trainable_parameters": train_parameter_counts,
                "total_parameter_count": train_parameter_counts["total"],
                "trainable_parameter_count": train_parameter_counts["trainable"],
                "ema_teacher_parameters": (
                    parameter_summary(ema_teacher) if ema_teacher is not None else None
                ),
                "train_acquisitions": len(train_acquisitions),
                "validation_acquisitions": len(validation_acquisitions),
                "train_batches_per_epoch": len(train_loader),
                "validation_batches_per_epoch": len(validation_loader),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "accumulation_steps": args.accumulation_steps,
                "effective_batch_size": args.batch_size * args.accumulation_steps,
                "frame_chunk_size": args.frame_chunk_size,
                "max_batches": args.max_batches,
                "num_workers": args.num_workers,
                "persistent_workers": False,
                "augmentation_seed_contract": "v2_worker_independent",
                "validation_augmentation_seed": 0,
                "log_interval": args.log_interval,
                "amp": bool(args.amp and device.type == "cuda"),
                "amp_init_scale": args.amp_init_scale,
                "two_view": bool(args.two_view),
                "representation_queue_capacity": args.representation_queue_capacity,
                "covariance_loss_weight": args.covariance_loss_weight,
                "covariance_min_samples": args.covariance_min_samples,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda" and device.index is not None
                    else (torch.cuda.get_device_name(0) if device.type == "cuda" else None)
                ),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "metrics_output": str(metrics_path.resolve()),
                "log_output": str(log_path.resolve()),
            }
        )
        best_deviance = float("inf")
        for epoch in range(args.epochs):
            epoch_started = time.perf_counter()
            train_data.set_epoch(epoch)
            validation_data.set_epoch(0)
            if args.two_view:
                assert ema_teacher is not None and consistency_head is not None
                train_metrics = train_two_view_epoch(
                    model,
                    train_loader,
                    optimizer,
                    ema_teacher,
                    consistency_head,
                    device=device,
                    max_batches=args.max_batches,
                    amp=args.amp,
                    scaler=scaler,
                    accumulation_steps=args.accumulation_steps,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    progress_callback=emit_event,
                )
                validation_metrics = evaluate_two_view_epoch(
                    model,
                    validation_loader,
                    ema_teacher,
                    consistency_head,
                    device=device,
                    max_batches=args.max_batches,
                    amp=args.amp,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    progress_callback=emit_event,
                )
            else:
                train_metrics = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    task="pretraining",
                    device=device,
                    max_batches=args.max_batches,
                    amp=args.amp,
                    scaler=scaler,
                    accumulation_steps=args.accumulation_steps,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    progress_callback=emit_event,
                )
                validation_metrics = evaluate_epoch(
                    model,
                    validation_loader,
                    task="pretraining",
                    device=device,
                    max_batches=args.max_batches,
                    amp=args.amp,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    progress_callback=emit_event,
                )
            row = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_deviance": train_metrics.deviance,
                "validation_loss": validation_metrics.loss,
                "validation_deviance": validation_metrics.deviance,
                "train_masked": train_metrics.masked,
                "train_visible": train_metrics.visible,
                "validation_masked": validation_metrics.masked,
                "validation_visible": validation_metrics.visible,
                "train_batches": train_metrics.batches,
                "validation_batches": validation_metrics.batches,
                "latent": train_metrics.latent,
                "gate": train_metrics.gate,
                "variance": train_metrics.variance,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "run_elapsed_seconds": time.perf_counter() - run_started,
            }
            row.update({f"train_health_{key}": value for key, value in train_metrics.health.items()})
            row.update(
                {f"validation_health_{key}": value for key, value in validation_metrics.health.items()}
            )
            with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            emit_event({"event": "epoch_end", **row})
            if validation_metrics.deviance < best_deviance:
                best_deviance = validation_metrics.deviance
                save_checkpoint(
                    args.checkpoint,
                    model,
                    optimizer,
                    epoch=epoch,
                    metrics=row,
                    metadata={
                        "task": "pretraining",
                        "seed": args.seed,
                        "device": str(device),
                        "amp": bool(args.amp and device.type == "cuda"),
                        "two_view": bool(args.two_view),
                        "representation_queue_capacity": args.representation_queue_capacity,
                        "covariance_loss_weight": args.covariance_loss_weight,
                        "covariance_min_samples": args.covariance_min_samples,
                        "accumulation_steps": args.accumulation_steps,
                        "frame_chunk_size": args.frame_chunk_size,
                        "num_workers": args.num_workers,
                        "log_interval": args.log_interval,
                        "log_output": str(log_path.resolve()),
                        "model_parameters": parameter_summary(model),
                        "trainable_parameters": train_parameter_counts,
                        "normalization": normalization.to_dict(),
                        "normalization_sha256": sha256_file(args.normalization),
                        "scanner_vocab": train_acquisitions.scanner_vocab,
                        "collimator_vocab": train_acquisitions.collimator_vocab,
                        "stress_rest_vocab": train_acquisitions.stress_vocab,
                        "manifest": str(Path(args.manifest).resolve()),
                        "splits": str(Path(args.splits).resolve()),
                        "manifest_sha256": sha256_file(args.manifest),
                        "splits_sha256": sha256_file(args.splits),
                    },
                    extra_state=(
                        {
                            "ema_teacher": ema_teacher.state_dict(),
                            "consistency_head": consistency_head.state_dict(),
                        }
                        if args.two_view and ema_teacher is not None and consistency_head is not None
                        else None
                    ),
                )
        emit_event(
            {
                "event": "run_end",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "metrics": str(metrics_path.resolve()),
                "log": str(log_path.resolve()),
                "best_validation_deviance": best_deviance,
            }
        )
        return 0
    if args.command == "denoise":
        from torch.utils.data import DataLoader

        from .data.torch_dataset import (
            NormalizationStats,
            PairDataset,
            collate_acquisitions,
        )
        from .models import ProjectionBackbone, ProjectionDenoiser
        from .training import (
            evaluate_denoising_patients,
            evaluate_epoch,
            parse_calibration_ranges,
            parameter_summary,
            save_checkpoint,
            train_epoch,
        )

        if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
            raise SystemExit("--epochs, --batch-size, and --accumulation-steps must be positive")
        if args.freeze_epochs < 0 or args.freeze_epochs > args.epochs:
            raise SystemExit("--freeze-epochs must lie between 0 and --epochs")
        if args.warmup_epochs < 0:
            raise SystemExit("--warmup-epochs cannot be negative")
        if args.early_stop_patience < 1:
            raise SystemExit("--early-stop-patience must be positive")
        if args.num_workers < 0:
            raise SystemExit("--num-workers cannot be negative")
        if args.max_batches is not None and args.max_batches < 1:
            raise SystemExit("--max-batches must be positive when supplied")
        if args.log_interval < 0:
            raise SystemExit("--log-interval cannot be negative")
        if args.frame_chunk_size is not None and args.frame_chunk_size < 1:
            raise SystemExit("--frame-chunk-size must be positive when supplied")

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        normalization = NormalizationStats.from_json(args.normalization)
        split_payload = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        _require_versioned_splits(split_payload)
        split_map = split_payload.get("dataset2", {})
        pair_records = read_jsonl(args.pairs)
        _require_versioned_rows(pair_records, "Dataset2 pair manifest")
        accepted = [row for row in pair_records if bool(row.get("accepted"))]
        train_records = [
            row for row in accepted if split_map.get(str(row.get("patient_key"))) == "train"
        ]
        validation_records = [
            row for row in accepted if split_map.get(str(row.get("patient_key"))) == "validation"
        ]
        if not train_records or not validation_records:
            raise SystemExit("locked Dataset 2 splits produced an empty train or validation set")
        if args.baseline_pseudocount <= 0:
            raise SystemExit("--baseline-pseudocount must be positive")
        if args.low_signal_threshold < 0:
            raise SystemExit("--low-signal-threshold must be nonnegative")
        try:
            calibration_ranges = parse_calibration_ranges(args.calibration_ranges)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        # Read pretrained metadata before constructing Dataset2 so categorical
        # IDs use exactly the vocabularies learned during Dataset1 pretraining.
        pretrained_payload = torch.load(
            Path(args.pretrained_checkpoint), map_location="cpu", weights_only=False
        )
        if not isinstance(pretrained_payload, Mapping):
            raise SystemExit("pretrained checkpoint must be a mapping or state dictionary")
        pretrained_metadata = dict(
            pretrained_payload.get("metadata", {})
            if isinstance(pretrained_payload.get("metadata", {}), Mapping)
            else {}
        )
        pretrained_checkpoint_sha256 = sha256_file(args.pretrained_checkpoint)
        pairs_sha256 = sha256_file(args.pairs)
        splits_sha256 = sha256_file(args.splits)
        migration_report_sha256 = (
            sha256_file(args.exposure_migration_report)
            if args.exposure_migration_report
            else None
        )
        validation_kwargs: dict[str, Any] = {
            "normalization_sha256": sha256_file(args.normalization),
            "allow_legacy_backbone_transfer": bool(args.legacy_backbone_transfer),
            "migration_report": args.exposure_migration_report,
            "legacy_checkpoint_sha256": pretrained_checkpoint_sha256,
        }
        if args.legacy_backbone_transfer:
            validation_kwargs.update(
                {
                    "manifest_sha256": pairs_sha256,
                    "pairs_sha256": pairs_sha256,
                    "splits_sha256": splits_sha256,
                }
            )
        validate_checkpoint_metadata(pretrained_metadata, **validation_kwargs)
        scanner_vocab = dict(pretrained_metadata.get("scanner_vocab", {}))
        collimator_vocab = dict(pretrained_metadata.get("collimator_vocab", {}))
        stress_vocab = dict(pretrained_metadata.get("stress_rest_vocab", {}))
        train_data = PairDataset(
            args.root,
            train_records,
            config=config,
            normalization=normalization,
            scanner_vocab=scanner_vocab,
            collimator_vocab=collimator_vocab,
            stress_rest_vocab=stress_vocab,
        )
        validation_data = PairDataset(
            args.root,
            validation_records,
            config=config,
            normalization=normalization,
            scanner_vocab=scanner_vocab,
            collimator_vocab=collimator_vocab,
            stress_rest_vocab=stress_vocab,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            **({"prefetch_factor": 2} if args.num_workers > 0 else {}),
            collate_fn=collate_acquisitions,
            generator=torch.Generator().manual_seed(args.seed + 1009),
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            **({"prefetch_factor": 2} if args.num_workers > 0 else {}),
            collate_fn=collate_acquisitions,
        )

        if isinstance(pretrained_payload.get("model"), Mapping):
            pretrained_state = pretrained_payload["model"]
        elif isinstance(pretrained_payload.get("backbone"), Mapping):
            pretrained_state = pretrained_payload["backbone"]
        else:
            # Also accept a raw ProjectionBackbone state_dict saved with
            # torch.save(model.state_dict(), path).
            pretrained_state = pretrained_payload
        raw_tensor_keys = [
            key.removeprefix("module.")
            for key, value in pretrained_state.items()
            if isinstance(key, str) and torch.is_tensor(value)
        ]
        has_backbone_prefix = any(key.startswith("backbone.") for key in raw_tensor_keys)
        backbone_state: dict[str, torch.Tensor] = {}
        for key, value in pretrained_state.items():
            if not isinstance(key, str) or not torch.is_tensor(value):
                continue
            normalized_key = key.removeprefix("module.")
            if has_backbone_prefix:
                if not normalized_key.startswith("backbone."):
                    continue
                normalized_key = normalized_key.removeprefix("backbone.")
            backbone_state[normalized_key] = value
        if not backbone_state:
            raise SystemExit("pretrained checkpoint contains no backbone parameters")

        def embedding_vocab_size(vocab: Mapping[str, Any], parameter_name: str) -> int:
            if vocab:
                return len(vocab) + 1
            weight = backbone_state.get(f"{parameter_name}_embedding.weight")
            if weight is not None and weight.ndim >= 1:
                return int(weight.shape[0])
            return 1

        checkpoint_frame_chunk = pretrained_metadata.get("frame_chunk_size")
        frame_chunk_size = (
            args.frame_chunk_size if args.frame_chunk_size is not None else checkpoint_frame_chunk
        )
        backbone = ProjectionBackbone(
            log_exposure_mean=normalization.log_exposure_mean,
            log_exposure_std=normalization.log_exposure_std,
            radius_mean_mm=normalization.radius_mean_mm,
            radius_std_mm=normalization.radius_std_mm,
            scanner_vocab_size=embedding_vocab_size(scanner_vocab, "scanner"),
            collimator_vocab_size=embedding_vocab_size(collimator_vocab, "collimator"),
            stress_rest_vocab_size=embedding_vocab_size(stress_vocab, "stress_rest"),
            frame_chunk_size=frame_chunk_size,
        )
        model = ProjectionDenoiser(
            backbone=backbone,
            baseline_residual=args.baseline_residual,
            baseline_pseudocount=args.baseline_pseudocount,
        )
        initialization = "scratch" if args.scratch else "pretrained"
        decoder_initialization_sha256 = state_dict_sha256(model.decoder)
        backbone_initialization_sha256 = state_dict_sha256(model.backbone)
        if not args.scratch:
            model.backbone.load_state_dict(backbone_state, strict=True)
        backbone_runtime_sha256 = state_dict_sha256(model.backbone)
        model = model.to(device)

        comparison_protocol = args.comparison_protocol or (
            "full_finetune" if args.freeze_epochs == 0 else "partial_freeze_transfer"
        )
        expected_protocol = (
            "full_finetune" if args.freeze_epochs == 0 else "partial_freeze_transfer"
        )
        if comparison_protocol != expected_protocol:
            raise SystemExit(
                "--comparison-protocol does not match --freeze-epochs; "
                f"expected {expected_protocol!r}"
            )

        def set_trainable_stage(epoch: int) -> None:
            if epoch < args.freeze_epochs:
                for parameter in model.backbone.parameters():
                    parameter.requires_grad_(False)
                for module in (model.backbone.angular2, model.backbone.temporal2):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
            else:
                for parameter in model.backbone.parameters():
                    parameter.requires_grad_(True)
            for parameter in model.decoder.parameters():
                parameter.requires_grad_(True)

        set_trainable_stage(0)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.backbone_learning_rate},
                {"params": model.decoder.parameters(), "lr": args.decoder_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )

        def schedule(epoch: int) -> float:
            if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                return float(epoch + 1) / float(args.warmup_epochs)
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        base_learning_rates = [args.backbone_learning_rate, args.decoder_learning_rate]
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(args.amp and device.type == "cuda"),
            init_scale=args.amp_init_scale,
        )
        metrics_path = (
            Path(args.metrics_output)
            if args.metrics_output
            else Path(args.checkpoint).with_name("denoise_metrics.jsonl")
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = (
            Path(args.log_output)
            if args.log_output
            else metrics_path.with_name(f"{metrics_path.stem}.events.jsonl")
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.resolve() == metrics_path.resolve():
            raise SystemExit("--log-output and --metrics-output must be different files")
        patient_metrics_path = (
            Path(args.patient_metrics_output) if args.patient_metrics_output else None
        )
        if patient_metrics_path is not None:
            patient_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        run_started = time.perf_counter()

        def emit_event(event: Mapping[str, Any]) -> None:
            payload = dict(event)
            payload.setdefault("run_elapsed_seconds", time.perf_counter() - run_started)
            line = json.dumps(payload, sort_keys=True, allow_nan=False)
            event_name = payload.get("event")
            if event_name == "run_start":
                print(
                    f"[denoise] model={payload['model']} trainable={payload['trainable_parameter_count']:,} "
                    f"train_pairs={payload['train_pairs']} validation_pairs={payload['validation_pairs']} "
                    f"device={payload['device']} amp={payload['amp']}",
                    flush=True,
                )
            elif event_name == "batch":
                print(
                    f"[denoise {payload['phase']}] epoch={int(payload['epoch']) + 1}/{args.epochs} "
                    f"batch={payload['batch']}/{payload.get('batches_total')} "
                    f"nll_cf/bin={float(payload['loss']):.6f} "
                    f"deviance/bin={float(payload['deviance']):.6f} "
                    f"bias={float(payload.get('absolute_total_count_bias_fraction', 0.0)):.4f} "
                    f"rate={float(payload.get('batches_per_second', 0.0)):.3f} batches/s",
                    flush=True,
                )
            elif event_name == "epoch_end":
                print(
                    f"[denoise epoch] {int(payload['epoch']) + 1}/{args.epochs} "
                    f"train_dev/bin={float(payload['train_deviance']):.6f} "
                    f"val_dev/bin={float(payload['validation_deviance']):.6f} "
                    f"val_abs_bias={float(payload.get('validation_health_absolute_total_count_bias_fraction', 0.0)):.4f} "
                    f"time={float(payload['epoch_seconds']):.1f}s",
                    flush=True,
                )
            elif event_name == "run_end":
                print(
                    f"[denoise] complete best_epoch={payload.get('best_epoch')} "
                    f"best_validation_deviance={float(payload['best_validation_deviance']):.6f}",
                    flush=True,
                )
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

        trainable_counts = parameter_summary(model)
        emit_event(
            {
                "event": "run_start",
                "task": "dataset2_denoising",
                "model": type(model).__name__,
                "loss_definition": "constant_free_poisson_nll_per_selected_detector_bin",
                "deviance_definition": "poisson_deviance_per_selected_detector_bin",
                "trainable_parameters": trainable_counts,
                "total_parameter_count": trainable_counts["total"],
                "trainable_parameter_count": trainable_counts["trainable"],
                "train_pairs": len(train_data),
                "validation_pairs": len(validation_data),
                "test_pairs": sum(
                    1
                    for row in accepted
                    if split_map.get(str(row.get("patient_key"))) == "test"
                ),
                "epochs": args.epochs,
                "freeze_epochs": args.freeze_epochs,
                "comparison_protocol": comparison_protocol,
                "batch_size": args.batch_size,
                "accumulation_steps": args.accumulation_steps,
                "effective_batch_size": args.batch_size * args.accumulation_steps,
                "backbone_learning_rate": args.backbone_learning_rate,
                "decoder_learning_rate": args.decoder_learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "early_stop_patience": args.early_stop_patience,
                "frame_chunk_size": frame_chunk_size,
                "amp": bool(args.amp and device.type == "cuda"),
                "amp_init_scale": args.amp_init_scale,
                "baseline_residual": bool(args.baseline_residual),
                "baseline_pseudocount": args.baseline_pseudocount,
                "initialization": initialization,
                "weights_transferred": not args.scratch,
                "decoder_initialization_sha256": decoder_initialization_sha256,
                "backbone_initialization_sha256": backbone_initialization_sha256,
                "backbone_runtime_sha256": backbone_runtime_sha256,
                "metadata_source_checkpoint": str(Path(args.pretrained_checkpoint).resolve()),
                "legacy_backbone_transfer": bool(args.legacy_backbone_transfer),
                "legacy_checkpoint_sha256": pretrained_checkpoint_sha256,
                "migration_report_sha256": migration_report_sha256,
                "migration_schema_version": 2 if args.exposure_migration_report else None,
                "source_checkpoint_scientific_semantics_version": int(
                    pretrained_metadata.get("scientific_semantics_version", 1)
                ),
                "exposure_migration_report": (
                    str(Path(args.exposure_migration_report).resolve())
                    if args.exposure_migration_report
                    else None
                ),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "pretrained_checkpoint": str(Path(args.pretrained_checkpoint).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "metrics_output": str(metrics_path.resolve()),
                "log_output": str(log_path.resolve()),
                "patient_metrics_output": (
                    str(patient_metrics_path.resolve()) if patient_metrics_path else None
                ),
            }
        )

        best_deviance = float("inf")
        best_bias = float("inf")
        best_epoch: int | None = None
        epochs_without_improvement = 0
        for epoch in range(args.epochs):
            set_trainable_stage(epoch)
            lr_scale = schedule(epoch)
            for group, base_learning_rate in zip(optimizer.param_groups, base_learning_rates):
                group["lr"] = base_learning_rate * lr_scale
            stage = "decoder_upper_warmup" if epoch < args.freeze_epochs else "full_finetune"
            emit_event(
                {
                    "event": "stage",
                    "epoch": epoch,
                    "stage": stage,
                    "backbone_trainable": any(
                        parameter.requires_grad for parameter in model.backbone.parameters()
                    ),
                    "angular2_temporal2_trainable": all(
                        parameter.requires_grad
                        for module in (model.backbone.angular2, model.backbone.temporal2)
                        for parameter in module.parameters()
                    ),
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                }
            )
            epoch_started = time.perf_counter()
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                task="denoising",
                device=device,
                max_batches=args.max_batches,
                amp=args.amp,
                scaler=scaler,
                accumulation_steps=args.accumulation_steps,
                log_interval=args.log_interval,
                epoch=epoch,
                progress_callback=emit_event,
            )
            validation_metrics = evaluate_epoch(
                model,
                validation_loader,
                task="denoising",
                device=device,
                max_batches=args.max_batches,
                amp=args.amp,
                log_interval=args.log_interval,
                epoch=epoch,
                progress_callback=emit_event,
            )
            if patient_metrics_path is not None:
                patient_rows = evaluate_denoising_patients(
                    model,
                    validation_loader,
                    device=device,
                    max_batches=args.max_batches,
                    amp=args.amp,
                    low_signal_threshold=args.low_signal_threshold,
                    calibration_ranges=calibration_ranges,
                )
                with patient_metrics_path.open(
                    "a", encoding="utf-8", newline="\n"
                ) as handle:
                    for patient_row in patient_rows:
                        handle.write(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "seed": args.seed,
                                    "initialization": initialization,
                                    "baseline_pseudocount": args.baseline_pseudocount,
                                    **patient_row,
                                },
                                sort_keys=True,
                                allow_nan=False,
                            )
                            + "\n"
                        )
            row = {
                "epoch": epoch,
                "stage": stage,
                "train_loss": train_metrics.loss,
                "train_deviance": train_metrics.deviance,
                "validation_loss": validation_metrics.loss,
                "validation_deviance": validation_metrics.deviance,
                "train_batches": train_metrics.batches,
                "validation_batches": validation_metrics.batches,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "run_elapsed_seconds": time.perf_counter() - run_started,
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "decoder_learning_rate": optimizer.param_groups[1]["lr"],
            }
            row.update({f"train_health_{key}": value for key, value in train_metrics.health.items()})
            row.update(
                {f"validation_health_{key}": value for key, value in validation_metrics.health.items()}
            )
            with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            emit_event({"event": "epoch_end", **row})

            validation_deviance = validation_metrics.deviance
            validation_bias = validation_metrics.health.get(
                "absolute_total_count_bias_fraction", float("inf")
            )
            is_new_best = (
                not math.isfinite(best_deviance)
                or validation_deviance < best_deviance * 0.999
                or (
                    validation_deviance <= best_deviance * 1.001
                    and validation_bias < best_bias
                )
            )
            if is_new_best:
                best_deviance = validation_deviance
                best_bias = validation_bias
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    args.checkpoint,
                    model,
                    optimizer,
                    epoch=epoch,
                    metrics=row,
                    metadata={
                        "task": "dataset2_denoising",
                        "loss_definition": "constant_free_poisson_nll_per_selected_detector_bin",
                        "deviance_definition": "poisson_deviance_per_selected_detector_bin",
                        "seed": args.seed,
                        "device": str(device),
                        "amp": bool(args.amp and device.type == "cuda"),
                        "amp_init_scale": args.amp_init_scale,
                        "baseline_residual": bool(args.baseline_residual),
                        "baseline_pseudocount": args.baseline_pseudocount,
                        "initialization": initialization,
                        "weights_transferred": not args.scratch,
                        "decoder_initialization_sha256": decoder_initialization_sha256,
                        "backbone_initialization_sha256": backbone_initialization_sha256,
                        "backbone_runtime_sha256": backbone_runtime_sha256,
                        "metadata_source_checkpoint": str(
                            Path(args.pretrained_checkpoint).resolve()
                        ),
                        "legacy_backbone_transfer": bool(args.legacy_backbone_transfer),
                        "legacy_checkpoint_sha256": pretrained_checkpoint_sha256,
                        "migration_report_sha256": migration_report_sha256,
                        "migration_schema_version": 2 if args.exposure_migration_report else None,
                        "source_checkpoint_scientific_semantics_version": int(
                            pretrained_metadata.get("scientific_semantics_version", 1)
                        ),
                        "exposure_migration_report": (
                            str(Path(args.exposure_migration_report).resolve())
                            if args.exposure_migration_report
                            else None
                        ),
                        "batch_size": args.batch_size,
                        "accumulation_steps": args.accumulation_steps,
                        "freeze_epochs": args.freeze_epochs,
                        "comparison_protocol": comparison_protocol,
                        "frame_chunk_size": frame_chunk_size,
                        "pretrained_checkpoint": str(Path(args.pretrained_checkpoint).resolve()),
                        "pairs": str(Path(args.pairs).resolve()),
                        "splits": str(Path(args.splits).resolve()),
                        "normalization": normalization.to_dict(),
                        "normalization_sha256": sha256_file(args.normalization),
                        "pairs_sha256": pairs_sha256,
                        "splits_sha256": splits_sha256,
                    },
                )
            else:
                epochs_without_improvement += 1
            if (
                epoch + 1 >= args.freeze_epochs
                and epochs_without_improvement >= args.early_stop_patience
            ):
                emit_event(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "patience": args.early_stop_patience,
                        "best_epoch": best_epoch,
                        "best_validation_deviance": best_deviance,
                    }
                )
                break
        emit_event(
            {
                "event": "run_end",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "metrics": str(metrics_path.resolve()),
                "log": str(log_path.resolve()),
                "best_epoch": best_epoch,
                "best_validation_deviance": best_deviance,
                "best_absolute_count_bias_fraction": best_bias if math.isfinite(best_bias) else None,
            }
        )
        return 0
    if args.command == "visualize-denoise":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .data.torch_dataset import (
            NormalizationStats,
            PairDataset,
            collate_acquisitions,
            model_batch,
        )
        from .models import ProjectionBackbone, ProjectionDenoiser

        if args.pair_index < 0:
            raise SystemExit("--pair-index cannot be negative")
        if args.gate_index < 0 or args.view_index < 0:
            raise SystemExit("--gate-index and --view-index cannot be negative")
        if not 0.0 < args.percentile <= 100.0:
            raise SystemExit("--percentile must be in (0, 100]")
        if args.dpi < 1:
            raise SystemExit("--dpi must be positive")
        if args.frame_chunk_size is not None and args.frame_chunk_size < 1:
            raise SystemExit("--frame-chunk-size must be positive when supplied")

        all_pair_records = read_jsonl(args.pairs)
        _require_versioned_rows(all_pair_records, "Dataset2 pair manifest")
        pair_records = [row for row in all_pair_records if bool(row.get("accepted"))]
        if args.pair_id is not None:
            pair_records = [row for row in pair_records if str(row.get("pair_id")) == args.pair_id]
        if args.patient_key is not None:
            pair_records = [
                row for row in pair_records if str(row.get("patient_key")) == args.patient_key
            ]
        if not pair_records:
            raise SystemExit("no accepted pair matched --pair-id/--patient-key")
        if args.pair_index >= len(pair_records):
            raise SystemExit(
                f"--pair-index {args.pair_index} is out of range for {len(pair_records)} matching pairs"
            )
        pair_record = pair_records[args.pair_index]

        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        normalization = NormalizationStats.from_json(args.normalization)
        pretrained_payload = torch.load(
            Path(args.pretrained_checkpoint), map_location="cpu", weights_only=False
        )
        if not isinstance(pretrained_payload, Mapping):
            raise SystemExit("pretrained checkpoint must be a mapping or state dictionary")
        pretrained_metadata = dict(
            pretrained_payload.get("metadata", {})
            if isinstance(pretrained_payload.get("metadata", {}), Mapping)
            else {}
        )
        donor_validation_kwargs: dict[str, Any] = {
            "normalization_sha256": sha256_file(args.normalization),
            "allow_legacy_backbone_transfer": bool(args.legacy_backbone_transfer),
            "migration_report": args.exposure_migration_report,
            "legacy_checkpoint_sha256": sha256_file(args.pretrained_checkpoint),
        }
        if args.legacy_backbone_transfer:
            donor_validation_kwargs.update(
                {
                    "manifest_sha256": sha256_file(args.pairs),
                    "pairs_sha256": sha256_file(args.pairs),
                    "splits_sha256": sha256_file(args.splits),
                }
            )
        validate_checkpoint_metadata(pretrained_metadata, **donor_validation_kwargs)
        scanner_vocab = dict(pretrained_metadata.get("scanner_vocab", {}))
        collimator_vocab = dict(pretrained_metadata.get("collimator_vocab", {}))
        stress_vocab = dict(pretrained_metadata.get("stress_rest_vocab", {}))
        if isinstance(pretrained_payload.get("model"), Mapping):
            pretrained_state = pretrained_payload["model"]
        elif isinstance(pretrained_payload.get("backbone"), Mapping):
            pretrained_state = pretrained_payload["backbone"]
        else:
            pretrained_state = pretrained_payload
        raw_tensor_keys = [
            key.removeprefix("module.")
            for key, value in pretrained_state.items()
            if isinstance(key, str) and torch.is_tensor(value)
        ]
        has_backbone_prefix = any(key.startswith("backbone.") for key in raw_tensor_keys)
        backbone_state: dict[str, torch.Tensor] = {}
        for key, value in pretrained_state.items():
            if not isinstance(key, str) or not torch.is_tensor(value):
                continue
            normalized_key = key.removeprefix("module.")
            if has_backbone_prefix:
                if not normalized_key.startswith("backbone."):
                    continue
                normalized_key = normalized_key.removeprefix("backbone.")
            backbone_state[normalized_key] = value
        if not backbone_state:
            raise SystemExit("pretrained checkpoint contains no backbone parameters")

        def embedding_vocab_size(vocab: Mapping[str, Any], parameter_name: str) -> int:
            if vocab:
                return len(vocab) + 1
            weight = backbone_state.get(f"{parameter_name}_embedding.weight")
            if weight is not None and weight.ndim >= 1:
                return int(weight.shape[0])
            return 1

        frame_chunk_size = (
            args.frame_chunk_size
            if args.frame_chunk_size is not None
            else pretrained_metadata.get("frame_chunk_size")
        )
        backbone = ProjectionBackbone(
            log_exposure_mean=normalization.log_exposure_mean,
            log_exposure_std=normalization.log_exposure_std,
            radius_mean_mm=normalization.radius_mean_mm,
            radius_std_mm=normalization.radius_std_mm,
            scanner_vocab_size=embedding_vocab_size(scanner_vocab, "scanner"),
            collimator_vocab_size=embedding_vocab_size(collimator_vocab, "collimator"),
            stress_rest_vocab_size=embedding_vocab_size(stress_vocab, "stress_rest"),
            frame_chunk_size=frame_chunk_size,
        )
        backbone.load_state_dict(backbone_state, strict=True)
        model = ProjectionDenoiser(
            backbone=backbone,
            baseline_residual=args.baseline_residual,
        ).eval()
        checkpoint_payload = torch.load(
            Path(args.checkpoint), map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint_payload, Mapping) or not isinstance(
            checkpoint_payload.get("model"), Mapping
        ):
            raise SystemExit("denoiser checkpoint has no model state")
        checkpoint_metadata = checkpoint_payload.get("metadata", {})
        if not isinstance(checkpoint_metadata, Mapping):
            raise SystemExit("denoiser checkpoint metadata is missing")
        validate_checkpoint_metadata(
            checkpoint_metadata,
            normalization_sha256=sha256_file(args.normalization),
            pairs_sha256=sha256_file(args.pairs),
            splits_sha256=sha256_file(args.splits),
        )
        trained_pseudocount = float(checkpoint_metadata.get("baseline_pseudocount", 0.5))
        if trained_pseudocount <= 0:
            raise SystemExit("denoiser checkpoint has an invalid baseline pseudocount")
        model.baseline_pseudocount = trained_pseudocount
        trained_residual: bool | None = None
        if isinstance(checkpoint_metadata, Mapping) and "baseline_residual" in checkpoint_metadata:
            trained_residual = bool(checkpoint_metadata["baseline_residual"])
        if trained_residual is not None and trained_residual != bool(args.baseline_residual):
            expected_flag = "--baseline-residual" if trained_residual else "--no-baseline-residual"
            supplied_flag = "--baseline-residual" if args.baseline_residual else "--no-baseline-residual"
            raise SystemExit(
                "checkpoint decoder mode mismatch: checkpoint was trained with "
                f"{expected_flag}, but visualization requested {supplied_flag}; "
                f"rerun with {expected_flag}"
            )
        if trained_residual is None:
            print(
                "[visualize] WARNING: checkpoint metadata does not record baseline_residual; "
                "verify --baseline-residual matches the mode used during training",
                flush=True,
            )
        checkpoint_metrics = checkpoint_payload.get("metrics", {})
        if isinstance(checkpoint_metrics, Mapping) and checkpoint_metrics.get("train_batches", 0) < 50:
            print(
                "[visualize] WARNING: checkpoint was trained on fewer than 50 batches; "
                "the denoised image is only a smoke-test result",
                flush=True,
            )
        checkpoint_state = {
            key.removeprefix("module."): value
            for key, value in checkpoint_payload["model"].items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        model.load_state_dict(checkpoint_state, strict=True)

        pair_dataset = PairDataset(
            args.root,
            [pair_record],
            config=config,
            normalization=normalization,
            scanner_vocab=scanner_vocab,
            collimator_vocab=collimator_vocab,
            stress_rest_vocab=stress_vocab,
        )
        batch = collate_acquisitions([pair_dataset[0]])
        low = read_nm_dicom(Path(args.root) / str(pair_record["low_relative_path"]), config)
        high = read_nm_dicom(Path(args.root) / str(pair_record["high_relative_path"]), config)
        gate, view = args.gate_index, args.view_index
        if gate >= low.counts.shape[0] or gate >= high.counts.shape[0]:
            raise SystemExit(f"--gate-index {gate} is outside the acquisition gate range")
        if view >= low.counts.shape[1] or view >= high.counts.shape[1]:
            raise SystemExit(f"--view-index {view} is outside the acquisition view range")

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)
        inputs = {
            name: value.to(device, non_blocking=device.type == "cuda")
            for name, value in model_batch(batch, include_counts_raw=True).items()
        }
        autocast_enabled = bool(args.amp and device.type == "cuda")
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            output = model(**inputs)
        predicted_rate = output.rate[0, gate, view].float().cpu().numpy()
        target_exposure_s = float(high.frame_exposure_s[gate, view])
        denoised_counts = np.clip(predicted_rate * target_exposure_s, 0.0, None)
        low_slice = low.counts[gate, view].astype(np.float32)
        high_slice = high.counts[gate, view].astype(np.float32)
        arrays = [low_slice, denoised_counts, high_slice]
        if args.scale == "log1p":
            display_arrays = [np.log1p(np.clip(array, 0.0, None)) for array in arrays]
            colorbar_label = "log1p(expected counts)"
        else:
            display_arrays = arrays
            colorbar_label = "expected counts"
        vmax = float(np.percentile(np.concatenate([array.reshape(-1) for array in display_arrays]), args.percentile))
        vmax = max(vmax, 1.0e-6)

        figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        titles = (
            f"10s input\nactual total={low.frame_exposure_s.sum():.2f}s",
            f"denoised expected 30s counts\nrequested frame={target_exposure_s:.3f}s",
            f"30s/high target\nactual total={high.frame_exposure_s.sum():.2f}s",
        )
        images = []
        for axis, image, title in zip(axes, display_arrays, titles):
            rendered = axis.imshow(image, cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            axis.set_title(title)
            axis.set_xlabel("detector column")
            axis.set_ylabel("detector row")
            images.append(rendered)
        figure.colorbar(images[-1], ax=axes, shrink=0.82, label=colorbar_label)
        figure.suptitle(
            f"pair={pair_record.get('pair_id', '')} gate={gate} view={view} "
            f"scale={args.scale}",
            fontsize=12,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=args.dpi, facecolor="white")
        plt.close(figure)
        print(
            f"[visualize] saved={output_path.resolve()} pair={pair_record.get('pair_id', '')} "
            f"gate={gate} view={view} scale={args.scale} vmax={vmax:.6f}",
            flush=True,
        )
        return 0
    if args.command == "export-denoise-dicom":
        from .data.dicom_export import write_nm_projection_dicom
        from .data.torch_dataset import (
            NormalizationStats,
            PairDataset,
            collate_acquisitions,
            model_batch,
        )
        from .models import ProjectionBackbone, ProjectionDenoiser

        if args.count < 1:
            raise SystemExit("--count must be positive")
        if args.frame_chunk_size is not None and args.frame_chunk_size < 1:
            raise SystemExit("--frame-chunk-size must be positive when supplied")

        config = _reader_config(
            argparse.Namespace(
                hash_key_env=args.hash_key_env,
                allow_unhashed=args.allow_unhashed,
                institution_id=args.institution_id,
                start_angle=None,
                angular_step=None,
                rotation_direction=None,
                view_dwell_s=None,
                spacing_row=None,
                spacing_col=None,
                radius_mm=None,
            )
        )
        normalization = NormalizationStats.from_json(args.normalization)
        split_payload = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        _require_versioned_splits(split_payload)
        split_map = split_payload.get("dataset2", {})
        all_pair_records = read_jsonl(args.pairs)
        _require_versioned_rows(all_pair_records, "Dataset2 pair manifest")
        accepted = [row for row in all_pair_records if bool(row.get("accepted"))]
        validation_records = [
            row for row in accepted if split_map.get(str(row.get("patient_key"))) == "validation"
        ]
        if len(validation_records) < args.count:
            raise SystemExit(
                f"requested {args.count} validation pairs, but only "
                f"{len(validation_records)} are available"
            )
        selected_records = validation_records[: args.count]

        pretrained_payload = torch.load(
            Path(args.pretrained_checkpoint), map_location="cpu", weights_only=False
        )
        if not isinstance(pretrained_payload, Mapping):
            raise SystemExit("pretrained checkpoint must be a mapping or state dictionary")
        pretrained_metadata = dict(
            pretrained_payload.get("metadata", {})
            if isinstance(pretrained_payload.get("metadata", {}), Mapping)
            else {}
        )
        donor_validation_kwargs: dict[str, Any] = {
            "normalization_sha256": sha256_file(args.normalization),
            "allow_legacy_backbone_transfer": bool(args.legacy_backbone_transfer),
            "migration_report": args.exposure_migration_report,
            "legacy_checkpoint_sha256": sha256_file(args.pretrained_checkpoint),
        }
        if args.legacy_backbone_transfer:
            donor_validation_kwargs.update(
                {
                    "manifest_sha256": sha256_file(args.pairs),
                    "pairs_sha256": sha256_file(args.pairs),
                    "splits_sha256": sha256_file(args.splits),
                }
            )
        validate_checkpoint_metadata(pretrained_metadata, **donor_validation_kwargs)
        scanner_vocab = dict(pretrained_metadata.get("scanner_vocab", {}))
        collimator_vocab = dict(pretrained_metadata.get("collimator_vocab", {}))
        stress_vocab = dict(pretrained_metadata.get("stress_rest_vocab", {}))
        if isinstance(pretrained_payload.get("model"), Mapping):
            pretrained_state = pretrained_payload["model"]
        elif isinstance(pretrained_payload.get("backbone"), Mapping):
            pretrained_state = pretrained_payload["backbone"]
        else:
            pretrained_state = pretrained_payload
        raw_tensor_keys = [
            key.removeprefix("module.")
            for key, value in pretrained_state.items()
            if isinstance(key, str) and torch.is_tensor(value)
        ]
        has_backbone_prefix = any(key.startswith("backbone.") for key in raw_tensor_keys)
        backbone_state: dict[str, torch.Tensor] = {}
        for key, value in pretrained_state.items():
            if not isinstance(key, str) or not torch.is_tensor(value):
                continue
            normalized_key = key.removeprefix("module.")
            if has_backbone_prefix:
                if not normalized_key.startswith("backbone."):
                    continue
                normalized_key = normalized_key.removeprefix("backbone.")
            backbone_state[normalized_key] = value
        if not backbone_state:
            raise SystemExit("pretrained checkpoint contains no backbone parameters")

        def embedding_vocab_size(vocab: Mapping[str, Any], parameter_name: str) -> int:
            if vocab:
                return len(vocab) + 1
            weight = backbone_state.get(f"{parameter_name}_embedding.weight")
            if weight is not None and weight.ndim >= 1:
                return int(weight.shape[0])
            return 1

        frame_chunk_size = (
            args.frame_chunk_size
            if args.frame_chunk_size is not None
            else pretrained_metadata.get("frame_chunk_size")
        )
        backbone = ProjectionBackbone(
            log_exposure_mean=normalization.log_exposure_mean,
            log_exposure_std=normalization.log_exposure_std,
            radius_mean_mm=normalization.radius_mean_mm,
            radius_std_mm=normalization.radius_std_mm,
            scanner_vocab_size=embedding_vocab_size(scanner_vocab, "scanner"),
            collimator_vocab_size=embedding_vocab_size(collimator_vocab, "collimator"),
            stress_rest_vocab_size=embedding_vocab_size(stress_vocab, "stress_rest"),
            frame_chunk_size=frame_chunk_size,
        )
        backbone.load_state_dict(backbone_state, strict=True)
        model = ProjectionDenoiser(
            backbone=backbone,
            baseline_residual=args.baseline_residual,
        )
        checkpoint_payload = torch.load(
            Path(args.checkpoint), map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint_payload, Mapping) or not isinstance(
            checkpoint_payload.get("model"), Mapping
        ):
            raise SystemExit("denoiser checkpoint has no model state")
        checkpoint_metadata = checkpoint_payload.get("metadata", {})
        if not isinstance(checkpoint_metadata, Mapping):
            raise SystemExit("denoiser checkpoint metadata is missing")
        validate_checkpoint_metadata(
            checkpoint_metadata,
            normalization_sha256=sha256_file(args.normalization),
            pairs_sha256=sha256_file(args.pairs),
            splits_sha256=sha256_file(args.splits),
        )
        trained_pseudocount = float(checkpoint_metadata.get("baseline_pseudocount", 0.5))
        if trained_pseudocount <= 0:
            raise SystemExit("denoiser checkpoint has an invalid baseline pseudocount")
        model.baseline_pseudocount = trained_pseudocount
        trained_residual: bool | None = None
        if isinstance(checkpoint_metadata, Mapping) and "baseline_residual" in checkpoint_metadata:
            trained_residual = bool(checkpoint_metadata["baseline_residual"])
        if trained_residual is not None and trained_residual != bool(args.baseline_residual):
            expected_flag = "--baseline-residual" if trained_residual else "--no-baseline-residual"
            supplied_flag = "--baseline-residual" if args.baseline_residual else "--no-baseline-residual"
            raise SystemExit(
                "checkpoint decoder mode mismatch: checkpoint was trained with "
                f"{expected_flag}, but export requested {supplied_flag}; "
                f"rerun with {expected_flag}"
            )
        if trained_residual is None:
            print(
                "[export] WARNING: checkpoint metadata does not record baseline_residual; "
                "verify --baseline-residual matches the training mode",
                flush=True,
            )
        checkpoint_state = {
            key.removeprefix("module."): value
            for key, value in checkpoint_payload["model"].items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        model.load_state_dict(checkpoint_state, strict=True)

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device).eval()
        output_root = Path(args.output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []
        for validation_index, pair_record in enumerate(selected_records):
            pair_dataset = PairDataset(
                args.root,
                [pair_record],
                config=config,
                normalization=normalization,
                scanner_vocab=scanner_vocab,
                collimator_vocab=collimator_vocab,
                stress_rest_vocab=stress_vocab,
            )
            item = pair_dataset[0]
            batch = collate_acquisitions([item])
            low_path = Path(args.root) / str(pair_record["low_relative_path"])
            high_path = Path(args.root) / str(pair_record["high_relative_path"])
            low_sample = read_nm_dicom(low_path, config)
            high_sample = read_nm_dicom(high_path, config)
            inputs = {
                name: value.to(device, non_blocking=device.type == "cuda")
                for name, value in model_batch(batch, include_counts_raw=True).items()
            }
            autocast_enabled = bool(args.amp and device.type == "cuda")
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                predicted_rate = model(**inputs).rate[0].float().cpu().numpy()

            target_exposure = np.asarray(item["target_exposure_s"].numpy(), dtype=np.float32)
            if str(pair_record.get("provenance", "independent")) == "nested":
                denoised_counts = (
                    low_sample.counts.astype(np.float32)
                    + predicted_rate * target_exposure[..., None, None]
                )
            else:
                denoised_counts = predicted_rate * high_sample.frame_exposure_s[..., None, None]

            pair_id = str(pair_record.get("pair_id", f"validation_{validation_index:03d}"))
            pair_root = output_root / f"pair_{validation_index:03d}_{pair_id[:12]}"
            pair_root.mkdir(parents=True, exist_ok=True)
            input_output = write_nm_projection_dicom(
                low_path,
                pair_root / "input_10s_projection.dcm",
                low_sample.counts.astype(np.float32),
                view_dwell_s=low_sample.view_dwell_s,
                series_description="gated-spect 10s input projections",
                derivation_description="Source low-count projection stack exported for commercial SPECT reconstruction.",
                series_number=900 + validation_index * 10 + 1,
                expected_counts=False,
                provenance=str(pair_record.get("provenance", "independent")),
            )
            denoised_output = write_nm_projection_dicom(
                high_path,
                pair_root / "denoised_expected_30s_projection.dcm",
                denoised_counts,
                view_dwell_s=high_sample.view_dwell_s,
                series_description="gated-spect denoised expected 30s projections",
                derivation_description=(
                    "ProjectionDenoiser expected high-count projections; values are counts "
                    "at the high-acquisition frame exposure."
                ),
                series_number=900 + validation_index * 10 + 2,
                expected_counts=True,
                provenance=str(pair_record.get("provenance", "independent")),
            )
            target_output = write_nm_projection_dicom(
                high_path,
                pair_root / "target_30s_projection.dcm",
                high_sample.counts.astype(np.float32),
                view_dwell_s=high_sample.view_dwell_s,
                series_description="gated-spect 30s high-count target projections",
                derivation_description="Source high-count projection stack exported as a reconstruction reference.",
                series_number=900 + validation_index * 10 + 3,
                expected_counts=False,
                provenance=str(pair_record.get("provenance", "independent")),
            )
            manifest_rows.append(
                {
                    "validation_index": validation_index,
                    "pair_id": pair_id,
                    "patient_key": str(pair_record.get("patient_key", "")),
                    "provenance": str(pair_record.get("provenance", "independent")),
                    "shape_gvhw": list(low_sample.counts.shape),
                    "input_total_exposure_s": float(low_sample.frame_exposure_s.sum()),
                    "target_total_exposure_s": float(high_sample.frame_exposure_s.sum()),
                    "input_total_counts": int(low_sample.counts.astype(np.int64).sum()),
                    "denoised_total_counts": int(np.rint(denoised_counts).astype(np.int64).sum()),
                    "target_total_counts": int(high_sample.counts.astype(np.int64).sum()),
                    "input_dicom": str(input_output.relative_to(output_root)),
                    "denoised_dicom": str(denoised_output.relative_to(output_root)),
                    "target_dicom": str(target_output.relative_to(output_root)),
                }
            )
            print(
                f"[export] {validation_index + 1}/{len(selected_records)} "
                f"pair={pair_id} shape={tuple(low_sample.counts.shape)} "
                f"input={int(low_sample.counts.sum())} "
                f"denoised={int(np.rint(denoised_counts).sum())} "
                f"target={int(high_sample.counts.sum())}",
                flush=True,
            )

        manifest = {
            "format": "gated_spect_derived_nm_projection_export_v1",
            "description": (
                "Each DICOM is a multi-frame projection stack for import into a "
                "commercial SPECT reconstruction application; these are not anatomical slices."
            ),
            "pretrained_checkpoint": str(Path(args.pretrained_checkpoint).resolve()),
            "denoiser_checkpoint": str(Path(args.checkpoint).resolve()),
            "baseline_residual": bool(args.baseline_residual),
            "pairs": manifest_rows,
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / "README.txt").write_text(
            "These files are multi-frame NM projection stacks, not reconstructed anatomical volumes.\n"
            "Import the three projection series for a pair into the commercial SPECT application\n"
            "and run its standard reconstruction. The denoised series contains expected counts at\n"
            "the high-acquisition frame exposure.\n",
            encoding="utf-8",
        )
        print(f"[export] complete output_dir={output_root} manifest={manifest_path}", flush=True)
        return 0
    if args.command == "validate-dicom":
        config = _reader_config(args)
        sample = read_nm_dicom(args.path, config)
        payload = {
            "sample_id": sample.sample_id,
            "shape": sample.shape,
            "total_counts": int(sample.counts.astype(np.int64).sum()),
            "geometry_hash": sample.geometry_hash,
            "patient_key": sample.patient_key,
            "study_key": sample.study_key,
            "exposure_source": sample.exposure_source,
            "metadata": dict(sample.metadata),
        }
        _write_or_print(payload, args.output)
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
