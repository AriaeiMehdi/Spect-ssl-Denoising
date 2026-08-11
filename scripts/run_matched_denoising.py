"""Launch matched Dataset2 pretrained-versus-scratch denoising runs.

The metadata donor checkpoint is always read so both conditions use identical
scanner/collimator/stress vocabularies.  ``--scratch`` only disables weight
transfer; it does not create a Dataset2-specific encoding.  This launcher
never reads or evaluates the locked test split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_matched_command(
    args: argparse.Namespace,
    *,
    initialization: str,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Build one matched command and its auditable protocol metadata."""

    if initialization not in {"pretrained", "scratch"}:
        raise ValueError("initialization must be pretrained or scratch")
    protocol = args.comparison_protocol or (
        "full_finetune" if args.freeze_epochs == 0 else "partial_freeze_transfer"
    )
    expected_protocol = (
        "full_finetune" if args.freeze_epochs == 0 else "partial_freeze_transfer"
    )
    if protocol != expected_protocol:
        raise ValueError(
            "comparison_protocol does not match freeze_epochs; "
            f"expected {expected_protocol!r}"
        )
    label = f"{initialization}_alpha_{args.pseudocount:g}_seed_{seed}"
    run_root = Path(args.output_root).resolve() / label
    command = [
        sys.executable,
        "-m",
        "gated_spect",
        "denoise",
        "--root",
        args.root,
        "--pairs",
        args.pairs,
        "--splits",
        args.splits,
        "--normalization",
        args.normalization,
        "--pretrained-checkpoint",
        args.pretrained_checkpoint,
        "--exposure-migration-report",
        args.migration_report,
        "--checkpoint",
        str(run_root / "best.pt"),
        "--metrics-output",
        str(run_root / "metrics.jsonl"),
        "--log-output",
        str(run_root / "events.jsonl"),
        "--patient-metrics-output",
        str(run_root / "patient_metrics.jsonl"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--accumulation-steps",
        str(args.accumulation_steps),
        "--num-workers",
        str(args.num_workers),
        "--frame-chunk-size",
        str(args.frame_chunk_size),
        "--baseline-pseudocount",
        str(args.pseudocount),
        "--low-signal-threshold",
        str(args.low_signal_threshold),
        "--calibration-ranges",
        args.calibration_ranges,
        "--freeze-epochs",
        str(args.freeze_epochs),
        "--comparison-protocol",
        protocol,
        "--seed",
        str(seed),
        "--legacy-backbone-transfer",
    ]
    dataset1_manifest = getattr(args, "dataset1_manifest", None)
    if dataset1_manifest:
        command[command.index("--normalization") + 2:command.index("--pretrained-checkpoint")] = [
            "--dataset1-manifest",
            dataset1_manifest,
        ] + command[command.index("--normalization") + 2:command.index("--pretrained-checkpoint")]
    dataset2_manifest = getattr(args, "dataset2_manifest", None)
    if dataset2_manifest:
        command[command.index("--pairs") + 2:command.index("--splits")] = [
            "--dataset2-manifest",
            dataset2_manifest,
        ] + command[command.index("--pairs") + 2:command.index("--splits")]
    if initialization == "scratch":
        command.append("--scratch")
    metadata = {
        "comparison_protocol": protocol,
        "freeze_epochs": args.freeze_epochs,
        "initialization": initialization,
        "seed": seed,
        "same_split": True,
        "same_normalization": True,
        "same_pseudocount": True,
        "same_optimizer_configuration": True,
        "same_decoder_initialization_seed": True,
        "different_backbone_initialization": initialization == "scratch",
        "scratch_flag": initialization == "scratch",
    }
    return command, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--dataset2-manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--dataset1-manifest", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--migration-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--seeds", default="20260806,20260807,20260808")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--frame-chunk-size", type=int, default=128)
    parser.add_argument("--low-signal-threshold", type=float, default=5.0)
    parser.add_argument(
        "--calibration-ranges", default="0,1,2,3-5,6-10,11-20,>20"
    )
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument(
        "--comparison-protocol",
        choices=("full_finetune", "partial_freeze_transfer"),
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pseudocount <= 0:
        raise SystemExit("--pseudocount must be positive")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    if args.freeze_epochs < 0:
        raise SystemExit("--freeze-epochs cannot be negative")
    if args.freeze_epochs > args.epochs:
        raise SystemExit("--freeze-epochs cannot exceed --epochs")
    if args.comparison_protocol is None:
        args.comparison_protocol = (
            "full_finetune" if args.freeze_epochs == 0 else "partial_freeze_transfer"
        )
    output_root = Path(args.output_root).resolve()
    for initialization in ("pretrained", "scratch"):
        for seed in seeds:
            command, metadata = build_matched_command(
                args, initialization=initialization, seed=seed
            )
            print("[matched]", " ".join(command), flush=True)
            print("[matched-protocol]", json.dumps(metadata, sort_keys=True), flush=True)
            if args.dry_run:
                continue
            run_root = output_root / f"{initialization}_alpha_{args.pseudocount:g}_seed_{seed}"
            run_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
