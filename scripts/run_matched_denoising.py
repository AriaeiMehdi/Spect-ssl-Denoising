"""Launch matched Dataset2 pretrained-versus-scratch denoising runs.

The metadata donor checkpoint is always read so both conditions use identical
scanner/collimator/stress vocabularies.  ``--scratch`` only disables weight
transfer; it does not create a Dataset2-specific encoding.  This launcher
never reads or evaluates the locked test split.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--normalization", required=True)
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pseudocount <= 0:
        raise SystemExit("--pseudocount must be positive")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    output_root = Path(args.output_root).resolve()
    for initialization in ("pretrained", "scratch"):
        for seed in seeds:
            label = f"{initialization}_alpha_{args.pseudocount:g}_seed_{seed}"
            run_root = output_root / label
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
                "--legacy-backbone-transfer",
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
                "--seed",
                str(seed),
            ]
            if initialization == "scratch":
                command.append("--scratch")
            print("[matched]", " ".join(command), flush=True)
            if args.dry_run:
                continue
            run_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
