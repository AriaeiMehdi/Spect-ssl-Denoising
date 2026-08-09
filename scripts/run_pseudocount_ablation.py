"""Launch the matched Dataset2 pseudocount ablation matrix.

This orchestrator deliberately writes one isolated checkpoint/log directory
per (alpha, seed). It does not evaluate the locked test split or select a
winner; selection remains a validation-only research decision.
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
    parser.add_argument("--migration-report")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--alphas", default="0.5,0.1,0.01,0.001")
    parser.add_argument("--seeds", default="20260806,20260807,20260808")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--frame-chunk-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not alphas or not seeds or any(alpha <= 0 for alpha in alphas):
        raise SystemExit("alphas and seeds must be nonempty; every alpha must be positive")
    output_root = Path(args.output_root).resolve()
    for alpha in alphas:
        for seed in seeds:
            label = f"alpha_{alpha:g}_seed_{seed}"
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
                str(alpha),
                "--seed",
                str(seed),
            ]
            if args.migration_report:
                command.extend(
                    [
                        "--legacy-backbone-transfer",
                        "--exposure-migration-report",
                        args.migration_report,
                    ]
                )
            print("[ablation]", " ".join(command), flush=True)
            if args.dry_run:
                continue
            run_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
