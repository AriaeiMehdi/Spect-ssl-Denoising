"""Create validation-only patient-level denoising experiment reports.

Inputs are the ``patient_metrics.jsonl`` files emitted by ``gated-spect
denoise --patient-metrics-output``.  The script selects an epoch using only
the validation rows, averages duplicate pairs within patient, and reports
patient-bootstrap intervals.  It never loads a DICOM test pair or reports
locked-test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"patient metrics file is empty: {path}")
    required = {"epoch", "patient_key", "model_deviance", "naive_deviance"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"patient metrics file {path} is missing fields: {sorted(missing)}")
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def select_best_epoch(rows: list[Mapping[str, Any]]) -> int:
    """Select the minimum mean validation deviance, with bias as tie-breaker."""

    epochs: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        epochs[int(row["epoch"])].append(row)
    ranked = []
    for epoch, values in epochs.items():
        bias = [abs(float(row.get("model_total_count_bias_fraction", 0.0))) for row in values]
        ranked.append((
            _mean(float(row["model_deviance"]) for row in values),
            _mean(bias),
            epoch,
        ))
    return int(min(ranked)[2])


def patient_means(rows: list[Mapping[str, Any]], epoch: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["epoch"]) == epoch:
            grouped[str(row["patient_key"])].append(row)
    if not grouped:
        raise ValueError(f"selected epoch {epoch} has no rows")
    result: dict[str, dict[str, float]] = {}
    for patient, values in grouped.items():
        result[patient] = {
            "model_deviance": _mean(float(row["model_deviance"]) for row in values),
            "naive_deviance": _mean(float(row["naive_deviance"]) for row in values),
            "absolute_bias": _mean(
                abs(float(row.get("model_total_count_bias_fraction", 0.0)))
                for row in values
            ),
        }
    return result


def bootstrap_mean(values: Iterable[float], *, seed: int, resamples: int) -> tuple[float, float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty patient set")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError("each --run must be LABEL=PATIENT_METRICS_JSONL")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError("each --run must have a nonempty label and path")
        parsed.append((label, Path(path)))
    if not parsed:
        raise ValueError("at least one --run is required")
    return parsed


def _load_run(label: str, path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    epoch = select_best_epoch(rows)
    first = rows[0]
    if "seed" not in first or "initialization" not in first:
        raise ValueError(f"{path} lacks seed/initialization metadata")
    return {
        "label": label,
        "path": str(path.resolve()),
        "rows": rows,
        "epoch": epoch,
        "seed": int(first["seed"]),
        "initialization": str(first["initialization"]),
        "pseudocount": float(first.get("baseline_pseudocount", 0.5)),
        "patients": patient_means(rows, epoch),
    }


def _matched_summary(runs: list[dict[str, Any]], resamples: int) -> dict[str, Any]:
    by_init: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        by_init[run["initialization"]][run["seed"]] = run
    if set(by_init) != {"pretrained", "scratch"}:
        raise ValueError("matched mode requires both pretrained and scratch runs")
    seeds = sorted(set(by_init["pretrained"]) & set(by_init["scratch"]))
    if len(seeds) < 3:
        raise ValueError("matched mode requires at least three common seeds")
    pseudocounts = {run["pseudocount"] for run in runs}
    if len(pseudocounts) != 1:
        raise ValueError("matched runs must use one fixed pseudocount")
    patients = sorted(
        set.intersection(
            *(set(by_init["pretrained"][seed]["patients"]) for seed in seeds)
        )
        & set.intersection(*(set(by_init["scratch"][seed]["patients"]) for seed in seeds))
    )
    if not patients:
        raise ValueError("matched runs have no common validation patients")
    deltas = []
    per_seed = []
    for seed in seeds:
        pre = by_init["pretrained"][seed]["patients"]
        scratch = by_init["scratch"][seed]["patients"]
        seed_delta = [pre[p]["model_deviance"] - scratch[p]["model_deviance"] for p in patients]
        mean, low, high = bootstrap_mean(seed_delta, seed=seed, resamples=resamples)
        deltas.extend(seed_delta)
        per_seed.append({"seed": seed, "pretrained_epoch": by_init["pretrained"][seed]["epoch"], "scratch_epoch": by_init["scratch"][seed]["epoch"], "mean_delta": mean, "ci95_low": low, "ci95_high": high})
    # The primary estimate gives each patient equal weight after averaging
    # independent seeds, rather than giving a patient extra weight for having
    # more pair rows or more seeds.
    patient_delta = {
        p: _mean(
            by_init["pretrained"][seed]["patients"][p]["model_deviance"]
            - by_init["scratch"][seed]["patients"][p]["model_deviance"]
            for seed in seeds
        )
        for p in patients
    }
    mean, low, high = bootstrap_mean(patient_delta.values(), seed=20260806, resamples=resamples)
    return {
        "mode": "matched_pretrained_vs_scratch",
        "pseudocount": next(iter(pseudocounts)),
        "seeds": seeds,
        "patient_count": len(patients),
        "mean_pretrained_minus_scratch_deviance": mean,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "ssl_improvement_supported": bool(high < 0.0),
        "per_seed": per_seed,
        "runs": [{k: v for k, v in run.items() if k not in {"rows", "patients"}} for run in runs],
    }


def _pseudocount_summary(runs: list[dict[str, Any]], resamples: int) -> dict[str, Any]:
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["pseudocount"]].append(run)
    summaries = []
    for alpha, alpha_runs in sorted(groups.items()):
        patient_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        for run in alpha_runs:
            for patient, values in run["patients"].items():
                patient_groups[patient].append(values)
        patient_values = {
            patient: {
                key: _mean(item[key] for item in values)
                for key in ("model_deviance", "naive_deviance", "absolute_bias")
            }
            for patient, values in patient_groups.items()
        }
        dev_mean, dev_low, dev_high = bootstrap_mean(
            (item["model_deviance"] for item in patient_values.values()),
            seed=20260806 + int(round(alpha * 1_000_000)),
            resamples=resamples,
        )
        improvement = [
            (item["naive_deviance"] - item["model_deviance"])
            / max(item["naive_deviance"], 1.0e-12)
            for item in patient_values.values()
        ]
        imp_mean, imp_low, imp_high = bootstrap_mean(
            improvement,
            seed=20260807 + int(round(alpha * 1_000_000)),
            resamples=resamples,
        )
        summaries.append(
            {
                "pseudocount": alpha,
                "seeds": sorted(run["seed"] for run in alpha_runs),
                "patient_count": len(patient_values),
                "mean_model_deviance": dev_mean,
                "model_deviance_ci95_low": dev_low,
                "model_deviance_ci95_high": dev_high,
                "mean_improvement_over_naive": imp_mean,
                "improvement_ci95_low": imp_low,
                "improvement_ci95_high": imp_high,
                "mean_absolute_count_bias_fraction": _mean(
                    item["absolute_bias"] for item in patient_values.values()
                ),
                "runs": [{k: v for k, v in run.items() if k not in {"rows", "patients"}} for run in alpha_runs],
            }
        )
    return {"mode": "pseudocount_ablation", "results": summaries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("matched", "pseudocount"), required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=patient_metrics.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        raise SystemExit("--bootstrap-resamples must be at least 100")
    runs = [_load_run(label, path) for label, path in _parse_runs(args.run)]
    report = (
        _matched_summary(runs, args.bootstrap_resamples)
        if args.mode == "matched"
        else _pseudocount_summary(runs, args.bootstrap_resamples)
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = report.get("results", report.get("per_seed", []))
        if report["mode"] == "matched_pretrained_vs_scratch":
            rows = [
                {
                    "seed": "pooled",
                    "mean_delta": report["mean_pretrained_minus_scratch_deviance"],
                    "ci95_low": report["bootstrap_ci95_low"],
                    "ci95_high": report["bootstrap_ci95_high"],
                },
                *rows,
            ]
        fields = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
