# Remediation status

This file distinguishes code implementation/testing from validation on real
data. Legacy manifests, caches, normalizations, and checkpoints remain
read-only inputs; incompatible artifacts are rejected rather than overwritten.
Rows marked `VALIDATED ON REAL DATA` refer to pre-existing artifacts from
earlier work; this remediation pass ran code/regression tests and one bounded
queue smoke only, not new full scientific experiments.

| Task | Status | Evidence / next gate |
| --- | --- | --- |
| Scientific semantics v2 constants | TESTED | `src/gated_spect/versioning.py`; persisted artifacts carry version metadata. |
| Cache schema/exposure versioning and stale-cache rejection | TESTED | `StaleCacheError` plus regression test in `tests/test_cache.py`. |
| DICOM `ActualFrameDuration` per angular position | TESTED | Synthetic 10 s/30 s tests and real-dataset sample tests pass. |
| Exposure-source provenance | TESTED | Reader records dwell source and raw duration; v2 manifests contain it. |
| v2 Dataset1/2 manifests | TESTED | Fresh create-only artifacts in `artifacts/v2/fresh_20260811/manifests/` (993/528 acquisitions and 264 pair-audit rows); recovered v1 inputs remain under `artifacts/recovery/v1_exact/`. |
| Canonical cache schema 3 | TESTED | `artifacts/v2/fresh_20260811_strict/cache_v3/dataset1` (992) and `dataset2` (528) contain source-relative paths, source SHA-256, manifest SHA-256, row SHA-256, complete checksum bundles, and index-level checksum SHA-256 bindings; restored schema-2 cache remains legacy and is rejected. |
| v1-to-v2 exposure migration report | TESTED | Strict create-only report `artifacts/v2/fresh_20260811/qc/exposure_migration_report_schema2_strict_v2.{json,csv}` compares 1,521 raw-DICOM rows and 1,138 exact split assignments, with zero errors, exact sample coverage, finite deltas, unchanged radius/count element statistics, and persisted numeric thresholds. |
| Legacy-backbone compatibility decision | TESTED | The strict report authorizes only the exact recovered checkpoint/runtime artifacts; mean abs standardized-exposure delta is 3.45e-15 and p99 is 1.07e-14. The legacy absolute-rate decoder remains explicitly not physically calibrated. |
| Checkpoint compatibility contract | TESTED | Versioned save/load and explicit legacy-transfer/migration validation; regression test rejects v1. |
| Legacy transfer artifact binding | TESTED | Runtime transfer now requires schema-2 exact checkpoint/manifests/splits/normalization hashes; the historical approval JSON is intentionally rejected until regenerated. |
| Worker-independent epoch-dependent augmentation | TESTED | Stream-based seed contract, `persistent_workers=False` for pretraining, and two-worker regression pass. |
| Deterministic validation augmentation | TESTED | Validation dataset is reset to epoch 0 every training epoch; reset/replay regression passes in `tests/test_torch_dataset.py`. |
| Covariance/SVD effective rank and participation rank | TESTED | Rank-1, independent-feature, correlation, and participation-rank regressions pass. |
| Queue capacity and collapse diagnostics | TESTED | Default queue is 2048; covariance/off-diagonal diagnostics are logged and tested. |
| SSL accumulation-window queue exact-once semantics | TESTED | Four-batch, partial-window, content-uniqueness, and failed-step regressions pass; bounded v2 smoke observed queue size 32 after 16 batches/8-way accumulation, with 2 optimizer steps and 0 AMP skips. |
| Queue-supported standardized covariance regularizer | TESTED | Current differentiable representations plus detached queue history, 512-sample warm-up, scale-invariance, duplicate/independent-feature, and gradient-isolation regressions pass. |
| Exact-artifact legacy migration validation | TESTED | Schema-2 report fields and SHA-256 binding reject wrong checkpoint, normalization, split, manifest, stale report, and bare compatibility approval. The historical report remains read-only; the fresh schema-2 report is the only transfer-authorizing artifact. |
| Immutable artifact writes and provenance binding | TESTED | Manifests, splits, inventories, pair reports, migration reports, and CLI outputs are create-only; collisions fail loudly. Runtime Dataset1 and accepted Dataset2 source checksums, split hashes, and normalization bindings are verified before training. |
| Resumable training/reproducibility state | TESTED | Checkpoints persist optimizer/scheduler, AMP scaler, EMA/queue, Python/NumPy/PyTorch RNG, dedicated DataLoader generators, run contract including epoch horizon, environment, and source-artifact hashes; incomplete state fails closed. Interrupted-after-epoch resume completed in `artifacts/v2/smoke_20260811g/`, and generator-order/strict-state regressions pass. |
| AMP successful-step synchronization | TESTED | Scale-drop helper suppresses EMA/queue updates and optimizer-step counts; skipped-step regression passes. |
| Centralized clipped exposure conditioning | TESTED | Shared standardized-log-exposure helper clamps to [-8,8]; model regression passes. |
| Dataset2 pretrained metadata vocabularies | TESTED | PairDataset accepts pretrained vocabularies, maps unknown categories to zero, and denoise loads them before dataset construction. |
| Correct global representation pooling | TESTED | Explicit valid spatial-token/frame mean helper and hand-computed regression pass. |
| Residual pseudocount configuration/ablation | VALIDATED ON REAL DATA | Historical pre-remediation artifact only: `artifacts/v2/qc/pseudocount_ablation_report.{json,csv}`. Expanded evaluator and deterministic selection-rule support are implemented, but no pseudocount study or value selection was performed in this remediation pass. |
| Derived NM DICOM semantics and G=1 restriction | TESTED | Exporter now writes PRIMARY derived type, per-view duration, provenance, and rejects G>1; focused tests pass. |
| Expanded patient-level denoising metrics | TESTED | `evaluate_denoising_patients` retains patient/pair identity while emitting per-view bias, zero-input bins, low-count strata, projection low-signal proxy, and configurable calibration ranges. |
| Matched launcher protocol and initialization fingerprints | TESTED | `--freeze-epochs` defaults to 0 in `scripts/run_matched_denoising.py`; protocol metadata and decoder/backbone SHA-256 fingerprints are recorded; dry-run regression passes. |
| Ideal OSEM labeling and reconstruction gate | IMPLEMENTED | Projector/OSEM modules remain explicitly documented as an ideal mathematical reference; clinical/unrolled use remains gated. |
| Corrected v2 representation smoke and collapse monitoring | TESTED | Bounded CUDA smoke/resume runs are under `artifacts/v2/smoke_20260811f/`, `smoke_20260811g/`, and `smoke_20260811i/`; logs include non-null train/validation masked-only checkpoint deviance, union deviance, rank, participation rank, correlation, AMP, and gradient diagnostics. Full queue-warm v2 pretraining remains a separate run. |
| Clean Dataset2 denoising under corrected v2 pipeline | VALIDATED ON REAL DATA | Historical pre-remediation artifact only: prior denoising outputs use v2 normalization/pairs/splits and patient-level rows; no denoising retraining was performed in this remediation pass. |
| Pretrained-versus-scratch >=3-seed comparison | VALIDATED ON REAL DATA | Historical pre-remediation artifact only: `artifacts/v2/qc/matched_pretrained_vs_scratch_report.{json,csv}`; no matched training or model-selection decision was performed in this remediation pass. |
| Full clean Dataset1 v2 pretraining | NOT STARTED | Only corrected v2 smoke runs have been executed; the legacy checkpoint remains an explicit migration baseline and its absolute-rate decoder is not physically calibrated. |
| Locked Dataset2 test evaluation | NOT STARTED | Deliberately untouched; evaluate once after development choices and geometry/export gates are frozen. |
