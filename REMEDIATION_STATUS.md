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
| v2 Dataset1/2 manifests | TESTED | `artifacts/v2/manifests/*.jsonl` generated without modifying v1 files. |
| Canonical cache v2 | TESTED | `artifacts/v2/cache_v2/dataset1` (992) and `dataset2` (528) written. |
| v1-to-v2 exposure migration report | IMPLEMENTED | The existing report is a preserved pre-remediation artifact and lacks schema-2 exact hashes; the new audit writes a non-overwriting schema-2 report and requires explicit legacy-checkpoint/manifest/split inputs. |
| Legacy-backbone compatibility decision | IMPLEMENTED | Historical standardized-input comparison is retained for analysis, but the legacy checkpoint is not reusable at runtime until a fresh schema-2 exact-artifact report (including the active Dataset2 pair manifest) passes validation. |
| Checkpoint compatibility contract | TESTED | Versioned save/load and explicit legacy-transfer/migration validation; regression test rejects v1. |
| Legacy transfer artifact binding | TESTED | Runtime transfer now requires schema-2 exact checkpoint/manifests/splits/normalization hashes; the historical approval JSON is intentionally rejected until regenerated. |
| Worker-independent epoch-dependent augmentation | TESTED | Stream-based seed contract, `persistent_workers=False` for pretraining, and two-worker regression pass. |
| Deterministic validation augmentation | TESTED | Validation dataset is reset to epoch 0 every training epoch; reset/replay regression passes in `tests/test_torch_dataset.py`. |
| Covariance/SVD effective rank and participation rank | TESTED | Rank-1, independent-feature, correlation, and participation-rank regressions pass. |
| Queue capacity and collapse diagnostics | TESTED | Default queue is 2048; covariance/off-diagonal diagnostics are logged and tested. |
| SSL accumulation-window queue exact-once semantics | TESTED | Four-batch, partial-window, content-uniqueness, and failed-step regressions pass; bounded v2 smoke observed queue size 32 after 16 batches/8-way accumulation, with 2 optimizer steps and 0 AMP skips. |
| Queue-supported standardized covariance regularizer | TESTED | Current differentiable representations plus detached queue history, 512-sample warm-up, scale-invariance, duplicate/independent-feature, and gradient-isolation regressions pass. |
| Exact-artifact legacy migration validation | TESTED | Schema-2 report fields and SHA-256 binding reject wrong checkpoint, normalization, split, manifest, stale report, and bare compatibility approval. The preserved historical report remains read-only and must be regenerated before transfer. |
| AMP successful-step synchronization | TESTED | Scale-drop helper suppresses EMA/queue updates and optimizer-step counts; skipped-step regression passes. |
| Centralized clipped exposure conditioning | TESTED | Shared standardized-log-exposure helper clamps to [-8,8]; model regression passes. |
| Dataset2 pretrained metadata vocabularies | TESTED | PairDataset accepts pretrained vocabularies, maps unknown categories to zero, and denoise loads them before dataset construction. |
| Correct global representation pooling | TESTED | Explicit valid spatial-token/frame mean helper and hand-computed regression pass. |
| Residual pseudocount configuration/ablation | VALIDATED ON REAL DATA | `scripts/run_pseudocount_ablation.py` completed 4 alphas × 3 seeds on the 22-patient validation cohort. Patient-bootstrap report: `artifacts/v2/qc/pseudocount_ablation_report.{json,csv}`; α=0.5 selected by validation deviance/improvement/bias. |
| Derived NM DICOM semantics and G=1 restriction | TESTED | Exporter now writes PRIMARY derived type, per-view duration, provenance, and rejects G>1; focused tests pass. |
| Expanded patient-level denoising metrics | TESTED | `evaluate_denoising_patients` retains patient/pair identity while emitting per-view bias, zero-input bins, low-count strata, projection low-signal proxy, and configurable calibration ranges. |
| Matched launcher protocol and initialization fingerprints | TESTED | `--freeze-epochs` defaults to 0 in `scripts/run_matched_denoising.py`; protocol metadata and decoder/backbone SHA-256 fingerprints are recorded; dry-run regression passes. |
| Ideal OSEM labeling and reconstruction gate | IMPLEMENTED | Projector/OSEM modules remain explicitly documented as an ideal mathematical reference; clinical/unrolled use remains gated. |
| Corrected v2 representation smoke and collapse monitoring | TESTED | Historical smoke artifacts remain read-only; the bounded post-fix queue smoke records queue size 32 after two successful optimizer steps and zero skips. Full queue-warm v2 pretraining remains a separate run. |
| Clean Dataset2 denoising under corrected v2 pipeline | VALIDATED ON REAL DATA | 12-run pseudocount matrix and fixed-α matched runs use v2 normalization/pairs/splits, corrected metadata vocabularies, residual formulation, and patient-level validation rows. |
| Pretrained-versus-scratch >=3-seed comparison | VALIDATED ON REAL DATA | `artifacts/v2/qc/matched_pretrained_vs_scratch_report.{json,csv}`: 3 common seeds, 22 validation patients, mean pretrained−scratch deviance `-0.03095/bin`, bootstrap 95% CI `[-0.03661,-0.02546]`; all seed CIs are below zero. |
| Full clean Dataset1 v2 pretraining | NOT STARTED | Only corrected v2 smoke runs have been executed; the legacy checkpoint remains an explicit migration baseline and its absolute-rate decoder is not physically calibrated. |
| Locked Dataset2 test evaluation | NOT STARTED | Deliberately untouched; evaluate once after development choices and geometry/export gates are frozen. |
