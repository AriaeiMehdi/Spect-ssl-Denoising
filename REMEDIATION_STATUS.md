# Remediation status

This file distinguishes code implementation/testing from validation on real
data. Legacy manifests, caches, normalizations, and checkpoints remain
read-only inputs; incompatible artifacts are rejected rather than overwritten.

| Task | Status | Evidence / next gate |
| --- | --- | --- |
| Scientific semantics v2 constants | TESTED | `src/gated_spect/versioning.py`; persisted artifacts carry version metadata. |
| Cache schema/exposure versioning and stale-cache rejection | TESTED | `StaleCacheError` plus regression test in `tests/test_cache.py`. |
| DICOM `ActualFrameDuration` per angular position | TESTED | Synthetic 10 s/30 s tests and real-dataset sample tests pass. |
| Exposure-source provenance | TESTED | Reader records dwell source and raw duration; v2 manifests contain it. |
| v2 Dataset1/2 manifests | TESTED | `artifacts/v2/manifests/*.jsonl` generated without modifying v1 files. |
| Canonical cache v2 | TESTED | `artifacts/v2/cache_v2/dataset1` (992) and `dataset2` (528) written. |
| v1-to-v2 exposure migration report | TESTED | `artifacts/v2/qc/exposure_migration_report.{json,csv}` generated; all 1,138 compared patient assignments unchanged and per-view dwell ratios are ~32 where the legacy division existed. |
| Legacy-backbone compatibility decision | TESTED | Compatibility is **true** after preserving legacy seed 20260806: mean abs z delta 3.45e-15, p99 1.07e-14; backbone transfer is allowed only through explicit migration mode. |
| Checkpoint compatibility contract | TESTED | Versioned save/load and explicit legacy-transfer/migration validation; regression test rejects v1. |
| Worker-independent epoch-dependent augmentation | TESTED | Stream-based seed contract, `persistent_workers=False` for pretraining, and two-worker regression pass. |
| Deterministic validation augmentation | IMPLEMENTED | Validation dataset is reset to epoch 0 every training epoch; final fixed-seed characterization remains. |
| Covariance/SVD effective rank and participation rank | TESTED | Rank-1, independent-feature, correlation, and participation-rank regressions pass. |
| Queue capacity and collapse diagnostics | TESTED | Default queue is 2048; covariance/off-diagonal diagnostics are logged and tested. |
| AMP successful-step synchronization | TESTED | Scale-drop helper suppresses EMA/queue updates and optimizer-step counts; skipped-step regression passes. |
| Centralized clipped exposure conditioning | TESTED | Shared standardized-log-exposure helper clamps to [-8,8]; model regression passes. |
| Dataset2 pretrained metadata vocabularies | TESTED | PairDataset accepts pretrained vocabularies, maps unknown categories to zero, and denoise loads them before dataset construction. |
| Correct global representation pooling | TESTED | Explicit valid spatial-token/frame mean helper and hand-computed regression pass. |
| Residual pseudocount configuration/ablation | IMPLEMENTED | `--baseline-pseudocount` is positive/configurable and checkpoint-versioned; `scripts/run_pseudocount_ablation.py` launches the planned 4-alpha x 3-seed validation matrix without test selection. |
| Derived NM DICOM semantics and G=1 restriction | TESTED | Exporter now writes PRIMARY derived type, per-view duration, provenance, and rejects G>1; focused tests pass. |
| Ideal OSEM labeling and reconstruction gate | IMPLEMENTED | Projector/OSEM modules remain explicitly documented as an ideal mathematical reference; clinical/unrolled use remains gated. |
| Full v2 pretraining/denoising experiments | NOT STARTED | P0/P1 tests and migration gate pass; clean Dataset2 denoising plus pseudocount study remain pending. Legacy absolute rate decoder is not physically calibrated. |
| Pretrained-versus-scratch >=3-seed comparison | NOT STARTED | Must occur before locked-test evaluation. |
