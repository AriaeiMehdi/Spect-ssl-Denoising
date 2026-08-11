# Gated cardiac SPECT agent policy

## Scope and authority

This repository implements self-supervised pretraining on gated cardiac SPECT Dataset1 and count-rate residual denoising on paired Dataset2 acquisitions. Treat `gated_spect_pretraining_implementation_plan.md` and `REMEDIATION_STATUS.md` as the active project specification and status record. Inspect implementation and tests before proposing or making changes.

Preserve correct existing behavior. Do not redesign the architecture, per-bin Poisson loss, count thinning, patient-level splits, or ideal projector/OSEM mathematics unless a demonstrated defect and regression test require it.

Do not silently reinterpret, overwrite, or migrate legacy caches, manifests, normalization files, reports, or checkpoints. Versioned incompatibilities must fail loudly and old artifacts must remain recoverable. Do not use the locked Dataset2 test set until development choices are frozen.

## Scientific invariants

- Interpret DICOM `ActualFrameDuration` as acquisition time per angular position, with explicit provenance.
- Keep OSEM/projector code labeled as an ideal mathematical baseline, not a clinically validated reconstruction system.
- Keep the Dataset2 residual count-rate formulation. Treat the baseline pseudocount as an ablation variable, not a known optimum.
- Preserve patient-level split isolation and pretrained Dataset1 metadata vocabularies during Dataset2 transfer.
- Update EMA teacher state and representation queues only after a successful optimizer step, including under AMP and gradient accumulation.
- Evaluate representation collapse with covariance/SVD effective rank, participation rank, covariance/correlation diagnostics, and queue-aware evidence.
- Restrict validated derived NM DICOM export to ungated `G=1` until geometry, orientation, physical-model, phantom, and clinical validation gates pass.
- Distinguish `IMPLEMENTED`, `TESTED`, and `VALIDATED ON REAL DATA` in `REMEDIATION_STATUS.md`.

## Required workflow

For scientific or training-critical changes, use three separated phases:

1. Theorist: inspect evidence and state the scientific contract, risks, and acceptance criteria without editing files.
2. Implementer: make the smallest justified change and add a regression test that fails under the old behavior.
3. Judge: independently inspect the diff and run relevant tests; issue `PASS`, `FAIL`, or `INSUFFICIENT EVIDENCE` without editing files.

Do not start full Dataset1 pretraining or final Dataset2 training during an audit or dry run. A full run requires all relevant P0/P1 corrections and tests to pass first.

## Model routing

- Root/implementation and training-log audit: prefer `gpt-5.6-luna` with `xhigh` reasoning when selectable.
- Scientific theorist and independent judge: prefer `gpt-5.6-sol` with `xhigh` reasoning.
- If the current Codex runtime does not expose Luna for subagents, use the inherited available model and disclose the substitution. Never claim an unavailable model was used.
- Parallelize only independent read-only investigations. Keep implementation ownership with one agent to prevent conflicting edits.

## Environment and verification

- Workspace: `D:\Dr.EntezarMahdi\3DRecon`
- Dataset root: `D:\Dr.EntezarMahdi\3DRecon\datasets`
- Python: `D:\global environment\torch_cuda124\Scripts\python.exe`
- Run focused tests after each logical change, then the full suite before declaring readiness.
- If Python fails with Windows error 1114 while loading PyTorch `shm.dll`, rerun the same command once before diagnosing a code failure.
- Preserve unrelated user changes and untracked files.
- Log commands, configuration, seeds, artifact hashes, environment, optimizer-step/AMP-skip counts, and training-health metrics required to reproduce conclusions.

## Decision format

Every audit or judgment must identify inspected artifacts/code, evidence-backed claims, unresolved risks, required tests or real-data validation, and whether training may proceed.
