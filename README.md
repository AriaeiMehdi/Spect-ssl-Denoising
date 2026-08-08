# Gated SPECT implementation

This repository is being implemented from
gated_spect_pretraining_implementation_plan.md. It is research-only software.

Use the requested interpreter for all commands:

    D:\global environment\torch_cuda124\Scripts\python.exe

The package uses a src layout. Until the editable install is complete, set the
workspace source path explicitly in PowerShell:

    $env:PYTHONPATH = (Join-Path (Get-Location) "src")

For an offline/local editable install in this environment:

    D:\global environment\torch_cuda124\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .

Run the current synthetic/unit suite:

    D:\global environment\torch_cuda124\Scripts\python.exe -m unittest discover -s tests -v

Save the runtime report:

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect environment-report --output artifacts\environment.json

Run an aggregate audit (the output contains no patient/study identifiers):

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect audit --root "D:\path\to\dataset1" --dataset-id dataset1 --output artifacts\qc\dataset1_audit.json

Validate filename pairs and write the de-identified pair manifest:

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect pair-audit --root "D:\path\to\denoising" --output artifacts\manifests\pairs.jsonl

Before inventory, set an institution-scoped HMAC secret in SPECT_HASH_KEY.
The inventory command stores relative paths, source-root IDs, and irreversible
patient/study keys; it does not store raw PatientID values.

    $env:SPECT_HASH_KEY = "<secret supplied by the data custodian>"
    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect inventory --root "D:\path\to\dataset1" --dataset-id dataset1 --output artifacts\manifests\dataset1_headers.jsonl

Do not use --allow-unhashed for real data.

Build the parsed acquisition manifests and lock the patient-level splits:

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect manifest --root "D:\Dr.EntezarMahdi\3DRecon\datasets\Pre-training dataset" --dataset-id dataset1 --output artifacts\manifests\dataset1_acquisition.jsonl
    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect manifest --root "D:\Dr.EntezarMahdi\3DRecon\datasets\Denoising dataset" --dataset-id dataset2 --output artifacts\manifests\dataset2_acquisition.jsonl
    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect lock-splits --dataset1-manifest artifacts\manifests\dataset1_acquisition.jsonl --dataset2-manifest artifacts\manifests\dataset2_acquisition.jsonl --pair-manifest artifacts\manifests\denoising_pairs.jsonl --output artifacts\manifests\locked_splits.json

Compute the frozen Dataset 1 training normalization statistics:

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect normalization --root "D:\Dr.EntezarMahdi\3DRecon\datasets\Pre-training dataset" --manifest artifacts\manifests\dataset1_acquisition.jsonl --splits artifacts\manifests\locked_splits.json --output artifacts\manifests\normalization.json

The PyTorch data layer is in
`src/gated_spect/data/torch_dataset.py`; `PretrainingDataset` performs the
deterministic binomial split and mask generation, and the one-batch training
steps are in `src/gated_spect/training/steps.py`.
`src/gated_spect/training/runner.py` provides finite-checked epoch loops,
gradient clipping, and atomic checkpoint save/load.

Run a full pretraining job (the smoke-test equivalent uses `--epochs 1
--max-batches 1`):

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect pretrain --root "D:\Dr.EntezarMahdi\3DRecon\datasets\Pre-training dataset" --manifest artifacts\manifests\dataset1_acquisition.jsonl --splits artifacts\manifests\locked_splits.json --normalization artifacts\manifests\normalization.json --checkpoint artifacts\runs\pretrain_best.pt --metrics-output artifacts\runs\pretrain_metrics.jsonl --epochs 250 --batch-size 1 --accumulation-steps 8 --num-workers 2

The pretraining command uses two independent student views, an unmasked EMA
teacher, latent/gate consistency, queue variance regularization, and CUDA AMP
by default. Use `--no-two-view` only for the masked-Poisson baseline.
If GPU memory is insufficient for a full acquisition, add
`--frame-chunk-size 64` (or `32` for a smaller peak). This uses activation
checkpointing while preserving all 512 projections and attention semantics;
it trades extra computation for lower peak GPU memory.
At startup it prints a JSON model/data summary, including total and trainable
parameter counts. It then prints a batch event every 10 batches by default and
an epoch summary with loss components, Poisson deviance, masking coverage,
prediction-rate statistics, representation variance/effective rank, gradient
norm, AMP scale, throughput, and elapsed time. Batch and epoch events are also
persisted to a sibling `*.events.jsonl` file; customize the cadence or path
with `--log-interval` and `--log-output`.

Fine-tune the pretrained backbone on the locked Dataset 2 pairs. The default
baseline-residual decoder learns a bounded multiplicative correction around the
exposure-scaled 10-second input, which preserves count scale while learning the
30-second structure:

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect denoise --root "D:\Dr.EntezarMahdi\3DRecon\datasets\Denoising dataset" --pairs artifacts\manifests\denoising_pairs.jsonl --splits artifacts\manifests\locked_splits.json --normalization artifacts\manifests\normalization.json --pretrained-checkpoint artifacts\runs\pretrain_best.pt --checkpoint artifacts\runs\denoise_residual_best40.pt --metrics-output artifacts\runs\denoise_residual_best40_metrics.jsonl --log-output artifacts\runs\denoise_residual_best40_events.jsonl --epochs 40 --freeze-epochs 8 --batch-size 1 --accumulation-steps 8 --num-workers 1 --frame-chunk-size 128

The denoiser command initializes only the backbone from the pretrained
checkpoint, creates a fresh rate decoder, freezes the frame encoder/first
attention blocks for the warm-up stage, then unfreezes the full backbone. It
selects the best checkpoint by validation Poisson deviance with total-count
bias as the tie-breaker. Terminal denoiser metrics report per-selected-bin
constant-free Poisson NLL (`nll_cf/bin`) and Poisson deviance (`deviance/bin`).

Save a detector-slice comparison for one accepted Dataset 2 pair. The
visualizer checks the checkpoint's recorded decoder mode and refuses to load a
residual checkpoint with the non-residual interpretation (or vice versa):

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect visualize-denoise --root "D:\Dr.EntezarMahdi\3DRecon\datasets\Denoising dataset" --pairs artifacts\manifests\denoising_pairs.jsonl --normalization artifacts\manifests\normalization.json --pretrained-checkpoint artifacts\runs\pretrain_best.pt --checkpoint artifacts\runs\denoise_residual_best40.pt --output artifacts\visualizations\denoise_residual_pair0.png --pair-index 0 --gate-index 0 --view-index 0 --frame-chunk-size 128

The saved PNG contains three panels: the 10-second low-count input, the
denoiser prediction converted to expected counts at the high acquisition's
actual exposure, and the high-count target. `--pair-index` can be replaced by
`--pair-id` or `--patient-key`; use `--gate-index` and `--view-index` to inspect
another detector projection. The default `--scale log1p` uses one common color
scale across all panels; use `--scale linear` for raw-count display.

Export three locked validation pairs as derived multi-frame Nuclear Medicine
DICOM projection stacks for import into a commercial SPECT reconstruction
application:

    D:\global environment\torch_cuda124\Scripts\python.exe -m gated_spect export-denoise-dicom --root "D:\Dr.EntezarMahdi\3DRecon\datasets\Denoising dataset" --pairs artifacts\manifests\denoising_pairs.jsonl --splits artifacts\manifests\locked_splits.json --normalization artifacts\manifests\normalization.json --pretrained-checkpoint artifacts\runs\pretrain_best.pt --checkpoint artifacts\runs\denoise_residual_best40.pt --output-dir artifacts\denoising_dicom_validation3 --count 3 --frame-chunk-size 128

Each `pair_*` directory contains `input_10s_projection.dcm`,
`denoised_expected_30s_projection.dcm`, and `target_30s_projection.dcm`.
These files are multi-frame projection stacks (for the validation data:
32 frames of 64x64 pixels), not reconstructed anatomical slices. Import the
three projection series into the commercial application's NM/SPECT projection
workflow and run its standard reconstruction there. `manifest.json` records
the pair IDs, exposures, and count totals.

The current implementation includes the contract-tested projection backbone and
rate decoder, deterministic Poisson/count-splitting losses, the ideal sparse
parallel-hole projector, conventional MLEM/OSEM, and a geometry-gated
projection-conditioned unrolled OSEM module. Real unrolled training remains
disabled until the projector orientation/phantom acceptance gate is signed off.
