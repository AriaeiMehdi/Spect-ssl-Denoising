# Implementation-Ready Plan: Self-Supervised Pretraining, Projection Denoising, and OSEM Reconstruction for Gated Cardiac SPECT

**Version:** 2.0  
**Date:** 2026-08-07  
**Status:** Ready to implement through the validated conventional OSEM baseline. Learned reconstruction remains gated on projector validation.  
**Use:** Research only; this plan does not define a clinical device or a diagnostic workflow.

**Implementation checkpoint (2026-08-08):** DICOM ingestion, de-identified
manifests, accepted-pair manifests, locked splits, frozen normalization,
projection backbone/decoder, Poisson objectives, deterministic data workers,
ideal projector, MLEM/OSEM, unrolled-OSEM wrapper, epoch runner, and atomic
checkpoints, two-view latent/gate consistency, and a CUDA AMP pretraining
launcher with per-epoch JSONL metrics are implemented and covered by 43 passing
tests. Gradient accumulation, CUDA AMP, pinned/non-blocking transfers, and
configurable Windows-safe workers are implemented and exercised by real-data
smoke runs. The Dataset 2 supervised denoiser launcher is now implemented: it
loads only the pretrained backbone, initializes a fresh denoising decoder,
freezes the backbone except angular2/temporal2 for the warm-up stage, then
fine-tunes all backbone layers, with validation-deviance/count-bias checkpoint
selection and terminal/JSONL logging. A short Dataset 2 AMP smoke checkpoint was
produced under artifacts/runs; full cohort training remains to be run. The
weighted Poisson reduction was subsequently corrected to normalize over selected
detector bins as specified in Section 10.3; denoiser training must be rerun after
this correction because the previous run used gradients inflated by 64x64.
Real unrolled training remains gated on the projector orientation/phantom
acceptance criteria in Section 12.6.

---

## 1. Objective and normative language

Build one reusable projection encoder from a large unlabeled raw-projection dataset, then use paired low/high-count raw projections to train:

1. a projection denoiser that estimates count rate; and
2. a projection-conditioned, unrolled OSEM reconstruction model.

The words MUST, MUST NOT, SHOULD, and MAY are requirements:

- **MUST/MUST NOT:** required for a valid run;
- **SHOULD:** default unless a reason is recorded in the run manifest;
- **MAY:** optional and disabled by default.

The implementation is complete only when the tests, manifests, QC outputs, locked splits, and acceptance gates in this document exist. Producing model code alone is not completion.

---

## 2. Locked scientific and engineering decisions

### 2.1 Dataset roles

- **Dataset 1:** unlabeled binned raw gated SPECT projections used only for self-supervised pretraining and its internal validation.
- **Dataset 2:** paired low/high-exposure binned raw gated projections used for supervised denoising, reconstruction training, validation, and a locked patient-level test.
- Dataset 2 validation and test patients MUST be excluded from Dataset 1 pretraining when cross-dataset identity matching is possible.
- If identity matching is impossible, final results MUST be labeled as having unresolved cross-dataset leakage risk.

### 2.2 Representation

- Detector rows and columns are spatial axes within a 2D projection.
- Projection view is an angular axis, not anatomical depth.
- Cardiac gate is a periodic temporal axis.
- The reusable model is therefore a factorized spatial-angular-temporal encoder, not a 2D image model and not a 3D convolution over view-as-depth.

### 2.3 Statistical semantics

- Original integer counts are immutable statistical observations.
- The projection denoiser predicts a nonnegative **total detected count rate per detector bin**, not a fixed-duration count image. Without a scatter measurement, that output includes scatter and other unseparated background counts.
- Projection-denoising likelihoods are evaluated as:

  \[
  \mu_{g,v,h,w}=E_{g,v}r^{total}_{g,v,h,w}.
  \]

  Reconstruction likelihoods are evaluated as:

  \[
  \mu_{g,v,h,w}=E_{g,v}\left([Ax]_{g,v,h,w}+s_{g,v,h,w}\right).
  \]

  Here \(E\) is frame exposure in seconds, \(x\) is activity rate, and \(s\) is an optional measured or validated scatter count rate.
- No normalized, denoised, or rescaled tensor may replace the original counts in a data-consistency term.

### 2.4 Reconstruction

- Conventional OSEM from original low-count measurements is the required baseline.
- Denoise-then-OSEM is a required comparison, not the preferred final method.
- The preferred unrolled method retains original low-count data in every EM update and uses learned projection features only as conditioning.
- Attenuation correction, scatter correction, and distance-dependent resolution recovery are disabled until their inputs or calibration parameters are validated.

### 2.5 Reproducibility

- Configuration, data manifests, splits, code revision, dependency lock, seed, and checkpoint selection metric MUST be stored with every run.
- Scanner geometry, tensor sizes, paths, durations, and loss weights MUST NOT be hardcoded in model modules.
- Default random seed: **20260806**.

---

## 3. Known sample and implications

The inspected example is binned raw gated SPECT projection data:

| Property | Observed value |
|---|---:|
| Canonical tensor | 16 gates x 32 views x 64 x 64 bins |
| DICOM frames | 512 |
| Pixel storage | unsigned 16-bit integer |
| Total counts | 3,660,504 |
| Mean count/bin | 1.745 |
| Zero-bin fraction | 63.16% |
| Maximum count/bin | 29 |
| Scan arc | 180 degrees |
| Rotation | clockwise |
| Start angle | 225 degrees |
| Angular increment | 5.625 degrees |
| Detector sampling | 6.4 x 6.4 mm |
| Orbit radius | view-dependent, 155-210 mm |
| Collimator | LEHR parallel-hole |
| Energy window | 126.45-154.55 keV |
| Gates | 16, nominally 64 ms each |
| Nominal R-R | 977 ms |
| Angular-stop duration | 27.282 s |

Consequences:

- Gates 15 and 16 have about 84% and 38% of normal gate counts. Exposure is a first-class input and loss weight.
- The frame index exposes one detector stream even though the detector-information sequence contains two entries. The parser MUST NOT invent a second detector stream.
- The example has no attenuation map and no scatter window. Reconstructions are non-attenuation-corrected and non-scatter-corrected until paired auxiliary series are found and validated.
- Frame order MUST come from frame-index vectors, never from an unconditional reshape.
- The provisional angle formula for this sample is:

  \[
  \theta_v = 225^\circ-v(5.625^\circ),\quad v=0,\ldots,31.
  \]

This formula is not considered geometrically verified until a phantom or vendor reconstruction confirms handedness, detector-bin direction, and volume orientation.

### 3.1 Audit of the local datasets

The local workspace audit on 2026-08-07 found:

- Denoising dataset: 528 files, exactly 264 filename pairs named with 10sec and 30sec suffixes. Every file is an explicitly ungated [1,32,64,64] acquisition; TimeSlotVector is absent and is represented as G=1 with gating_mode=ungated. The 10sec files have 10,000 ms actual rotation duration. The files labeled 30sec have 19,479-23,736 ms actual rotation duration, so training uses DICOM ActualFrameDuration rather than the filename label.
- Pre-training dataset: 993 files, all complete [16,32,64,64] gated acquisitions. Detector-sequence radii are present in 992 files and one file uses a rotation-sequence scalar radius. All have the 126.45-154.55 keV energy window.
- All 1,521 files parse successfully with the current reader and all have one detector stream. All Denoising CountsAccumulated values match pixel sums. One Pre-training file has a CountsAccumulated mismatch and is marked training-ineligible pending vendor-tag interpretation; 992 Pre-training files remain eligible.
- Pair QC under the locked independent-pair rules accepted 146 of 264 filename pairs. It rejected 67 for energy-window mismatch, 46 for motion/alignment thresholds, and 5 for radius differences over 2 mm. No pair is treated as nested.

These findings change the implementation path: Dataset 2 uses an ungated G=1 denoiser head, and the pair manifest is independent-acquisition supervision with target exposure taken from the high file. It is not valid to reshape Dataset 2 into 16 gates or to treat the 30sec filename as a guaranteed 30-second exposure.

---

## 4. Facts that the audit must resolve

These facts cannot be safely inferred from the current single-file workspace. The audit records them rather than allowing code to guess:

| Fact | Required resolution | Blocks |
|---|---|---|
| Dataset roots and supported SOP classes | configuration plus inventory | all data work |
| Dataset 2 pair provenance | nested, independent, or synthetic for every accepted pair | supervised training |
| True gate/view live time | vendor tag/metadata, or documented fallback | calibrated rate estimates |
| Detector identity inconsistency | vendor documentation or confirmed frame mapping | multi-detector modeling |
| Projection and volume handedness | phantom or registered vendor reconstruction | learned reconstruction |
| LEHR FWHM-vs-distance parameters | measurement, manual, or vendor data | resolution recovery |
| Attenuation map and scatter data | paired-series search and registration | AC/SC reconstruction |
| Compute limits | GPU model, memory, and storage recorded in environment report | final batch/worker tuning |

Safe defaults while unresolved:

- one detector stream;
- ideal parallel-hole projector without attenuation, scatter, or PSF;
- batch size 1 with gradient accumulation;
- quarantining instead of imputing malformed acquisitions;
- no learned reconstruction training before geometry validation.

---

## 5. Technology choices and repository layout

### 5.1 Runtime

Use:

- Python 3.11;
- PyTorch with a CUDA build compatible with the target machine;
- pydicom for DICOM;
- NumPy, SciPy, pandas, and PyArrow for audit/manifests;
- Hydra/OmegaConf for configuration composition;
- Matplotlib and seaborn for static QC;
- pytest for tests;
- Ruff for formatting/linting;
- mypy for public data-contract and physics interfaces;
- TensorBoard-compatible event logs.

The bootstrap step MUST resolve and write exact versions to requirements.lock. Package ranges belong in pyproject.toml; exact versions belong only in the lock and environment report. Training scripts use plain PyTorch so checkpoint and optimization behavior are not coupled to a training framework.

### 5.2 Final layout

    pyproject.toml
    requirements.lock
    README.md
    configs/
      audit.yaml
      split.yaml
      model/backbone.yaml
      pretrain.yaml
      finetune_denoiser.yaml
      physics/ideal_parallel.yaml
      physics/lehr_psf.yaml
      osem_baseline.yaml
      train_unrolled_osem.yaml
      evaluation.yaml
    src/gated_spect/
      cli.py
      types.py
      data/
        dicom_reader.py
        frame_indexing.py
        exposure.py
        inventory.py
        pairing.py
        splits.py
        transforms.py
        dataset.py
      physics/
        geometry.py
        operator.py
        siddon.py
        psf.py
        osem.py
        phantoms.py
      models/
        frame_encoder.py
        mixers.py
        projection_backbone.py
        pretraining_head.py
        denoising_head.py
        projection_conditioner.py
        volume_regularizer.py
        unrolled_osem.py
      losses/
        poisson.py
        consistency.py
        reconstruction.py
      training/
        common.py
        pretrain.py
        finetune_denoiser.py
        train_unrolled_osem.py
      evaluation/
        projection_metrics.py
        reconstruction_metrics.py
        bootstrap.py
        reports.py
    tests/
      fixtures/
      test_dicom_reader.py
      test_frame_order.py
      test_exposure.py
      test_pairing.py
      test_splits.py
      test_count_splitting.py
      test_masking.py
      test_model_shapes.py
      test_geometry.py
      test_adjoint.py
      test_osem.py
      test_no_target_leakage.py
      test_smoke_training.py
    artifacts/
      cache/
      manifests/
      splits/
      qc/
      geometry/
      runs/

artifacts is ignored by version control except for small schemas and README files. DICOM files, PHI, secrets, and model checkpoints MUST NOT enter version control.

---

## 6. Coordinate, tensor, count, and exposure contracts

### 6.1 Axis conventions

| Object | Axis order | Meaning |
|---|---|---|
| Projection counts | [G,V,H,W] | gate, view, detector axial row, detector tangential column |
| Batched projections | [B,G,V,C,H,W] | channel is explicit |
| Volume | [G,Z,Y,X] | gate, patient superior-inferior, posterior-anterior, right-left in the declared internal frame |
| Batched volume | [B,G,C,Z,Y,X] | channel is explicit |

The geometry object stores the transform between DICOM patient LPS coordinates and the internal volume frame. Array order alone MUST NOT define physical orientation.

### 6.2 Dtypes

- Raw counts: int32 in the canonical on-disk cache and CPU dataset object.
- Network inputs and model outputs: float32; mixed precision is allowed inside models.
- Exposure and geometry: float32 operationally, float64 in numerical validation.
- Masks: bool.
- Accumulations used for total-count QC: int64.

### 6.3 Canonical acquisition record

    @dataclass(frozen=True)
    class GatedSPECTSample:
        sample_id: str
        counts: Tensor                 # int32 [G,V,H,W]
        frame_exposure_s: Tensor       # float32 [G,V], strictly positive
        relative_exposure_q: Tensor    # float32 [G,V], typical stable frame near 1
        valid_frames: Tensor           # bool [G,V]

        angles_rad: Tensor             # float32 [V], acquisition order
        radii_mm: Tensor               # float32 [V]
        detector_spacing_mm: tuple[float, float]
        view_dwell_s: Tensor           # float32 [V]
        gate_phase_rad: Tensor         # float32 [G]

        energy_window_kev: tuple[float, float]
        scanner_id: str
        collimator_id: str
        stress_rest: str               # stress, rest, unknown
        detector_stream: str           # default detector_0

        patient_key: str               # HMAC, not source PatientID
        study_key: str
        geometry_hash: str
        exposure_source: str           # vendor, derived, or synthetic
        source_path: str               # relative path or opaque source ID
        metadata: dict[str, Any]

Invariants checked at construction:

- counts.shape equals [G,V,H,W];
- exposure, relative_exposure_q, and valid_frames equal [G,V];
- angle/radius length equals V and phase length equals G;
- accepted records have valid_frames.all();
- counts are nonnegative;
- accepted exposure values are finite and greater than zero;
- accepted relative-exposure values are finite and lie in the configured [0.02,1.25] range;
- the geometry hash includes shape, spacing, angle sequence, radius sequence, detector orientation, energy window, and collimator.

valid_frames exists to represent quarantined/investigational data and controlled masking. Version 1 training accepts complete acquisitions only; it does not fill missing clinical frames.

### 6.4 Exposure definition

frame_exposure_s is the estimated live time allocated to a specific gate and view. The expected counts for rate r are:

  \[
  \mu = E r.
  \]

Priority:

1. use vendor frame live time when its semantics are verified;
2. derive it from accepted-beat/gate occupancy metadata;
3. otherwise use the count-based fallback in Section 8.

For fallback exposure, relative gate/view occupancies \(q_{g,v}\) are normalized per view:

  \[
  E_{g,v}=T_v\frac{q_{g,v}}{\sum_{g'}q_{g',v}},
  \]

where \(T_v\) is view dwell time. Thus exposure sums to the known dwell time for every view and a rate remains interpretable as counts per second while that cardiac state is active.

relative_exposure_q stores the pre-normalization occupancy ratio used for low-information thresholds. For verified vendor live time, compute q by dividing each frame live time by the median frame live time among stable gates at that view, then apply the same [0.02,1.25] bounds. This keeps q semantics consistent across vendor and fallback sources without altering the vendor exposure used in likelihoods.

gate_phase_rad uses verified trigger-time bin centers normalized by nominal R-R when those times exist. Otherwise use \(2\pi(g+0.5)/G\) in validated TimeSlotVector order. Record the phase source in metadata.

Training and evaluation MUST use exposure matrices, not labels such as 10, 20, or 30 seconds in likelihood equations. A complementary target uses:

  \[
  E_{\mathrm{target}}=E_{\mathrm{high}}-E_{\mathrm{low}}.
  \]

This eliminates ambiguity when gates have unequal occupancy or nominal duration differs from actual duration.

### 6.5 Network input transform

The immutable statistical target remains integer counts. The input channels are:

1. standardized log1p(input_counts);
2. standardized log(input_exposure_s + 1e-6), spatially broadcast;
3. visibility mask as 0/1.

Compute one scalar mean and standard deviation for log1p counts over all eligible Dataset 1 training bins and one scalar pair for log exposure over eligible frames, using a streaming float64 Welford accumulator. Write them to normalization.json with sample-manifest checksum and eligible element counts. Standardized network channels are clipped to [-8,8]. Exposure is clipped only at 1e-6 before the log transform; likelihood exposure is not standardized or clipped.

No geometric flip, arbitrary image rotation, or view reversal is allowed as augmentation because those operations change acquisition geometry. Count thinning and masking are the default augmentations.

---

## 7. Manifests, identity protection, and split policy

### 7.1 Acquisition manifest

artifacts/manifests/acquisitions.parquet has one row per source acquisition:

- sample_id, dataset_id, relative source path, SOP/Series/Study opaque keys;
- patient_key and study_key;
- stress_rest, scanner_id, software_version, collimator_id;
- G, V, H, W, frame count, BitsStored, pixel spacing;
- start angle, angular increment, direction, scan arc, full angle list;
- radius minimum/maximum and full radius-list artifact path;
- energy windows and auxiliary-series availability;
- view dwell, gate timing, R-R limits, rejected beats;
- total counts, gate/view count artifact path, zero fraction, maximum value;
- detector-vector summary;
- exposure source;
- parse status, QC status, exclusion reason;
- geometry_hash and source checksum.

The manifest contains no raw PatientID, name, birth date, accession number, or absolute clinical path.

### 7.2 Canonical cache

Each accepted sample is cached under artifacts/cache/<sample_id>/ as:

- counts.npy: int32 [G,V,H,W], uncompressed and memory-mappable;
- frame_exposure_s.npy and relative_exposure_q.npy: float32 [G,V];
- valid_frames.npy: bool [G,V];
- geometry.npz: angles, radii, phases, spacing, and coordinate transforms;
- metadata.json: only de-identified fields from the canonical record;
- checksums.json: SHA-256 for every cache file and the source checksum.

Cache creation is atomic: write to a sibling temporary directory, validate invariants and checksums, then rename to the final sample directory. An existing cache with matching checksums is reused; a mismatch is an error rather than an overwrite.

### 7.3 Patient keys

Use HMAC-SHA256 over the institution-scoped stable patient identifier:

  patient_key = HMAC_SHA256(secret, institution + "|" + stable_id)

The secret is supplied through SPECT_HASH_KEY, is never logged, and is never stored beside the manifest. Plain unsalted hashes are not accepted.

### 7.4 Pair manifest

artifacts/manifests/pairs.parquet contains:

- pair_id, patient_key, low_sample_id, high_sample_id;
- provenance: nested, independent, or synthetic;
- low/high geometry hashes;
- low/high exposure totals;
- motion-QC metrics;
- nonnegative-difference result for nested pairs;
- accepted flag and rejection reason.

Every accepted supervised example MUST appear in this table. Pair type is never inferred during model training.

### 7.5 Locked split algorithm

1. Generate Dataset 2 patient groups.
2. Give each patient one stratum key: scanner majority category, stress/rest category (stress, rest, mixed, or unknown), and patient total-count quartile. Compute quartile boundaries from all eligible Dataset 2 patients before model development and store them.
3. Within each stratum, seed-shuffle patients, then greedily assign each patient to the split with the largest remaining stratum target; break ties by largest remaining global target, then fixed order train, validation, test. Use largest-remainder integer targets for 70%, 15%, and 15%.
4. Keep all studies, stress/rest scans, and repeat acquisitions from a patient together.
5. If any Dataset 2 split contains fewer than 10 patients, stop and require a recorded statistical-design decision; do not silently change percentages.
6. Exclude Dataset 2 validation and test patient keys from Dataset 1.
7. Seed-shuffle remaining Dataset 1 patients within scanner/stress strata. Set pretraining-validation size to max(10, round(0.05N)), capped at N-1; if N is below 20, stop for a recorded design decision. Assign the rest to pretraining.
8. Write immutable JSONL files with patient_key, sample_id, split, seed, manifest checksum, and creation time.

Regenerating a split after seeing validation or test results creates a new study version and invalidates prior claims.

---

## 8. Phase 0-1: audit, DICOM ingestion, exposure, and QC

### 8.1 DICOM reader behavior

Public API:

    read_nm_dicom(path: Path, config: ReaderConfig) -> GatedSPECTSample

The reader:

1. verifies an allowed SOP class/modality and decodable pixel data;
2. reads all frames as an array [F,H,W];
3. reads TimeSlotVector and AngularViewVector;
4. verifies both vectors have length F;
5. when FrameIncrementPointer is present, verifies that its declared indexing dimensions are consistent with the vectors used;
6. verifies each vector contains the contiguous one-based label set 1 through max(label), then converts to zero-based indices;
7. accepts DetectorVector only when it contains one unique detector label in version 1; multiple unique streams raise UnsupportedDetectorLayout;
8. verifies unique gate-view pairs and complete Cartesian coverage;
9. maps each frame into [G,V,H,W];
10. extracts angles, direction, radii, spacing, gate timing, view dwell, energy window, collimator, and detector metadata;
11. computes exposure and geometry hashes;
12. returns the immutable record or a typed rejection.

Required typed failures include UnsupportedSOPClass, UnsupportedDetectorLayout, PixelDecodeError, MissingFrameIndex, NonContiguousFrameIndex, DuplicateFrame, MissingFrame, InvalidPixelValue, InvalidExposure, and UnresolvedGeometry.

Frame construction:

    counts = zeros((G, V, H, W), int32)
    occupied = zeros((G, V), bool)
    for frame, gate, view in zip(pixels, gate_index, view_index):
        if occupied[gate, view]:
            raise DuplicateFrame(gate, view)
        counts[gate, view] = frame
        occupied[gate, view] = True
    if not occupied.all():
        raise MissingFrame(missing_indices)

G and V are derived from validated index labels. The expected [16,32,64,64] shape is a protocol check, not a parser assumption.

If RescaleSlope/Intercept are present, accepted raw-count data require slope 1 and intercept 0. Any other transform is quarantined until its count semantics are documented.

### 8.2 Angle and detector orientation

- Preserve acquisition order in angles_rad.
- Clockwise sequences decrement the angle under the provisional convention.
- Do not connect the endpoints of a 180-degree orbit with circular padding.
- radii_mm stores one value per view.
- For ideal parallel-hole projection, radius does not change ray direction; it is retained for distance-dependent PSF and QC.
- A geometry_verified flag remains false until the orientation validation gate passes.
- Geometry hashes use canonical JSON with angles rounded to 1e-6 rad and distances/spacings rounded to 1e-3 mm. Hashing code and its schema version are included in the hash input.

### 8.3 Count-based exposure fallback

Use only when verified live-time metadata is absent.

1. Compute \(c_{g,v}=\sum_{h,w}y_{g,v,h,w}\).
2. Compute gate totals \(C_g=\sum_v c_{g,v}\).
3. Let \(m=\operatorname{median}(\log(C_g+0.5))\) and \(d=1.4826\operatorname{median}(|\log(C_g+0.5)-m|)\).
4. Select stable gates satisfying \(|\log(C_g+0.5)-m|\le\max(2.5d,\log(1.25))\).
5. If fewer than half the gates are stable, reject derived exposure and require manual review.
6. For each view, compute \(b_v=\operatorname{median}_{g\in G_{\mathrm{stable}}}c_{g,v}\).
7. Compute \(q^{raw}_{g,v}=\operatorname{clip}(c_{g,v}/\max(b_v,1),0.02,1.25)\).
8. For each gate, compute the view-wise coefficient of robust variation as \(1.4826\,MAD(q^{raw}_{g,:})/\max(median(q^{raw}_{g,:}),10^{-6})\), and compare it with 0.10:
   - at or below 0.10, replace the row with its median;
   - above 0.10, apply a non-circular width-5 median filter along view and flag view-varying occupancy.
9. Normalize q to frame_exposure_s with the equation in Section 6.4.

This estimator is a documented fallback and may absorb small physiological count changes. exposure_source is set to derived, and all final metrics are stratified by exposure source.

Frames with \(q_{g,v}<0.20\) are allowed as model inputs but have zero target-loss weight during initial training. They remain in gate-specific evaluation.

### 8.4 QC artifacts

Produce one HTML report and machine-readable JSON per acquisition with:

- total counts and CountsAccumulated comparison when its semantics are verified;
- gate totals, view totals, and [G,V] count heatmap;
- [G,V] exposure heatmap and exposure source;
- projection montage for every gate;
- zero fraction, maximum value, and saturation warning when max equals the BitsStored limit;
- missing/duplicate/index-range results;
- angle/radius plots and geometry hash;
- detector-header discrepancy;
- abrupt view-to-view count changes, flagged when the log count difference exceeds five robust MADs within a gate;
- pair link and pair QC, when applicable.

Initial thresholds:

- no negative count;
- no duplicate or missing frame;
- all accepted exposure finite and positive;
- exact pixel-sum agreement with a verified total-count tag;
- saturation warning at the representable maximum;
- parse success at least 99% of intended in-scope files after documented exclusions;
- manual review of min(50, number of accepted acquisitions) reports spanning both datasets, scanners, stress/rest, low/high count, and low-exposure gates.

### 8.5 Pair verification

**Nested pair**

- geometry hashes MUST match;
- metadata or acquisition provenance MUST establish that low is a subset of high;
- counts_high - counts_low MUST be nonnegative in every bin;
- exposure_high - exposure_low MUST be positive in every accepted frame.

The difference becomes the independent target.

**Independent pair**

- geometry hashes MUST match;
- gate definitions and energy windows MUST match;
- for each view, sum over gates, divide by summed exposure, apply log1p and a 2D Hann window, then estimate detector-plane translation by phase correlation;
- accept only if median absolute shift is at most 1 detector bin and 95th percentile is at most 2 bins;
- do not warp count data by interpolation for the primary analysis; reject or analyze misaligned pairs separately.

**Synthetic pair**

- generate low counts on the fly from high counts;
- for desired low exposure \(E_l\) and source exposure \(E_h\), sample each bin with \(p_{g,v}=E_l/E_h\);
- require \(0<p_{g,v}<1\);
- low = Binomial(high,p), target = high-low, target exposure = high exposure-low exposure;
- never store one fixed thinning as the only training realization.

Phase acceptance gate: no unresolved frame-order issue, every supervised pair has accepted provenance/QC, and the split files are locked.

---

## 9. Phase 2: projection backbone

### 9.1 Input batch

    counts_input       float32 [B,G,V,1,H,W]
    input_exposure_s   float32 [B,G,V]
    target_exposure_s  float32 [B,G,V] or absent
    visibility         bool    [B,G,V,H,W]
    relative_exposure_q float32 [B,G,V]
    valid_frames       bool    [B,G,V]
    angles_rad         float32 [B,V]
    radii_mm           float32 [B,V]
    gate_phase_rad     float32 [B,G]
    split_fraction     float32 [B,1]
    scanner_index      int64   [B]
    collimator_index   int64   [B]
    stress_rest_index  int64   [B]

Unknown metadata categories map to index 0. Metadata vocabularies are built from training data only and saved with checkpoints.

### 9.2 Exact default architecture

**Frame encoder**

Apply one shared 2D encoder independently to B x G x V frames:

- 3x3 convolution, 3 to 32 channels;
- two residual blocks at 32 channels, 64x64;
- stride-2 3x3 convolution, 32 to 48; two residual blocks, 32x32;
- stride-2 3x3 convolution, 48 to 64; two residual blocks, 16x16;
- stride-2 3x3 convolution, 64 to 96; two residual blocks, 8x8.

Each residual block uses GroupNorm with 8 groups, SiLU, 3x3 convolution, GroupNorm, SiLU, 3x3 convolution. Convolutions use padding 1. There is no BatchNorm because patient batch size is small.

**Metadata embedding**

At each bottleneck token, add the output of a two-layer MLP from:

- sin(angle), cos(angle);
- standardized radius;
- sin(gate phase), cos(gate phase);
- standardized log input exposure;
- split fraction.

Categorical scanner, collimator, and stress/rest embeddings are each 8-dimensional, concatenated, projected to 96, and added. A category embedding is disabled if the training manifest contains only one known category.

**Angular and temporal mixing**

Use four pre-normalized transformer blocks in order A-T-A-T:

- width 96;
- four attention heads;
- feed-forward width 384;
- GELU;
- attention and residual dropout 0.10.

Angular attention operates over V for every [B,G,H/8,W/8] location and uses valid-view masks. Temporal attention operates over G for every [B,V,H/8,W/8] location and uses valid-gate masks. Angles and phases enter through metadata embeddings. Angular attention has no circular endpoint connection. Gate phase is periodic through sine/cosine embeddings, but attention still receives exposure/validity masks.

**Outputs**

- dense bottleneck features [B,G,V,96,H/8,W/8];
- frame-encoder skips at 64, 32, and 16 resolution;
- global representation: validity-weighted mean of bottleneck tokens, LayerNorm, MLP 96 to 256 to 256;
- optional task decoder outputs.

### 9.3 Projection decoder

Use a U-Net-style decoder per gate/view:

- upsample 8 to 16, concatenate the 64-channel skip, residual blocks to 64 channels;
- upsample 16 to 32, concatenate the 48-channel skip, residual blocks to 48;
- upsample 32 to 64, concatenate the 32-channel skip, residual blocks to 32;
- 1x1 output convolution to one rate channel.

Bilinear upsampling plus 3x3 convolution is preferred over transposed convolution. Rate output is:

  \[
  \hat r=\operatorname{softplus}(a)+10^{-6}.
  \]

The projection-conditioning head uses the same decoder topology with a separate 1x1 output to eight signed feature channels.

### 9.4 Model tests

- exact shape tests for the sample protocol and synthetic variable-G/V/H/W cases;
- padding/mask invariance for padded gates/views;
- no NaN/Inf under all-zero input with positive exposure;
- positive rate output;
- gradients reach frame, angular, and temporal blocks;
- parameter count reported and required to remain between 1 and 5 million for the specified backbone unless an architecture change is recorded.

Version 1 accepts detector H and W divisible by eight. Other sizes are quarantined rather than silently resized; adding pad/crop support requires a versioned data-contract change. The listed 64, 32, 16, and 8 spatial sizes are the exact default for the inspected 64x64 protocol.

---

## 10. Phase 3: self-supervised pretraining

### 10.1 Count splitting

For each raw bin \(y\):

  \[
  y^{(a)}\sim\operatorname{Binomial}(y,p), \qquad y^{(b)}=y-y^{(a)},
  \]

with one \(p\sim U(0.4,0.7)\) per acquisition per epoch.

Student input is \(y^{(a)}\) with exposure \(pE\). The prediction target is \(y^{(b)}\) with exposure \((1-p)E\). A fresh split is generated in the dataset worker every epoch from a deterministic seed derived from global seed, epoch, sample ID, and worker ID.

Required tests:

- a+b equals y exactly;
- neither split is negative;
- Monte Carlo mean and variance match binomial expectations;
- samples change across epochs and reproduce under the same seed.

### 10.2 Masking distribution

Choose exactly one primary mask per sample:

| Task | Probability | Exact initial rule |
|---|---:|---|
| Whole views | 0.50 | hide min(3,V-1) unique views across all gates |
| Angular block | 0.25 | hide two adjacent views without endpoint wrap |
| Detector patches | 0.15 | hide 25% of 8x8 tiles; reuse tile positions across gates to prevent temporal copying |
| Complete gate | 0.10 | hide one gate whose median q is at least 0.20 |

Masked input bins are set to zero and visibility is false. Input exposure remains \(pE\) so the model knows sampling effort; visibility distinguishes missing evidence from observed zeros. If a sampled mask has no eligible target bin, resample once, then skip the sample with a logged reason.

Select an additional 5% uniform random subset of visible eligible bins for the visible-bin loss.

### 10.3 Poisson loss

For target count y and mean \(\mu\), train with constant-free Poisson NLL:

  \[
  \ell(\mu,y)=\mu-y\log(\mu+10^{-8}).
  \]

Report Poisson deviance:

  \[
  D=2\left[\mu-y+y\log\frac{y}{\mu}\right],
  \]

with the final term defined as zero for y=0.

Loss reduction is a weighted mean over selected bins, followed by an unweighted mean over patients. Frames with q below 0.20 have target weight zero. This prevents high-count patients or frames from dominating by bin count.

For predicted rate \(\hat r\):

  \[
  \mu^{(b)}=(1-p)E\hat r.
  \]

### 10.4 Representation and gate consistency

**Global latent consistency**

- Create two independently thinned and masked inputs.
- Pass their 256-dimensional global representations through a two-layer projection MLP and predictor MLP using LayerNorm, never BatchNorm.
- Use symmetric stop-gradient cosine loss as in a SimSiam-style objective.
- Maintain a FIFO queue of the previous 127 stop-gradient projected representations and combine it with the current representation for the variance estimate.
- After the queue contains at least 31 entries, add \(\operatorname{mean}(\max(0,0.1-\operatorname{std}(z)))\); before then the variance term is zero.
- Update the queue only after a successful optimizer step and store/restore it in training checkpoints. In distributed training, all-gather the newly detached representations before queue insertion.

**Hidden-gate consistency**

- Maintain an exponential-moving-average teacher with decay 0.996.
- The teacher sees the independent complement with no artificial gate mask.
- Pool one teacher bottleneck vector per gate.
- The student hidden-gate output at the same phase is matched to the stop-gradient teacher vector with cosine loss.
- Apply only to gates with q at least 0.20.

The EMA teacher is an optimization aid and is not the saved reusable artifact.

### 10.5 Objective

  \[
  L_{\mathrm{SSL}} =
  L_{\mathrm{mask}}
0.25L_{\mathrm{visible}}
0.05L_{\mathrm{latent}}
0.05L_{\mathrm{gate}}
\lambda_{\mathrm{physics}}L_{\mathrm{physics}}.
  \]

Default \(\lambda_{\mathrm{physics}}=0\). After the projector acceptance gate, a separate experiment MAY ramp it linearly from 0 to 0.10 over 20 epochs. Physics pretraining never replaces the masked count objective.

### 10.6 Training defaults

| Setting | Default |
|---|---:|
| Optimizer | AdamW |
| Peak learning rate | 2e-4 |
| Weight decay | 0.05 |
| Warmup | 10 epochs |
| Schedule | cosine to 1e-6 |
| Physical patient batch | 1 |
| Effective batch | 8 by gradient accumulation |
| Precision | AMP mixed precision |
| Gradient clip norm | 1.0 |
| Maximum epochs | 250 |
| Validation frequency | every epoch |
| Early-stop patience | 30 epochs after warmup |
| EMA decay | 0.996 |

Use DistributedDataParallel when multiple GPUs exist. Do not use DataParallel.

Checkpoint selection uses lowest Dataset 1 validation masked Poisson deviance. Latent metrics and calibration are reported as guardrails; a checkpoint is rejected if global representation standard deviation is below 0.01 or any validation output contains NaN/Inf.

Saved pretraining artifact:

- projection-backbone state;
- architecture and normalization files;
- metadata vocabularies;
- split and acquisition-manifest checksums;
- code revision, lockfile checksum, seeds, and environment report;
- validation metrics and selected epoch.

The pretraining decoder and EMA teacher are not part of the reusable checkpoint.

---

## 11. Phase 4: supervised projection denoising

### 11.1 Training example contract

Each example contains:

    low_counts          int32   [G,V,H,W]
    low_exposure_s      float32 [G,V]
    target_counts       int32   [G,V,H,W]
    target_exposure_s   float32 [G,V]
    target_weight       float32 [G,V]
    geometry and metadata
    pair_id and provenance

Construction:

- **Nested:** target_counts = high_counts-low_counts and target_exposure = high_exposure-low_exposure.
- **Independent:** target_counts = high_counts and target_exposure = high_exposure.
- **Synthetic:** binomially generate low and use the complement as target.

Training code receives the constructed target and MUST NOT branch on filenames, nominal duration text, or hidden access to high counts. test_no_target_leakage changes the high target while holding the low input fixed and verifies the model input is byte-identical.

### 11.2 Model initialization

- Load the selected Dataset 1 backbone.
- Initialize a new rate decoder.
- Epochs 1-8: train the decoder plus the second angular and second temporal blocks; freeze the frame encoder and first A/T blocks.
- From epoch 9: unfreeze all backbone parameters.
- The scratch comparison uses the same architecture, optimizer, batches, and schedule with random initialization.

### 11.3 Objective

Input is low_counts with low_exposure. Predicted target mean is:

  \[
  \hat\mu_{\mathrm{target}}=E_{\mathrm{target}}\hat r.
  \]

The default objective is only the weighted Poisson NLL:

  \[
  L_{\mathrm{FT}}=\operatorname{mean}_{eligible}\ell(\hat\mu_{\mathrm{target}},y_{\mathrm{target}}).
  \]

An optional log-count Huber term is disabled in the primary experiment and may be enabled only as an ablation with weight 0.05 and delta 0.2. SSIM and pixel MSE are not training losses.

Target eligibility requires:

- valid pair/frame;
- target exposure greater than zero;
- occupancy q at least 0.20;
- no pair-QC rejection.

### 11.4 Training defaults

| Setting | Decoder/upper blocks | Full fine-tuning |
|---|---:|---:|
| Epochs | 8 | up to 72 additional |
| Backbone LR | 1e-5 for unfrozen blocks | 1e-5 |
| Decoder LR | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| Warmup | 2 epochs | none |
| Schedule | cosine | cosine |
| Effective batch | 8 | 8 |
| Early-stop patience | not used | 15 validation checks |
| Gradient clip | 1.0 | 1.0 |

Select the checkpoint with lowest Dataset 2 validation Poisson deviance. In a tie within 0.1%, select lower absolute total-count bias.

### 11.5 Inference contract

The denoiser returns rate and optional conditioning features:

    output = denoiser(low_counts, low_exposure_s, geometry, metadata)
    rate_cps = output.rate
    expected_counts = requested_exposure_s[..., None, None] * rate_cps

requested_exposure_s is built from a requested view dwell and a declared gate-occupancy template. A scalar such as 30 MUST NOT be multiplied directly by rate without constructing the [G,V] exposure matrix.

For paired validation/test inference, use the actual high-acquisition exposure matrix so predictions and acquired targets have identical exposure. For a hypothetical duration without a paired acquisition, use the same patient's low-acquisition relative_exposure_q, normalize it per view with Section 6.4, and record that choice. A cohort-median occupancy template is allowed only when patient occupancy is unavailable and must be labeled inferred.

Expected counts remain floating point and are not sampled or rounded for evaluation. Conventional OSEM applied to denoised projections is explicitly labeled as model-mean OSEM.

### 11.6 Denoising progression gate

On Dataset 2 validation, compare with the exposure-scaled raw baseline
\(\hat r_{\mathrm{naive}}=y_{\mathrm{low}}/\max(E_{\mathrm{low}},10^{-8})\) and
\(\hat\mu_{\mathrm{target}}=E_{\mathrm{target}}\hat r_{\mathrm{naive}}\):

- mean patient-level Poisson deviance improves by at least 5%;
- absolute total-count bias is at most 3%;
- absolute gate-level count bias is at most 5% for at least 95% of eligible gates;
- no scanner/protocol subgroup with at least 10 patients degrades by more than 5% in deviance;
- pretrained validation deviance is no worse than the identically trained scratch model.

A claim that pretraining improves performance additionally requires the patient-bootstrap 95% confidence interval for pretrained-minus-scratch deviance to lie below zero. Pretrained-versus-scratch results are reported even when this criterion fails.

Failure does not permit threshold tuning on the test set. It triggers error analysis or a new versioned development experiment.

---

## 12. Phase 5: projector and conventional OSEM

### 12.1 Operator interface

    class SPECTProjector(Protocol):
        def forward(
            self, activity_rate: Tensor, geometry: Geometry
        ) -> Tensor:
            # [B,G,Z,Y,X] -> [B,G,V,H,W]

        def adjoint(
            self, projections: Tensor, geometry: Geometry
        ) -> Tensor:
            # [B,G,V,H,W] -> [B,G,Z,Y,X]

forward and adjoint exclude exposure. Exposure is applied by the statistical model and OSEM sensitivity calculation. The operator is linear, fixed during primary training, and supplies both float64 CPU validation and float32 operational paths.

### 12.2 Reference ideal projector

Implement an ideal parallel-hole projector first:

- reconstruction grid: 64 x 64 x 64;
- voxel spacing: 6.4 mm isotropic;
- detector sampling: from DICOM, initially expected 6.4 x 6.4 mm;
- 2D ray integration independently for each axial plane using a Siddon-style sparse system matrix;
- axial detector row maps through the declared geometry transform, with no axial interpolation when grids match;
- one sparse matrix per unique geometry hash;
- operational forward uses sparse matrix multiplication and adjoint uses its exact transpose.

This reference is deliberately simple and testable. For an ideal parallel-hole collimator, orbit radius does not change the ray line; the radius is used by the PSF model below.

Cache matrices under artifacts/geometry using a hash of angles, grid, spacing, detector bins, orientation, algorithm version, and numerical tolerance. Never reuse a cache when the hash differs.

### 12.3 Progressive physical effects

1. ideal ray geometry and detector orientation;
2. exposure-aware sensitivity;
3. distance-dependent LEHR response;
4. attenuation, only with a registered validated map;
5. scatter, only with measured or independently validated estimates.

LEHR response uses the calibrated form:

  \[
  \operatorname{FWHM}(d)=a+bd,\qquad
  \sigma(d)=\frac{\operatorname{FWHM}(d)}
  {2\sqrt{2\log 2}}.
  \]

The depth-dependent implementation groups ray depth into eight initial slabs, applies the appropriate axial/tangential Gaussian response before summing contributions, and implements the adjoint by reversing the same operations. The number of slabs is increased until a 16-slab reference changes phantom projections by less than 1%.

No numeric a or b is guessed. psf.mode remains none until calibration provenance is stored. The same rule applies to attenuation and scatter.

### 12.4 Forward model

  \[
  \mu=E(Ax+s).
  \]

- x is activity/count rate on the gated volume grid. With s=0 on uncorrected raw data, x also absorbs scatter and other unmodeled background; it is not quantitatively corrected activity.
- A is the selected fixed projector.
- s is scatter rate and defaults exactly to zero.
- E is frame exposure, broadcast over detector bins.

### 12.5 OSEM update

For subset S and current positive volume x:

  \[
  \mu_S=E_S(A_Sx+s_S),
  \]

  \[
  x \leftarrow x
  \frac{A_S^T\left(E_S\,y_S/\max(\mu_S,10^{-8})\right)}
       {\max(A_S^T E_S,10^{-8})}.
  \]

This form intentionally keeps exposure in both the mean and sensitivity. With zero scatter, the numerator is algebraically equivalent to \(A_S^T[y_S/(A_Sx)]\), but the implemented form is clearer and supports nonzero scatter.

For 32 views, default subsets are interleaved:

- S0 = [0,4,8,12,16,20,24,28]
- S1 = [1,5,9,13,17,21,25,29]
- S2 = [2,6,10,14,18,22,26,30]
- S3 = [3,7,11,15,19,23,27,31]

For other V, set \(K=\min(4,V)\), assign view v to subset v mod K, and require every subset nonempty. Subset sizes may differ by one. One OSEM iteration means one complete sweep through all subsets.

Initialization:

1. start with an all-ones positive volume inside a cylindrical FOV mask;
2. compute predicted primary total for that unit volume;
3. scale it by \(\max(\sum y-\sum Es,10^{-8})/\max(\sum EA1,10^{-8})\);
4. clamp to at least 1e-8.

Default baseline:

- four subsets;
- eight full iterations;
- no post-filter for primary quantitative comparison;
- optional Gaussian post-filter selected only on validation and reported with FWHM;
- reconstruction of summed ungated data and all individual gates.

### 12.6 Required validation

The ideal projector MUST pass:

- float64 adjoint relative error at most 1e-4 over five random trials;
- float32 adjoint relative error at most 1e-3 over five trials;
- point-source centroid error below 0.5 voxel in noiseless synthetic tests;
- point-source FWHM error below 10% of the discretized reference;
- uniform-cylinder noiseless reprojection normalized RMSE below 1%;
- correct recovery of left/right, anterior/posterior, and superior/inferior asymmetric markers;
- finite nonnegative OSEM output;
- non-increasing full-data Poisson NLL for ten MLEM iterations;
- sensitivity-cache equality to uncached calculation.

The real-data geometry gate additionally requires:

- phantom or vendor comparison confirms angle direction and detector-column direction;
- registered myocardial/phantom centroid agrees within one voxel;
- reprojection residuals from min(10, validation-patient count) cases selected deterministically across scanner, stress/rest, and count quartiles have no systematic angular reversal or axial flip;
- the validation report is signed off and geometry_verified is set true.

Vendor image intensity need not match because corrections and filtering may differ. Orientation and spatial correspondence are the purpose of that comparison.

Unrolled reconstruction training is blocked until these criteria pass.

---

## 13. Phase 6: OSEM integration strategies

### 13.1 Required baseline: denoise then OSEM

1. Estimate rate from original low-count projections.
2. Form expected projections at a declared exposure matrix.
3. Reconstruct them with the same geometry, iterations, subsets, and filtering policy used for raw low-count OSEM.

Because these are model means rather than Poisson measurements, this branch is a baseline only. The report MUST not describe its input as acquired high-count data.

### 13.2 Preferred model: projection-conditioned unrolled OSEM

Inputs:

- original low_counts and low_exposure;
- fixed validated projector;
- pretrained projection backbone;
- patient geometry and metadata.

Initialization is one full conventional OSEM iteration. Each of five unrolled stages performs:

1. one full four-subset OSEM sweep using original low counts;
2. projection-residual construction;
3. projection-feature lifting into volume space;
4. one shared 3D spatiotemporal regularizer update.

For stage k:

  \[
  u^k=\operatorname{OSEMFullSweep}(x^k,y_{\mathrm{low}},E_{\mathrm{low}},A),
  \]

  \[
  \rho^k=\frac{y_{\mathrm{low}}-\mu^k}{\sqrt{\mu^k+1}},
  \]

  \[
  b^k=\frac{A^T(E_{\mathrm{low}}\rho^k)}
  {\sqrt{A^TE_{\mathrm{low}}+10^{-8}}}.
  \]

The projection conditioner produces eight signed full-resolution detector feature maps per gate/view. Lift each channel with:

  \[
  c_j=\frac{A^T(E_{\mathrm{low}}f_j)}
  {A^TE_{\mathrm{low}}+10^{-8}}.
  \]

The regularizer input per gate has 11 channels:

1. log(u+1e-8);
2. standardized log sensitivity;
3. normalized backprojected residual b;
4. eight lifted projection-feature channels.

### 13.3 Volume regularizer

Use one shared 3D U-Net per gate with temporal mixing at the bottleneck:

- input 11 to 24 channels, two 3D residual blocks at 64 cubed;
- downsample to 48 channels at 32 cubed, two blocks;
- downsample to 96 channels at 16 cubed, two blocks;
- temporal attention over G at every bottleneck voxel, width 96, four heads, feed-forward width 384;
- symmetric decoder with skip connections;
- one signed output channel \(\delta^k\).

All convolution weights are shared across gates. The temporal block processes gates jointly and uses phase/exposure masks.

Use a bounded multiplicative update:

  \[
  x^{k+1}=\max\left(
  u^k\exp\left(0.5\tanh(\delta^k)\right),10^{-8}
  \right).
  \]

This preserves positivity and limits a learned stage to a multiplicative factor in \([\exp(-0.5),\exp(0.5)]\), or [0.6065,1.6487] to four decimals. The limit is configurable but fixed before final evaluation.

Regularizer weights are shared across all five stages in the primary model. Unshared weights are an ablation. Gradients pass through every OSEM and regularizer stage; no stage output is detached. Activation checkpointing is enabled for every 3D encoder/decoder level and unrolled stage by default.

---

## 14. Phase 7: train the unrolled reconstruction model

### 14.1 Independent projection supervision

The model input is always low_counts. The target is constructed by the same pair logic as denoising.

  \[
  \hat x=f_{\mathrm{unrolled}}(y_{\mathrm{low}},E_{\mathrm{low}},A),
  \]

  \[
  \hat\mu_{\mathrm{target}}
  =E_{\mathrm{target}}(A\hat x+s).
  \]

Primary loss:

  \[
  L_{\mathrm{recon}}=
  \operatorname{PoissonNLL}
  (y_{\mathrm{target}},\hat\mu_{\mathrm{target}})
  +10^{-4}\operatorname{mean}_{k}((\delta^k)^2).
  \]

The correction penalty is for optimization stability, not image ground truth. No reconstructed-volume target is required.

### 14.2 Training schedule

| Phase | Epochs | Trainable parameters | LR |
|---|---:|---|---:|
| A | 1-20 | volume regularizer and projection conditioner | 1e-4 |
| B | 21-40 | above plus final A/T backbone blocks | 5e-5 regularizer, 5e-6 backbone |
| C | 41-80 | full model if validation is stable | 2e-5 regularizer, 1e-6 backbone |

Other defaults:

- AdamW, weight decay 0.01;
- physical and effective batch size 1;
- gradient clip 1.0;
- AMP, except projector validation;
- early-stop patience 15 after phase A;
- lowest validation target-projection Poisson deviance selects checkpoint;
- fixed projector, exposure, subset definitions, and correction configuration.

Phase C begins only if phase B has no NaN/Inf, validation deviance has not worsened for five consecutive checks, and count bias is within 5%. Otherwise the backbone remains partially frozen and the reason is recorded.

### 14.3 Reconstruction progression gate

Compared with raw low-count conventional OSEM on Dataset 2 validation:

- target-projection Poisson deviance improves at least 5%;
- total reprojection count bias is at most 3%;
- output is finite and nonnegative for every case;
- no subgroup with at least 10 patients degrades more than 5% in deviance;
- image QC finds no systematic orientation, truncation, checkerboard, or gate-flicker artifact.

This is a development gate, not a claim of clinical superiority.

---

## 15. Evaluation protocol

### 15.1 Locked comparisons

Use the same locked Dataset 2 test patients:

1. conventional OSEM from original low-count data;
2. denoised model-mean projections followed by conventional OSEM;
3. projection-conditioned unrolled OSEM from original low-count data;
4. conventional OSEM from acquired high-count data;
5. vendor reconstruction when available and comparable.

The OSEM iteration/subset/filter choice for each baseline is selected on validation before test evaluation. High-count OSEM is a reference, not assumed ground truth.

### 15.2 Projection metrics

Primary:

- independent-target Poisson deviance per patient;
- total count-rate bias:

  \[
  100\frac{\sum\hat\mu-\sum y}{\sum y};
  \]

- gate-level and view-level count bias.

Secondary:

- mean absolute error in rate space;
- residual mean and variance within predicted-count deciles;
- angular count-profile error;
- metrics stratified by q, especially q below 0.20 and gates 15-16;
- SSIM/PSNR on log1p counts, clearly labeled descriptive.

### 15.3 Reconstruction metrics

Technical primary:

- independent/high-count projection Poisson deviance after forward projection;
- reprojection bias and residuals by gate and angle;
- background coefficient of variation;
- wall sharpness measured with one predefined profile method;
- gate-to-gate ventricular-volume curve smoothness.

Clinical/derived, only when validated reference methods exist:

- myocardial-to-background contrast;
- EDV, ESV, and LVEF agreement;
- polar-map agreement;
- defect contrast/recovery;
- blinded expert artifact and diagnostic-quality ratings.

The segmentation method and any thresholds MUST be selected using training/validation data and frozen before the test set.

### 15.4 Statistical analysis

- Unit of resampling and inference: patient.
- Paired method differences: report mean, median, and 95% confidence interval.
- Bootstrap: 10,000 patient-level resamples, seed 20260806.
- Report all patients and predeclared strata; do not hide small unfavorable strata.
- A subgroup significance claim requires at least 20 patients; 10-19 is descriptive only.
- Missing metrics and failed reconstructions are reported, not silently removed.
- Final test evaluation runs once from a script that refuses to overwrite an existing locked result without a new study version.

---

## 16. Required ablations

Run on the locked validation protocol, then evaluate preselected ablations on test only if declared before unlocking:

1. random initialization versus Dataset 1 pretraining;
2. spatial only versus spatial+angular;
3. spatial+angular versus spatial+angular+temporal;
4. no exposure channel/loss weighting versus full exposure handling;
5. masked prediction versus masked+count-split+latent consistency;
6. physics-consistency pretraining off versus on after validation;
7. frozen, partial, and full backbone fine-tuning;
8. raw OSEM, denoise-then-OSEM, and projection-conditioned unrolled OSEM;
9. shared versus unshared unrolled-stage regularizer;
10. target-loss masking versus inclusion of q below 0.20.

The primary scientific ablation is pretrained versus scratch with identical Dataset 2 examples and optimization budget.

---

## 17. Configuration defaults

All values below appear in Hydra configs and are copied into each run directory.

### 17.1 audit.yaml

    seed: 20260806
    allowed_modalities: [NM]
    require_complete_gate_view_grid: true
    raw_rescale_slope: 1.0
    raw_rescale_intercept: 0.0
    exposure:
      priority: [vendor, beat_metadata, count_fallback]
      q_clip: [0.02, 1.25]
      low_information_q: 0.20
      view_variation_threshold: 0.10
    qc:
      manual_review_minimum: 50
      parse_success_minimum: 0.99

### 17.2 split.yaml

    seed: 20260806
    dataset2:
      train_fraction: 0.70
      validation_fraction: 0.15
      test_fraction: 0.15
      minimum_patients_per_split: 10
      strata: [scanner, stress_rest, count_quartile]
    dataset1:
      validation_fraction: 0.05
      minimum_validation_patients: 10
      exclude_dataset2_splits: [validation, test]

### 17.3 pretrain.yaml

    seed: 20260806
    model: backbone
    count_split_p: [0.4, 0.7]
    visible_bin_fraction: 0.05
    loss_weights:
      mask: 1.0
      visible: 0.25
      latent: 0.05
      gate: 0.05
      physics: 0.0
    optimizer:
      name: adamw
      lr: 0.0002
      weight_decay: 0.05
    epochs: 250
    warmup_epochs: 10
    early_stop_patience: 30
    physical_batch_size: 1
    accumulation_steps: 8
    amp: true
    activation_checkpointing: true
    gradient_clip: 1.0

### 17.4 finetune_denoiser.yaml

    seed: 20260806
    freeze_epochs: 8
    max_epochs: 80
    backbone_lr: 0.00001
    decoder_lr: 0.0001
    weight_decay: 0.01
    early_stop_patience: 15
    structural_loss_weight: 0.0
    low_information_q: 0.20
    physical_batch_size: 1
    accumulation_steps: 8
    amp: true

### 17.5 osem_baseline.yaml

    grid_shape: [64, 64, 64]
    voxel_spacing_mm: [6.4, 6.4, 6.4]
    projector: ideal_parallel
    attenuation: none
    scatter: none
    psf: none
    subsets: 4
    iterations: 8
    epsilon: 1.0e-8
    post_filter_fwhm_mm: null

### 17.6 train_unrolled_osem.yaml

    seed: 20260806
    stages: 5
    initialization_osem_iterations: 1
    osem_sweeps_per_stage: 1
    shared_regularizer: true
    projection_feature_channels: 8
    max_log_update: 0.5
    correction_penalty: 0.0001
    max_epochs: 80
    early_stop_patience: 15
    physical_batch_size: 1
    accumulation_steps: 1
    amp: true
    activation_checkpointing: true

---

## 18. Public APIs and command sequence

### 18.1 Required public APIs

    build_inventory(config) -> Path
    read_nm_dicom(path, config) -> GatedSPECTSample
    estimate_exposure(counts, metadata, config) -> ExposureResult
    verify_pair(low, high, provenance, config) -> PairResult
    create_locked_splits(acquisitions, pairs, config) -> SplitArtifacts

    ProjectionBackbone.forward(batch) -> BackboneOutput
    ProjectionDenoiser.forward(batch) -> DenoiserOutput

    SPECTProjector.forward(x, geometry) -> projections
    SPECTProjector.adjoint(y, geometry) -> volume
    osem_reconstruct(y, exposure, geometry, config) -> volume
    UnrolledOSEM.forward(batch) -> ReconstructionOutput

All public tensor APIs document units, shapes, dtype, device, and masking behavior in docstrings.

### 18.2 Execution sequence

    python -m gated_spect.cli environment-report
    python -m gated_spect.cli inventory --config-name audit
    python -m gated_spect.cli qc --config-name audit
    python -m gated_spect.cli verify-pairs --config-name audit
    python -m gated_spect.cli create-splits --config-name split
    pytest -q

    python -m gated_spect.cli pretrain --config-name pretrain
    python -m gated_spect.cli finetune-denoiser --config-name finetune_denoiser

    python -m gated_spect.cli build-projector-cache --config-name osem_baseline
    python -m gated_spect.cli validate-projector --config-name osem_baseline
    python -m gated_spect.cli reconstruct-osem --config-name osem_baseline

    python -m gated_spect.cli train-unrolled --config-name train_unrolled_osem
    python -m gated_spect.cli evaluate --config-name evaluation

Dataset roots and artifact roots are passed through configuration or environment-specific config files excluded from version control. Commands fail fast when a required manifest, checksum, split, geometry-validation flag, or checkpoint is absent.

---

## 19. Test and continuous-integration gates

### 19.1 Unit tests

- DICOM vector indexing, one-based conversion, duplicate/missing rejection;
- exact count preservation and dtype;
- exposure normalization: sum over gates equals view dwell;
- count-fallback stable-gate selection and low-q behavior;
- nested complement exactness and synthetic thinning;
- patient grouping and cross-dataset exclusion;
- mask rules and no angular endpoint wrap;
- Poisson NLL/deviance against hand-computed cases;
- model shapes, positivity, padding invariance, and gradients;
- system-matrix cache invalidation;
- projector adjointness;
- OSEM sensitivity, positivity, and MLEM likelihood behavior;
- target leakage;
- checkpoint round-trip and one-batch smoke training.

### 19.2 Integration fixtures

Create synthetic DICOM fixtures with:

- 2 gates x 4 views x 8 x 8 bins;
- intentionally shuffled frame order;
- known TimeSlotVector/AngularViewVector;
- one duplicate, one missing-frame, and one invalid-rescale variant;
- known angle/radius sequence;
- known pixel sum and exposure metadata.

Also create asymmetric point, cylinder, and cardiac-like numerical phantoms. Fixtures contain no patient data.

### 19.3 CI policy

CPU CI runs lint, type checks, unit tests, float64 adjoint tests, and a one-batch model smoke test. GPU CI, when available, runs float32 adjoint, AMP, sparse-operator, and one-step unrolled backward tests.

No milestone is complete with skipped required tests. Hardware-dependent tests may be marked separately only when CPU coverage verifies the same mathematics.

---

## 20. Milestones, deliverables, and acceptance gates

### M0 - Bootstrap

Deliver:

- repository skeleton, pyproject, exact dependency lock;
- environment report and CLI;
- test/lint/type-check commands.

Gate: clean installation and CI smoke test on the target machine.

### M1 - Audit and canonical data

Deliver:

- acquisition and pair manifests;
- canonical reader/cache;
- exposure estimation;
- QC reports;
- locked patient splits.

Gate: Section 8 acceptance criteria pass; no unresolved frame-order or pair-provenance issue for accepted data.

### M2 - Conventional physics

Deliver:

- geometry object and cache;
- ideal forward/adjoint operator;
- phantom tests;
- conventional MLEM/OSEM;
- orientation/vendor comparison report.

Gate: all Section 12.6 criteria pass and geometry_verified is true.

### M3 - Dataset 1 pretraining

Deliver:

- backbone and SSL heads;
- count split/masking;
- selected checkpoint and validation report;
- scratch/untrained representation guardrail.

Gate: validation masked deviance improves over a baseline that predicts one exposure-corrected scalar rate per acquisition from its visible input bins, no collapse/NaN occurs, and the checkpoint is reproducible.

### M4 - Projection denoising

Deliver:

- paired loader;
- pretrained and scratch denoisers;
- expected-count inference;
- denoise-then-OSEM baseline;
- calibration and count-bias report.

Gate: Section 11.6 or a documented failure-analysis decision.

### M5 - Unrolled reconstruction

Deliver:

- projection conditioner;
- shared 3D temporal regularizer;
- five-stage unrolled OSEM;
- independent projection-likelihood training;
- validation comparison.

Gate: Section 14.3 and no projector regression.

### M6 - Locked evaluation

Deliver:

- one immutable test result bundle;
- confidence intervals, ablations, failure cases;
- manuscript-ready tables/figures;
- configs, checksums, checkpoints, and environment metadata.

Gate: another researcher can reproduce tables from locked manifests and checkpoints without manual data reshaping or undocumented choices.

---

## 21. Risk register with concrete triggers

| Risk | Trigger | Required response |
|---|---|---|
| Cross-dataset leakage | validation/test patient key in Dataset 1 | remove before pretraining and regenerate checkpoint |
| Unresolved identity | stable cross-dataset key unavailable | label risk; do not claim strict inductive evaluation |
| Invalid low/high pair | geometry mismatch, motion threshold failure, or unknown provenance | reject or analyze separately |
| Count scaling ambiguity | exposure source unknown or inconsistent | block calibrated rate claims |
| Low gate occupancy | q below 0.20 | input allowed, target masked, report separately |
| Detector ambiguity | frame vector and detector sequence disagree | keep one stream; block dual-detector assumptions |
| Incorrect geometry | adjoint/orientation/phantom gate fails | block unrolled training |
| Missing PSF calibration | no a,b provenance | run psf=none and label no resolution recovery |
| Missing AC/SC inputs | no validated map/window | run without correction and label explicitly |
| Denoiser count bias | absolute validation bias over 3% | do not advance selected checkpoint |
| Learned artifact | systematic gate flicker/orientation/checkerboard | stop, inspect conditioning/update limits, retrain on validation only |
| Domain shift | subgroup deviance degrades over 5% | report, condition/adapt in a versioned experiment |
| Test-set reuse | locked output already exists | require new study version, never overwrite |

---

## 22. Definition of done

The plan is implemented when:

- data can be audited from source DICOM to an immutable [G,V,H,W] count tensor with explicit exposure and geometry;
- every included patient and pair is traceable through de-identified, checksummed manifests;
- splits prevent patient leakage and cannot be silently regenerated;
- the backbone, SSL losses, denoiser, projector, OSEM, and unrolled model obey the documented shape and unit contracts;
- raw counts remain available and unchanged in every statistical likelihood;
- the projector passes adjoint, phantom, and orientation tests before learned reconstruction;
- validation gates are applied without consulting the locked test set;
- all mandatory baselines and ablations are reproducible from saved configs;
- final results include failures, subgroup behavior, uncertainty, and explicit correction status.

The implementation order is:

1. bootstrap and tests;
2. inventory, DICOM reader, exposure, QC, and pair audit;
3. locked patient splits;
4. projection backbone and Dataset 1 pretraining;
5. Dataset 2 projection denoising;
6. ideal projector validation, conventional OSEM, and denoise-then-OSEM;
7. optional calibrated PSF/AC/SC additions;
8. projection-conditioned unrolled OSEM;
9. locked evaluation.

This order allows useful denoising work to proceed while geometry is being validated, but it never allows an unvalidated projector to supervise or justify learned reconstruction.
