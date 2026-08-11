
---

# Part B — Experimental Roadmap to Perform After the Code Is Fixed

This second plan is for **me**, not for the coding agent.

The objective is to establish whether the corrected SSL backbone learns diverse, useful representations and whether those representations improve low-count cardiac SPECT denoising and ultimately reconstruction.

---

# Stage 0 — Freeze the corrected codebase

After Codex finishes the code fixes:

1. review its diff;
2. verify all tests pass;
3. record the Git commit hash;
4. save the Python/PyTorch/CUDA environment;
5. freeze v2 manifests;
6. freeze locked patient splits;
7. freeze v2 normalization;
8. verify the strengthened migration report;
9. do not change these artifacts during experiments.

Create a scientific experiment version such as:

```text
spect_ssl_v2_experiments_01
```

All experiments should reference that code revision.

---

# Stage 1 — Verify corrected SSL mechanics before long training

Run a short SSL smoke/health study.

This is no longer bug testing; it is the first scientific sanity check.

Run sufficiently long to:

```text
fill representation queue ≥512
```

and observe several optimizer updates afterward.

Monitor:

```text
masked Poisson deviance
visible Poisson deviance
global consistency loss
gate consistency loss
feature variance
true SVD effective rank
effective-rank fraction
participation-rank fraction
mean off-diagonal correlation
p95 off-diagonal correlation
maximum correlation
gradient norm
AMP skipped-step rate
exposure-clipping fraction
```

Check that:

```text
no NaN/Inf
queue grows at expected rate
no obvious representation collapse
validation metrics finite
```

Do not use the locked Dataset2 test set.

---

# Stage 2 — Decide whether covariance regularization is necessary

Run a small controlled SSL comparison:

```text
covariance weight = 0
0.001
0.01
```

Everything else identical:

```text
same split
same seed
same batches
same optimizer
same masking policy
same augmentation
same training duration
```

Each run must continue long enough that:

```text
queue >= 512
```

and representation metrics stabilize.

Compare:

```text
validation masked deviance
effective-rank fraction
participation rank
mean/p95 correlation
minimum component std
```

Decision rule:

> Use the smallest covariance weight necessary to avoid unhealthy representation collapse without materially degrading Poisson validation performance.

If:

```text
weight = 0
```

already has healthy representation statistics, keep:

```text
weight = 0
```

for the main experiment.

Do not add regularization merely because it exists.

---

# Stage 3 — Full clean Dataset1 v2 SSL pretraining

Once covariance configuration is selected, launch clean v2 pretraining.

Use:

```text
v2 manifests
v2 normalization
corrected queue
corrected diagnostics
corrected AMP state management
```

Train using the complete Dataset1 training cohort.

Save:

```text
best validation checkpoint
latest checkpoint
metrics JSONL
event log
configuration snapshot
code commit
artifact hashes
```

Monitor representation health throughout training.

Do not select the checkpoint solely by representation rank.

Primary model selection should remain based on the predefined validation SSL criterion, while representation diagnostics act as health constraints.

---

# Stage 4 — Evaluate representation quality directly

Before Dataset2 fine-tuning, determine whether the SSL backbone actually learned meaningful features.

## 4A. Covariance-spectrum analysis

On held-out Dataset1 validation patients:

```text
extract ≥1024 representations
```

calculate:

```text
SVD spectrum
effective rank
participation rank
feature correlation matrix
feature variance
```

Plot the eigenvalue/singular-value spectrum.

---

## 4B. Same-patient invariance

Generate multiple valid versions of the same acquisition:

```text
different Poisson thinning
different masks
different hidden gates
```

Compare cosine similarity:

```text
same patient / different augmentation
different patients
```

Calculate a separation metric:

[
\Delta_\mathrm{sep}
===================

## E[s_\mathrm{same}]

E[s_\mathrm{different}].
]

A useful representation should show:

```text
Δsep > 0
```

with clear patient-level separation.

---

## 4C. Frozen probes

Freeze the backbone and evaluate simple heads for useful attributes.

Possible probes:

```text
gate phase
angular information
patient retrieval
scanner/collimator
exposure/count level
```

Interpret nuisance variables carefully.

High scanner prediction alone is not evidence of good anatomical representation.

---

## 4D. Spatial feature probe

Because reconstruction/denoising relies on the spatial bottleneck:

```text
[B,G,V,96,8,8]
```

not only the global vector, train a very small decoder on frozen bottleneck features.

Compare:

```text
SSL frozen backbone
vs
random frozen backbone
```

using the same decoder.

This is one of the strongest direct tests that SSL spatial representations contain useful SPECT information.

---

# Stage 5 — Full-finetuning pretrained-vs-scratch study

Use the corrected matched launcher.

Primary experiment:

```text
freeze_epochs = 0
```

for both:

```text
pretrained
scratch
```

Use at least:

```text
3 seeds
```

for example the same three existing seeds.

Everything identical except initial backbone weights.

Use:

```text
102 Dataset2 training patients
22 Dataset2 validation patients
```

Do not touch the 22-patient locked test cohort.

Evaluate patient-level:

```text
Poisson deviance/bin
total-count bias
per-view count bias
correlation
MAE/NRMSE where appropriate
```

Primary comparison:

[
\Delta D_i
==========

## D_{\mathrm{SSL},i}

D_{\mathrm{scratch},i}.
]

Calculate patient-bootstrap 95% CI.

Strong evidence of SSL utility:

```text
mean ΔD < 0
95% CI entirely below 0
```

---

# Stage 6 — Keep the partial-freeze transfer study separately

The existing:

```text
freeze_epochs = 8
```

experiment should not be discarded.

Treat it as a separate question:

> Does preserving pretrained features during early fine-tuning improve transfer?

Report:

```text
full fine-tuning result
partial-freeze result
```

separately.

If both favor SSL, the transfer evidence becomes substantially stronger.

---

# Stage 7 — Frozen-backbone downstream comparison

For stronger evidence of representation quality:

```text
freeze complete backbone
train only denoising head
```

Compare:

```text
SSL frozen backbone
vs
random frozen backbone
```

This asks a different and important question:

> Does the pretrained feature representation itself contain useful information before supervised adaptation?

Expected qualitative hierarchy for a strong SSL model:

```text
Frozen:
SSL >> random

Partial FT:
SSL > scratch

Full FT:
SSL ≥ scratch
```

---

# Stage 8 — Label-efficiency study

This is especially valuable because Dataset2 contains only 102 accepted training patients.

Use training subsets such as:

```text
10%
25%
50%
100%
```

patient-wise.

For every fraction compare:

```text
pretrained
scratch
```

using identical data.

Do not change the validation cohort.

If SSL is useful, its advantage should be especially visible when paired data are scarce.

Example scientific question:

> How much paired low/high-count data does SSL pretraining save?

This can become one of the strongest contributions of the study.

---

# Stage 9 — Expanded pseudocount analysis

Now run or reanalyze:

```text
α = 0.5
0.1
0.01
0.001
```

across ≥3 seeds.

Use the expanded evaluator.

Assess:

```text
Poisson deviance
total-count bias
per-view bias
zero-input-bin behavior
1-count/2-count/3–5-count strata
low-signal/background proxy
calibration
```

Selection hierarchy:

1. deviance within 1% of best;
2. total-count bias ≤3%, if feasible;
3. minimize zero-bin/background/per-view biases;
4. if practically tied, prefer smaller pseudocount.

At this stage you may select the final pseudocount.

Do not call it clinically optimal yet.

---

# Stage 10 — Freeze the downstream model-selection pipeline

Once you have selected:

```text
SSL checkpoint
covariance weight
pseudocount
fine-tuning protocol
learning schedule
checkpoint-selection criterion
```

freeze all model decisions.

Write them into a final experiment configuration.

Only after this point should the locked test cohort be used.

---

# Stage 11 — Locked Dataset2 test evaluation

Evaluate once on the untouched:

```text
22 test patients
```

Compare at minimum:

```text
naive exposure-scaled low-count baseline
scratch model
SSL-pretrained model
acquired high-count target
```

Report:

```text
patient-level Poisson deviance
deviance improvement
total-count bias
per-view bias
correlation
NRMSE/MAE
low-count calibration
background proxy metrics
```

Use patient-bootstrap 95% CIs.

Do not tune anything after seeing this result.

If a bug is discovered afterward, document it and restart the evaluation protocol appropriately rather than quietly retuning.

---

# Stage 12 — Validate DICOM interoperability

Before judging reconstruction performance:

1. export an acquired high-count projection series through your new DICOM exporter;
2. validate the generated DICOM;
3. import both:

   * original high-count acquisition;
   * exported copy;
4. reconstruct both with the exact same commercial reconstruction settings.

The two should reconstruct essentially identically.

This proves the exporter itself is not altering interpretation.

Only after this is demonstrated should AI-denoised DICOM projections be reconstructed.

---

# Stage 13 — Commercial reconstruction comparison

For each test case reconstruct:

```text
A. acquired low-count
B. AI-denoised low-count
C. acquired high-count
```

Optionally include naive exposure-scaled counts if meaningful to the workstation.

Use exactly identical reconstruction parameters.

Record:

```text
iterations
subsets
attenuation correction
scatter correction
resolution recovery
filter
voxel size
matrix
post-filter
```

Do not change them between methods.

---

# Stage 14 — Reconstruction-domain metrics

Where technically/clinically available, evaluate:

```text
NRMSE
SSIM
background CV
myocardial/background contrast
LV wall uniformity
regional uptake
polar-map error
defect contrast
defect recovery
```

If clinical readers are available, later consider blinded reader evaluation.

Projection-domain performance and reconstructed-image performance should be reported separately.

---

# Stage 15 — Real projector validation

In parallel with or after the denoising study, validate the physics reconstruction branch.

Required before learned/unrolled reconstruction:

```text
adjointness
orientation
rotation direction
center of rotation
radius/orbit
asymmetric phantom
point-source phantom
known geometry
vendor comparison
```

Then progressively validate:

```text
attenuation
scatter
collimator-detector response
system sensitivity
```

according to available data.

---

# Stage 16 — Unrolled reconstruction only afterward

Do not train the unrolled OSEM model until the projector verification gate is green.

The training code should require:

```text
geometry_verified = true
orientation_verified = true
phantom_verified = true
adjoint_verified = true
```

Then evaluate unrolled reconstruction separately from the projection-denoising project.

---

# Final scientific evidence hierarchy

At the end of the project, evidence that SSL learned useful representations should come from several independent levels:

### Representation health

```text
healthy SVD spectrum
healthy effective rank
healthy participation rank
low cross-feature correlation
```

### Representation invariance

```text
same patient across Poisson/mask augmentations remains close
different patients remain distinguishable
```

### Frozen representation utility

```text
SSL frozen backbone > random frozen backbone
```

### Downstream transfer

```text
SSL pretrained > scratch
```

across multiple seeds.

### Label efficiency

```text
SSL advantage increases as paired training data decrease
```

### Final clinical/reconstruction relevance

```text
denoised projections improve reconstructed image quality
toward the acquired high-count reference
```

No single metric should be used alone to claim that the SSL representation is good.

The strongest eventual statement would be supported only if these levels agree:

> The self-supervised gated-SPECT backbone learned non-collapsed, augmentation-stable and transferable projection representations that improved low-count denoising relative to matched scratch training, particularly under limited paired-data conditions, and these improvements propagated to reconstructed cardiac SPECT images.
