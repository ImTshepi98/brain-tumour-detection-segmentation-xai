# Domain-gap check, BraTS vs IXI (Phase 3b/3c)

Kernel `leweletp/brats-dual-qc-domain` v1 (CPU, 4 workers, 2605 s), code `src/qc_domain.py` at commit 211a044. Inputs are the Phase 1 and Phase 2 2 mm arrays, **train+val only**: 1063 BraTS and 490 IXI (test never read). Classifier: HistGradientBoosting, 5-fold stratified CV, seed 42. Raw results are in `phase3/cv_results.json`, per-subject features in `phase3/features_base.csv`, and out-of-fold predictions in `phase3/oof_base_plan.csv`.

Tumour-free region: BraTS brain minus WT dilated by 1 voxel; IXI whole brain. "Plan" = PLAN 3b feature set (T1/T2 32-bin histograms, brain volume, bounding box, surface/volume, outer 2-voxel shell mean/SD; 73 features). "Extended" adds sharpness and low-frequency bias features (10).

## AUROC (5-fold mean ± SD)

| Feature set | Before | it1: mask fill/close + IXI histogram-matched to BraTS | it2: it1 + BraTS matched to the same reference |
|---|---|---|---|
| **Plan (decides 3c)** | **1.000 ± 0.000** | **1.000 ± 0.000** | **0.999 ± 0.000** |
| Plan + extended | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Histograms only | 1.000 | 1.000 | 0.990 |
| Shape only (volume, bbox, surface/volume) | 0.989 | 0.993 | 0.993 |
| Outer shell only | 0.989 | 0.996 | 0.985 |
| Sharpness only | 0.948 | 0.944 | 0.952 |
| Low-frequency bias only | 0.960 | 0.961 | 0.958 |

The threshold was 0.80. Both allowed mitigation iterations ran, and **the gap was not reduced**. Per PLAN 3c, the result is reported as is. No further iterations were run, and nothing was tuned afterwards.

## What separates the two cohorts (medians, tumour-free region, before mitigation)

| Feature | BraTS | IXI Guys | IXI HH | IXI IOP |
|---|---|---|---|---|
| Brain volume (mL, mask incl. tumour for BraTS) | 1480 | 1582 | 1610 | 1590 |
| Surface / volume | 0.082 | 0.079 | 0.078 | 0.078 |
| Outer-shell T1 mean (z) | −1.00 | −1.36 | −1.40 | −1.29 |
| Outer-shell T2 mean (z) | 0.11 | 0.02 | −0.15 | −0.56 |
| T1 gradient sharpness | 15.5 | 16.6 | 13.7 | 12.4 |
| T1 Laplacian variance | 2.03 | 2.33 | 1.54 | 0.93 |
| T1 low-frequency spread | 0.42 | 0.40 | 0.56 | 0.64 |
| T1 quadratic bias fit R² | 0.17 | 0.17 | 0.45 | 0.51 |

The most important single feature (permutation importance) is **outer-shell T1 mean**, before mitigation and after it2, followed by surface/volume and brain volume.

## Interpretation

1. **The skull-stripping mask is the strongest cue.** IXI brains are about 100–130 mL larger than BraTS brains after affine registration to a BraTS-space template, and their outer 2-voxel shell is much darker on T1. This is consistent with SynthStrip keeping a rim of CSF (dark on T1) that the BraTS skull-stripping removed. Mask hole-filling and closing (it1) cannot remove a rim of real CSF voxels, so the gap stays.
2. **Histogram matching does not help, and it1 is partly self-defeating.** Matching every IXI subject to one reference makes the IXI histograms nearly identical to each other, while BraTS histograms still vary. That uniformity is itself separable. it2 matches both groups, and histograms alone still reach 0.990. The 201-quantile piecewise mapping keeps small within-segment and tail differences, and the boosted trees pick these up across about 1500 subjects.
3. **Sharpness and inhomogeneity are site effects, not a clean IXI-vs-BraTS cue.** Guys T1 is about as sharp as BraTS. HH and IOP are smoother and show a much stronger low-frequency bias (R² 0.45–0.51 vs 0.17); these sites are where the Phase 2 panels showed visible gradients. Each feature group alone still reaches 0.94–0.96 AUROC.
4. **Every feature group alone separates the cohorts at AUROC ≥ 0.94.** Two different acquisition and preprocessing pipelines (BraTS: multi-institution, co-registered to SRI24, skull-stripped by the challenge; IXI: three London sites, our SynthStrip + ANTs) leave many independent fingerprints. Matching two summary statistics cannot remove them.

## Consequences for the rest of the project

- **Detection results cannot be read as tumour detection on their own.** A classifier with no knowledge of tumours separates the cohorts almost perfectly from tumour-free tissue. A high F1 for the detection head is therefore expected whether or not it looks at tumours. The dissertation must say this directly.
- The planned **Phase 7 shortcut check** (Grad-CAM++ mass in a 3-voxel brain-boundary shell) targets exactly the strongest cue found here. It moves from a sanity check to a central result.
- Added controls (logged in DECISIONS.md, no extra GPU training):
  - **Phase 6 tumour-removal control:** on BraTS **val**, the detection head's probability with the WT region replaced by mirrored contralateral tissue versus unmodified. A large drop means the model uses the tumour; little drop means it uses domain cues.
  - **Phase 6 boundary control:** the detection probability on IXI val after eroding the brain mask by 1 voxel (2 mm) to remove the CSF rim.
- Preprocessing is **not changed now**. The arrays stay as produced in Phases 1–2: neither PLAN mitigation helped, and the plan does not allow a third iteration. One option is to re-strip IXI with a tighter mask, or to re-strip BraTS with SynthStrip so both cohorts share one skull-stripper, which would target cue 1 directly. It would cost a re-run of Phases 1–2 (about 3 CPU hours). It is offered as an option to the owner (PROGRESS.md), not done automatically.

## Age-only classifier (3d)
Not possible: BraTS 2021 Task 1 has no per-case ages. BraTS ages are unavailable for all cases unless `survival_info.csv` from the BraTS 2020 survival task is supplied (USER ACTION 7, optional). The age distribution of the negatives is in `age_ixi.md`.
