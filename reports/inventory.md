# Phase 0 inventory (30 Oct 2026)

Checks PLAN.md section A against the owner's Kaggle notebooks (pulled read-only into `legacy/`) and three new CPU kernels (`brats-dual-env-check`, `brats-dual-split-ids`, `brats-dual-hf-token-check`).

Legend: **OK** = confirmed, **CORRECTED** = section A was wrong or incomplete, **NEW** = not in section A.

## Data sources

| Item | Status | Evidence |
|---|---|---|
| BraTS tar `dschettler8845/brats-2021-task1` | OK | Mounted at `/kaggle/input/datasets/dschettler8845/brats-2021-task1/BraTS2021_Training_Data.tar`. **1251 subjects, 5 members each** (t1, t1ce, t2, flair, seg). The dataset also contains two stray single-case tars (`BraTS2021_00495.tar`, `BraTS2021_00621.tar`), which are ignored. |
| IXI HF zip `Joe0204/dataset-ixi@94862e0d…` | CORRECTED | **401 even without a token**, from this PC and from Kaggle. The repo is private or gated, so `HF_TOKEN` must belong to an account with access. Not on the critical path any more (see next row). |
| IXI fallback `mdnakib/ixi-dataset` | NEW: now primary | T1 581 / T2 578 / PD 578, **577 paired T1+T2**, sites Guys / HH / IOP. Same counts as section A for the HF zip. Byte-identity with the HF zip is not verified. |
| IXI demographics | NEW | Official `IXI.xls` from `http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls` (fetched 30 Oct 2026 14:15 UTC, 210 432 bytes, saved in `reports/phase0/split-ids/`). 619 rows, 26 duplicated IDs. **559 of 577 paired subjects have an age** (mean 48.7, SD 16.5, range 20.0 to 86.3). 18 have none (train 11, val 4, test 3). |
| IXI licence | TODO (Phase 9) | Verify CC BY-SA 3.0 on the official IXI page and record the URL in README. |

## Splits: where the IDs live

All IDs are consolidated in **`data_splits/splits.csv`** (1828 rows: subject, owner hash id, source, site, split, label, has_seg, owner_excluded). It is built by subtraction, so no image data were read for test.

| Cohort | train | val | test | Source of IDs |
|---|---|---|---|---|
| BraTS | 875 | 188 | 188 | train/val: `brats_trainval_scope.json` (output of `brats-supplied-grid-full-cohort-qc`, identical copy in `brats-fp32-checkpoint-continuation`). Test = the 188 tar subjects not in it. **OK** |
| IXI | 404 (Guys 223 / HH 130 / IOP 51) | 86 (48 / 27 / 11) | 87 (48 / 28 / 11) | train/val: `ixi_scope.json`, embedded in the `ixi-full-native-preprocessing-qc` notebook source (extracted to `legacy/derived/ixi_scope_trainval.json`). Test = the 87 paired subjects not in it. **OK**, matches section A exactly. |

The owner's subject `id` is a 20-hex hash; the hashing scheme isn't recorded. Matching therefore uses subject names (BraTS2021_xxxxx, IXInnn).

## Cohort-size discrepancy (1542 vs 1828): reconciled

- **1828** = 1251 BraTS + 577 IXI, the full cohort including test.
- **1542** = train+val only. The parent manifest held 1553 (1063 BraTS + 490 IXI). The owner's eligibility step (`eligibility_sha256 8bb1f6c7…`) then excluded **11 IXI subjects**: IXI020, IXI210, IXI326, IXI445, IXI504, IXI556, IXI584, IXI585, IXI626, IXI642, IXI652. That gives IXI train/val 395/84. No pulled output records the reason.
- **This project uses 1828** (PLAN B1). The 11 are flagged (`owner_excluded=1`) and judged by Phase 2 QC. See DECISIONS.md.

## Notebook-by-notebook

| Notebook | Section A claim | Status | Notes |
|---|---|---|---|
| `brats-supplied-grid-full-cohort-qc` | 1063/1063 numerical QC pass, signature `brats2021-supplied-grid-zscore-v1`, visual review not complete | **OK** | `summary.json`: passed 1063, failed 0, `visual_review_complete: false`, 15 300 s. `modality_stats.csv`: 4252 rows = 1063 × 4, columns `id, split, modality, nonzero, mean, std, normalized_min, normalized_max, source_sha256` (keyed by the hash `id`, not the subject name). Preprocessing: native 1 mm, per-channel non-zero z-score, no clipping. |
| `brats-fp32-checkpoint-continuation` | Val Dice epoch 4: WT 0.784, TC 0.692, ET 0.699, ~13 min/epoch | **OK** | `history.json` epochs 0–4: WT .615/.756/.773/.796/.784, TC .564/.647/.656/.687/.692, ET .571/.658/.689/.696/.699. Epochs 3–4 took ~13 min; epoch 2 took 22 min (includes start-up). Note: **epoch 3 had the best WT (0.796)**. torch 2.11+cu128, MONAI 1.5.1, 2× T4. Checkpoints (`best.pt`, `last.pt`, `epoch_00{2,3,4}.pt`) exist in the output but were not downloaded. |
| `brats-supplied-grid-segmentation-baseline` | AMP parent run failed with a non-finite grad norm in epoch 2 | Not re-checked | Output listing is empty via the API (the failed run kept no output). Epochs 0–1 match the continuation's history. |
| `notebook6794c43f52` | IXI-to-BraTS-like smoke test, engineering only | **OK + CORRECTED** | Embedded job (decoded: `legacy/notebook6794c43f52/decoded_0.py`), signature `candidate-fresh-rigid-v8-brats-raw-support-ixi-corestats-2mm-pad32-v1`. Pipeline: **SynthStrip on CPU** (`mri_synthstrip.py` + `synthstrip.1.pt`, SHA-pinned) → ANTs rigid T2→T1 → **ANTs Affine to the MNI152 2 mm template (nilearn)**, applied to BraTS and IXI alike → percentile clip 0.5/99.5 → z-score → fp16. Section A says "HD-BET and SynthStrip and ANTs all tried"; only SynthStrip and ANTs appear in this code. Requires a **Python 3.12** image. |
| `notebook55f6d89c44` | Audit of 64 candidate arrays, `BATCH002_64_ARRAY_AUDITS_PASSED_NOT_COHORT_QC` | OK (not re-read in depth) | Output `candidate_batch_002_array_audit.json`. |
| `fork-of-notebook6794c43f52` | 1 subject prepared, `expected_subjects: 1542` | **OK** | 1542 explained above. |
| `ixi-full-native-preprocessing-qc` | Failed after 123 s with HF 401 | **OK** | Contains the IXI train/val scope; no arrays were produced. |
| `ixi-readable-qc-diagnostic-strips` | Display aid only | OK | Not used. |
| `stage-1-ixi-healthy-anatomy-pretraining-self-su` | not in section A | **NEW** | Sept 2026, self-supervised autoencoder on IXI + BraTS T1/T2 at 128×128×96. IXI comes from `ixi_preprocessed_sri24.zip` in HF repo `Joe0204/brats-3d-unet-checkpoint` (its provenance is undocumented; also 401 without a token). Random 90/10 split. |
| `stage-2-classification-healthy-ixi-vs-tumor-br` | not in section A ("no dual-head model or Grad-CAM++ found") | **NEW / CORRECTED** | Sept 2026 prototype `DualHeadAttentionUNet3D` with attention gates and a `GradCAMPlusPlus3D` class; 80/20 split by folder order, **no locked test**. Prior work only; not reused for results (DECISIONS.md). |

## Kaggle environment (measured 30 Oct, CPU kernel)

Python 3.13.15, torch 2.11.0+cpu, numpy 2.1.3, nibabel 5.4.2, SimpleITK 2.5.6, huggingface_hub 1.29.0. **MONAI and ANTs are not preinstalled.** 4 CPUs, 32 GB RAM, `/kaggle/working` 21 GB free, `/tmp` about 1.1 TB free, **no `/kaggle/temp`**. Datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`. The tar header walk over the 13.4 GB BraTS tar took about 60 s. According to kaggle CLI 2.2.5, the P100 is retired, so the GPU is T4 x2.

## Phase 0 acceptance

- [x] `reports/inventory.md` exists.
- [x] Split IDs located for BraTS train/val/test and IXI train/val/test (`data_splits/splits.csv`).
- [x] A trivial CPU kernel ran; output saved (`reports/phase0/env-check/env.json`).
- [x] Private code dataset `leweletp/brats-dual-code` created.
- [ ] HF token verified: **not needed** after switching to `mdnakib/ixi-dataset`. Optional USER ACTION.
