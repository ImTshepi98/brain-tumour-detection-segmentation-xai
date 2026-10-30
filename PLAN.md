# PLAN.md (v3): Dual-head 3D CNN for brain tumour detection + segmentation, built on the owner's existing Kaggle work

Deadline: submission 6 Nov 2026. Work window: 31 Oct to 5 Nov (6 Nov is buffer only).
Owner: Tshepiso Lewele (Kaggle user `leweletp`). Supervisor: Dr. Tite Tuyikeze. BSc Hons Computer Science, Sol Plaatje University.

This version replaces the earlier plans. It reuses what already exists on Kaggle instead of starting over.

---

## A. What already exists (read from the owner's Kaggle logs on 30 Oct; VERIFY in Phase 0, do not trust blindly)

| Item | State |
|---|---|
| BraTS source | Kaggle dataset `dschettler8845/brats-2021-task1` (`BraTS2021_Training_Data.tar`, about 13.4 GB, 1251 cases). |
| BraTS split | Train 875 / val 188 / test 188. Only train+val (1063 cases) are in `brats_trainval_scope.json`; **test = the 188 tar cases not in that file. Test is LOCKED and has never been read.** |
| BraTS full-cohort QC | Notebook `brats-supplied-grid-full-cohort-qc`: 1063/1063 passed numerical QC, signature `brats2021-supplied-grid-zscore-v1`. Outputs `summary.json`, `subjects.json`, `panels.json`, `modality_stats.csv` (per-subject per-modality mean/std over non-zero voxels). **Visual review was NOT marked complete.** |
| Segmentation baseline | Notebook `brats-fp32-checkpoint-continuation` (script `brats_supplied_grid_training.py`): MONAI residual UNet (16-32-64-128-256, 2 res units, InstanceNorm), 1 mm, random 96^3 patches (50% centred on tumour), batch 2, AdamW lr 3e-4, squared Dice + BCE on WT/TC/ET, FP32, 2x T4 (DataParallel), reads straight from the tar. Validation Dice after epoch 4 (5 epochs trained): WT 0.784, TC 0.692, ET 0.699. About 13 min per epoch. Has protocol hashing, resume (`last.pt`), finite-gradient checks. **No attention gates, no classification head.** |
| IXI source | HF repo `Joe0204/dataset-ixi`, file `ixi-dataset.zip`, pinned revision `94862e0d6ce1234b1a63a49ac2b9af04020fbdeb`. 1737 members: T1 581, T2 578, PD 578; **577 subjects have both T1 and T2**. Fallback source: Kaggle dataset `mdnakib/ixi-dataset` (the owner has it bookmarked). IXI licence: CC BY-SA 3.0 (verify on the official IXI page). |
| IXI split | Site-stratified: train 404 (Guys 223, HH 130, IOP 51), val 86 (48/27/11), test 87 (48/28/11). |
| IXI to BraTS-like preprocessing | Only **engineering smoke tests**: notebook `notebook6794c43f52` (stage `t2_coverage_qc_v2`, 192 prepared cases, 64^3 patch, 1 epoch; its own config says "engineering smoke only, source confounding unresolved") and `notebook55f6d89c44` (audit of 64 candidate arrays, shape 4x131x149x127 float16, IXI channels 2 and 3 are zero; ended `BATCH002_64_ARRAY_AUDITS_PASSED_NOT_COHORT_QC`). The code is in these notebooks' outputs (`experiment.py`, `t2_coverage_qc_v2.py`). HD-BET and SynthStrip and ANTs were all tried. |
| **Full IXI run** | Notebook `ixi-full-native-preprocessing-qc` **failed after 123 s with HTTP 401 from Hugging Face** (private/gated repo or token not attached to that notebook). So there are **no full-cohort IXI arrays yet.** |
| Not found | Dual-head model, Grad-CAM++, GUI, BraTS-vs-IXI domain-gap report, any test-set evaluation. |
| Baseline history | Notebook `brats-supplied-grid-segmentation-baseline` (the AMP parent run, T4x2) **failed after 8652 s in epoch 2 with a non-finite gradient norm** (`clip_grad_norm_` with `error_if_nonfinite=True`). Epochs 0 and 1 had finished (val Dice epoch 1: WT 0.756, TC 0.647, ET 0.658). The FP32 continuation resumed from that checkpoint, which is why it started at epoch 2. Epoch 0 ended at 5318 s of run time but epoch 1 only about 830 s later, so a fresh run carries a large one-off start-up cost (probably cold tar reading; unverified). |
| Other notebooks | `ixi-readable-qc-diagnostic-strips`: 34 s CPU run that rendered 129 fixed QC display rows, "QC false; training false" (a display aid, not data). `fork-of-notebook6794c43f52`: 198 s smoke test, `expected_subjects: 1542`, 1 subject prepared, `preprocessing_ready: false`, uses SynthStrip. **The cohort size here (1542) differs from the 1828 (1251 BraTS + 577 IXI) in `notebook6794c43f52`; reconcile in Phase 0.** |

## B. Decisions made for this version (log each in `DECISIONS.md`)

1. **Keep all 577 IXI subjects (no age filter).** Instead: report the IXI age distribution, and report detection results overall **and** stratified by IXI age band and by site (Phase 6). Reason: filtering discards data and cannot make BraTS age-matched anyway (BraTS 2021 Task 1 has no per-case age; only a survival subset does).
2. **The dual-head model runs at 2 mm on whole volumes** (BraTS 1 mm volumes average-pooled 2x). Reason: the detection head needs whole-volume context (a random 96^3 patch may not contain the tumour, which would make labels noisy), and 1 mm whole volumes would make an epoch several times slower than the owner's 13 min. The owner's 1 mm patch baseline stays as the high-resolution segmentation reference. Report both honestly; they are not like-for-like. Log as a deviation from the proposal (1 mm).
3. **Reuse, never modify.** Existing notebooks and their outputs are the owner's QC-signed artifacts. Claude Code reads them (`kaggle kernels pull`, `kaggle kernels output`) and builds NEW notebooks with new slugs.
4. **Test stays locked until the final evaluation (Phase 6)**, and is read once per model with settings frozen from validation.
5. **FP32 by default.** The owner's AMP run crashed on a non-finite gradient norm on T4s, and the pipeline then migrated to FP32 and kept finite-gradient checks. Try fp16 AMP only if the smoke-test benchmark shows it is needed, and keep the finite checks.

## C. USER ACTION items (Claude Code: collect these in `PROGRESS.md`, keep working on everything else)

- **IXI access:** the owner reports attaching the `HF_TOKEN` secret on 30 Oct (not yet verified). Kaggle secrets are attached **per notebook**, so any new notebook that touches Hugging Face needs the toggle switched on in its editor (Add-ons, Secrets), which only the owner can do. To limit this, Phase 2 downloads the IXI zip **once** and every later notebook reads the cached copy. Claude Code must never read, print or store the token. **Fallback that needs no token:** attach the Kaggle dataset `mdnakib/ixi-dataset` and check that it has paired T1/T2 for about 577 subjects.
- **Visual QC sign-off:** the owner's QC gate requires a human to review panels. Claude Code produces the panels and a flagged-cases list; **only the owner sets any `visual_review_complete` flag.**
- Kaggle phone verification (for GPU and internet) and the real weekly GPU quota (shown in account settings).
- If available: BraTS survival metadata with ages (`survival_info.csv` from BraTS Task 2). Optional (Phase 3d).

## 1. Operating rules for Claude Code

1. Work autonomously; do not stop to ask. Ambiguity: choose the most reasonable option and append a line to `DECISIONS.md`. The only exceptions are USER ACTION items.
2. Claude Code runs on the owner's computer; Kaggle is only compute (CLI: push kernel, poll status, download output). Keep `PROGRESS.md` current (done / running / blocked / next command, kernel slugs and last status).
3. **Before every kernel push, run a local CPU dry run** on tiny synthetic data (`--debug`, volumes 4x32x32x32) covering the whole script.
4. Every phase has an **acceptance check**; do not move on until it passes unless the fallback says otherwise. Save check outputs under `reports/`.
5. Seeds 42; log git commit and config for every run; each run writes to `runs/<name>/`; keep the owner's protocol-hash style (hash the config, store with checkpoints).
6. Honesty: never tune on test, never hide failed checks, record every deviation.
7. Verify external facts (Kaggle quota and CLI flags, dataset layouts, licences). Record the licence URL or text used in `README.md`.
8. Never print or commit credentials. Never edit or delete the owner's existing notebooks.
9. Tier labels: **[MUST]**, **[SHOULD]**, **[COULD]**.

## 2. Kaggle constraints (re-verify at start)

- GPU quota is about 30 h per week but floats; plan for at most 25 h and check the real number. Sessions are limited to 12 h. Never run two GPU notebooks at once; CPU-only notebooks do not use GPU quota.
- Accelerators: 2x T4 (as the owner used) or P100. Use whichever is offered; benchmark iterations per second in the smoke test.
- `/kaggle/working` holds about 20 GB of saved output; `/kaggle/temp` is scratch; `/kaggle/input` is read-only. Internet must be switched on per notebook.
- Logs and outputs come back only after a run completes: every training script needs `--max-minutes`, a checkpoint every epoch, and `--resume`.
- Pass data between notebooks with notebook outputs as sources and datasets as sources. Everything private.

## 3. Fixed design

| Item | Decision |
|---|---|
| Input | 4 channels in order [T1, T1ce, T2, FLAIR]. IXI subjects have only T1 and T2; channels 2 and 3 are zero. |
| Resolution | 2 mm whole volumes. BraTS 1 mm z-scored volumes cropped to the brain box, then 2x average-pooled; padded to multiples of 16 (expected about 80x80x64; compute from the data). |
| Labels | Native 1 = NCR, 2 = ED, 4 = ET. ET = {4}, TC = {1,4}, WT = {1,2,4}; 3 overlapping binary channels, sigmoid. Pool the 3 binary channels with threshold 0.5. |
| Normalisation | The owner's QC-approved per-subject per-modality mean/std over non-zero voxels (`modality_stats.csv`), background stays 0. IXI: same formula on its own non-zero voxels. No N4. |
| Network | 5-level 3D residual U-Net, channels [16,32,64,128,256], InstanceNorm + LeakyReLU, **attention gates (Oktay 2018) on all skip connections**. |
| Heads | Seg: 1x1x1 conv to 3 channels. Cls: global average pool of the bottleneck, then MLP (256, 64, 1). |
| Passes | One shared encoder, two passes per step. **Detection pass:** T1ce and FLAIR zeroed for ALL samples. **Segmentation pass:** all 4 channels, BraTS cases only. |
| Loss | `L = L_seg + 0.5 * L_cls`. `L_seg` = squared Dice + BCE (as the owner's baseline). `L_cls` = BCE-with-logits with `pos_weight` or a balanced sampler. Healthy subjects contribute only `L_cls`. |
| Optimiser | AdamW lr 3e-4 (as baseline) with cosine decay, wd 1e-5, grad clip 1.0, finite-gradient checks, FP32. |
| Why not a transformer or foundation model | Compute and time, and Grad-CAM++ is cleaner on conv layers; matches the proposal. |

## 4. Repository layout

```
brats_dual/
  PLAN.md PROGRESS.md DECISIONS.md README.md requirements.txt
  legacy/      (read-only pulls of the owner's notebooks and outputs)
  configs/     kaggle.yaml debug.yaml
  kaggle/      one folder per new notebook: kernel-metadata.json + thin script
  src/         prep_brats_2mm.py prep_ixi_2mm.py age_tables.py qc_domain.py splits.py
               datasets.py model.py losses.py metrics.py train.py evaluate.py
               gradcam3d.py explain.py synthetic.py
  app/         streamlit_app.py
  reports/ runs/ tests/ data_local/
```

## 5. Phases

### Phase 0: Inventory and plumbing (about 2 h) [MUST]
- Local venv; install `kaggle`, `torch` (CPU), `monai`, `nibabel`, `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `pandas`, `matplotlib`, `streamlit`, `pyyaml`, `pytest`. Confirm `kaggle kernels list --mine` works.
- Pull code and outputs of the owner's notebooks into `legacy/` (`kaggle kernels pull`, `kaggle kernels output`). Read: `experiment.py`, `t2_coverage_qc_v2.py`, `brats_supplied_grid_training.py`, the QC outputs, the scope file and any IXI split file. Write `reports/inventory.md` confirming or correcting section A, including exactly where the BraTS and IXI split IDs live.
- Create `kaggle/` notebook folders and a private code dataset `<user>/brats-dual-code` (`src/`, `configs/`); version it after every code change.
- Write `src/synthetic.py` for dry runs.
- **Verify the HF token (about 1 minute, CPU):** push a tiny notebook (owner toggles `HF_TOKEN` on once) that calls the Hugging Face API for `Joe0204/dataset-ixi` at the pinned revision and prints only the status and file count, never the token. A 401 again means the token or repo access is wrong; switch to the `mdnakib/ixi-dataset` fallback and log it.
- Reconcile the cohort-size discrepancy (1542 expected subjects in the fork vs 1828 elsewhere): find out which IXI subjects each count includes and record which definition this project uses.
- **Accept:** `reports/inventory.md` exists; split IDs for BraTS train/val/test and IXI train/val/test are located (or the missing ones are listed as USER ACTION); a trivial CPU kernel ran and its output (versions, disk, RAM, `ls /kaggle/input`) is saved.

### Phase 1: BraTS 2 mm arrays (CPU notebook) [MUST]
`src/prep_brats_2mm.py`.
1. For each of the 1251 cases (train/val from the scope file; test = remaining cases): load the 4 modalities and seg from the tar; assert (240,240,155); z-score per modality with the owner's `modality_stats.csv` (train/val); for the 188 test cases compute the same statistics using the same formula (non-zero voxels), without reading test labels for anything but storing them.
2. Brain-box crop (compute the union bounding box over train+val only, reuse for test), pad to a multiple of 2, average-pool 2x (re-zero background where the pooled brain fraction is below 0.5), cast to fp16, and write `.npy` per case plus uint8 label bits (bit0 ET, bit1 TC, bit2 WT). Total size must fit under 12 GB (it should be far below).
3. Write `manifest.csv` (id, source=brats, split, label=1, has_seg=1) and `crop.json`.
- **Accept:** 1251 arrays with equal shape; label-mapping unit test on a toy volume; QC montage of 6 cases with seg overlay; the owner's 1 mm numerical QC (1063 passed) is cited, not redone.

### Phase 2: IXI to BraTS-like 2 mm arrays [MUST] (blocked until IXI access works)
`src/prep_ixi_2mm.py`, based on the owner's working code in `legacy/` (skull strip with whichever of HD-BET or SynthStrip the legacy code used successfully; ANTs rigid T2-to-T1, affine T1 to a BraTS-space template).
0. **Cache once:** download `ixi-dataset.zip` (T1 and T2 members only) at the pinned revision and save it as the output of a small CPU notebook; later notebooks use that output as a source and never touch Hugging Face. Reason: in the owner's first notebook, indexing the remote zip took about 39 minutes (361 s to 2699 s in the log) on every run.
1. All 577 paired subjects, **no age filter**. Read each subject's age, sex and site from the IXI demographics file (find it in the HF zip or on the official IXI site; record the source).
2. Reorient to the BraTS orientation code (from `legacy/` header notes or `nib.aff2axcodes` on a BraTS file), skull-strip native T1, rigid-register T2 to T1 and apply the mask, affine-register to the BraTS template, resample onto the BraTS 1 mm grid, then apply the **same crop, 2x pooling and padding as Phase 1**, z-score over non-zero voxels per modality, store 4-channel fp16 arrays with channels 2 and 3 zero.
3. Process in resumable shards (4 CPU workers); checkpoint progress; failures to `reports/ixi_failed.txt`; keep going on single failures.
4. QC panels: 12 IXI and 12 BraTS T1/T2 middle slices, intensity histograms, and a list of flagged subjects (low mask overlap with BraTS template, odd brain volume). **Owner reviews the panels.**
- **Accept:** at least 500 of 577 subjects pass; every array has the BraTS shape; panels produced. If more than 15% fail registration, change the template and rerun only the failures.
- **Fallback if this phase exceeds 6 h wall-clock:** stop with the subjects completed, log the count, and keep the existing split proportions.

### Phase 3: Age tables and domain-gap check (CPU) [MUST]
`src/age_tables.py`, `src/qc_domain.py`.
- **3a Age table:** IXI age histogram, mean, SD and counts per site and per 10-year band, written to `reports/age_ixi.md`.
- **3b Domain gap:** gradient-boosting classifier BraTS vs IXI on **tumour-free voxels only** (BraTS outside the tumour mask, IXI whole brain). Features: T1/T2 intensity histograms (32 bins), brain volume, bounding-box size, mask surface-to-volume ratio, mean/SD of an outer 2-voxel shell. 5-fold CV AUROC to `reports/domain_gap.md`.
- **3c Mitigation (max two iterations) if AUROC > 0.80:** same mask hole-filling and closing for both groups; histogram matching of IXI T1/T2 to a BraTS reference over non-zero voxels. Recompute the AUROC. Report honestly even if still high.
- **3d [COULD]:** if BraTS survival metadata with ages is available, train an age-only classifier (IXI vs that BraTS subset) and report AUROC; otherwise state that BraTS ages are unavailable for most cases.
- **3e Define the age band** for Phase 6 from the data (default 40 to 75 years inclusive; log the reason) and the number of IXI subjects inside and outside it.
- **Accept:** `reports/age_ixi.md` and `reports/domain_gap.md` with before/after AUROC and a short interpretation.

### Phase 4: Splits and data loading [MUST]
- BraTS split unchanged. IXI split: reuse the owner's (train 404, val 86, test 87) if the IDs are found; otherwise site-stratified 70/15/15 with seed 42. Assert no overlap and that every id in the manifest belongs to exactly one split.
- `datasets.py`: load arrays into RAM (fp16), augmentation on GPU (flips on all axes, intensity scale 0.9 to 1.1 and shift +/-0.1, small Gaussian noise), balanced sampler (about 50% healthy). `has_seg` mask passed to the training loop.
- **[COULD]** hold out one IXI site for a generalisation test.
- **Accept:** batch-shape unit test on synthetic data; split assertions pass.

### Phase 5: Model and training (GPU) [MUST]
- `model.py` per section 3 with `forward_cls(x)` (zeroes channels 1 and 3 internally) and `forward_seg(x)`. Test shapes and parameter count on CPU.
- `train.py`: config-driven, `--max-minutes`, `--resume`, protocol hash, `best.pt` (validation `0.5*mean_DSC + 0.5*F1`) and `last.pt` every epoch, `metrics.jsonl`.
- **Smoke test first** (GPU, at most 15 min): 1 epoch on 40 cases including validation; log iterations per second and the one-off start-up cost (the owner's runs showed a slow first epoch; cached 2 mm arrays in RAM should avoid the tar-reading part). If out of memory, reduce batch, then enable activation checkpointing, then lower base channels to 12. Set the epoch count for each run from the measured speed.
- **Runs (priority order):**
  1. **[MUST] `main`:** dual-head + attention gates. Budget about 4 h or the epoch count measured to fit.
  2. **[SHOULD] `noattn`:** identical, plain skip concatenation (ablation). About 3 h.
  3. **[SHOULD] `segonly` and `clsonly`:** single-task baselines at 2 mm, about 1.5 h each.
  4. **[COULD]** continue the owner's 1 mm baseline for more epochs and evaluate on test as the high-resolution reference.
- **Accept:** `main` completes; val mean DSC rises over the first 20 epochs (or the measured equivalent) and val F1 above 0.9. If DSC stays near zero after 15 epochs, check label alignment, loss weights and learning rate first.

### Phase 6: Evaluation (GPU, test split read once per model) [MUST]
`src/evaluate.py` with threshold and checkpoint frozen from validation.
- **Detection (overall):** Recall, Precision, F1, Specificity, AUROC, confusion matrix, bootstrap 95% CIs (1000 resamples).
- **Detection by group (the age/site integrity check):** specificity and false-positive rate on IXI **inside the age band**, on IXI **outside the band (older and younger)**, and **per IXI site**; recall on all BraTS. State clearly that BraTS has no per-case age, so these results show how age and site affect the negatives; they do not age-match the two groups.
- **Segmentation:** per-case DSC and HD95 (MONAI, `percentile=95`, spacing 2 mm) for ET/TC/WT as mean +/- SD and median. BraTS conventions: both empty gives DSC 1, HD95 0; empty ground truth with non-empty prediction gives DSC 0, HD95 373.13.
- **Inference time:** mean ms per case on GPU at batch 1 after warm-up; also on CPU locally; report hardware.
- Outputs: `results_<run>.json`, `reports/tables.md` (comparison across runs), figures (ROC, confusion matrix, DSC boxplots, specificity by age band and site).
- **Accept:** one `reports/tables.md` containing every number used in the dissertation.

### Phase 7: Explainability [MUST]
`src/gradcam3d.py`, `src/explain.py`: our own 3D Grad-CAM++ with hooks (alpha = g^2 / (2 g^2 + sum(A) g^3), weights = sum(alpha * relu(g)), cam = relu(sum(weights * A)), trilinear upsample, normalise to [0,1]).
- **Detection pathway:** target the cls logit, last encoder stage, T1/T2 input.
- **Segmentation pathway:** target = sum of the class logit inside the predicted mask for ET/TC/WT, last decoder stage before the head, 4-channel input.
- 6 test cases (mixed difficulty, 2 healthy including one outside the age band, any false positives/negatives); save heatmaps and axial/coronal/sagittal overlays.
- **Quantitative checks** (`reports/xai_metrics.md`): (1) fraction of heatmap mass inside the ground-truth tumour vs the tumour's volume fraction; (2) **shortcut check:** heatmap mass in a 3-voxel shell at the brain boundary, for both classes; (3) **[SHOULD]** deletion test (zero the top 10/20/30% CAM voxels, report the probability drop).
- **Accept:** overlays, the metrics table and an honest `reports/xai_notes.md`.

### Phase 8: Streamlit GUI (local, CPU) [SHOULD]
- Download `main/best.pt` and about 10 test cases (`.npy`) into `data_local/`. Choose a bundled case or upload preprocessed arrays; show tumour probability and decision (T1/T2 only), segmentation overlay (ET/TC/WT colours) when 4 modalities exist, slice slider, Grad-CAM++ toggle, region volumes in mL (voxel = 8 mm3 at 2 mm), inference time. Banner: "research prototype, not for clinical use". State that uploads must already be BraTS-like (skull-stripped, SRI24-registered, 2x-pooled and cropped); provide `src/preprocess_one.py` for raw BraTS-format cases.
- **Accept:** `streamlit run app/streamlit_app.py` runs a bundled case end to end; 3 screenshots in `reports/gui/`.

### Phase 9: Package results [MUST]
- `README.md`: setup, data sources and licences, kernel order and commands per phase, hardware, run times.
- `reports/results.md`: Results and Discussion skeleton with real numbers (data summary per split, preprocessing, age tables, domain-gap, model/training, detection tables overall and by group, segmentation table, ablation, XAI findings, GUI screenshots, limitations).
- **Ethics/licensing notes:** IXI CC BY-SA 3.0 (attribute the original IXI source, not only the HF copy); the BraTS 2021 dataset page lists Conditions for Use (acknowledgement and a publication-embargo clause aimed at challenge participants): read them and cite as required; both datasets are de-identified secondary data.
- **Limitations that must appear (adjust if untrue):** single seed; 2 mm resolution for the dual-head model (1 mm only for the reference baseline); healthy and tumour data come from different sources so residual domain bias is possible (state the Phase 3 AUROC); BraTS ages are not available per case so detection is not age-matched (state the IXI age distribution and the by-band results); detection excludes T1ce and FLAIR; BraTS-style preprocessing assumed for any new data; research prototype only.
- Turn `DECISIONS.md` into a "Deviations from proposal" list.

## 6. GPU budget (cap about 25 h; check the real quota first)

| Job | Hours |
|---|---|
| Smoke test and benchmark | 0.5 |
| IXI skull-stripping (only if the legacy code used the GPU) | 1.0 |
| `main` | 4.0 |
| `noattn` | 3.0 |
| `segonly` + `clsonly` | 3.0 |
| Evaluation + XAI (all runs) | 2.0 |
| Reserve for crashes and reruns | 8.0 |

These per-run times are estimates, not measurements. Replace them with numbers from the smoke test. If quota is short, cut in this order: continuing the 1 mm baseline, `segonly`/`clsonly`, `noattn` (reduce), then `main`. Never cut: Phases 3, 6, 7 or the limitations.

## 7. Schedule

| Day | Date | Target |
|---|---|---|
| 1 | Sat 31 Oct | Phase 0; fix IXI access (owner); Phase 1 on CPU; start Phase 2; write model/train code and dry-run locally |
| 2 | Sun 1 Nov | Finish Phase 2; Phases 3 and 4; smoke test; launch `main` overnight |
| 3 | Mon 2 Nov | Evaluate `main`; launch `noattn`; write explainability code and GUI |
| 4 | Tue 3 Nov | Phase 7 on `main`; `segonly`/`clsonly`; evaluate `noattn` |
| 5 | Wed 4 Nov | Remaining evaluations; Phase 8; fill `reports/tables.md` |
| 6 | Thu 5 Nov | Phase 9, final checks, freeze everything by evening |
| 7 | Fri 6 Nov | Submission day: buffer only |

**Biggest schedule risk:** Phase 2 (IXI registration on 577 subjects) and the 401. If IXI access is not fixed on day 1, switch to the `mdnakib/ixi-dataset` fallback immediately and log it.

## 8. Definition of done

- [ ] `main` trained; test metrics in `reports/tables.md`
- [ ] Age tables and domain-gap report with before/after AUROC
- [ ] Detection results reported overall and by IXI age band and site
- [ ] At least one ablation (`noattn`) or a logged reason why not
- [ ] 3D Grad-CAM++ overlays and the boundary shortcut check for both pathways
- [ ] Working local Streamlit demo with screenshots
- [ ] `README.md` reproducible, licences and conditions of use noted
- [ ] `reports/results.md` draft with real numbers and limitations
- [ ] Deviations list written; test set unlocked once, after freeze
