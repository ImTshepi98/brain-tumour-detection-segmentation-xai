# PLAN2.md (v1): A shortcut-controlled dual-head 3D CNN for tumour detection + segmentation

Drafted 1 Nov 2026. Deadline unchanged: submission 6 Nov 2026 (6 Nov is buffer only).
Owner: Tshepiso Lewele (`leweletp`). Supervisor: Dr. Tite Tuyikeze. Proposal: `22983_..._Proposal_...docx` (9 Apr 2026).

PLAN.md is still finished to completion. PLAN2 is a second model built next to it, not a replacement. The owner decides which model the dissertation leads with **after** both results exist (USER ACTION C3).

**Guiding rule: trust before score.** A lower but honest detection score beats a perfect score that comes from a shortcut. Every check that decides whether the model can be trusted is fixed in this file **before** any PLAN2 model is trained (section 4), and failing a check is reported, never hidden or tuned away on test.

---

## A. Why PLAN2 exists (what PLAN taught us; read `reports/xai_notes.md`)

| Finding in PLAN | Consequence | What PLAN2 does about it |
|---|---|---|
| Label = dataset: every BraTS scan is a tumour, every IXI scan is healthy | Detection can be solved by recognising the dataset | Both classes must occur in **both** datasets (section 3: hemispheres + insertion) |
| Simple features separate BraTS and IXI with AUROC 1.000 even after the shared SynthStrip boundary and histogram matching (`reports/domain_gap.md`) | Preprocessing alone cannot remove the gap; BraTS raw scans are not distributed, so "one pipeline from raw" is **not possible** | Make dataset identity useless for the label by design, then test that the model ignores it |
| Removing the tumour (mirroring) moved the logit only -0.14; 186/188 still positive | Tumour was not necessary for the decision | Necessity check is pass/fail (S4) |
| Inserting a tumour into IXI moved the logit +18.15 (61/86 crossed); sham +0.24 (5/86) | Model is tumour-sensitive, so the signal is learnable | Insertion becomes training data, with sham pastes as negatives |
| Detection Grad-CAM++ put 70% of its mass outside the brain; deletion no better than random | Heatmap did not explain a tumour decision | Localisation and faithfulness checks are pass/fail (S7), plus methods that do not depend on gradients (occlusion, SHAP) |
| F1 = AUROC = 1.000 on val and test | A perfect score was the first warning sign | Perfect detection on PLAN2 triggers a review before acceptance (rule 1.7) |
| Detection uses T1 + T2 only (IXI has no T1ce/FLAIR) and zero channels mark IXI | A missing-modality pattern is itself a dataset marker | Detection input is T1 + T2 for **every** sample; no zero channels reach the detection pass |
| Controls existed only after training, on val | Checks could only explain, not guard | Checks are pre-registered and run on val **and** test (section 4) |

## B. Distance from the proposal (log each as a deviation in `DECISIONS.md`)

Unchanged: unified 3D CNN with a shared encoder, detection and segmentation heads, attention gates, BraTS 2021 ET/TC/WT, F1/Recall/DSC/HD95, single-task comparison, Grad-CAM++ on both pathways, Streamlit GUI.
Closer to the proposal than PLAN: SHAP and LIME (proposal timeline, "Model Evaluation"); the ethics section's promise to check for "spurious correlations".

| # | Deviation | Size | Approval |
|---|---|---|---|
| D1 | Detection unit is a **hemisphere**, not a whole scan. Whole-scan results are still reported (scan positive if either hemisphere is positive) | Moderate | Supervisor (USER ACTION C1) |
| D2 | **Synthetic tumour insertion** (and sham pastes) used as training data | Moderate | Supervisor (C1) |
| D3 | Group-balanced sampling; optional domain-adversarial branch | Small | DECISIONS.md |
| D4 | A segmentation-derived detector is compared with the detection head | Small (an ablation) | DECISIONS.md |
| D5 | 2 mm resolution, as in PLAN (proposal: 1 mm) | Carried over | Already logged |
| D6 | Test subjects are the same as PLAN's and were read once for PLAN's models. No PLAN2 design choice uses PLAN's test numbers (the design comes from val controls) | Disclosure | DECISIONS.md |

## C. USER ACTION items (collect in `PROGRESS.md`)

- **C1** Supervisor approval of D1 and D2 (an email-ready paragraph is in `reports/p2/deviations_for_supervisor.md`, Phase P0). *Status 1 Nov: the owner reports that the supervisor will review the findings together with the owner when the project is done; the owner authorised going ahead. Recorded as approval deferred to that review, not as given.*
- **C2** Review the pre-registered thresholds in section 4. The owner may change any number **before** the tag `plan2-prereg` is created; after that, changes are deviations. *Done 1 Nov: owner gave full go-ahead with section 4 as written (chat).*
- **C3** After both plans finish: choose which model the dissertation leads with. Claude Code must not choose this.
- **C4** Only the owner sets any `visual_review_complete` flag (insertion QC panels, Phase P1).

## 1. Operating rules (PLAN.md section 1 applies in full; additions below)

1. All PLAN rules hold: autonomous work, `DECISIONS.md`, CPU dry run before every push, seed 42, never tune on test, never print tokens, never modify the owner's notebooks, one GPU notebook at a time (shared with PLAN's remaining runs).
2. **Pre-registration first.** Section 4 is committed and tagged `plan2-prereg` before any PLAN2 training kernel is pushed. The tag's commit hash is quoted in every PLAN2 report.
3. **One remedy iteration, decided in advance.** If a [MUST] check fails on val, exactly one pre-declared remedy is allowed (P5: enable the domain-adversarial branch). No other iteration is driven by check results.
4. **Test is read once** for each PLAN2 model, with checkpoint, threshold and check code frozen. The shortcut checks run on test too, in the same read.
5. **Split isolation for synthetic data.** Donor tumours and hosts always come from the same split (train→train, val→val, test→test). Donor and host pairing is fixed by seed and written to a file before use.
6. Reports for PLAN2 go to `reports/p2/`, runs to `runs/p2-*`, kernels to `kaggle/p2-*` (Kaggle slugs `brats-dual-p2-*`).
7. **A perfect score is a warning, not a result.** If any group shows recall and specificity both equal to 1.000 on val, write a short review in `reports/p2/perfect_score_review.md` (which checks rule out a shortcut, which do not) before accepting.

## 2. Data design

### 2.1 Hemisphere samples
- Arrays: PLAN's 2 mm SRI24 arrays (4x96x112x80) with the shared SynthStrip boundary (border 0 mm). No new preprocessing of BraTS or IXI.
- Midline: the x index that maximises the left-right symmetry of the mean **train** brain mask (expected about 41; record the value and a montage). Each hemisphere is cut at the midline, padded with zeros to a fixed width (expected 48), and **right hemispheres are flipped** so the midline is always at the same edge.
- Labels, fixed in advance:
  - **positive:** WT volume in that hemisphere >= 1.0 mL (>= 125 voxels at 2 mm);
  - **negative:** zero WT voxels in that hemisphere **and** no WT voxel within 4 mm (2 voxels) of the midline;
  - **excluded:** anything in between (tumour < 1 mL in that half, or tumour touching the midline on the negative side). Report how many hemispheres each rule excludes, per split.
- Normalisation: re-z-score T1 and T2 **within each hemisphere's brain mask**, so global scan intensity statistics no longer carry over.

### 2.2 Synthetic positives and sham negatives
Reuse `evaluate.insert_tumour` (intensity matched to the host relative to the donor's healthy brain, re-z-scored; sham mode pastes the donor's healthy mirror tissue at the same voxels). Tests already cover both.

| Cell | Dataset | Label | How it is made |
|---|---|---|---|
| B+r | BraTS | 1 | Real tumour hemisphere |
| B+p | BraTS | 1 | Tumour from another BraTS case (same split) pasted into a BraTS negative hemisphere |
| B-r | BraTS | 0 | Real tumour-free hemisphere |
| B-s | BraTS | 0 | Sham paste into a BraTS negative hemisphere |
| I+p | IXI | 1 | BraTS tumour (same split) pasted into an IXI hemisphere |
| I-r | IXI | 0 | Real IXI hemisphere |
| I-s | IXI | 0 | Sham paste into an IXI hemisphere |

- Pasting runs on the fly during training (new donor and host each epoch, seeded). Val and test use one fixed pairing file each.
- Donors are hemisphere tumours (the WT mask on the donor side), mirrored if needed to match the host side.
- Training labels for the segmentation heads on pasted samples = the pasted WT mask (and ET/TC for BraTS hosts, where all 4 channels are pasted).
- **Known limit:** pasted tumours have no mass effect and no peritumoural oedema beyond the WT mask. Only BraTS has real positives, so "real tumour with mass effect" stays BraTS-only. This is stated in the limitations, and the checks in section 4 measure what it costs.

### 2.3 Sampling
Each batch draws dataset x label equally (25% B+, 25% B-, 25% I+, 25% I-). Within BraTS positives 50% real / 50% pasted; within negatives 50% real / 50% sham, in both datasets. So, in the training distribution, the dataset says nothing about the label, and "pasted" says little about the label.

### 2.4 Augmentation (on GPU)
PLAN's flips (except left-right, which hemispheres make meaningless), intensity scale and shift, and noise, plus **random brain-boundary erosion of 0-2 voxels** (so that the boundary shape is not a stable cue).

## 3. Model

| Item | Decision |
|---|---|
| Network | PLAN's 5-level residual U-Net with attention gates, input 4x48x112x80 |
| Detection pass | T1 + T2 only, for every sample (no zero channels in this pass) |
| Segmentation pass A | 4 channels, BraTS samples (real and pasted), ET/TC/WT, as in PLAN |
| Segmentation pass B | T1 + T2 only, **WT only**, every sample (IXI and sham included, empty targets for negatives). This pass drives the segmentation-derived detector |
| Loss | `L = L_segA + L_segB + 0.5 * L_cls` (+ `0.1 * L_adv` only in `p2-adv`) |
| Domain-adversarial (`p2-adv`) | Gradient-reversal head on the pooled detection features, predicting BraTS vs IXI **within each label** (Ganin 2016) |
| Model selection | `best.pt` by `0.5 * mean DSC (pass A, real BraTS) + 0.5 * worst-group balanced accuracy at 0.5` |
| Threshold | Chosen on val to maximise the **worst-group** balanced accuracy (not F1 on the pooled set) |
| Optimiser, precision, resume, memory budget | As PLAN (AdamW 3e-4 cosine, FP32, `--max-rss-gb 29`, resume across sessions) |

**Segmentation-derived detector:** a hemisphere is positive if the pass-B WT volume >= v*, with v* chosen on val by the same worst-group rule. It gets the same metrics and the same checks as the head.

## 4. Pre-registered checks (fixed before training; tag `plan2-prereg`)

All checks run on val (for the decision) and are repeated in the single test read. The val pairing files for S4-S6 are written in P1. "Group" = the cells in 2.2. Logit changes are reported alongside probabilities.

| ID | Check | Pass rule | Tier |
|---|---|---|---|
| S1 | Dataset from the detection logit, within each label (real negatives B-r vs I-r; pasted positives B+p vs I+p) | AUROC <= 0.70 for both | MUST |
| S2 | Group gap at the frozen threshold | TNR(B-r) vs TNR(I-r) differ by <= 0.10, and TPR(B+p) vs TPR(I+p) differ by <= 0.10 | MUST |
| S3 | Within-dataset discrimination | AUROC(B+r vs B-r) >= 0.80, and AUROC(I+p vs I-r) >= 0.80 | MUST |
| S4 | Necessity: real BraTS positives detected at threshold, tumour replaced with mirrored healthy tissue (PLAN's `mirror_lr`) | >= 70% drop below threshold and median logit change <= -2 | MUST |
| S5 | Sufficiency on held-out negatives: tumour pasted into val negatives of **both** datasets | >= 60% cross the threshold in each dataset, and the rates differ by <= 0.15 | MUST |
| S6 | Sham: sham paste into val negatives of both datasets | <= 5% cross in each dataset and median abs logit change < 1 | MUST |
| S7 | Localisation on true positives (detection Grad-CAM++ and occlusion) | median tumour ratio >= 2 and median mass outside the brain <= 20% for at least one of the two; deletion of top-30% voxels drops the probability more than random deletion (paired Wilcoxon p < 0.05) | MUST |
| S8 | Boundary: 1-voxel erosion of the brain mask | median abs logit change < 1 and <= 5% crossings in each group | SHOULD |
| S9 | Probe: logistic regression on pooled detection features, BraTS vs IXI within label | **Report only** (expected high, since the scans differ; S1 asks whether the *decision* uses it) | SHOULD |
| G1 | Segmentation false positives on real negatives (pass B) | median predicted WT <= 1 mL in both B-r and I-r | MUST |

**Operational definitions (added 1 Nov before the tag; they make the rules above computable and change no threshold):**
- *Hemisphere geometry.* Axis 0 of the arrays is left-right (as `evaluate.LR_AXIS`). The midline m is the cut between voxels m-1 and m that maximises the Dice overlap of the mean train brain mask (> 0.5) with its mirror image (voxel i <-> 2m-1-i). Left hemisphere = x[m-48:m], right = x[m:m+48] flipped, both zero-padded to 48, so the midline is always the last index. The brain share lost by the 48-voxel crop is reported.
- *Groups and worst group.* Per dataset d in {BraTS, IXI}: BA_d = (TPR over all positive cells of d + TNR over all negative cells of d) / 2 (BraTS: B+r, B+p vs B-r, B-s; IXI: I+p vs I-r, I-s). Worst-group balanced accuracy = min(BA_BraTS, BA_IXI). Used for model selection (at 0.5) and for the threshold (maximised on val; ties go to the candidate closest to 0.5).
- *S1.* AUROC is direction-free: max(AUROC, 1 - AUROC) of the detection logit separating the two datasets, so a value far below 0.5 also fails.
- *S3/S5/S6 cells.* S3 uses real hemispheres only for BraTS (B+r vs B-r) and pasted vs real for IXI (I+p vs I-r). S5/S6 paste into every real val negative (B-r and I-r), one fixed donor each (seed 42); "cross" means the probability is below the frozen threshold before and at or above it after.
- *S4.* Applied to the B+r hemispheres detected at the threshold: the tumour is replaced with `mirror_tumour` on the full scan, then the hemisphere is cut again.
- *S7.* Occlusion and Grad-CAM++ maps on the B+r hemispheres detected at the threshold. Tumour ratio and outside-brain mass as in PLAN (`explain.mass_stats`); deletion = zero the top 30% of brain voxels by the map vs the same number of random brain voxels (seed 42), probability drop compared with a one-sided paired Wilcoxon test. Passes if at least one of the two maps meets the ratio and outside-mass rule, and that map also passes deletion.
- *G1.* Predicted WT volume of pass B (sigmoid > 0.5) in mL at 2 mm, on B-r and I-r.

**Verdict rule, fixed now:** the model is called **"shortcut-controlled"** only if every MUST check passes on val. If one fails: apply the single remedy (P5) and re-run val; if it still fails, the model is reported as **"not shortcut-controlled"**, naming the failed checks, and the test read still happens once so the failure is measured, not hidden. A test result that disagrees with val is reported as is.

## 5. Explainability (answer to "can SHAP/LIME be added to Grad-CAM++?")

Yes: as **complementary** explanations shown side by side, not fused into one map. Whether one is better is measured, not assumed.

| Method | What it explains | Settings (fixed now) |
|---|---|---|
| Grad-CAM++ (kept, primary per proposal) | Both pathways | PLAN's implementation; detection target layer moved **one stage up** (about 8 mm cells, previously 16 mm); the old layer reported too |
| Occlusion sensitivity | Detection | 16 mm cube, stride 8 mm, filled with the blurred baseline below; gradient-free |
| SHAP over regions | Detection and pass-B WT volume | Features = predicted-tumour component(s) + SRI24 atlas regions (if the licence allows; otherwise 16 mm cubes inside the brain) + outside-brain; permutation Shapley, 2000 evaluations; baseline = Gaussian blur sigma 8 mm of the input (zero baseline as a sensitivity analysis) |
| LIME | Detection | 3D SLIC supervoxels on T1/T2 (about 100), 1000 samples, same baseline; run with 3 seeds and report top-5 stability (Jaccard) |

Comparison (`reports/p2/xai_compare.md`, val; repeated on test in the single read):
- **Faithfulness:** deletion and insertion AUC (most-relevant-first) for each method, against random ordering.
- **Localisation:** tumour ratio, and a pointing game (does the top region or voxel fall in the tumour).
- **Sanity:** cascading weight randomisation (Adebayo 2018). The maps of a randomised model should change (Spearman with the trained model's map should be low).
- **Agreement:** region-level Spearman between methods.
- **Cost:** seconds per case.

GUI [SHOULD]: hemisphere scores for both sides, a SHAP region bar chart next to the Grad-CAM++ overlay, and an occlusion toggle.

## 6. Phases

### P0: Pre-registration and deviations (CPU, about 1 h) [MUST]
- Write `reports/p2/deviations_for_supervisor.md` (D1-D6, one paragraph each).
- Owner reviews section 4 (C2). Then commit and tag `plan2-prereg`.
- **Accept:** tag exists; its hash is in `PROGRESS.md`.

### P1: Hemisphere index and pairing files (CPU) [MUST]
`src/hemi.py`, `src/p2_pairs.py`.
- Midline, hemisphere crops, labels (2.1), exclusion counts per split, montage of 6 hemispheres per cell.
- Fixed val and test pairing files for insertion, sham and S4-S6, plus 12 QC panels of pasted samples (owner reviews, C4).
- **Accept:** unit tests (midline flip is an involution; the label rules on toy masks; the pairing never crosses splits); counts per cell and split in `reports/p2/data.md`; at least 150 val hemispheres in each of B+r, B-r and I-r.

### P2: Training code and dry run (CPU) [MUST]
- `datasets.py` gets a hemisphere mode with the 2.3 sampler, on-the-fly pasting and boundary erosion; `model.py` gets pass B and the optional adversarial head; `train.py` gets the worst-group selection score.
- `src/shortcut_checks.py` implements S1-S9 and G1 as one function used for both val and test.
- **Accept:** full local dry run (train, resume, val checks, test step) passes; existing tests still pass.

### P3: `p2-main` (GPU) [MUST]
- Smoke test (at most 15 min), then train to the epoch count the smoke test supports (target 40), across sessions with the memory budget, as in PLAN.
- **Accept:** finishes; val mean DSC (pass A) rises; worst-group balanced accuracy above 0.5.

### P4: Val evaluation and checks (GPU) [MUST]
- Metrics per group and overall (detection: recall, specificity, F1, balanced accuracy and AUROC per dataset; segmentation: DSC/HD95 on real BraTS as in PLAN, WT DSC on pasted samples, FP volume on negatives) and S1-S9 and G1, for both the head and the segmentation-derived detector.
- Grad-CAM++, occlusion, SHAP and LIME on 60 val hemispheres (20 B+r, 10 B+p, 10 I+p, 10 B-r, 10 I-r).
- **Accept:** `reports/p2/val_checks.md` with a pass/fail line per check and the verdict.

### P5: Single remedy, only if a MUST check failed (GPU) [SHOULD]
- Train `p2-adv` (same as `p2-main` plus the adversarial branch). Re-run P4 on val. The better of the two by the selection score **among those that pass the most MUST checks** is frozen.

### P6: Test read, once (GPU) [MUST]
- Frozen checkpoint and threshold; all metrics, all checks and the XAI comparison on test in one kernel. Whole-scan results reported too, with the caveat that whole-scan BraTS negatives do not exist.
- **Accept:** `reports/p2/tables.md` with every PLAN2 number, and the check table for val and test side by side.

### P7: Write-up material [MUST]
- `reports/p2/results.md`: PLAN vs PLAN2 side by side (scores, checks, heatmaps). The expected story: PLAN2 scores lower and passes the checks, PLAN scores perfectly and fails them; if that does not happen, say what did.
- Limitations: pasted tumours lack mass effect; real positives only in BraTS; 2 mm; single seed; test subjects reused from PLAN (D6); T1 + T2-only detection.
- [SHOULD] GUI additions (section 5).

### P8 [COULD]
- 4-channel detector within BraTS only (B+r vs B-r) as a reference.
- Leave-one-IXI-site-out check of S2.

## 7. GPU budget (shared with PLAN's remaining runs; check the real quota first)

About 13 h of the week's quota will be used by PLAN after `segonly`/`clsonly` and their evaluations.

| Job | Hours (estimates; replace with smoke-test numbers) |
|---|---|
| P3 smoke + `p2-main` | 4.5 |
| P4 val checks + XAI | 1.5 |
| P5 `p2-adv` + val (only if needed) | 5.0 |
| P6 test read + checks + XAI | 1.5 |
| Reserve | 2.0 |

MUST total about 9.5 h; with P5, about 14.5 h. If quota runs short, cut in this order: P8, LIME, P5 (report the failed checks instead), SHAP's zero-baseline sensitivity. Never cut: the pre-registration, S1-S7, G1 or the test-read rule.

## 8. Schedule (interleaved with PLAN; one GPU notebook at a time)

| Date | PLAN | PLAN2 |
|---|---|---|
| Sun 1 Nov | `segonly` sessions | P0 drafted; owner review (C2) |
| Mon 2 Nov | `clsonly`; evaluations of both | P1, P2 (CPU), dry run; smoke test |
| Tue 3 Nov | Phase 9 drafting (CPU) | P3 `p2-main` training |
| Wed 4 Nov | | P4 val checks; P5 if needed |
| Thu 5 Nov | Final tables | P6 test read; P7 |
| Fri 6 Nov | Buffer | Buffer |

**Biggest risks:** the schedule (P5 may not fit; then report the failed checks instead), and whether S3/S5 can pass at all with pasted tumours (an honest negative result is still a result).

## 9. Definition of done

- [ ] `plan2-prereg` tag before any PLAN2 training
- [ ] `p2-main` trained (and `p2-adv` if a MUST check failed)
- [ ] S1-S9 and G1 reported for val and test, with the verdict rule applied as written
- [ ] Head vs segmentation-derived detector compared
- [ ] Grad-CAM++, occlusion, SHAP and LIME compared on faithfulness, localisation, sanity and cost
- [ ] `reports/p2/results.md` with PLAN vs PLAN2 and the limitations
- [ ] Deviations D1-D6 in `DECISIONS.md`; supervisor approval recorded or listed as outstanding
