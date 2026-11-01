# Results and Discussion (draft skeleton with real numbers)

Every number below comes from `reports/tables.md` (generated) or the report it names. Sections marked **PENDING** wait for the `segonly`/`clsonly` runs.
Split sizes: BraTS train 875 / val 188 / test 188; IXI train 404 / val 86 / test 87. Seed 42 throughout; test read once per model.

## 1. Data
- **BraTS 2021 Task 1:** 1251 cases, all four sequences, ET/TC/WT labels. Pooled to 2 mm whole volumes, 4x96x112x80 (`reports/phase1.md`):
  - 1 ET region was lost by pooling (BraTS2021_01512, 32 voxels at 1 mm);
  - 5 train+val cases have no TC at all.
- **IXI:** 577 healthy subjects with T1 + T2 (Guys 319, HH 185, IOP 73), registered to a BraTS-space template built from 60 BraTS train cases. 577/577 processed; template Dice median 0.953-0.960 per site; 34 flagged and all kept after the owner's review (`reports/phase2.md`).
- **Ages (IXI only):** 558/577 have an age; mean 48.7, SD 16.5, range 20-86. Age band 40-75: 341 inside (`reports/age_ixi.md`). BraTS Task 1 has no per-case ages.

## 2. Domain gap (Phase 3, `reports/domain_gap.md`)
- A gradient-boosting classifier separates BraTS from IXI on **tumour-free tissue alone** with AUROC 1.000 (5-fold CV, train+val).
- It stays at 0.999-1.000 after:
  - mask hole-filling and closing;
  - histogram matching (both directions);
  - re-stripping both cohorts with one shared SynthStrip boundary (border 0 mm).
- Strongest cue: the outer-shell T1 intensity (a rim of CSF), then brain volume and surface/volume.
- **Discussion point:** with healthy and tumour scans from different sources, any detector can in principle separate them without looking at the tumour. Section 5 tests whether ours does.

## 3. Model and training
- Dual-head 3D residual U-Net with attention gates on all skip connections, 5.88 M parameters, FP32, AdamW 3e-4 with cosine decay, batch 4, 40 epochs.
- Detection pass: T1 + T2 only. Segmentation pass: all 4 sequences, BraTS only. Loss `L_seg + 0.5 L_cls`.
- Best checkpoints:

| run | best epoch | val score |
|---|---|---|
| `main` | 32 | 0.932 |
| `noattn` | 31 | 0.940 |

- **Engineering note:** host memory leaked about 1 GB per epoch. Training therefore stops cleanly at a memory budget and resumes in a new session from `last.pt`. The protocol hash is identical across sessions.

## 4. Main results

### 4.1 Detection (test, threshold frozen on val)
| run | recall | specificity | F1 | AUROC |
|---|---|---|---|---|
| `main` (thr 0.017) | 1.000 (188/188) | 1.000 (87/87) | 1.000 | 1.000 |
| `noattn` (thr 0.5) | 1.000 | 1.000 | 1.000 | 1.000 |
| `clsonly` | PENDING | | | |

- Specificity is 1.000 in every IXI age band and site; Wilson lower bounds range from 0.44 (n=3) to 0.96 (n=87).
- The margin is wide: the lowest tumour probability is 0.99994 and the highest healthy probability 0.0011.
- **Do not read these as tumour-detection performance on their own:** see section 5.

### 4.2 Segmentation (BraTS test, 2 mm, n = 188)
| run | ET DSC | TC DSC | WT DSC | WT HD95 median (mm) |
|---|---|---|---|---|
| `main` | 0.849 ± 0.215 | 0.900 ± 0.169 | 0.921 ± 0.089 | 2.8 |
| `noattn` | 0.844 ± 0.219 | 0.892 ± 0.185 | 0.921 ± 0.086 | 2.8 |
| `segonly` | PENDING | | | |

- Medians for `main`: 0.920 / 0.957 / 0.948.
- Failures are rare: 2 of 188 test cases have WT DSC < 0.5. The worst is BraTS2021_01293, a large cystic tumour, dark on T1 and FLAIR, where only rim fragments were segmented.
- Context: the owner's earlier 1 mm patch baseline reached WT 0.784, TC 0.692, ET 0.699 on val after 5 epochs. That is a different resolution, training length and protocol, so the comparison is not like-for-like.

### 4.3 Ablations
- **Attention gates (`noattn` − `main`, paired per case on test):**

| region | mean DSC difference (95% CI) |
|---|---|
| ET | −0.005 (−0.017 to 0.010) |
| TC | −0.008 (−0.017 to −0.001) |
| WT | 0.000 (−0.003 to 0.003) |

  - On val, the direction is reversed: ET +0.015, TC +0.018, both CIs touching 0.
  - **Conclusion:** no meaningful effect of attention gates at this resolution. The hypothesis that attention gates improve segmentation is not supported.
  - Attention gates cost 14 ms per case on a T4 (107 vs 94 ms).
- **Dual-head vs single-task (`segonly`, `clsonly`):** PENDING. This tests the proposal's hypothesis that shared features improve both tasks.

### 4.4 Efficiency
- Detection + segmentation, batch 1: 107 ms per case on a T4 GPU and 2.23 s on a local 12-core CPU.

## 5. Explainability and shortcut analysis (`reports/xai_notes.md`)
Counterfactual controls on val (`main`; `noattn` agrees):

| control | question | median logit change | crossings |
|---|---|---|---|
| Tumour replaced by mirrored healthy tissue | Is the tumour **necessary**? | −0.14 | 186/188 still positive |
| BraTS tumour pasted into healthy IXI | Is the tumour **sufficient**? | +18.15 | 61/86 become positive |
| Healthy BraTS tissue pasted at the same voxels (sham) | Tumour, or just a BraTS patch? | +0.24 | 5/86 |
| Brain mask eroded 1 voxel | Boundary cue? | IXI +1.23, BraTS +0.03 | IXI 2/86 |

- Insertion effect grows with tumour volume (Spearman 0.65).
- Sham control, by the rule fixed before the run: classed as **"mixed"**. The logit criterion was met; the crossings criterion was missed by one case. Paired on the same host and donor, the tumour moves the logit 16.1 more than the sham, and further in 84/86 pairs.

Grad-CAM++:
- **Segmentation pathway:** concentrates on the tumour (tumour ratio median 7.6 val, 7.9 test).
- **Detection pathway:** does **not**:
  - 70% of its mass lies outside the brain;
  - the tumour gets less than its volume share (ratio 0.33);
  - deleting the top-CAM voxels changes the probability no more than deleting random voxels.
- The detection target layer is coarse (16 mm cells).

**Finding:** the detector uses redundant cues, either of which is enough.
- For BraTS scans, the tumour-free remainder of the image already decides: dataset identity, consistent with section 2.
- For healthy scans, adding a tumour is usually enough to flip the decision.
- So perfect detection scores reflect both tumour sensitivity and a dataset shortcut. The study cannot separate the two with these data.
- **This is a contribution, not only a limitation.** Standard metrics (F1 = AUROC = 1) did not reveal the problem, and Grad-CAM++ alone hinted at it. Counterfactual controls fixed before the runs made it measurable.
- The design of a shortcut-controlled follow-up is in `PLAN2.md`.

## 6. GUI (`reports/gui/`)
- Streamlit prototype: bundled or uploaded BraTS-like case, tumour probability and decision, ET/TC/WT overlay with volumes in mL, Grad-CAM++ toggle, slice slider, inference time, research-prototype banner. Three screenshots are in `reports/gui/`.

## 7. Deviations from the proposal (full list with reasons in `DECISIONS.md`)
1. **2 mm whole volumes instead of 1 mm.** The detection head needs whole-volume context, and 1 mm whole volumes would be too slow for the GPU budget.
2. **Healthy data from IXI.** The proposal names only BraTS, which has no healthy scans, so a detection head needs an external healthy cohort.
3. **Detection uses T1 + T2 only.** IXI has no T1ce or FLAIR.
4. **Shared SynthStrip brain boundary for both cohorts.** Owner decision after the domain-gap result.
5. **Training split into sessions with a memory budget.** Engineering only; it changes no result.
6. **Added counterfactual controls** (mirroring, insertion, sham, erosion), on val only. Insertion and sham were approved by the owner and their rules fixed before the runs.
7. **SHAP and LIME (proposal timeline) not done in PLAN.** They are designed into PLAN2 with a faithfulness comparison.
8. **Hypothesis wording ("significantly improve", "superior").** The attention-gate part is not supported. The dual-head vs single-task part is PENDING.

## 8. Limitations
- Single seed.
- 2 mm resolution.
- Detection without T1ce or FLAIR.
- Healthy and tumour scans come from different sources, with domain AUROC 1.000 and a shown dataset shortcut (section 5), so detection scores are not evidence of tumour detection alone.
- No per-case BraTS ages, so detection is not age-matched; results are reported by IXI age band and site instead.
- Insertion and sham controls paste tissue without mass effect.
- Grad-CAM++ detection maps are coarse.
- BraTS-style preprocessing is assumed for any new data.
- Research prototype; not for clinical use.
