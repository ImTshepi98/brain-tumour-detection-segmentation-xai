# PLAN2 P4: shortcut checks on the validation split (`p2-main`)

Source: Kaggle kernel `brats-dual-p2-eval-main-r10-val` (output in `runs/p2-eval-main-r10-val/eval/`). The kernel's own table is `val_checks.md` there; this file restates it with the reasoning behind each check, and adds the one labelled sensitivity analysis.

- **Model:** `p2-main`, best checkpoint = epoch 35 of 40 (selection score 0.873), protocol hash `ba5dc6671f66`.
- **Pre-registration:** tag `plan2-prereg` (commit 6e2fbd9). Every rule below was fixed before any PLAN2 result existed.
- **Data:** 966 val samples in 7 cells (B+r 249, B+p 67, B-r 67, B-s 67, I+p 172, I-r 172, I-s 172).
- **Thresholds, chosen on val by the pre-registered worst-group rule:**
  - detection head: probability 0.5;
  - segmentation-derived detector (pass-B WT volume ≥ v\*): v\* = 1.544 mL.
- **Cost:** 5.9 GPU-h (checks 0.6 h, XAI comparison on 60 cases 5.2 h).

## 1. Verdict (rule applied as written)

| Detector | Verdict | Failed MUST checks |
|---|---|---|
| Segmentation-derived (pass-B WT volume ≥ 1.544 mL) | **shortcut-controlled** | none |
| Detection head (probability ≥ 0.5) | **not shortcut-controlled** | **S7 only** (explanation localisation) |

**What this means.**
- *Dataset shortcut (S1–S6, G1, all passed by both detectors).* The decisions do not depend on which dataset a hemisphere came from. They follow the tumour: they flip when a tumour is added or removed, and stay put when healthy tissue is pasted.
- *Head localisation (S7).* The head's heat maps (Grad-CAM++ and occlusion) do not concentrate on the tumour. Deleting the regions they rank highest also does not lower the probability more than deleting random brain voxels.

So the head's *behaviour* passes every counterfactual test, but its *explanations* fail. The verdict rule makes no exception for that, so the head is reported as not shortcut-controlled.

**The remedy step (P5) was not run.** PLAN2 section 7 pre-registers the cut order P8, then LIME, then P5, "report the failed checks instead".
- About 7.3 GPU-h were left.
- P5 would need about 4 h of training plus about 5.9 h of val evaluation.
- The test read (a MUST) needs about 5.9 h.

P5 did not fit before the 6 Nov deadline, even with LIME removed (DECISIONS, 3 Nov). The frozen model is therefore `p2-main`, epoch 35, with the thresholds above.

## 2. Headline numbers (val)

**Balanced accuracy (BA).** BA is the mean of recall and specificity (Brodersen et al., 2010), so chance is 0.5 whatever the class mix. Worst-group BA is the lower of BA on BraTS and BA on IXI (Sagawa et al., 2020). A detector that only recognises the dataset scores 0.5 on it, because inside each dataset it would call everything the same.

| Detector | Worst-group BA | BA BraTS | BA IXI | Recall (all) | Specificity (all) | AUROC (all) |
|---|---|---|---|---|---|---|
| Head @ 0.5 | **0.944** | 0.944 | 0.948 | 0.902 | 0.985 | 0.971 |
| Segmentation-derived @ 1.544 mL | **0.940** | 0.940 | 0.951 | 0.912 | 0.973 | 0.963 |

**Positive rate per cell.** This is the share of samples called "tumour". Ideal values are high for the + cells and about 0 for the − cells.

| Cell | n | Head | Segmentation-derived |
|---|---|---|---|
| B+r (real BraTS tumour) | 249 | 0.892 | 0.900 |
| B+p (tumour pasted into BraTS healthy hemisphere) | 67 | 0.910 | 0.910 |
| I+p (tumour pasted into IXI) | 172 | 0.913 | 0.930 |
| B-r (real BraTS tumour-free hemisphere) | 67 | 0.000 | 0.000 |
| B-s (healthy-tissue sham into BraTS) | 67 | 0.015 | 0.045 |
| I-r (real IXI hemisphere) | 172 | 0.006 | 0.012 |
| I-s (healthy-tissue sham into IXI) | 172 | 0.029 | 0.047 |

**Segmentation (pass A, 4-channel, real BraTS tumours B+r, n = 249):**
- WT DSC 0.849 (median 0.918);
- TC DSC 0.793 (median 0.926);
- ET DSC 0.764 (median 0.897).

**Pass B (T1/T2 only):**
- WT DSC: B+r 0.768, B+p 0.845, I+p 0.841.
- False-positive volume on real negatives: median 0 mL on both B-r and I-r. Only 1.5% (B-r) and 1.2% (I-r) are over 1 mL.

**Whole-scan view (the two hemispheres combined).**
- Head: recall 0.968 on 188 BraTS scans, specificity 0.988 on 86 IXI scans.
- Segmentation-derived: 0.973 / 0.977.

These scan-level numbers are reported for comparison with PLAN. They have the PLAN confound, though: no BraTS scan is tumour-free.

**Speed:** 111 ms per hemisphere pair on a T4, covering the detection pass, pass B and pass A.

**No perfect-score review needed.** No dataset group reaches recall = specificity = 1.000. The B-r specificity of 1.000 is paired with a BraTS recall of 0.90.

## 3. Check by check

Each row gives what the check does, its pre-registered rule, the result for the head and for the segmentation-derived detector, and the goal: which shortcut it would expose.

| Check | What is done | Rule (MUST unless noted) | Head | Seg-derived | Goal |
|---|---|---|---|---|---|
| **S1** | Within one label, can the score tell BraTS from IXI? AUROC of B-r vs I-r (negatives) and of B+p vs I+p (pasted positives), direction-free (max(AUROC, 1−AUROC)) | both ≤ 0.70 | PASS 0.535 / 0.543 | PASS 0.557 / 0.523 | A dataset-recogniser separates the datasets even when the label is the same; a tumour-detector cannot |
| **S2** | Gap between datasets in specificity on real negatives (B-r vs I-r) and recall on pasted positives (B+p vs I+p) | both gaps ≤ 0.10 | PASS 0.006 / 0.002 | PASS 0.012 / 0.020 | Equal error rates in both datasets; a shortcut makes one dataset "easy" |
| **S3** | Tumour vs no tumour inside each dataset: AUROC of B+r vs B-r and of I+p vs I-r | both ≥ 0.80 | PASS 0.980 / 0.970 | PASS 0.964 / 0.971 | The task must be solvable where dataset identity carries no label information |
| **S4** | Remove the tumour: each detected real BraTS tumour hemisphere replaced by its mirror (healthy side) | ≥ 70% fall below threshold; median logit change ≤ −2 | PASS 79.3% (n 222), −12.2 logits | PASS 79.9% (n 224), −76 mL | The tumour must be **necessary** for the call |
| **S5** | Add a tumour: real negatives receive a pasted tumour (same host voxels as their pair) | ≥ 60% cross in each dataset; rates differ ≤ 0.15 | PASS 0.910 BraTS / 0.912 IXI | PASS 0.910 / 0.929 | The tumour must be **sufficient**, and equally so in both datasets |
| **S6** | Sham: real negatives receive healthy tissue from the same donor, same voxels and intensity matching | ≤ 5% cross in each dataset; median abs logit change < 1 | PASS 1.5% / 2.3%; 0.09 / 0.06 logits | PASS 4.5% / 3.5% | Pasting by itself (seams, texture, BraTS tissue) must **not** trigger a call; only tumour appearance may |
| **S7** | On detected B+r (40 cases): do Grad-CAM++ or occlusion maps concentrate on the tumour, and are they faithful? | one map: median tumour ratio ≥ 2, outside-brain mass ≤ 0.20 and top-30% deletion beats random (Wilcoxon p < 0.05) | **FAIL** (see 3.1) | n/a | The explanation must point at the tumour and must be the reason for the decision |
| **S8** (SHOULD) | Erode the brain mask by 1 voxel (removes skull-strip edge texture) | median abs logit change < 1 and ≤ 5% crossings per real cell | PASS (B+r 2.0%, 0.31; B-r 0%, 0.11; I-r 0%, 0.09) | PASS (0.8% / 1.5% / 0.6%) | Skull-stripping edges differ between datasets and must not drive the call |
| **S9** (report only) | Linear probe on the pooled detection features: BraTS vs IXI within a label | none | AUROC 0.984 (negatives), 0.802 (pasted positives) | – | Shows how much dataset identity the features still *contain* (see 3.2) |
| **G1** | False tumour volume from pass B on real negatives | median ≤ 1 mL on B-r and on I-r | PASS 0 / 0 mL | PASS 0 / 0 mL | Pass B must not paint tumours into healthy brains |

### 3.1 Why S7 fails

In the medians over the 40 detected B+r cases:

| Map | Tumour ratio (needs ≥ 2) | Mass outside brain (needs ≤ 0.20) | Probability drop, top-30% deletion | Drop, random deletion | Wilcoxon p (needs < 0.05) |
|---|---|---|---|---|---|
| Grad-CAM++ (enc[-2]) | 1.07 | 0.53 | 0.37 | 0.66 | 0.97 |
| Occlusion (16 mm cube) | 1.12 | 0.25 | 0.03 | 0.66 | 1.00 |

- **Tumour ratio.** This is the share of map mass on the tumour divided by the tumour's share of the volume. A value near 1 means the map is spread almost evenly over the hemisphere: it does not favour the tumour. Only 8/40 occlusion maps and 2/40 Grad-CAM++ maps reach 2.
- **Outside-brain mass.** Grad-CAM++ still puts about half of its mass outside the brain, as in PLAN. Occlusion is much better (0.25) but misses the 0.20 bar.
- **Deletion.** Zeroing the 30% of brain voxels a map ranks highest lowers the probability *less* than zeroing the same number of random brain voxels.

  Two facts, reported rather than used to excuse the failure:
  - The detected cases are saturated: median probability 1.000.
  - Random scattered zeroing is a strong, out-of-distribution perturbation.

  The prediction survives losing the top-ranked region. That fits a decision supported by several redundant regions, for example tumour plus oedema plus mass-effect cues. It does not fit a decision resting on one compact area.

Behaviourally the head is tumour-driven (S4: −12 logits when the tumour is mirrored away; S5: 91% flip when one is added; S6: shams do nothing). The failure lies in the *pixel-attribution maps*, not in the decision. It is still a failure under the rule as written.

### 3.2 S9: dataset information still in the features

The probe separates BraTS from IXI negatives at AUROC 0.98 using the pooled features. The features therefore still *encode* the dataset (scanner and preprocessing differences are visible). S1, S2 and S6 show that the decision does not *use* that information.

This is the distinction the adversarial remedy (P5) was meant to test. P5 was cut, so the question stays open and is listed as a limitation.

## 4. Sensitivity analysis: S6 on shams of at least 1 mL (labelled, not pre-registered)

This analysis was announced before the evaluation (DECISIONS, 3 Nov). The reason: a sham can be near-empty when the donor's other hemisphere has little tumour-free brain in the region (31 of 172 I-s and 8 of 67 B-s are under 1 mL), and an empty sham passes S6 trivially. S6 was therefore recomputed after dropping those shams (`src/p2_s6_sensitivity.py`; output `s6_sensitivity_val.json`).

| Detector | BraTS: crossings (n) | IXI: crossings (n) | Median abs logit change BraTS / IXI | Under the S6 rule |
|---|---|---|---|---|
| Head | 1.7% (59) | 2.9% (140) | 0.11 / 0.08 | would pass |
| Segmentation-derived | **5.1%** (3 of 59) | 4.3% (139) | – | would **narrowly fail** (5.1% > 5%) |

**Reading.**
- The head's sham result does not depend on the near-empty shams.
- For the segmentation-derived detector, three real-size BraTS shams produced more than 1.544 mL of predicted tumour. Pass B is slightly sensitive to pasted healthy tissue in BraTS hosts.

This does not change the pre-registered verdict, which uses all shams. It is reported next to it as a caveat on the segmentation-derived detector's S6 margin.

## 5. XAI comparison (60 val cases: 20 B+r, 10 each of B+p, I+p, B-r, I-r)

**What is measured.**
- *Deletion AUC* (lower is better): area under the probability curve as the most relevant regions are removed first.
- *Insertion AUC* (higher is better): regions are added starting from a blurred baseline.
- *Random baseline*: the same removal or addition in random order.
- *Tumour ratio and pointing game*: positives only. Pointing counts a hit when the map's maximum falls on the tumour.
- *Sanity check* (Adebayo et al., 2018): Spearman correlation between the trained model's map and the map after randomising the detection head (stage 1) or the whole network (stage 2). A faithful method should drop towards 0.

| Method | Deletion AUC (random 0.396) | Insertion AUC (random 0.962) | Tumour ratio | Outside-brain mass | Pointing | Sanity stage 1 / stage 2 | Seconds per case |
|---|---|---|---|---|---|---|---|
| Grad-CAM++ (enc[-2], new layer) | 0.776 | 0.990 | 1.09 | 0.52 | 0.83 | 0.998 / 0.539 | 0.02 |
| Grad-CAM++ (enc[-1], PLAN's layer) | 0.593 | 0.981 | 0.48 | 0.69 | 0.25 | – | 0.02 |
| Occlusion | 0.470 | 0.999 | 1.12 | 0.25 | 0.43 | 0.284 / −0.036 | 34 |
| Region SHAP | 0.603 | 0.998 | 0.001 | 0.99 | 0.03 | 0.502 / 0.017 | 96 |
| LIME (3 seeds; top-5 Jaccard 0.51) | 0.735 | 0.999 | 0.94 | 0.00 | 0.18 | 0.213 / 0.014 | 158 |

Median Spearman agreement between methods is low (0.19–0.42).

**Reading.**
1. **Deletion.** No method beats random deletion (every deletion AUC is above 0.396). The decision is robust to losing any single top-ranked part, which matches the redundancy noted in 3.1.
2. **Grad-CAM++ fails the sanity check.** Its map hardly changes when the detection head is randomised (0.998), and is still half-correlated with a fully random network (0.54). It mostly reflects the architecture and the input, not the learned decision. Moving it one stage up improved pointing (0.25 → 0.83) and the tumour ratio (0.48 → 1.09), but the sanity result says to read those gains cautiously.
3. **The perturbation methods pass the sanity check** (occlusion, SHAP, LIME: about 0 after full randomisation). They are the more trustworthy family here. They are also 1,700–8,000× slower.
4. **Region SHAP.** It puts almost all of its value on the outside-brain region (0.99). The blurred-baseline region "outside brain" acts as a single large player. This is a property of the region design, reported as found.

## 6. What happens next

- **Test read.** P6 runs once with the frozen epoch-35 checkpoint and the thresholds above. It runs the same checks and the same XAI comparison (kernel `brats-dual-p2-eval-main-r10-test`), and writes `test_lock.json` before reading any test array.
- **Reporting.** A test result that disagrees with val is reported as is. The val and test check tables are side by side in `reports/p2/tables.md`.
