# PLAN2 results: a shortcut-controlled follow-up to PLAN

All numbers come from `reports/p2/tables.md`, which `src/p2_tables.py` generates from the kernel outputs. Val: `brats-dual-p2-eval-main-r10-val`. Test: `brats-dual-p2-eval-main-r10-test`, read once; `test_lock.json` was written at 21:47 on 3 Nov, and the val results inside it are byte-identical to the val run.

## 1. Question and answer

**PLAN** scored perfectly: test F1 = AUROC = 1.000 on whole scans. But its counterfactual controls showed a **dataset shortcut**. Every tumour scan came from BraTS and every healthy scan from IXI, so "which dataset is this?" was enough to decide.

**PLAN2** asked whether detection can be made to depend on the tumour itself, and whether that can be shown with pass/fail checks fixed in advance. Three changes made this possible:
- each hemisphere is treated as its own sample;
- tumours, and healthy-tissue shams, are pasted so that both classes occur in both datasets;
- sampling is balanced across the four dataset-by-label groups.

**Answer, by the verdict rule fixed before training (tag `plan2-prereg`):**

| Detector | Val | Test |
|---|---|---|
| Segmentation-derived (pass-B tumour volume ≥ 1.544 mL) | shortcut-controlled | **shortcut-controlled** |
| Detection head (probability ≥ 0.5) | not shortcut-controlled (S7) | **not shortcut-controlled (S7)** |

The expected story was "PLAN2 scores lower and passes the checks; PLAN scores perfectly and fails them". It held for the segmentation-derived detector. It held only in part for the detection head:
- the head's **decisions** pass every behavioural check;
- its **explanations** (heat maps) fail the localisation check S7 on both splits.

## 2. PLAN vs PLAN2 side by side (test split)

| | PLAN `main` | PLAN2 head | PLAN2 seg-derived |
|---|---|---|---|
| Unit of decision | whole scan | hemisphere | hemisphere |
| Tumour-free samples from BraTS? | no (0) | yes (real B-r, sham B-s) | yes |
| Tumours in IXI? | no | yes (pasted I+p) | yes |
| F1 | 1.000 | 0.942 | 0.951 |
| AUROC | 1.000 | 0.980 | 0.978 |
| Worst-group BA (Sagawa et al., 2020) | not defined: one class per dataset | 0.937 | 0.945 |
| Can the score tell the datasets apart within a label? (S1, ≤ 0.70) | not testable; dataset AUROC 1.000 | 0.527 / 0.518 | 0.575 / 0.519 |
| Tumour necessary? (removing it lowers the score) | no: median −0.14 logit, 186/188 stay positive | yes: −12.4 logit, 79% fall below | yes: 83% fall below |
| Tumour sufficient? (adding one raises it) | yes: +18.2 logit, 61/86 cross | yes: 94% / 90% cross | yes: 95% / 91% cross |
| Healthy-tissue sham triggers a call? | 5/86 | 3.1% / 4.6% | 4.7% / 4.0% |
| Heat map on the tumour? (Grad-CAM++ tumour ratio) | 0.33, with 70% of mass outside the brain | 1.17, 53% outside | n/a |
| Segmentation WT / TC / ET DSC | 0.921 / 0.900 / 0.849 (whole scans) | 0.872 / 0.807 / 0.773 (tumour hemispheres B+r) | same pass A |

**Reading.**
- PLAN2's detection numbers are lower, but they are honest: they hold within each dataset, where dataset identity carries no label information.
- PLAN's perfect numbers could not be separated from the shortcut.
- The segmentation DSC is lower in PLAN2 for two reasons. Each hemisphere is segmented without the other side as context. And PLAN2 spends half of every batch on tumour-free hemispheres and synthetic samples.
- The two DSC rows are computed on different units (hemispheres vs whole scans), so the comparison is indicative only.

## 3. The checks on test

Full side-by-side tables are in `tables.md`. Summary:

- **S1–S6 and G1, MUST: pass for both detectors on val and test.**
  - Within a label, the score cannot tell BraTS from IXI (AUROC 0.52–0.58; 0.5 = chance).
  - Error rates match across datasets (gaps ≤ 0.045).
  - Mirroring the tumour away lowers the head's score by a median of 12.4 logits.
  - A pasted tumour flips 90–95% of hemispheres in both datasets.
  - Healthy-tissue shams flip ≤ 4.7%.
  - Pass B paints no tumour into real healthy hemispheres (median 0 mL).
- **S7, MUST, head only: fails on both splits.**
  - Grad-CAM++: tumour ratio 1.17, 53% of mass outside the brain.
  - Occlusion: tumour ratio 1.19, 22% outside the brain.
  - Deleting a map's top 30% does not lower the probability more than random deletion (p ≈ 1.00).
- **S8, SHOULD: passes on val, fails on test for both detectors.**
  - Eroding the brain mask by one voxel flips 4 of 67 real tumour-free BraTS hemispheres (6.0%, limit 5%). The other cells are 0–1.6%.
  - A recommended check, so it does not enter the verdict. It points to a small sensitivity to the skull-strip boundary in BraTS hosts.
- **S9, report only:** the pooled features still encode the dataset (probe AUROC 0.993 on negatives). The decision does not use it (S1, S2, S6). The adversarial remedy that would have tested removing it was cut (section 5).
- **S6 on shams ≥ 1 mL** (labelled sensitivity analysis, not pre-registered): both detectors narrowly exceed 5% on test (head IXI 5.6%, seg-derived BraTS 5.4%). The pre-registered S6 uses all shams and passes, but its margin is thin.

**Whole-scan view.** The segmentation-derived detector reaches recall 1.000 (188 BraTS scans) and specificity 1.000 (87 IXI scans) at scan level. This is reported, not celebrated: it is the PLAN setting, where scan-level BraTS negatives do not exist, so dataset identity cannot be ruled out at that level. The hemisphere-level checks above are the evidence. No hemisphere-level group is perfect.

## 4. Explainability comparison (60 cases per split)

| Method | Faithful? (deletion AUC vs random; lower is better) | Passes the randomisation sanity check? | On the tumour? (ratio, pointing) | Time per case |
|---|---|---|---|---|
| Grad-CAM++ (one stage up) | no (0.82 vs 0.35) | **no**: 1.00 after randomising the head, 0.53 after randomising everything | 1.06, 0.60 | 0.02 s |
| Grad-CAM++ (PLAN's layer) | no (0.74) | – | 0.50, 0.35 | 0.02 s |
| Occlusion | no (0.47) | yes (0.16 / 0.08) | 1.24, 0.33 | 37 s |
| Region SHAP | no (0.49) | yes (0.41 / −0.03) | about 0 (99% on "outside brain") | 102 s |
| LIME | no (0.77) | yes (0.26 / −0.04) | 1.05, 0.12 | 171 s |

**Reading.**
- No method beats random deletion. The decision survives losing any one top-ranked region, which suggests redundant tumour-related cues (tumour, oedema, the hemisphere's intensity profile) rather than one compact spot.
- Grad-CAM++ is the method the proposal named, and the fastest. But it fails the Adebayo et al. (2018) sanity check: its map reflects the architecture and the input more than the learned decision.
- The perturbation methods pass the sanity check and are the more trustworthy family. They are 1,800–8,500 times slower.
- Region SHAP's mass on "outside brain" comes from its region design: one large region replaced by a blurred baseline. That is a property of the design, not evidence for the model.

## 5. Deviations and cuts during P4–P6

- **P5 (adversarial remedy) cut** under the pre-registered cut order. About 7.3 GPU-h were left; P5 plus its val run needed about 10 h, and the test read, which is mandatory, needed about 6 h. The remedy targets dataset identity, which already passes (S1–S6). It does not target S7 (DECISIONS, 3 Nov).
- **S6 sensitivity analysis on shams ≥ 1 mL:** announced before the evaluation, labelled as not pre-registered.
- **Code changes after the evaluation:** only optional arguments for the GUI, with unchanged defaults. The evaluation kernels ran the code dataset built from commit c749c18.

## 6. Comparison with the literature

Compare PLAN2 with published work on the **metric that fits the claim**:
- **Segmentation:** BraTS DSC per region (ET/TC/WT) and HD95. These are directly comparable with published BraTS 2021 results, with the caveats that we work at 2 mm and on hemispheres.
- **Detection:** published "tumour vs healthy" classifiers report accuracy, F1 or AUROC, usually on mixed-source data. Our F1/AUROC can be put beside them. Only worst-group BA and the checks show whether a score survives dataset control, and most papers do not report anything comparable. That is the point to make.

| Work | Data | Task | Metric | Their value | Ours (PLAN) | Ours (PLAN2) |
|---|---|---|---|---|---|---|
| _(owner to fill from the citation manager)_ | | segmentation | WT / TC / ET DSC | | 0.921 / 0.900 / 0.849 | 0.872 / 0.807 / 0.773 |
| | | detection | accuracy / F1 / AUROC | | 1.000 / 1.000 | 0.942 F1, 0.980 AUROC |
| | | detection, controlled | worst-group BA | usually not reported | not defined | 0.937 (head), 0.945 (seg) |

## 7. Limitations

- Pasted tumours have no mass effect. Shams can be near-empty (30 of 174 IXI test shams are under 1 mL).
- Real positives exist only in BraTS. IXI positives are all synthetic.
- 2 mm resolution, single seed, one training run.
- Detection uses T1 and T2 only, because IXI has no T1ce or FLAIR.
- The test subjects are the same as PLAN's (D6), so PLAN and PLAN2 test numbers are paired, not independent.
- S8 fails on test (boundary sensitivity in BraTS tumour-free hemispheres, 4/67).
- The S7 failure was not remedied (P5 cut). The features still encode the dataset (S9).
- Research prototype; not for clinical use.
