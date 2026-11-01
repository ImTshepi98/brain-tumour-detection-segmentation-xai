# Explainability notes (Phase 7)

Model: `main` best.pt (epoch 32 of 40; protocol 948372a86b6e; shared SynthStrip boundary, border 0 mm).
Numbers: `reports/tables.md` (Grad-CAM++ and controls sections) and `reports/eval_val/xai_metrics_val.md`.
Overlays: `reports/eval_val/overlays/` (val) and `reports/eval_test/overlays/` (test, the PLAN Phase 7
cases). The test split was read once (1 Nov, `eval-main-r5-test`) with the checkpoint and threshold
frozen from val.

## Validation (n = 274: 188 BraTS, 86 IXI)

### Detection pathway (T1 + T2 only, last encoder stage)

The detection head separates val perfectly (threshold 0.0168 frozen from val: 188/188 tumours,
86/86 healthy; AUROC 1.000). Two counterfactual controls test *why* (changes are reported in
probability and in the logit, because probabilities near 0 or 1 hide large logit moves):

- **Is the tumour necessary? No.** Replacing each tumour with mirrored contralateral tissue (median
  98% of the tumour filled from valid brain) barely moves the decision: median logit change -0.14
  from a median of +12.7; 186 of 188 cases stay above the threshold.
- **Is the tumour sufficient? Largely yes.** Pasting a BraTS val tumour into each healthy IXI scan
  (same SRI24 voxels, intensity-matched to the host; median 83 mL, 99.7% inside the host brain) moves
  the median logit by +18.2 (from -12.6 to +5.6); 61 of 86 healthy scans cross the threshold (median
  probability change +0.996). Larger tumours flip more often (Spearman rho 0.65 between inserted
  volume and probability change, from controls_val.csv). By the rule fixed before the run
  (DECISIONS 2026-11-01) this is clear tumour sensitivity, not a 'not sufficient' result.
- **Boundary cue: small.** A 1-voxel erosion moves BraTS logits by +0.03 and IXI logits by +1.23
  (2 of 86 IXI cross the threshold); the outer 6 mm shell gets less than its volume share of the
  heatmap (shell ratio 0.24 tumour, 0.34 healthy).

Reading: the detector uses **redundant cues, either of which is enough**. For BraTS scans the
tumour-free remainder of the image already carries decisive evidence (dataset identity: scanners,
protocols, preprocessing history, consistent with the Phase 3 result that a simple feature classifier
separates the cohorts with AUROC 1.000, `reports/domain_gap.md`; peritumoural changes outside the WT
mask, such as mass effect, may also contribute). For healthy scans, inserting tumour tissue is enough
to flip the decision. So the val (and expected test) scores cannot be read as tumour detection alone,
but the model is not blind to tumours either: the sham control below shows the insertion effect comes
almost entirely from tumour appearance.

**Sham insertion (is it the tumour or the BraTS patch?).** Same hosts, donors, voxels and intensity
matching, but pasting the donor's *healthy* mirror-side tissue (median 97% of the WT voxels, 81 mL):
median logit change +0.24 (IQR +0.10 to +0.55) and 5 of 86 healthy scans cross the threshold,
against +18.15 and 61 of 86 for the tumour. Paired on the same host and donor, the tumour moves the
logit 16.1 more than the sham (median), and further in 84 of 86 pairs. Under the rule fixed before
the run (DECISIONS 2026-11-01) this is classed as **mixed**: the logit criterion for 'tumour-specific'
is met (0.24 < 4.5) but the crossing criterion is missed by one case (5 > 4). In size, the insertion
effect is almost entirely due to tumour appearance, with a small patch (BraTS texture) effect.

Heatmaps agree with 'not necessary': the detection Grad-CAM++ puts most of its mass outside the brain
(median 70% for tumour cases, 58% for healthy), the tumour region gets less than its volume share
(median tumour ratio 0.33), and zeroing the top 10/20/30% CAM voxels changes the probability as
little as zeroing random voxels (median ~0 for both; the BraTS cases are saturated). Removing the top
30% CAM voxels *raises* the tumour probability for many IXI cases (median -0.015, IQR -0.45 to 0),
i.e. changing global intensity statistics moves a healthy scan toward the BraTS side.

Limitation of the map itself: the target layer is 6 x 7 x 5 voxels (16 mm cells), so the heatmap is
coarse; its localisation is read together with the controls above, not on its own.

### Segmentation pathway (4 channels, last decoder stage)

The segmentation maps do concentrate on the tumour: median tumour ratio 7.6 (IQR 6.0 to 11.3), i.e.
the predicted-WT region holds 7.6 times its volume share of the heatmap, and the shell is not enriched
(ratio 0.33). This matches the segmentation scores (val WT DSC 0.914, TC 0.869, ET 0.820).

### Selected cases (val)

No false positives or false negatives at the frozen threshold, so the selection is best / median /
worst WT DSC plus one healthy case inside and one outside the 40-75 age band:

| case | role | tumour prob | WT DSC | note |
|---|---|---|---|---|
| BraTS2021_00413 | best WT | 1.000 | 0.985 | seg map fills the tumour; detection map peaks in the other hemisphere |
| BraTS2021_01192 | median WT | 1.000 | 0.947 | |
| BraTS2021_01035 | worst WT | 1.000 | 0.000 | small infratentorial (brainstem/cerebellar) tumour missed completely; detection still certain |
| IXI550 | healthy, Guys | 0.000 | n/a | |
| IXI517 | healthy, IOP | 0.000 | n/a | detection map peaks outside the head |

Val segmentation failures are rare: 3 of 188 cases have WT DSC < 0.5 (01035, 00493, 01490).

## Test (n = 275: 188 BraTS, 87 IXI; read once)

The test results repeat the val pattern (all numbers in `reports/tables.md`, TEST split):

- **Detection:** 188/188 tumours and 87/87 healthy scans at the frozen threshold 0.0168 (F1 =
  AUROC = 1.000), with a wide margin: lowest tumour probability 0.99994, highest healthy 0.0011.
  As on val, this reflects the redundant cues described above (the counterfactual controls are
  val-only by design, PLAN 6), so it is not evidence of tumour detection on its own.
- **Detection heatmaps:** median 70% of the mass outside the brain for tumour cases (58% healthy),
  tumour ratio 0.34, shell ratio 0.25 / 0.35, deletion of the top CAM voxels changes the probability
  no more than random deletion; the same as val.
- **Segmentation:** WT DSC 0.921, TC 0.900, ET 0.849 (means; medians 0.948 / 0.957 / 0.920), slightly
  above val; heatmaps on the tumour (median tumour ratio 7.9). 2 of 188 cases have WT DSC < 0.5.

### Selected test cases

PLAN Phase 7 asks for 6 test cases including any false positives/negatives. There were none, so the
selection rule (fixed in `explain.py` before the read) gives 5: best / median / worst WT DSC and one
healthy case inside and one outside the age band.

| case | role | tumour prob | WT / TC / ET DSC | note |
|---|---|---|---|---|
| BraTS2021_00809 | best WT | 1.000 | 0.982 / 0.976 / 0.970 | |
| BraTS2021_01043 | median WT | 1.000 | 0.948 / 0.982 / 0.937 | |
| BraTS2021_01293 | worst WT | 1.000 | 0.351 / 0.081 / 0.400 | large cystic tumour, dark on T1 and FLAIR; only rim fragments segmented; the detection map is cold over the tumour |
| IXI225 | healthy, Guys | 0.000 | n/a | |
| IXI071 | healthy, Guys | 0.000 | n/a | |

