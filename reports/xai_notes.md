# Explainability notes (Phase 7)

Model: `main` best.pt (epoch 32 of 40; protocol 948372a86b6e; shared SynthStrip boundary, border 0 mm).
Numbers: `reports/tables.md` (Grad-CAM++ and controls sections) and `reports/eval_val/xai_metrics_val.md`.
Overlays: `reports/eval_val/overlays/`. This section covers **validation** only; the test-set overlays
and metrics (PLAN Phase 7: 6 test cases) are added after the single test read.

## Validation (n = 274: 188 BraTS, 86 IXI)

### Detection pathway (T1 + T2 only, last encoder stage)

The detection head separates val perfectly (threshold 0.0168 frozen from val: 188/188 tumours,
86/86 healthy; AUROC 1.000). The evidence below says that this separation does **not** come from the
tumour:

- **Tumour mirroring control.** Replacing each tumour with the mirrored contralateral tissue (median
  98% of the tumour filled from valid brain) left the tumour probability essentially unchanged
  (mean change -0.007, median -0.000); 186 of 188 cases stay above the threshold.
- **Heatmap location.** The detection Grad-CAM++ puts most of its mass *outside* the brain
  (median 70% for tumour cases, 58% for healthy), and the tumour region gets *less* than its volume
  share (median tumour ratio 0.33; 1 = no enrichment). In the overlays the detection map is broad and
  peaks away from the tumour (often in the other hemisphere or above the head).
- **Deletion test.** Zeroing the top 10/20/30% CAM voxels inside the brain changes the tumour
  probability by about as little as zeroing random voxels (median drop ~0 for both).
- **Boundary shortcut.** The obvious Phase 3 cue (the brain boundary) is not what the model uses
  either: the outer 6 mm shell gets less than its volume share of the heatmap (shell ratio 0.24
  tumour, 0.34 healthy), and a 1-voxel erosion barely moves the probability (mean change +0.005
  BraTS, +0.012 IXI; 2 of 86 IXI cross the threshold).

Reading: the T1 + T2 classifier recognises the *dataset* (BraTS vs IXI: scanners, protocols and
preprocessing history) from diffuse, global properties of the image, which is consistent with the
Phase 3 domain-gap result (a simple feature classifier also reached AUROC 1.000, before and after the
shared boundary; `reports/domain_gap.md`). Its val (and expected test) scores therefore measure
dataset separation, not tumour detection, and must be reported as such. The healthy-case deletion
result points the same way: removing the top 30% CAM voxels *raises* the tumour probability for many
IXI cases (median -0.015, IQR -0.45 to 0), i.e. changing global intensity statistics moves a healthy
scan toward the BraTS side.

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
