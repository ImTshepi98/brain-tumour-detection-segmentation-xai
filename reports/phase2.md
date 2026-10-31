# Phase 2: IXI to BraTS-like 2 mm arrays (acceptance)

Kernels `leweletp/brats-dual-prep-ixi-s0/s1/s2` (CPU, 4 workers each, subject index mod 3; s0 7800 s wall-clock). Code `src/prep_ixi_2mm.py`, signature `ixi-bratslike-2mm-v1`. Pipeline: SynthStrip (owner's SHA-pinned files) → ANTs rigid T2→T1 → affine T1-brain → BraTS-space T1 template (60 BraTS train cases, `phase2/template.json`, identical in all 3 shards), both estimated at 2 mm and applied at full resolution → BraTS crop box, masked 2× pooling, z-score (owner's formula) → 4×96×112×80 fp16, channels [T1, 0, T2, 0]. Arrays stay on Kaggle as the shard notebooks' outputs; QC rows, manifest and panels are in `reports/phase2/`.

## Acceptance

| Check | Result |
|---|---|
| ≥ 500 of 577 subjects produce arrays | **Pass: 577/577**, 0 failures, no reruns needed (train 404 / val 86 / test 87). |
| > 15% registration failures → change template | Not triggered (fail rate 0.0). |
| Flagged list for review | `phase2/ixi_flagged.csv`: 34 subjects (details below). |
| Visual panels IXI vs BraTS (train/val only) | `phase2/panels_ixi.png`, `phase2/panels_brats.png`, `phase2/histograms.png`. **Owner visual review pending** (`visual_review_complete=false`). |
| Shape/dtype match BraTS | All rows report 4×96×112×80 fp16; the 12 downloaded panel arrays were re-hashed against their row's `x_sha256`. |

## Metrics (all 577)

| site | n | template Dice median (min) | T1–template corr median (min) | warped brain mL median |
|---|---|---|---|---|
| Guys | 319 | 0.960 (0.938) | 0.647 (0.523) | 1570 |
| HH | 185 | 0.953 (0.927) | 0.544 (0.455) | 1598 |
| IOP | 73 | 0.958 (0.937) | 0.645 (0.543) | 1575 |

Median time 134.5 s/subject. Lowest template Dice overall: 0.927 (IXI526).

## Flags (34)
- **11 owner-eligibility exclusions** (kept, flagged; see DECISIONS.md). Only IXI556 also trips a metric.
- **19 with T1–template correlation < 0.5, all from HH** (0.455–0.4999). Their template Dice is normal. HH is the 3 T Philips site and its T1 contrast differs from the BraTS-mean template, so corr is lower at that site overall (median 0.544 vs 0.647). This looks like a contrast effect, not misregistration. The 0.5 threshold was set before seeing the data and is **not changed after the fact**.
- **3 warped brain volume > median + 3 MAD** (IXI451, IXI526, IXI544, all HH, 1668–1694 mL vs median 1579). These are also the 3 lowest template Dice values (0.927–0.93): a mildly generous SynthStrip mask or slight over-scaling.
- **1 native brain volume outlier** (IXI457, 2105 mL vs median 1519): a large head or loose mask. Its warped volume is normal.

Nothing is excluded automatically. Pending the owner's review, flagged subjects stay in the data with `flagged=1` in `ixi_manifest.csv`, and Phase 4 can report results with and without them.

## Observations from the panels (for Phase 3)
- Skull stripping and alignment look right in all 12 IXI panels. Orientation matches BraTS, and the T1/T2 z-score histograms overlap BraTS closely. IXI T1 shows a slightly larger low-intensity (CSF) shoulder.
- **Residual intensity inhomogeneity**: no N4 bias correction was applied. IXI315 and IXI238 show a visible front–back brightness gradient on T1. Together with the sharper IXI T1 noted earlier, this is a candidate non-tumour shortcut. Phase 3's domain classifier should include low-frequency intensity features (bias field) as well as sharpness.

## Test split
Test subjects went through the same deterministic pipeline. Their QC metrics appear in the tables above (aggregate QC only). No choice was tuned on them.

**Visual review:** OK, by the owner on 2026-10-31 (T2 fine; T1 histograms discussed; all 34 flagged subjects kept). `visual_review_complete: true`.
