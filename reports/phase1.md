# Phase 1: BraTS 2 mm arrays (acceptance)

Kernel `leweletp/brats-dual-prep-brats-2mm` **v3** (CPU, 4 workers, 3525 s), code `src/prep_brats_2mm.py`, signature `brats2021-2mm-wholevol-v1`. Earlier versions: v1 was a 15-case smoke run; v2 failed with the odd-length z-box bug, which is now fixed with a regression test (DECISIONS.md). Small outputs are in `runs/prep-brats-2mm/brats2mm/`; the arrays stay on Kaggle as that notebook's output.

## Acceptance

| Check | Result |
|---|---|
| 1251 arrays with equal shape | **Pass**: 1251/1251, all `x` = 4×96×112×80 fp16, `y` = 96×112×80 uint8 bits (bit0 ET, bit1 TC, bit2 WT). 9.68 GB in total. |
| Label-mapping unit test on a toy volume | **Pass**: `tests/test_prep_brats.py::test_label_mapping_toy_volume` (plus pooling, z-score, even-box regression and end-to-end synthetic tests; 11 tests in the suite). |
| QC montage of 6 cases with seg overlay | **Pass**: `runs/prep-brats-2mm/brats2mm/qc/montage.png` (train/val cases only). **Owner visual review pending.** |
| Owner's 1 mm numerical QC cited, not redone | 1063/1063 passed (`brats-supplied-grid-full-cohort-qc`, signature `brats2021-supplied-grid-zscore-v1`). The z-score here reuses its `modality_stats.csv`. Recomputing the statistics per case gave a max relative difference of **0.0** against the owner's file. |

## Geometry (`crop.json`)
- Native axcodes **LPS**, 1 mm grid 240×240×155.
- Crop box = union of the train+val brain boxes: x 38..216, y 10..230, z 0..156 (z extended one zero-filled slice past the end, so the length is even).
- 2× brain-masked mean pooling → 89×110×78, padded at the end to multiples of 16 → **96×112×80**.

## Data checks
| | train+val (1063) | test (188, stored only) |
|---|---|---|
| Tumour voxels outside brain support | 0 | 0 |
| Cases with brain outside crop box | 0 | 1 (fraction 3.2e-6 of its brain) |
| Cases with tumour outside crop box | 0 | 0 |
| WT empty at 2 mm | 0 | 0 |
| TC empty at 2 mm | 5 (all 5 have no TC at 1 mm) | n/a |
| ET empty at 2 mm | 27: 26 have no ET at 1 mm; **1 lost by pooling** (BraTS2021_01512, 32 ET voxels at 1 mm) | 7 (not inspected further) |

Test-set note: test images were z-scored with their own non-zero-voxel statistics (the same formula) and stored with their labels. Only the aggregate counts above were computed; nothing was tuned on them.

**Visual review:** OK, by the owner on 2026-10-31 (`visual_review_complete: true`).
