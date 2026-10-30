# DECISIONS.md

One line per decision or deviation. Format: date | phase | decision | reason.

## From PLAN.md section B (pre-agreed)
- 2026-10-30 | B1 | Keep all 577 IXI subjects, no age filter; report age distribution and stratified detection results. | Filtering discards data and cannot age-match BraTS (no per-case age in Task 1).
- 2026-10-30 | B2 | Dual-head model at 2 mm whole volumes (BraTS 1 mm average-pooled 2x). **Deviation from proposal (1 mm).** | Detection head needs whole-volume context; 1 mm whole volumes too slow. Owner's 1 mm patch baseline kept as high-res reference.
- 2026-10-30 | B3 | Reuse, never modify, the owner's notebooks; new notebooks get new slugs (`brats-dual-*`). | Owner's QC-signed artifacts.
- 2026-10-30 | B4 | Test split locked until Phase 6, read once per model. | Honesty.
- 2026-10-30 | B5 | FP32 by default; AMP only if the smoke benchmark demands it, with finite-gradient checks kept. | Owner's AMP run died on a non-finite grad norm.

## Made during execution
- 2026-10-30 | 0 | Repo root is `C:\Users\Admin\Honours Research Project` itself (the plan's `brats_dual/` layout lives directly in it). | Folder was empty; avoids a redundant nesting level.
- 2026-10-30 | 0 | Local env: Python 3.12.10 venv in `.venv`, torch 2.15.0+cpu, monai 1.6.1, kaggle CLI 2.2.5; full pin list in `requirements.lock.txt`. | Reproducibility.
- 2026-10-30 | 0 | GPU accelerator = T4 x2 only. | kaggle CLI 2.2.5 help states P100 is retired ("accepted but warn, since the session runs on a replacement"). Plan's "or P100" is moot.
- 2026-10-30 | 0 | Phase 0 split into two kernels: `brats-dual-env-check` (no secrets, also lists the `mdnakib/ixi-dataset` fallback layout and counts paired T1/T2) and `brats-dual-hf-token-check` (needs owner's HF_TOKEN toggle). | The env check and fallback assessment can proceed without waiting for the owner.
- 2026-10-30 | 0 | HF check tries repo_type `dataset` then `model`. | The plan does not say which repo type `Joe0204/dataset-ixi` is.
