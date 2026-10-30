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
- 2026-10-30 | 0 | Kaggle auth via new-style API token in `~/.kaggle/access_token`; both legacy `kaggle.json` keys are dead (401). | Owner generated a new token.
- 2026-10-30 | 0 | **Cohort-size reconciliation:** 1828 = 1251 BraTS + 577 IXI (full cohort incl. test). 1542 = train+val only (1063 BraTS + 490 IXI = 1553) minus 11 IXI subjects excluded by the owner's eligibility step (eligibility_sha256 8bb1f6c7...). Excluded: IXI020, IXI210, IXI326, IXI445, IXI504, IXI556, IXI584, IXI585, IXI626, IXI642, IXI652 (reason not recorded in any pulled output). **This project uses the 1828 definition** (all 577 IXI per PLAN B1); the 11 are carried as a flagged list into Phase 2 QC and excluded only if they fail this project's own QC. | PLAN B1 (keep all 577); the exclusion reason is undocumented, so dropping them silently would be unjustified.
- 2026-10-30 | 0 | IXI source for Phase 2 = Kaggle dataset `mdnakib/ixi-dataset` (primary), not the HF zip. | Env check verified it has T1 581 / T2 578 / PD 578 and 577 paired T1+T2 subjects, identical counts to the HF zip; it needs no per-notebook secret, and Joe0204/dataset-ixi still returns 401. Caveat: byte-identity with the HF zip is unverified; subject IDs map 1:1 by IXI number.
- 2026-10-30 | 0 | Owner's Sept notebooks `stage-1-...`/`stage-2-...` (an earlier dual-head attention U-Net + Grad-CAM++ prototype) are **not reused for results**. | 80/20 split by folder order, no locked test, IXI from an undocumented `ixi_preprocessed_sri24.zip`. Mentioned in the inventory as prior work only.
- 2026-10-30 | 0 | New notebooks run on Kaggle's current default image (Python 3.13, torch 2.11) and pip-install monai/antspyx/nilearn as needed; they do not pin the owner's 3.12 image. | Owner's IXI jobs assert Python 3.12 ("inherited June30 original runtime"); not portable to new notebooks. Revisit in Phase 2 if antspyx has no 3.13 wheel.
- 2026-10-30 | 0 | Kaggle input path convention is `/kaggle/input/datasets/<owner>/<slug>/`; there is no `/kaggle/temp`, use `/tmp` (about 1.1 TB free) as scratch. | Measured in env-check.
