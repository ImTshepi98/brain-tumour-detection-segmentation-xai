# Dual-head 3D CNN for brain tumour detection and segmentation

Honours project, BSc Hons Computer Science, Sol Plaatje University (Tshepiso Lewele; supervisor Dr. Tite Tuyikeze).
Plan: `PLAN.md` (a second, shortcut-controlled plan is in `PLAN2.md`). Status: `PROGRESS.md`. Deviations and every judgement call: `DECISIONS.md`.
All numbers: `reports/tables.md`. Write-up skeleton: `reports/results.md`. Explainability findings: `reports/xai_notes.md`.

**Research prototype, not for clinical use.**

## What it does
One 3D residual U-Net with attention gates (5.88 M parameters) and two heads, run on 2 mm whole volumes (4x96x112x80):
- **detection:** tumour vs healthy from T1 + T2 (T1ce and FLAIR zeroed, because the healthy IXI scans have no such sequences);
- **segmentation:** ET / TC / WT from T1, T1ce, T2 and FLAIR (BraTS only).

Explainability: our own 3D Grad-CAM++ (`src/gradcam3d.py`) on both pathways, plus counterfactual controls (tumour mirroring, tumour and sham insertion, mask erosion). **Main caveat:** detection scores are perfect, but the controls show that dataset identity alone decides the BraTS cases (`reports/xai_notes.md`), so the detection scores are not evidence of tumour detection on their own.

## Setup (local, Windows, CPU)
```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q tests
```
Exact versions: `requirements.lock.txt`. Kaggle CLI: `.venv\Scripts\kaggle` with a token in `~/.kaggle/` (never committed).

## Data sources and licences
- **BraTS 2021 Task 1** (RSNA-ASNR-MICCAI), 1251 cases, read from the Kaggle mirror `dschettler8845/brats-2021-task1` (licence field on the mirror: "Data files © Original Authors"). Its Conditions for Use (from the challenge data description reproduced on that page) require:
  - citing Baid et al. 2021 (arXiv:2107.02314), Menze et al. 2015 (IEEE TMI 34(10), DOI 10.1109/TMI.2014.2377694) and Bakas et al. 2017 (Sci Data 4:170117, DOI 10.1038/sdata.2017.117), and where allowed the TCIA data citations (DOI 10.7937/K9/TCIA.2017.KLXWJJ1Q and 10.7937/K9/TCIA.2017.GJQ7R0EF);
  - acknowledging the challenge, using its wording: "Data used in this publication were obtained as part of the RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge project through Synapse ID (syn25829067)."
  - The publication embargo and the "no additional data" rule apply to challenge participants' ranked submissions; this project is not a challenge entry. It uses extra data (IXI), as the terms allow for scientific publication when stated, and its BraTS-only segmentation results are reported separately.
  - The official pages (Synapse, CBICA) could not be loaded on 1 Nov 2026; the terms were read from the Kaggle copy.
- **IXI** (Biomedical Image Analysis Group, Imperial College London), T1 + T2 of 577 healthy subjects, read from the Kaggle mirror `mdnakib/ixi-dataset` (same paired counts as the Hugging Face copy `Joe0204/dataset-ixi`, which needed a token; DECISIONS, Phase 0). Ages and sites come from the official `IXI.xls`. Licence **CC BY-SA 3.0** (https://creativecommons.org/licenses/by-sa/3.0/legalcode), confirmed on https://brain-development.org/ixi-dataset/ on 1 Nov 2026. That page asks users to acknowledge the source of the IXI data (for example the page itself); acknowledge the original IXI source, not only the mirror.
- Both datasets are de-identified secondary data; no patient contact.
- SynthStrip model files: the owner's SHA-pinned copy (DECISIONS, Phase 2).

## Pipeline: kernel order and commands
Compute runs on Kaggle (all kernels private, user `leweletp`); this machine only generates, dry-runs, pushes and downloads.

How it runs:
- Code reaches Kaggle as the private dataset `leweletp/brats-dual-code`. After any change to `src/` or `configs/`, run `bash kaggle/sync_code.sh "msg"`, and wait for `kaggle datasets status` to show ready before pushing.
- Push a kernel with `kaggle kernels push -p kaggle/<folder>`.
- **Before every push:** `.venv\Scripts\python kaggle/dryrun.py` runs the train, resume and eval wrappers on synthetic data; it must end with `KERNEL DRY RUN OK`.

| # | Phase | Kernel folder(s) / generator | Device | Notes |
|---|---|---|---|---|
| 0 | Inventory, environment, split ids | `env-check`, `split-ids` | CPU | `reports/inventory.md`, `data_splits/splits.csv` |
| 1 | BraTS 2 mm arrays | `prep-brats-2mm` (`src/prep_brats_2mm.py`) | CPU | 58 min; 1251 arrays |
| 2 | IXI to BraTS-like 2 mm arrays | `python kaggle/prep-ixi/make_shards.py 3` → `prep-ixi/s0..s2` (`src/prep_ixi_2mm.py`) | CPU | 577/577, 34 flagged and kept |
| 3 | Shared SynthStrip boundary, domain gap | `python kaggle/sdt-brats/make_shards.py 3` → `sdt-brats/s0..s2`; `src/calibrate_border.py` (border 0 mm); `qc-domain` | CPU | `reports/domain_gap.md` (AUROC 1.000) |
| 5 | Training (one or more sessions per run) | `python kaggle/train/make_kernel.py NAME CONFIG BORDER MAX_MINUTES [EXTRA] [CONTINUE_FROM]` → `train-<name>` | T4 | Continuing sessions attach the previous session's output and resume from its `last.pt` |
| 6-7 | Evaluation + Grad-CAM++ | `python kaggle/eval/make_kernel.py RUN val --controls`, then `RUN test` (once) → `eval-<run>-<stage>` | T4 | Threshold and checkpoint frozen from val; the test stage refuses controls |
| 6 | Tables | `python src/make_tables.py --runs main=runs/eval-main noattn=runs/eval-noattn ... --cpu main=reports/cpu_inference.json` | local | writes `reports/tables.md` |
| 6 | Local CPU timing | `python src/cpu_timing.py --ckpt data_local/main/best.pt --cases data_local/cases --out reports/cpu_inference.json` | local | |
| 8 | GUI | `.venv\Scripts\streamlit run app/streamlit_app.py` | local | needs `data_local/main/best.pt`, `threshold.json`, `cases/*.npz` (exported by `eval-main-r5-test --export`) |

Local debug entry points (synthetic data):
- `python src/train.py --debug --out runs/debug-train`
- `python src/evaluate.py --debug --out runs/debug-eval --controls`
- `python src/explain.py --debug --out runs/debug-explain`

## Hardware and run times
- **Kaggle GPU:** NVIDIA Tesla T4 (16 GB), about 31 GB host RAM; FP32. Quota: 30 GPU h for the whole project (the next reset falls after the deadline).
- **Training memory:** host RAM grows about 1 GB per epoch (a leak that `gc` and malloc settings did not fix). Training therefore stops cleanly at a 29 GB budget (`--max-rss-gb`) and continues in a new session. This changes no result: the protocol hash excludes these runtime settings.
- **Training runs** (40 epochs each; best checkpoint by val `0.5*mean DSC + 0.5*F1`, or by the single task's metric for `segonly` and `clsonly`):

| run | sessions (epochs) | best epoch |
|---|---|---|
| `main` | r1 0-14, r3 15-25, r4 26-36, r5 37-39 | 32 |
| `noattn` | 0-14, 15-29, 30-39 | 31 |
| `segonly` | in progress (session 1: 0-18) | |
| `clsonly` | pending | |

- **Inference:** batch 1, detection + segmentation: T4 107 ms (`main`), local CPU (AMD, 12 logical cores) 2.23 s.
- **Local machine:** Windows 11, 7.3 GB RAM. Enough for dry runs, tables and the GUI; not for full-data training.

## Repository layout
```
PLAN.md PLAN2.md PROGRESS.md DECISIONS.md README.md requirements*.txt
legacy/    read-only pulls of the owner's earlier notebooks
configs/   kaggle.yaml debug.yaml train/{main,noattn,segonly,clsonly,debug}.yaml
kaggle/    one folder per kernel (kernel-metadata.json + thin run.py), generators, dryrun.py, sync_code.sh
src/       preprocessing, datasets, model, losses, metrics, train, evaluate, Grad-CAM++, explain, tables, GUI inference
app/       streamlit_app.py
reports/   every report, table and figure used in the write-up
runs/      downloaded kernel outputs (not committed)
data_local/ GUI checkpoint and bundled cases (not committed)
tests/     pytest suite
```

## Integrity rules followed
- Seed 42.
- The test split was read **once per model**, with the checkpoint and threshold frozen from val.
- Counterfactual controls run on val only.
- The owner's existing Kaggle notebooks were never modified.
- Every deviation from the proposal is listed in `DECISIONS.md` and summarised in `reports/results.md`.
