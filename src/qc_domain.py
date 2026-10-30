"""Phase 3b/3c: domain-gap check, BraTS vs IXI on tumour-free voxels (CPU).

Train+val subjects only (the test split stays locked). Per subject, from the 2 mm arrays:
  tumour-free region = brain minus WT dilated by 1 voxel (BraTS); whole brain (IXI).
  plan features (PLAN 3b): T1/T2 histograms (32 bins), brain volume, bounding-box size,
    mask surface-to-volume ratio, mean/SD of T1/T2 in the outer 2-voxel shell.
  extended features (Phase 2 observations): sharpness (gradient magnitude, Laplacian
    variance) and low-frequency bias (polynomial-fit R^2, smoothed-intensity spread).
Classifier: HistGradientBoosting, 5-fold stratified CV, seed 42, AUROC.
Mitigation (3c), at most two iterations, each recomputed from the arrays:
  it1: mask hole-filling + closing for both groups; IXI T1/T2 histogram-matched to a BraTS
       reference (mean quantile function of BraTS *train* tumour-free voxels).
  it2: it1 plus BraTS also matched to the same reference (both groups on one histogram).
Writes features_*.csv, cv_results.json, oof_*.csv to --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SEED = 42
NBINS, HRANGE = 32, (-3.0, 5.0)
NQ = 201
QS = np.linspace(0, 1, NQ)
CH = {"t1": 0, "t2": 2}
BALL = ndi.generate_binary_structure(3, 1)

# populated in workers by _init
_REF = None


def regions(x, y, fill):
    brain = (x != 0).any(0)
    if fill:
        brain = ndi.binary_closing(ndi.binary_fill_holes(brain), BALL, iterations=1, border_value=0)
        brain = ndi.binary_fill_holes(brain)
    tumour = np.zeros_like(brain) if y is None else ndi.binary_dilation((y & 4) > 0, BALL)
    free = brain & ~tumour
    shell = brain & ~ndi.binary_erosion(brain, BALL, iterations=2)
    interior = ndi.binary_erosion(brain, BALL, iterations=2) & free
    return brain, free, shell & free, interior


def match(img, mask, ref_q):
    """Monotone quantile mapping of img[mask] onto the reference quantile function."""
    v = img[mask]
    if v.size == 0:
        return img
    src_q = np.quantile(v, QS)
    out = img.copy()
    out[mask] = np.interp(v, src_q, ref_q)
    return out


def bias_feats(img, mask):
    idx = np.argwhere(mask).astype(np.float32)
    v = img[mask].astype(np.float64)
    if len(v) < 50:
        return dict(bias_r2_lin=np.nan, bias_r2_quad=np.nan)
    c = (idx - idx.mean(0)) / (idx.std(0) + 1e-6)
    one = np.ones((len(c), 1))
    lin = np.hstack([one, c])
    quad = np.hstack([lin, c ** 2, c[:, [0]] * c[:, [1]], c[:, [0]] * c[:, [2]], c[:, [1]] * c[:, [2]]])
    out = {}
    for name, A in (("lin", lin), ("quad", quad)):
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = v - A @ coef
        out[f"bias_r2_{name}"] = 1 - r.var() / (v.var() + 1e-12)
    return out


def features(x, y, fill=False, match_ixi=None, match_all=None):
    x = x.astype(np.float32)
    brain, free, shell, interior = regions(x, y, fill)
    f = {}
    imgs = {}
    for m, c in CH.items():
        img = x[c]
        ref = match_all if match_all is not None else (match_ixi if y is None else None)
        if ref is not None:
            img = match(img, free & (img != 0), ref[m])
        imgs[m] = img
    # plan features
    for m, img in imgs.items():
        v = img[free & (img != 0)]
        h, _ = np.histogram(np.clip(v, *HRANGE), bins=NBINS, range=HRANGE)
        h = h / max(h.sum(), 1)
        for i, hv in enumerate(h):
            f[f"hist_{m}_{i:02d}"] = hv
        sv = img[shell & (img != 0)]
        f[f"shell_{m}_mean"] = float(sv.mean()) if sv.size else np.nan
        f[f"shell_{m}_sd"] = float(sv.std()) if sv.size else np.nan
    nb = int(brain.sum())
    f["shape_brain_ml"] = nb * 8 / 1000.0
    lo, hi = np.argwhere(brain).min(0), np.argwhere(brain).max(0) + 1
    for a, (l, h_) in zip("xyz", zip(lo, hi)):
        f[f"shape_bbox_{a}"] = int(h_ - l)
    surf = brain & ~ndi.binary_erosion(brain, BALL)
    f["shape_surf_to_vol"] = surf.sum() / max(nb, 1)
    # extended features
    for m, img in imgs.items():
        sm = interior & (img != 0)
        g = np.sqrt(sum(ndi.sobel(img, axis=a) ** 2 for a in range(3)))
        lap = ndi.laplace(img)
        sd = img[sm].std() + 1e-6
        f[f"sharp_{m}_grad"] = float(g[sm].mean() / sd)
        f[f"sharp_{m}_lapvar"] = float(lap[sm].var() / sd ** 2)
        low = ndi.gaussian_filter(np.where(free, img, 0), 6) / (ndi.gaussian_filter(free.astype(np.float32), 6) + 1e-3)
        f[f"bias_{m}_lowfreq_sd"] = float(low[sm].std() / sd)
        for k, v in bias_feats(img, sm).items():
            f[f"{k.replace('bias_', f'bias_{m}_')}"] = v
    return f


def quantiles(x, y):
    x = x.astype(np.float32)
    _, free, _, _ = regions(x, y, False)
    out = {}
    for m, c in CH.items():
        v = x[c][free & (x[c] != 0)]
        out[m] = np.quantile(v, QS) if v.size else np.full(NQ, np.nan)
    return out


def _load(job):
    x = np.load(job["x"])
    y = np.load(job["y"]) if job.get("y") else None
    return x, y


def _init(ref):
    global _REF
    _REF = ref


def task_pass1(job):
    x, y = _load(job)
    q = quantiles(x, y)
    return job["subject"], features(x, y), q


def task_pass2(job):
    x, y = _load(job)
    return (job["subject"], features(x, y, fill=True, match_ixi=_REF),
            features(x, y, fill=True, match_all=_REF))


def run_pool(fn, jobs, workers, ref=None):
    if workers <= 1:
        _init(ref)
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(ref,)) as ex:
        return list(ex.map(fn, jobs, chunksize=4))


GROUPS = {
    "hist": lambda c: c.startswith("hist_"),
    "shape": lambda c: c.startswith("shape_"),
    "shell": lambda c: c.startswith("shell_"),
    "sharp": lambda c: c.startswith("sharp_"),
    "bias": lambda c: c.startswith("bias_"),
}
PLAN = ("hist", "shape", "shell")


def cv_auc(X, y, sites, cols, n_splits=5, importance=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    aucs, imps = [], []
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200, learning_rate=0.05)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
        if importance:
            pi = permutation_importance(clf, X[te], y[te], scoring="roc_auc", n_repeats=3, random_state=SEED)
            imps.append(pi.importances_mean)
    res = dict(auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)), auc_folds=[float(a) for a in aucs],
               auc_pooled=float(roc_auc_score(y, oof)), n_features=len(cols))
    res["ixi_site_mean_p_brats"] = {s: float(oof[(y == 0) & (sites == s)].mean())
                                    for s in sorted(set(sites[y == 0]))}
    if importance:
        m = np.mean(imps, 0)
        order = np.argsort(-m)[:12]
        res["top_permutation_importance"] = [(cols[i], float(m[i])) for i in order]
    return res, oof


def evaluate(rows, label, sites):
    cols = sorted(rows[0])
    X = np.array([[r[c] for c in cols] for r in rows], float)
    out, oofs = {}, {}
    sets = {"plan": [c for c in cols if any(GROUPS[g](c) for g in PLAN)],
            "plan+extended": cols}
    sets.update({f"only_{g}": [c for c in cols if fn(c)] for g, fn in GROUPS.items()})
    for name, cs in sets.items():
        idx = [cols.index(c) for c in cs]
        out[name], oofs[name] = cv_auc(X[:, idx], label, sites, cs, importance=name in ("plan", "plan+extended"))
    return out, oofs


def write_csv(path, subjects, rows, extra):
    cols = sorted(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", *extra[0].keys(), *cols])
        for s, r, e in zip(subjects, rows, extra):
            w.writerow([s, *e.values(), *[r[c] for c in cols]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats-dir", required=True, help="dir with x/ and y/")
    ap.add_argument("--ixi-dirs", nargs="+", required=True, help="dirs with x/")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0, help="per source, for smoke runs")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.splits) as f:
        sp = {r["subject"]: r for r in csv.DictReader(f)}
    jobs = []
    bd = Path(args.brats_dir)
    for s in sorted(sp):
        r = sp[s]
        if r["split"] not in ("train", "val"):
            continue  # test split locked
        if r["source"] == "brats":
            if (bd / "x" / f"{s}.npy").exists():
                jobs.append(dict(subject=s, source="brats", x=str(bd / "x" / f"{s}.npy"), y=str(bd / "y" / f"{s}.npy")))
        else:
            for d in map(Path, args.ixi_dirs):
                if (d / "x" / f"{s}.npy").exists():
                    jobs.append(dict(subject=s, source="ixi", x=str(d / "x" / f"{s}.npy"), y=None))
                    break
    if args.limit:
        jobs = [j for j in jobs if j["source"] == "brats"][: args.limit] + \
               [j for j in jobs if j["source"] == "ixi"][: args.limit]
    n_b = sum(j["source"] == "brats" for j in jobs)
    n_i = len(jobs) - n_b
    expect_b = sum(r["source"] == "brats" and r["split"] != "test" for r in sp.values())
    expect_i = sum(r["source"] == "ixi" and r["split"] != "test" for r in sp.values())
    print(f"jobs: brats {n_b}/{expect_b}, ixi {n_i}/{expect_i}", flush=True)
    subjects = [j["subject"] for j in jobs]
    label = np.array([j["source"] == "brats" for j in jobs], int)
    sites = np.array([sp[s]["site"] for s in subjects])
    meta = [dict(source=j["source"], split=sp[j["subject"]]["split"], site=sp[j["subject"]]["site"]) for j in jobs]

    p1 = run_pool(task_pass1, jobs, args.workers)
    base = [r[1] for r in p1]
    print(f"pass1 done {time.time() - t0:.0f}s", flush=True)
    train_brats = [i for i, j in enumerate(jobs) if j["source"] == "brats" and sp[j["subject"]]["split"] == "train"]
    ref = {m: np.nanmean([p1[i][2][m] for i in train_brats], 0) for m in CH}
    np.savez(out / "reference_quantiles.npz", **ref, qs=QS)
    write_csv(out / "features_base.csv", subjects, base, meta)

    results = dict(n_brats=n_b, n_ixi=n_i, expected_brats=expect_b, expected_ixi=expect_i,
                   threshold=args.threshold, reference_from=f"{len(train_brats)} BraTS train subjects")
    results["base"], oofs = evaluate(base, label, sites)
    print("base", {k: round(v["auc_mean"], 3) for k, v in results["base"].items()}, flush=True)
    iterations = []
    if results["base"]["plan"]["auc_mean"] > args.threshold:
        p2 = run_pool(task_pass2, jobs, args.workers, ref)
        print(f"pass2 done {time.time() - t0:.0f}s", flush=True)
        for it, k in (("it1", 1), ("it2", 2)):
            feats = [r[k] for r in p2]
            write_csv(out / f"features_{it}.csv", subjects, feats, meta)
            results[it], o = evaluate(feats, label, sites)
            print(it, {kk: round(v["auc_mean"], 3) for kk, v in results[it].items()}, flush=True)
            iterations.append(it)
            if results[it]["plan"]["auc_mean"] <= args.threshold:
                break
    results["iterations_run"] = iterations
    with open(out / "oof_base_plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "source", "site", "p_brats"])
        for s, m, p in zip(subjects, meta, oofs["plan"]):
            w.writerow([s, m["source"], m["site"], round(float(p), 4)])
    results["elapsed_s"] = round(time.time() - t0)
    (out / "cv_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, dict)}, indent=2))
    return results


if __name__ == "__main__":
    main()
