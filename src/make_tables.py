"""Phase 6 acceptance: collect every reported number into reports/tables.md.

  python src/make_tables.py --runs main=runs/eval-main noattn=runs/eval-noattn ... --out reports/tables.md
Each run dir holds evaluate.py outputs (results_val.json, results_test.json) and explain.py outputs
under xai/ (xai_summary_<split>.json). Missing files are shown as "not run".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = ("ixi_all", "ixi_in_band", "ixi_below_band", "ixi_above_band", "ixi_no_age")


def f(v, d=3):
    return "n/a" if v is None or v != v else f"{v:.{d}f}"


def ci(m, ci_, k):
    lo, hi = ci_.get(k, [None, None])
    return f"{f(m.get(k))} ({f(lo)} to {f(hi)})"


def load(d, split):
    p = Path(d) / f"results_{split}.json"
    return json.loads(p.read_text()) if p.exists() else None


def tables(runs, split):
    res = {name: load(d, split) for name, d in runs.items()}
    have = {k: v for k, v in res.items() if v}
    L = [f"## {split.upper()} split", ""]
    if not have:
        return L + ["Not run yet.", ""]
    missing = [k for k, v in res.items() if not v]
    if missing:
        L += [f"Not run yet: {', '.join(missing)}.", ""]
    L += ["### Detection (T1 + T2 only), 95% bootstrap CIs", "",
          "| run | n | threshold | recall | precision | F1 | specificity | AUROC |", "|---|---|---|---|---|---|---|---|"]
    for k, r in have.items():
        d, c = r["detection"], r.get("detection_ci", {})
        if not d:
            continue
        L.append(f"| {k} | {r['n']} | {f(r['threshold'])} | " + " | ".join(
            ci(d, c, m) for m in ("recall", "precision", "f1", "specificity", "auroc")) + " |")
    L += ["", "### Detection by group (IXI specificity, Wilson 95% CI; BraTS recall)", ""]
    sites = sorted({g for r in have.values() for g in r.get("groups", {}) if g.startswith("ixi_site_")})
    cols = list(GROUPS) + sites
    L += ["| run | " + " | ".join(c.replace("ixi_", "") for c in cols) + " | BraTS recall |",
          "|---" * (len(cols) + 2) + "|"]
    for k, r in have.items():
        g = r.get("groups", {})
        if not r["detection"]:
            continue
        cells = [f"{f(g[c]['specificity'])} [{f(g[c]['specificity_ci'][0], 2)}, {f(g[c]['specificity_ci'][1], 2)}] "
                 f"(n={g[c]['n']})" if c in g and g[c]["n"] else "n/a" for c in cols]
        b = g.get("brats_all", {})
        L.append(f"| {k} | " + " | ".join(cells) + f" | {f(b.get('recall'))} (n={b.get('n')}) |")
    L += ["", "BraTS has no per-case age, so these groups show how age and site affect the negatives; "
          "they do not age-match the two cohorts.", "",
          "### Segmentation (BraTS), 2 mm", "",
          "| run | region | DSC mean ± SD | DSC median | HD95 mm mean ± SD | HD95 median | n |", "|---|---|---|---|---|---|---|"]
    for k, r in have.items():
        s = r.get("segmentation", {})
        for reg in ("ET", "TC", "WT"):
            if f"dsc_{reg}" in s:
                a, h = s[f"dsc_{reg}"], s[f"hd95_{reg}"]
                L.append(f"| {k} | {reg} | {f(a['mean'])} ± {f(a['sd'])} | {f(a['median'])} | "
                         f"{f(h['mean'], 1)} ± {f(h['sd'], 1)} | {f(h['median'], 1)} | {a['n']} |")
    L += ["", "### Inference time (batch 1, after warm-up; detection + segmentation pass)", "",
          "| run | ms per case | hardware |", "|---|---|---|"]
    for k, r in have.items():
        i = r.get("inference", {})
        L.append(f"| {k} | {f(i.get('ms_per_case_mean'), 1)} ± {f(i.get('ms_per_case_sd'), 1)} | {i.get('hardware')} |")
    ctl = {k: r["controls"] for k, r in have.items() if r.get("controls")}
    if ctl:
        L += ["", "### Controls (val): change in tumour probability", "",
              "| run | control | n | mean change | median change | above threshold before -> after |", "|---|---|---|---|---|---|"]
        for k, c in ctl.items():
            for name in ("tumour_mirrored_brats", "eroded_ixi", "eroded_brats"):
                v = c[name]
                L.append(f"| {k} | {name} | {v['n']} | {f(v['mean_change'])} | {f(v['median_change'])} | "
                         f"{v['above_thr_before']} -> {v['above_thr_after']} |")
            L.append(f"| {k} | mirror filled from valid contralateral tissue (median share) | | "
                     f"{f(c['mirror_valid_frac_median'])} | | |")
    xai = {k: Path(runs[k]) / "xai" / f"xai_summary_{split}.json" for k in have}
    xai = {k: json.loads(p.read_text()) for k, p in xai.items() if p.exists()}
    if xai:
        L += ["", "### Grad-CAM++ checks (median per case)", "",
              "| run | class | det shell ratio | det outside-brain mass | det tumour ratio | seg tumour ratio | "
              "det drop top-20% CAM | det drop random 20% | n |", "|---|---|---|---|---|---|---|---|---|"]
        keys = ("det_shell_ratio", "det_outside_mass", "det_tumour_ratio", "seg_tumour_ratio",
                "det_drop_cam_20", "det_drop_random_20")
        for k, x in xai.items():
            for cls, s in x["summary"].items():
                vals = [s.get(m) for m in keys]
                n = max((v[3] for v in vals if v), default=0)
                L.append(f"| {k} | {cls} | " + " | ".join(f(v[0]) if v else "n/a" for v in vals) + f" | {n} |")
        L += ["", "shell ratio = share of heatmap mass in the outer 6 mm of the brain / that shell's share of brain "
              "volume (1 = no enrichment; > 1 = the map concentrates on the boundary)."]
    return L + [""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--out", default="reports/tables.md")
    args = ap.parse_args(argv)
    runs = dict(r.split("=", 1) for r in args.runs)
    L = ["# Results tables", "", "Generated by `src/make_tables.py` from evaluate.py / explain.py outputs. "
         "Every number used in the dissertation comes from this file.", ""]
    for split in ("test", "val"):
        L += tables(runs, split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


if __name__ == "__main__":
    print(main())
