"""S6 sensitivity analysis (labelled, not pre-registered; DECISIONS 3 Nov): S6 recomputed on shams whose
pasted healthy tissue is >= 1 mL, since near-empty shams pass trivially.
python src/p2_s6_sensitivity.py EVAL_DIR SPLIT"""
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shortcut_checks import decide  # noqa: E402

MIN_ML = 1.0


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k in ("logit", "prob", "wt_ml_b", "ml"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
        r["sample"] = int(r["sample"])
        rows.append(r)
    return rows


def main(eval_dir, split):
    d = Path(eval_dir)
    res = json.loads((d / f"results_{split}.json").read_text())
    rows = load(d / f"cases_{split}.csv")
    kept = [r for r in rows if not (r["cell"] in ("B-s", "I-s") and not r["ml"] >= MIN_ML)]
    out = {"min_ml": MIN_ML, "shams_dropped": {c: sum(r["cell"] == c for r in rows) - sum(r["cell"] == c for r in kept)
                                               for c in ("B-s", "I-s")}}
    for det, thr in (("head", res["threshold_head"]), ("seg", res["threshold_seg"])):
        s6 = [c for c in decide(kept, thr, detector=det) if c["id"] == "S6"][0]
        out[det] = dict(value=s6["value"], passed=bool(s6["passed"]))
    (d / f"s6_sensitivity_{split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
