"""PLAN2 section 4 decision rules on hand-made predictions: a tumour-driven detector passes, a dataset-driven
one fails S1/S2, and the worst-group threshold rule picks a separating cut."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import p2_metrics as pm  # noqa: E402
import shortcut_checks as sc  # noqa: E402

LOGIT = {"B+r": 4.0, "B+p": 3.6, "I+p": 3.6, "B-r": -4.0, "B-s": -3.9, "I-r": -4.0, "I-s": -3.9}  # no dataset offset


def _rows(logit_of, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cell, host in (("B+r", None), ("B-r", None), ("I-r", None), ("B+p", "B-r"), ("B-s", "B-r"),
                       ("I+p", "I-r"), ("I-s", "I-r")):
        for k in range(n):
            lg = logit_of(cell) + rng.normal(0, 0.3)
            rows.append(dict(cell=cell, dataset="brats" if cell[0] == "B" else "ixi",
                             label=int("+" in cell), sample=f"{host or cell}{k}", logit=lg, prob=1 / (1 + np.exp(-lg)),
                             wt_ml_b=(20.0 if "+" in cell else 0.0) + abs(rng.normal(0, 0.1))))
    return rows


def _s4(rows):
    return [dict(id=k, logit=r["logit"], prob=r["prob"], wt_ml_b=r["wt_ml_b"], logit_after=-4.0,
                 prob_after=0.018, wt_ml_b_after=0.0) for k, r in enumerate(rows) if r["cell"] == "B+r"]


def _s7(n=10):
    return [dict(gradcam_tumour_ratio=3.0, gradcam_outside_mass=0.05, gradcam_drop_map=0.5 + 0.01 * k,
                 gradcam_drop_random=0.01, occlusion_tumour_ratio=1.0, occlusion_outside_mass=0.3,
                 occlusion_drop_map=0.0, occlusion_drop_random=0.0) for k in range(n)]


def test_tumour_driven_detector_passes():
    rows = _rows(LOGIT.get)
    checks = sc.decide(rows, 0.5, _s4(rows), _s7(), None, None, "head")
    assert sc.verdict(checks) == dict(shortcut_controlled=True, failed_must=[]), checks
    seg = sc.decide(rows, 1.0, _s4(rows), None, None, None, "seg")
    assert sc.verdict(seg)["shortcut_controlled"], seg


def test_dataset_driven_detector_fails():
    rows = _rows(lambda c: 4.0 if c[0] == "B" else -4.0)        # "is it BraTS?"
    v = sc.verdict(sc.decide(rows, 0.5, _s4(rows), _s7(), None, None, "head"))
    assert not v["shortcut_controlled"] and {"S1", "S2"} <= set(v["failed_must"])


def test_s1_is_direction_free():
    rows = _rows(lambda c: LOGIT[c] + (-1.5 if c[0] == "B" else 1.5))   # IXI scores higher within each label
    s1 = next(c for c in sc.decide(rows, 0.5, detector="head") if c["id"] == "S1")
    assert not s1["passed"] and s1["value"]["neg_B_vs_I"] > 0.9


def test_worst_group_threshold():
    rows = _rows(LOGIT.get)
    thr, wg = pm.choose_threshold(rows, [r["prob"] for r in rows])
    assert wg == 1.0 and 0.05 < thr < 0.95
