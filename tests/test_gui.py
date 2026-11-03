import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import datasets as ds  # noqa: E402
import infer  # noqa: E402
import preprocess_one as po  # noqa: E402
import synthetic as syn  # noqa: E402
import train  # noqa: E402


@pytest.fixture(scope="module")
def debug_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    train.main(["--debug", "--out", str(out), "--set", "model.channels=[4,8,16,32,64]", "optim.epochs=1"])
    d = out / "debug_data"
    rows = ds.read_splits(d / "splits.csv")
    val = ds.Cohort.load(rows, ("val",), d / "brats2mm", [d / "ixi2mm"], [d / "sdt_brats"], border=0, workers=1)
    infer.export_cases(val, val.ids, out / "cases")
    return out, val


def test_analyse_and_export(debug_run):
    out, val = debug_run
    model, st, thr = infer.load(out / "best.pt")
    for p in sorted((out / "cases").glob("*.npz")):
        x, y, meta = infer.load_case(p)
        r = infer.analyse(model, x.astype(np.float32), thr, "detection")
        assert 0 <= r["prob"] <= 1 and r["cam"].shape == x.shape[1:] and meta["shared_boundary"]
        assert r["can_segment"] == (meta["source"] == "brats")
        if r["can_segment"]:
            assert set(r["volumes_ml"]) == {"ET", "TC", "WT"}
            assert r["volumes_ml"]["ET"] <= r["volumes_ml"]["TC"] <= r["volumes_ml"]["WT"]
    with pytest.raises(AssertionError, match="T1 and T2"):
        infer.analyse(model, np.zeros((4, 32, 32, 32), np.float32))


def test_preprocess_one_matches_phase1_steps():
    shape = (40, 40, 30)
    crop_cfg = {"box_1mm": [[4, 36], [4, 36], [0, 30]], "target_shape": [16, 16, 16]}
    brain = syn._ellipsoid(shape, [20, 20, 15], [14, 14, 11])
    rng = np.random.default_rng(0)
    vols = {m: np.where(brain, rng.normal(500, 80, shape), 0).astype(np.float32) for m in ("t1", "t2", "t1ce")}
    seg = syn.native_seg(shape, rng) * brain
    x, y, meta = po.preprocess(vols, crop_cfg, expected=shape, seg=seg)
    assert x.shape == (4, 16, 16, 16) and x.dtype == np.float16 and not x[3].any() and y.any()
    assert meta["modalities"] == ["t1", "t1ce", "t2"] and not meta["shared_boundary"]
    sdt = syn.signed_distance(brain, spacing=1.0).astype(np.float32)
    x2, _, meta2 = po.preprocess(vols, crop_cfg, border=-3, sdt=sdt, seg=seg, expected=shape)
    assert meta2["shared_boundary"] and (x2 != 0).any(0).sum() < (x != 0).any(0).sum()
    with pytest.raises(AssertionError, match="SynthStrip"):
        po.preprocess(vols, crop_cfg, border=0, expected=shape)


def test_streamlit_app_runs(debug_run):
    from streamlit.testing.v1 import AppTest
    out, _ = debug_run
    os.environ["BRATS_DUAL_CKPT"] = str(out / "best.pt")
    os.environ["BRATS_DUAL_CASES"] = str(out / "cases")
    # AppTest swaps sys.modules["__main__"] for the app script; restore it, or later Windows spawn
    # workers (multiprocessing in the prep tests) re-run the app and hang
    main_mod = sys.modules["__main__"]
    try:
        at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()
        assert not at.exception, at.exception
        if not any("Tumour probability" in m.label for m in at.metric):  # PLAN2 is the default model
            at.sidebar.radio[0].set_value("PLAN: whole scan").run()
        assert not at.exception, at.exception
        assert any("Tumour probability" in m.label for m in at.metric)
        next(b for b in at.selectbox if b.label == "Grad-CAM++ overlay").select("segmentation").run()
        assert not at.exception, at.exception
    finally:
        sys.modules["__main__"] = main_mod
        os.environ.pop("BRATS_DUAL_CKPT")
        os.environ.pop("BRATS_DUAL_CASES")


def test_make_tables(tmp_path):
    import evaluate
    import make_tables
    evaluate.main(["--debug", "--out", str(tmp_path / "main"), "--controls"])
    md = make_tables.main(["--runs", f"main={tmp_path / 'main'}", f"noattn={tmp_path / 'none'}",
                           "--out", str(tmp_path / "tables.md")])
    assert "## TEST split\n\nNot run yet." in md and "Not run yet: noattn." in md
    assert "| main | ET |" in md and "tumour_mirrored_brats" in md and "in_band" in md
