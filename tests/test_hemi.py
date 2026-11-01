"""PLAN2 P1 acceptance: midline flip is an involution, label rules on toy masks, pairing never crosses splits."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import datasets as ds  # noqa: E402
import hemi  # noqa: E402
import p2_data  # noqa: E402
import synthetic as syn  # noqa: E402


def test_find_midline_symmetric_mask():
    m = np.zeros((40, 8, 8), np.float32)
    m[11:29] = 1          # symmetric about the cut between 19 and 20
    assert hemi.find_midline(m)[0] == 20


def test_cut_is_canonical_and_invertible():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 30, 5, 4)).astype(np.float32)
    m, w = 14, 16
    left, right = hemi.cut(a, m, "L", w), hemi.cut(a, m, "R", w)
    assert left.shape == right.shape == (2, w, 5, 4)
    np.testing.assert_array_equal(left[:, -1], a[:, m - 1])       # midline neighbours at the last index
    np.testing.assert_array_equal(right[:, -1], a[:, m])
    np.testing.assert_array_equal(right[:, 0], a[:, m + w - 1])     # far edge of the right crop
    np.testing.assert_array_equal(hemi.cut(a, 20, "R", w)[:, :6], 0)  # 20 + 16 - 1 = 35 > 29: zero padding
    np.testing.assert_array_equal(hemi.cut(hemi.uncut(right, m, "R", 30), m, "R", w), right)
    back = hemi.uncut(left, m, "L", 30) + hemi.uncut(right, m, "R", 30)
    np.testing.assert_array_equal(back[:, m - w:m + w], a[:, m - w:m + w])
    # flipping twice (cut right, mirror back) is the identity on the covered slab
    np.testing.assert_array_equal(right[:, ::-1][:, :30 - m], a[:, m:])


def test_side_label_rules():
    bits = np.zeros((20, 6, 6), np.uint8)
    m, w = 10, 10
    bits[2:7, 0:5, 0:5] = 4          # 125 voxels in the left hemisphere, far from the midline
    assert hemi.side_label(bits, m, "L", w) == (1, 125)
    assert hemi.side_label(bits, m, "R", w) == (0, 0)
    bits[11, 0, 0] = 4               # 1 voxel on the right, inside the 2-voxel margin
    assert hemi.side_label(bits, m, "R", w) == (-1, 1)
    bits[11, 0, 0] = 0
    bits[9, 0, 0] = 4                # left tumour touching the midline: right side excluded
    assert hemi.side_label(bits, m, "R", w)[0] == -1
    small = np.zeros_like(bits)
    small[3, 0, 0:4] = 4             # under 1 mL: excluded, not positive
    assert hemi.side_label(small, m, "L", w) == (-1, 4)


def test_paste_and_sham():
    rng = np.random.default_rng(1)
    shape = (16, 12, 12)
    host = np.zeros((4, *shape), np.float32)
    host[:, 2:14, 2:10, 2:10] = rng.normal(size=(4, 12, 8, 8))
    host[1] = host[3] = 0            # IXI-like host: T1ce, FLAIR absent
    donor = np.zeros_like(host)
    donor[:, 1:15, 1:11, 1:11] = rng.normal(size=(4, 14, 10, 10))
    dbits = np.zeros(shape, np.uint8)
    dbits[5:9, 4:8, 4:8] = 7         # ET+TC+WT
    donor[:, dbits > 0] += 3
    x, t, frac, ml = hemi.paste(host, donor, dbits)
    assert frac == 1.0 and ml == 64 * 8 / 1000
    np.testing.assert_array_equal(t, dbits)
    assert not x[1].any() and not x[3].any()                       # absent channels stay absent
    hb = (host != 0).any(0)
    assert abs(x[0][hb].mean()) < 1e-4                              # host brain re-z-scored
    assert x[0][dbits > 0].mean() > x[0][hb & (dbits == 0)].mean() + 1
    sham_src = donor.copy()
    sham_src[:, dbits > 0] -= 3      # healthy tissue at the same voxels
    xs, ts, fs, _ = hemi.paste(host, donor, dbits, sham_src=sham_src, sham_bits=np.zeros_like(dbits))
    assert fs == 1.0 and not ts.any()
    assert abs(xs[0][dbits > 0].mean() - xs[0][hb & (dbits == 0)].mean()) < 1


def _hemisets(tmp_path):
    p = syn.write_fake_2mm(tmp_path, n_brats=16, n_ixi=8, lateral=True)
    rows = ds.read_splits(p["splits"])
    kw = dict(brats_dir=p["brats_dir"], ixi_dirs=[p["ixi_dir"]], brats_sdt_dirs=[p["brats_sdt_dir"]], border=0,
              width=16, workers=1)
    return rows, {s: hemi.HemiSet.load(rows, (s,), 16, **kw) for s in ("train", "val")}


def test_hemiset_cells_and_pairs_stay_in_split(tmp_path):
    rows, sets = _hemisets(tmp_path)
    split_of = {r["subject"]: r["split"] for r in rows}
    for split, hs in sets.items():
        assert hs.of("B+r") and hs.of("B-r") and hs.of("I-r"), (split, hs.counts)
        assert len(hs.pool) % 2 == 0 and all(hs.pmeta[j]["side"] == "L" for j in range(0, len(hs.pmeta), 2))
        pairs = p2_data.fixed_pairs(hs, split)
        assert {p["cell"] for p in pairs} == {"B+p", "B-s", "I+p", "I-s"}
        for p in pairs:
            assert split_of[p["host_id"].split(":")[0]] == split_of[p["donor_id"].split(":")[0]] == split
            assert hs.cell[p["donor"]] == "B+r"
        ev = p2_data.EvalSet(hs, pairs)
        assert len(ev) == len(hs) + len(pairs)
        for r, b in zip(ev.rows, ev.bits):
            assert (r["label"] == 1) == (r["wt_voxels"] > 0 if r["cell"] != "B+r" else True)
            if r["cell"].endswith("s"):
                assert not b.any()


def test_sample_batch_groups(tmp_path):
    _, sets = _hemisets(tmp_path)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(20):
        batch = p2_data.sample_batch(sets["train"], rng)
        assert len(batch) == 4
        assert [hemi.POSITIVE[c] for _, _, c in batch] == [1, 0, 1, 0]
        assert [hemi.DATASET[c] for _, _, c in batch] == ["brats", "brats", "ixi", "ixi"]
        seen |= {c for _, _, c in batch}
        for x, b, c in batch:
            assert x.shape[1] == 16 and np.isfinite(x).all()
            assert (((b & 4) > 0).any()) == bool(hemi.POSITIVE[c])
    assert seen == set(hemi.CELLS)
