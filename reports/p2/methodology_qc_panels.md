# Methodology section draft: visual quality control of synthetic tumour insertion

Draft for the write-up (PLAN2 methodology chapter). Written to be pasted into the report and edited; numbers
come from `reports/p2/qc_panels/panels.json`, `reports/p2/p1/` and the val pairing file. The 12 panels are in
`reports/p2/qc_panels/` (files `00_...png` to `11_...png`; panel numbers below are 1-12 in the same order).

---

## X.Y Visual quality control of synthetic tumour insertion

### X.Y.1 Why a visual check was needed

In the first study every BraTS scan contained a tumour and every IXI scan was healthy, so the label and the
dataset were the same thing and the detector could succeed by recognising the dataset. The second study
breaks this link by making both classes occur in both datasets. Positives are created in healthy tissue by
pasting a real BraTS tumour into a tumour-free hemisphere, and "sham" negatives are created by pasting
healthy tissue in the same way. These synthetic samples make up half of the BraTS positives, all of the IXI
positives and half of all negatives during training, so their quality directly limits what the model can
learn. If a pasted tumour looked like an obvious artefact (a hard-edged block, a brightness jump, tissue
outside the brain), the model could learn to detect "something was pasted" instead of "a tumour is present",
which would replace one shortcut with another.

The automatic checks (unit tests of the pasting code and the shortcut checks in Section X) test the rules
of the procedure but not whether the result looks anatomically plausible. A visual review by a person was
therefore required before the evaluation, and the protocol fixed in advance that only the project owner can
mark this review as complete.

**Goal:** confirm by eye, on a random sample, that pasted tumours look like tumours inside a brain, that
sham pastes look like healthy tissue, and that nothing outside the pasted region is changed.

### X.Y.2 How a synthetic sample is made

1. **Hemispheres.** Each 2 mm scan is split at the midline into a left and a right hemisphere. The midline
   was found once from the average brain mask of the training scans, as the left-right position that makes
   the mask most symmetric (index 41 of 96). Each hemisphere is a block 48 voxels wide (96 mm). The right
   hemisphere is mirrored, so every hemisphere is stored in the same "canonical" orientation with the cut
   midline face on the same side. Each hemisphere's T1 and T2 (and T1ce and FLAIR for BraTS) are
   re-normalised to zero mean and unit variance over that hemisphere's own brain voxels, so that whole-scan
   intensity statistics cannot reveal the dataset.
2. **Hemisphere labels.** A hemisphere is positive if it contains at least 125 whole-tumour voxels (1 mL).
   It is negative if it contains no tumour and no tumour lies within 2 voxels of the midline. Hemispheres in
   between are excluded. A BraTS negative is therefore the healthy side of a patient whose tumour lies
   entirely in the other hemisphere.
3. **Pairing.** Every real negative hemisphere in the validation and test splits (the "host") is assigned
   one tumour-bearing BraTS hemisphere from the same split (the "donor"), using a fixed random order (seed
   42). Each host is used twice with the same donor: once for a tumour paste and once for a sham paste. The
   pairing is written to a file before any model is evaluated, so it cannot be changed after seeing results.
4. **Tumour paste.** The donor's whole-tumour region is copied into the host at the same voxel positions
   (the canonical orientation means the tumour lands in the anatomically matching place). Only voxels that
   lie inside the host's brain are pasted. The intensity of each sequence is matched as follows: the donor's
   tumour intensities are expressed relative to the mean and standard deviation of the donor's own healthy
   brain tissue, and are then placed on the host's scale using the host's brain mean and standard deviation.
   The tumour therefore keeps its contrast relative to normal tissue (for example, darker than white matter
   on T1, brighter on T2) without carrying over the donor scan's absolute brightness. Finally the whole host
   hemisphere is re-normalised again, exactly as a real hemisphere is. For IXI hosts only T1 and T2 exist,
   so only those sequences are pasted; for BraTS hosts all four sequences are pasted.
5. **Sham paste.** The same region (the donor's tumour outline) is used, but the tissue copied into the
   host comes from the donor's **other**, healthy hemisphere, at the mirrored position. Voxels where that
   healthy hemisphere is outside the brain or itself contains tumour are skipped. Intensity matching and
   re-normalisation are identical to the tumour paste. A sham is labelled negative. **Goal:** the model sees
   pasted-but-healthy tissue labelled "no tumour", so the act of pasting (edges, intensity matching) carries
   no information about the label; only the tissue content does.
6. **Segmentation labels.** For a tumour paste, the segmentation target is the donor's tumour labels inside
   the pasted region. For a sham the target is empty.

Four synthetic cell types result, named by dataset, label and origin:

| Code | Host | Pasted tissue | Label | Purpose |
|---|---|---|---|---|
| I+p | IXI hemisphere | BraTS tumour | tumour | Gives IXI, which has no real tumours, a positive class |
| B+p | BraTS healthy hemisphere | tumour from another BraTS patient | tumour | Makes pasted positives occur in both datasets, so "pasted" does not mean "IXI" |
| I-s | IXI hemisphere | healthy tissue (sham) | no tumour | Controls for the pasting procedure itself in IXI |
| B-s | BraTS healthy hemisphere | healthy tissue (sham) | no tumour | Controls for the pasting procedure itself in BraTS |

### X.Y.3 How the panels were chosen

Twelve panels were drawn at random (seed 42) from the fixed validation pairs, not hand-picked, so that they
show typical results rather than the best ones: four I+p, two B+p, three I-s and three B-s. IXI receives
more panels because IXI hosts make up most of the synthetic samples (172 IXI vs 67 BraTS validation hosts).
Validation data were used so that the test split stayed untouched.

### X.Y.4 How to read a panel

Each panel is a grid of six images (Figure X.Y):

- **Title line:** cell code (for example `I+p`); host identifier with side (for example `IXI293:R` = the
  right hemisphere of IXI subject 293); donor identifier with side; the volume actually pasted in mL (pasted
  voxels x 8 mm3 at 2 mm); and the fraction of the donor's tumour region that was pasted (1.00 = all of it;
  lower values mean part of the region fell outside the host brain or, for shams, could not be taken from
  the healthy side).
- **Columns:** left = the host hemisphere **before** pasting; middle = the **same** hemisphere **after**
  pasting (titled with the cell code); right = the **donor** hemisphere that supplied the tumour outline.
  For tumour pastes the tissue in the middle column comes from this donor's tumour. For shams the donor
  column still shows the tumour hemisphere, because its outline defines where to paste, but the pasted
  tissue comes from the donor's other, healthy hemisphere, which is not displayed.
- **Rows:** top = T1-weighted image; bottom = T2-weighted image. These two sequences are shown because they
  are the only ones present in both datasets and the only ones the detector uses.
- **Slice:** one axial slice, the one where the donor's tumour is largest, so the pasted region is shown at
  its biggest. The same slice is shown in all six images.
- **Orientation:** the straight vertical edge on the right of each image is the cut midline; the curved
  edge on the left is the outer surface of the brain. All hemispheres appear on the same side because right
  hemispheres were mirrored into the canonical orientation.
- **Greyscale:** all images are plain greyscale intensities; no colour is added to the tissue, and the
  pasted tumour is not tinted. Brighter grey means higher signal on that sequence. On T1, tumour and
  oedema usually appear darker than white matter; on T2 they usually appear brighter, because they contain
  more water. Each image is scaled to its own minimum and maximum, so overall brightness can differ slightly
  between columns; this is a display effect only.
- **Background:** outside the brain the data are zero, which after normalisation equals the average brain
  intensity, so the background appears mid-grey rather than black. Its shade differs between images for the
  same display-scaling reason.
- **Yellow outline (the only colour in the panel):** the boundary of the donor's whole-tumour region on that
  slice, drawn identically on all three columns. On the donor it marks the real tumour. On the host
  "before" image it shows where the paste will go. On the "after" image it shows where the paste went.
  The outline marks the region that pasting was allowed to change; the actual pasted area is the part of
  it inside the host brain (and, for shams, inside healthy tissue of the donor's other hemisphere).

### X.Y.5 Review criteria

The criteria were fixed before the review. A panel passes if:

1. **Location:** the pasted region lies inside the host brain, with nothing pasted into the background.
2. **Appearance of tumour pastes (I+p, B+p):** the region looks like tumour tissue (heterogeneous texture,
   tumour-like contrast on T1 and T2) and not like a uniform block, noise or an obvious brightness jump.
3. **Appearance of shams (I-s, B-s):** the region looks like ordinary healthy brain tissue and not like a
   tumour.
4. **Host preserved:** outside the yellow outline the "after" image matches the "before" image.

Known limitations of the method were accepted in advance and were **not** reasons to reject a panel: the
pasted tumour has sharp edges; there is no mass effect (ventricles and sulci are not displaced, so a large
tumour can cover where a ventricle was); and in shams the folding pattern of the pasted healthy tissue does
not continue the host's folds, because it comes from a different brain.

### X.Y.6 What each panel shows and why it was reviewed

| Panel | Type | Host / donor | Pasted | What it shows | Goal of reviewing it |
|---|---|---|---|---|---|
| 1 | I+p | IXI293:R / BraTS2021_00824:R | 15.8 mL, 100% | Small tumour next to the midline, bright on T2 | Check that a small paste near the cut face stays inside the brain and still reads as tumour |
| 2 | I+p | IXI070:R / BraTS2021_00538:L | 172.8 mL, 99% | Very large, heterogeneous tumour covering much of the posterior hemisphere, including where the host ventricle was | Check the extreme of the size range; shows the no-mass-effect limitation most clearly |
| 3 | I+p | IXI426:R / BraTS2021_01048:R | 53.4 mL, 96% | Mid-sized tumour with a bright T2 core | Typical-size paste (validation median about 56 mL); check the intensity matching against IXI tissue |
| 4 | I+p | IXI524:R / BraTS2021_00406:L | 68.4 mL, 96% | Well-defined round tumour, darker on T1 and bright on T2 | Check that a left-hemisphere donor maps correctly into a right-hemisphere host through the canonical orientation |
| 5 | B+p | BraTS2021_00321:R / BraTS2021_01465:L | 140.7 mL, 96% | Large posterior tumour pasted into a healthy BraTS hemisphere | Check pasting between two BraTS patients (all four sequences pasted, T1 and T2 shown) |
| 6 | B+p | BraTS2021_01397:R / BraTS2021_01607:L | 61.8 mL, 87% | Tumour where 13% of the donor region fell outside the host brain and was skipped | Check that the brain-mask rule clips the paste at the brain edge instead of pasting into background |
| 7 | I-s | IXI360:R / BraTS2021_01182:R | 7.2 mL, 13% | Barely visible change; most of the outline could not be filled because the donor's other side has tumour there | Check that a sham looks like healthy tissue; revealed that some shams are small |
| 8 | I-s | IXI642:L / BraTS2021_01244:L | 0.1 mL, 7% | Essentially no change | As panel 7; donor tumour at the midline leaves almost no healthy mirror tissue |
| 9 | I-s | IXI484:L / BraTS2021_01141:L | 0 mL, 0% | No change at all; the sample equals the real host | As panel 7; an empty sham is effectively a real negative |
| 10 | B-s | BraTS2021_01179:L / BraTS2021_01498:R | 1.5 mL, 3% | Barely visible change | Check shams in BraTS hosts; same midline effect as panels 7-9 |
| 11 | B-s | BraTS2021_01461:L / BraTS2021_00773:L | 89.7 mL, 93% | Large region of healthy-looking tissue; folds do not continue the host's | Check that a full-size sham contains no tumour-like tissue and that edges do not look like a tumour |
| 12 | B-s | BraTS2021_00270:R / BraTS2021_01298:L | 88.1 mL, 90% | As panel 11 | Second full-size BraTS sham, to confirm panel 11 is not a one-off |

### X.Y.7 Outcome

The project owner reviewed all twelve panels against the criteria above and approved them on 3 November
2026 ("panels approved"). The approval is recorded in `reports/p2/qc_panels/panels.json`
(`visual_review_complete: true`).

The review also exposed one weakness, which is reported rather than corrected, because the procedure had
been fixed before training. Four of the six sham panels (7-10) pasted little or no tissue. When the donor's
tumour lies at or across the midline, the donor's other hemisphere contains tumour at the mirrored
position, and those voxels are skipped. Across all validation pairs:

| Sham type | n | Median pasted volume | Under 1 mL | Empty |
|---|---|---|---|---|
| I-s | 172 | 51.5 mL | 31 (18%) | 4 |
| B-s | 67 | 61.3 mL | 8 (12%) | 1 |
| Tumour pastes, for comparison (I+p / B+p) | 172 / 67 | 55.7 / 61.8 mL | 0 | 0 |

Most shams are therefore as large as the tumour pastes, but a minority are near-empty and behave as real
negatives. These are not wrong samples, but they make the sham check (S6, "a sham paste must not flip the
decision") easier to pass. S6 is therefore reported both as pre-registered and, as an additional
non-pre-registered analysis, on shams of at least 1 mL only.

### X.Y.8 Limitations of the visual check

- Twelve panels at one axial slice each are a sample, not an exhaustive check of every synthetic case.
- The review was done by the project owner, not a radiologist.
- Only T1 and T2 were inspected; for BraTS hosts the pasted T1ce and FLAIR were not displayed.
- The known limitations of the insertion itself (no mass effect, sharp edges, no oedema beyond the
  tumour mask) remain, and their cost is measured by the shortcut checks rather than by this review.

**Suggested figure:** panels 4 (I+p) and 11 (B-s) side by side as Figure X.Y, captioned: "Quality-control
panels for synthetic insertion (validation). Columns: host before pasting, after pasting, donor; rows: T1,
T2. Yellow line: donor whole-tumour outline. Left: BraTS tumour pasted into an IXI hemisphere (68.4 mL).
Right: sham paste of healthy tissue into a BraTS hemisphere (89.7 mL)."
