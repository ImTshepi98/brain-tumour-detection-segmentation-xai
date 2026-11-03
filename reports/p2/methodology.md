# Methodology draft: the shortcut-controlled model (PLAN2)

Draft for the write-up, in the same detailed style as `methodology_qc_panels.md` (each step: what it is, how
it works, and the goal behind it). Section numbers are placeholders (M.x). The insertion quality-control
section is a separate file and slots in after M.3. Results are not in this file; they go in the results
chapter once the single test read is done. Citations follow `writeup_citations.md`.

---

## M.1 Overview and guiding rule

The first model (PLAN) reached perfect detection on validation and test, but controls run after training
showed that it could separate tumour from healthy scans by recognising the dataset: every BraTS scan
contains a tumour and every IXI scan is healthy, so "which dataset is this?" answers "is there a tumour?".
Removing a tumour left 186 of 188 BraTS scans positive. The second model (PLAN2) was designed so that
dataset identity carries no information about the label, and so that this can be tested.

**Guiding rule: trust before score.** A lower but honest detection score is preferred to a perfect score
that comes from a shortcut (Geirhos et al., 2020). To make this rule binding, every check that decides
whether the model can be trusted, together with its pass threshold, was written down and committed to the
project repository **before** the model was trained (Git tag `plan2-prereg`, commit 6e2fbd9). Results
were not allowed to change these rules. Any later change is reported as a deviation.

**Goal:** design out the dataset shortcut and test, with rules fixed in advance, whether the model uses
the tumour rather than the dataset.

The model architecture, segmentation targets (ET, TC, WT), 2 mm resolution, data splits and preprocessing
are the same as in PLAN, so that differences between the two models come from the data design and training
objective, not from the network.

## M.2 Hemisphere samples

**What.** Each 2 mm scan (4 x 96 x 112 x 80 voxels: T1, T1ce, T2, FLAIR) is split at the brain midline into
a left and a right hemisphere. The hemisphere, not the whole scan, is the unit the model classifies.

**Why.** BraTS contains no tumour-free scans, so at whole-scan level the healthy class can only come from
IXI. Many BraTS tumours lie entirely in one hemisphere, so the other hemisphere of the same patient is a
real tumour-free sample from the BraTS dataset. Working with hemispheres therefore gives real negatives in
both datasets.

**How.**
1. *Midline.* The average brain mask of the training scans was computed, and the left-right position at
   which this mask is most symmetric was chosen (the cut that maximises the Dice overlap between the mask
   and its mirror image). The result was index 41 of 96, with a mirror Dice of 0.983. One midline is used
   for every scan because all scans are registered to the same SRI24 atlas grid.
2. *Crop.* Each hemisphere is a block 48 voxels (96 mm) wide. The crop lost no brain voxels in any scan
   (maximum loss 0.0%).
3. *Canonical orientation.* Right hemispheres are mirrored so that every hemisphere has its midline at the
   same edge. **Goal:** the model learns one orientation, and left or right gives no information.
4. *Labels* (fixed before training):
   - **positive:** at least 125 whole-tumour (WT) voxels in that hemisphere (1.0 mL at 2 mm);
   - **negative:** no WT voxel in that hemisphere and none within 2 voxels (4 mm) of the midline;
   - **excluded:** anything in between (a tumour smaller than 1 mL in that half, or a tumour touching the
     midline from the other side). **Goal:** the margin keeps tumours that cross or touch the midline out of
     the negative class, so no "negative" contains part of a tumour.
5. *Normalisation.* Each sequence of each hemisphere is re-normalised to zero mean and unit variance over
   that hemisphere's own brain voxels. **Goal:** scan-level intensity statistics, which differ between
   BraTS and IXI, are not carried into the sample.

**Counts.** BraTS hemispheres are B+r (real tumour) or B-r (real tumour-free); IXI hemispheres are I-r
(real healthy).

| Split | B+r | B-r | I-r | Excluded (BraTS) |
|---|---|---|---|---|
| Train | 1179 | 319 | 808 | 252 |
| Validation | 249 | 67 | 172 | 60 |

The plan asked for at least 150 validation hemispheres in each real cell. B-r reached 67, because most
BraTS tumours are large or near the midline. This shortfall was recorded and the rules were not changed.

## M.3 Synthetic positives and sham negatives

Hemispheres alone still leave IXI without positives. Synthetic samples were therefore created by pasting
real BraTS tumours into tumour-free hemispheres of both datasets, and "sham" samples were created by
pasting healthy tissue in the same way. The procedure, the seven cells it produces (B+r, B+p, B-r, B-s,
I+p, I-r, I-s), the intensity matching and the visual quality control are described in Section X.Y
(`methodology_qc_panels.md`).

During training, pastes are made on the fly: a new host and donor are drawn for every sample, from the
training split only. Validation and test use one fixed pairing file each, written before evaluation. Donors
and hosts always come from the same split, so no test tumour is ever seen in training.

**Known limitation.** A pasted tumour has no mass effect (surrounding structures are not displaced) and no
oedema beyond the tumour mask. Real tumours with mass effect exist only in BraTS. The checks in M.8 measure
what this costs.

## M.4 Group-balanced sampling

**What.** Each training batch holds four samples, one from each group: BraTS positive (B+), BraTS negative
(B-), IXI positive (I+) and IXI negative (I-). Within B+, half are real tumours and half are pasted. Within
B- and I-, half are real and half are shams. I+ is always pasted, because IXI has no real tumours. An epoch
is 320 batches (1280 samples).

**Goal.** In the training distribution, knowing the dataset gives exactly zero information about the label
(each dataset is 50% positive), and knowing that a sample was pasted gives little information either
(pasted samples are both positive and negative). The only cue that consistently predicts the label is the
presence of tumour tissue.

## M.5 Augmentation

The augmentations from PLAN were kept: random flips along the front-back and top-bottom axes, random
intensity scaling (up to 10%) and shift (up to 0.1), and Gaussian noise (standard deviation 0.05).
Left-right flips were removed, because all hemispheres are already in one orientation. One augmentation was
added: **random erosion of the brain boundary by 0 to 2 voxels**, after which the hemisphere is re-normalised.

**Goal of the erosion.** BraTS and IXI were skull-stripped by different tools, so the shape of the brain
edge could identify the dataset. Randomly eroding the edge makes its exact shape an unreliable cue.

## M.6 Model and training

**Network.** The same five-level residual 3D U-Net with attention gates as PLAN (channels 16, 32, 64, 128,
256; 5.9 million parameters), with input 4 x 48 x 112 x 80 (one hemisphere). The encoder is shared by
three outputs:

| Output | Input | Samples | Target | Purpose |
|---|---|---|---|---|
| Detection head | T1 and T2 only | every sample | tumour yes/no | The classifier being tested |
| Segmentation pass A | all four sequences | BraTS samples (real and pasted) | ET, TC, WT | Full tumour segmentation, as in PLAN |
| Segmentation pass B | T1 and T2 only | every sample (empty target for negatives) | WT only | Tumour outline available for IXI too; drives the segmentation-derived detector |

**Why T1 and T2 only for detection.** IXI has no T1ce or FLAIR. In PLAN those channels were zero for IXI,
so "two empty channels" itself marked a healthy scan. Using only T1 and T2 for every sample removes this
marker.

**Loss.** L = L_segA + L_segB + 0.5 x L_cls. Each segmentation loss is a squared Dice loss plus binary
cross-entropy (BCE); the Dice part is computed over the whole batch rather than per sample, so that the empty
targets of negative hemispheres do not each count as a failed case. L_cls is the BCE of the detection head. A domain-adversarial variant (p2-adv) adds 0.1 x L_adv, a
head that predicts the dataset from the detection features through a gradient-reversal layer (Ganin et
al., 2016), which pushes the features to carry no dataset information. It was declared in advance as the
**single remedy**, used only if a required check fails (M.8).

**Training.** AdamW (learning rate 3 x 10^-4, cosine schedule, weight decay 10^-5), gradient clipping at
1.0, full precision, 40 epochs of 320 steps, seed 42, on two NVIDIA T4 GPUs on Kaggle. Kaggle sessions
have limited host memory, and the training process's memory grew by about 2 GB per epoch, so training ran
as a chain of ten sessions of about four epochs each. Each session resumed exactly from the previous one's
saved model, optimiser, scheduler and random-number state, so the run is equivalent to one continuous run.

## M.7 Metrics, model selection and threshold

**Balanced accuracy.** For a set of samples, balanced accuracy (BA) is the mean of sensitivity (the share
of positives called positive) and specificity (the share of negatives called negative) (Brodersen et al.,
2010). It is 0.5 for a model that guesses or always gives the same answer, and 1.0 for a perfect model,
whatever the class balance.

**Worst-group balanced accuracy.** BA is computed separately within each dataset:

- BraTS: positives B+r and B+p; negatives B-r and B-s;
- IXI: positives I+p; negatives I-r and I-s.

The worst-group BA is the lower of the two (Sagawa et al., 2020). **Goal:** a model that only recognises
the dataset ("BraTS means tumour") scores BA = 0.5 in each dataset and therefore 0.5 overall, even though
it scored perfectly in PLAN. A high worst-group BA can only come from detecting tumours within each
dataset.

**Model selection.** After every epoch the model was scored on validation as 0.5 x (mean DSC of pass A on
real BraTS tumours) + 0.5 x (worst-group BA at probability 0.5), and the best epoch was kept. **Goal:**
reward segmentation and shortcut-free detection equally.

**Threshold.** The decision threshold was chosen on validation as the value that maximises worst-group BA
(ties go to the value closest to 0.5), not the value that maximises F1 on the pooled data. **Goal:** a
pooled threshold could favour the larger dataset; this rule protects the weaker group.

**Segmentation-derived detector.** As a comparison (ablation), a hemisphere is also called positive if
pass B predicts a WT volume of at least v* mL, with v* chosen on validation by the same worst-group rule.
**Goal:** test whether a separate detection head adds anything over simply asking "did the segmentation
find a tumour?".

**Reported metrics.** Detection: sensitivity, specificity, F1, BA and AUROC per dataset and per cell, and
whole-scan results (a scan is positive if either hemisphere is positive; whole-scan specificity can only be
measured on IXI, since BraTS has no tumour-free scans). Segmentation: Dice similarity coefficient (DSC) and
95th-percentile Hausdorff distance (HD95) for ET, TC and WT on real BraTS tumours, as in PLAN; WT DSC on
pasted tumours; predicted tumour volume on real negatives. The standard detection metrics are the ones
compared with published work; worst-group BA and the checks below are what PLAN2 adds.

## M.8 Pre-registered shortcut checks

Ten checks were fixed before training. "MUST" checks decide the verdict; "SHOULD" checks are reported.
Each runs on validation and again, unchanged, in the single test read.

| ID | What is measured | Pass rule | Tier | Goal |
|---|---|---|---|---|
| S1 | Can the detection score tell the datasets apart within a label? AUROC of the logit for B-r vs I-r and for B+p vs I+p, direction-free | Both <= 0.70 | MUST | The decision itself should not carry dataset identity |
| S2 | Do the datasets get equal treatment at the threshold? | Specificity B-r vs I-r and sensitivity B+p vs I+p each differ by <= 0.10 | MUST | Equal error rates across datasets |
| S3 | Can the model separate tumour from healthy within each dataset? | AUROC(B+r vs B-r) >= 0.80 and AUROC(I+p vs I-r) >= 0.80 | MUST | Real discrimination inside each dataset |
| S4 | **Necessity**: detected real tumours are replaced by mirrored healthy tissue from the other hemisphere | >= 70% drop below the threshold and median logit change <= -2 | MUST | The tumour must be needed for the decision (PLAN failed this: 186 of 188 stayed positive) |
| S5 | **Sufficiency**: a tumour is pasted into every validation negative of both datasets | >= 60% cross the threshold in each dataset, rates differ by <= 0.15 | MUST | A tumour alone should be enough, equally in both datasets |
| S6 | **Sham**: healthy tissue is pasted into every validation negative | <= 5% cross in each dataset and median absolute logit change < 1 | MUST | The act of pasting must not trigger a detection |
| S7 | **Localisation**: Grad-CAM++ and occlusion maps on detected real tumours | For at least one map: median tumour ratio >= 2 and median mass outside the brain <= 20%, and deleting its top 30% of voxels lowers the probability more than deleting random voxels (one-sided paired Wilcoxon, p < 0.05) | MUST | The explanation should point at the tumour and actually matter to the decision (PLAN's map put 70% of its mass outside the brain) |
| S8 | **Boundary**: brain mask eroded by one voxel | Median absolute logit change < 1 and <= 5% crossings per group | SHOULD | The brain edge should not drive the decision |
| S9 | **Probe**: logistic regression predicting the dataset from the pooled detection features | Report only | SHOULD | Shows how much dataset information the features hold; S1 asks whether the *decision* uses it |
| G1 | **Segmentation false positives**: pass-B predicted WT volume on real negatives | Median <= 1 mL in both B-r and I-r | MUST | The segmentation should not invent tumours in healthy tissue |

The tumour ratio is the share of the map's total mass that falls inside the tumour, divided by the share of
the brain volume that the tumour occupies; 1 means the map is no more concentrated on the tumour than on any
other brain tissue, and 2 means twice as concentrated.
Necessity, sufficiency and sham use the fixed validation pairing file, so every comparison is between the
same hemisphere before and after one change. Because 18% of IXI and 12% of BraTS shams pasted less than
1 mL (Section X.Y.7), S6 is also reported on shams of at least 1 mL as an additional analysis that was not
pre-registered.

**Verdict rule (fixed in advance).** The model is called **shortcut-controlled** only if every MUST check
passes on validation. If one fails, the single remedy (p2-adv) is trained and validation is repeated; if a
check still fails, the model is reported as **not shortcut-controlled**, naming the failed checks. The test
split is read exactly once in either case, so a failure is measured, not hidden. If any group shows
sensitivity and specificity both equal to 1.000, a written review of which checks rule out a shortcut is
required before the result is accepted, because a perfect score was the first warning sign in PLAN.

## M.9 Explainability comparison

**Why.** In PLAN only Grad-CAM++ (Chattopadhay et al., 2018) was used, and its detection map did not explain
the decision. The proposal also named SHAP and LIME. PLAN2 therefore runs four methods side by side and
**measures** which one explains the detector best, instead of assuming it.

| Method | Idea | Settings (fixed in advance) |
|---|---|---|
| Grad-CAM++ | Weights a network layer's feature maps by gradients of the detection score | One stage above the bottleneck (cells of about 8 mm instead of PLAN's 16 mm); PLAN's layer also reported |
| Occlusion | Hides one part of the input at a time and records how much the score drops; uses no gradients | 16 mm cube, moved in 8 mm steps, filled with a blurred copy of the input (Gaussian, sigma 8 mm) |
| Region SHAP | Shares out the score among regions as Shapley values (Lundberg and Lee, 2017) | Regions = each predicted tumour component, 16 mm cubes of the remaining brain, and outside-brain; 2000 model evaluations; hidden regions replaced by the blurred input |
| LIME | Fits a simple weighted linear model to the score over random on/off patterns of supervoxels (Ribeiro, Singh and Guestrin, 2016) | About 100 3D SLIC supervoxels from T1 and T2 inside the brain; 1000 samples; same blurred baseline; three random seeds, with the stability of the top five supervoxels reported (Jaccard index) |

**How they are compared**, on 60 validation hemispheres (20 B+r, 10 B+p, 10 I+p, 10 B-r, 10 I-r, seed 42):

- **Faithfulness:** deletion and insertion curves. The input is removed (or restored) in order of the map's
  importance, and the area under the probability curve is compared with a random order. **Goal:** a map is
  faithful if removing what it marks changes the decision faster than removing random voxels.
- **Localisation:** tumour ratio, and a pointing game (does the map's top region fall in the tumour). Only
  on positives.
- **Sanity check** (Adebayo et al., 2018): the weights of the detection head, and then of the whole
  network, are randomised, and the maps are recomputed on 5 cases. **Goal:** a map that barely changes when
  the model is destroyed explains the image, not the model.
- **Agreement:** Spearman correlation between the methods' region rankings.
- **Cost:** seconds per case.

## M.10 Evaluation protocol

1. **Validation stage.** The best checkpoint is evaluated on validation. The threshold for the detection
   head and v* for the segmentation-derived detector are chosen there. All metrics, checks S1 to S9 and G1,
   the verdict and the explainability comparison are computed.
2. **Remedy (only if needed).** If a MUST check fails, p2-adv is trained and step 1 is repeated for it.
3. **Test stage, once.** The checkpoint, thresholds and check code are frozen. One evaluation run on the
   test split computes every metric, check and explanation. The run writes a lock file and the evaluation
   code refuses to read the test split a second time. Results that disagree with validation are reported as
   they are.

The test subjects are the same as in PLAN, which had already read them once for its own models. No PLAN2
design choice used PLAN's test results; the design came from PLAN's validation controls only.

## M.11 Deviations from the proposal

| # | Deviation | Reason |
|---|---|---|
| D1 | Detection on hemispheres, not whole scans (whole-scan results also reported) | Gives real tumour-free BraTS samples |
| D2 | Synthetic tumour and sham pastes used as training data | Gives IXI positives and controls for pasting |
| D3 | Group-balanced sampling; optional domain-adversarial branch | Makes the dataset uninformative about the label |
| D4 | Segmentation-derived detector compared with the detection head | Tests whether the head is needed |
| D5 | 2 mm resolution, as in PLAN | Carried over from PLAN |
| D6 | Test subjects shared with PLAN | Disclosed; no design choice used PLAN's test numbers |

The supervisor will review D1 and D2 with the project owner at the end of the project; this approval is
recorded as deferred, not as given.
