# Cargo Release Analysis — complete documentation

Automated classification of HeLa cells by **calcein cargo release** from
iron-fumarate MOF nanoparticles.

This file explains *everything*: what the pipeline measures, every formula, every
processing step, every decision and the evidence behind it, the full development
history, the CNN classifier, and how the code is version-controlled. It is meant
to be read by you in six months, by a supervisor checking a methods section, or by
the next student taking the project over.

| | |
|---|---|
| Analysis version | **3.1** (rule) · CNN module **1.0** (optional) |
| Document state | 2026-11-14 |
| Code | this repository; every result records the git commit that produced it |

---

## Contents

1. [The biological question](#1-the-biological-question)
2. [Quick start](#2-quick-start)
3. [Data layout and file naming](#3-data-layout-and-file-naming)
4. [Processing, step by step, with the maths](#4-processing-step-by-step-with-the-maths)
5. [How rules are evaluated against hand labels](#5-how-rules-are-evaluated-against-hand-labels)
6. [Statistics in the Figures tab](#6-statistics-in-the-figures-tab)
7. [Sample naming from a plate layout](#7-sample-naming-from-a-plate-layout)
8. [Provenance: run records and settings hashes](#8-provenance-run-records-and-settings-hashes)
9. [The CNN classifier](#9-the-cnn-classifier)
10. [History: how the pipeline got here](#10-history-how-the-pipeline-got-here)
11. [Settings reference](#11-settings-reference)
12. [Version control: how to work with this repository](#12-version-control-how-to-work-with-this-repository)
13. [Known limitations and open questions](#13-known-limitations-and-open-questions)
14. [File reference](#14-file-reference)
15. [Glossary](#15-glossary)

---

## 1. The biological question

HeLa cells are incubated with iron-fumarate metal–organic-framework (MOF)
nanoparticles loaded with **calcein**, a green fluorescent dye. Two outcomes look
different under the microscope:

| | **Released** | **Not released** |
|---|---|---|
| What happened | the particle broke down / escaped the endosome and the dye spread through the cytosol | the particle was taken up but stayed intact, cargo trapped |
| How it looks | even ("homogeneous") green fill of the whole cell, **nucleus included** | discrete bright dots (puncta), often near the periphery; **dark nucleus** |
| Why it looks so | free calcein diffuses everywhere and is no longer self-quenched | calcein packed in a particle is concentrated (self-quenching) and, possibly, quenched by Fe³⁺ |

A third group took nothing up (no signal) and is **outside the question**: such a
cell is neither a release nor a failed release.

The pipeline answers: *of the cells that took up cargo, what fraction released
it?*, for each well, so conditions (e.g. ± histidine, ± BSA, before/after
glucose) can be compared.

Typical images: RGB 8-bit TIFF, 5958 × 4463 px, ~1600–2400 cells per field,
one brightfield and one fluorescence image per sample.

---

## 2. Quick start

### Install (once per computer)

1. Miniconda, then a conda environment with Cellpose. On the lab PC this is the
   **ROCm** environment `rocm312` (AMD Radeon RX 7800 XT):
   ```
   conda activate rocm312
   pip install -r requirements.txt
   ```
2. Get the code:
   ```
   git clone https://github.com/<owner>/<repo>.git
   ```

### Run (GUI)

Double-click **`Cargo Analysis.bat`**. It finds conda and tests every named
environment with `tools/env_probe.py`, each through `conda run` in a separate
process (GPU-capable = 2, CPU-only = 1, missing packages = 0). The best one is
remembered in `launcher_environment.txt` (also the place to force a choice) and
the GUI is started with **that environment's own `python.exe` by full path**. Then: **Setup check** tab → everything green → **Run** tab → choose
the *parent* folder (the one containing A1, A2, …) → tick samples → **Run
analysis**.

### Run (command line)

```
python run_cargo_pipeline.py "D:\data\pre_glucose" --samples all
python run_cargo_pipeline.py "D:\data\pre_glucose" --samples A1,B2 --force   # re-segment
python run_cargo_pipeline.py --gpu-info                                      # check GPU
```

### CNN (optional)

```
python -m cargo_cnn.train   --data "D:\data\Slide I"
python -m cargo_cnn.predict --model models\cnn_<date>\model.pt --parent "D:\data\pre_glucose"
```
Read [section 9](#9-the-cnn-classifier) before using CNN numbers.

---

## 3. Data layout and file naming

### Input

```
<parent>\                          e.g. 8_cond_pre_glucose   <- choose THIS folder
    A1\
        A1_brightfield.tif         segmentation input
        A1_fluorescence.tif        measured (green channel = calcein)
        A1_overlay.tif             ignored (a derived image)
    A2\ ...
```

**Channel identification** (`run_cargo_pipeline.find_channels`), in order:

1. *Explicit names* (preferred): a filename containing `brightfield` or `_bf` is
   the brightfield; one containing `fluor`, `_fl` or `calcein` is the
   fluorescence.
2. *Positional fallback* (older data): of the remaining tifs (excluding anything
   named `overlay`), the alphabetically **first** is brightfield, the **last**
   is fluorescence. Works with timestamped names because brightfield is
   acquired first.

Files the pipeline itself writes (`_cp_masks`, `_classified_overlay`,
`_brightfield_for_cellpose`, `_bbox`) are always excluded from this search. (They
once were not — see [history](#10-history-how-the-pipeline-got-here), bug B2.)

### Output

Per sample, in `<sample>\edited_results\` (or under the folder chosen with
**Save results to** / `--output`):

| file | contents |
|---|---|
| `<base>_results.csv` | one row per cell: all features + `Class` |
| `<base>_summary.csv` | one row: counts, both release percentages, method, threshold used, analysis version, settings hash |
| `<base>_classified_overlay.png` | QC image at ¼ scale: outline colour = class |
| `<base>_log.txt` | everything the measurement step printed |

Next to the images: `<base>_cp_masks.tif` (16-bit label mask from Cellpose) and
`<base>_cp_masks_bbox.csv`.

At parent level: `<parent>_batch_summary.csv` (all summaries stacked) and
`run_manifest_<date>_<time>.json/.txt` (the run record, [section 8](#8-provenance-run-records-and-settings-hashes)).

`<base>` is `<parent>_<well>` (e.g. `8_cond_pre_glucose_A1`), or, with a plate
layout, `<well>_<condition>_<parent>` (e.g.
`A1_Fe-fum-plus-Histidin-5uL_8_cond_pre_glucose`).

CSV files use **semicolons**, so they open directly in a German/Austrian-locale
Excel.

**Overlay colours:** green = released · red = not released · cyan = released
(high-signal, legacy rule only) · grey = no signal · yellow = too small.

---

## 4. Processing, step by step, with the maths

Notation: `I(p)` is the intensity of pixel `p`; a cell `C` is a set of pixels;
`|C|` its pixel count; `mean_C(I) = (1/|C|) Σ_{p∈C} I(p)`.

### 4.1 Brightfield preparation (`load_brightfield`)

1. **Reduce to 2-D.** The two largest axes are height and width. Every other
   axis is removed: axes of length ≤ 4 (RGB channels) by averaging, longer ones
   (z or time) by maximum projection. *Why:* assuming the channel axis is first
   shreds `(H, W, 3)` images.
2. **Percentile stretch to 8-bit.**
   ```
   lo = P1(I),  hi = P99(I)                       (1st and 99th percentile)
   I8 = 255 · clip( (I − lo) / (hi − lo), 0, 1 )
   ```
   Percentiles, not min/max, so a few hot or dead pixels cannot dictate the
   range.
3. **Downscale** by `CELLPOSE_SCALE_FACTOR` (default 0.5, bilinear). Cells are
   ~85 px across at full resolution, so ~43 px at 0.5×, well inside Cellpose's
   working range; segmentation time scales with pixel count, so this is ~4×
   faster.
4. **Contrast check:** if the standard deviation of `I8` is < 3 the image is
   nearly flat and Cellpose will probably find nothing; a warning is printed.

### 4.2 Segmentation (Cellpose 4, model `cpsam`)

```
masks = model.eval(I8, flow_threshold = 0.4, cellprob_threshold = 0.0, normalize = True)
```

* `cellprob_threshold` is the main knob for **missed cells**: lower (−1, −2)
  accepts fainter, less certain cells.
* `flow_threshold` is how far a mask may deviate from the predicted flow field;
  higher accepts more irregular shapes.
* `cpsam` is largely scale-invariant, so no diameter is given.
* If **0 cells** are found, the run is retried once with
  `cellprob_threshold = −2`.
* **Coverage** = fraction of pixels inside any cell. < 25 % prints a note,
  because a confluent HeLa field should be well covered.
* The mask is saved as a **16-bit TIFF**. 8-bit would wrap label ids above 255
  (one field had 1670 cells → 1415 merged), and ImageJ misreads 16-bit PNGs as
  RGB.
* GPU choice (`pick_torch_device`): CUDA/ROCm → DirectML → Apple MPS → CPU.
  PyTorch-for-ROCm reports AMD cards through the `torch.cuda` API, recognisable
  by `torch.version.hip`.

A mask is **reused** if it already exists, unless `--force` / *Re-run Cellpose*
is ticked. Change any Cellpose setting → tick it, or nothing changes.

### 4.3 Fluorescence and mask at full resolution (`cargo_measure.run_sample`)

* RGB → keep the **green** channel (calcein). A multi-plane stack → maximum
  projection.
* The half-resolution mask is upscaled with **nearest neighbour**:
  ```
  row index in mask = floor( y · H_mask / H_full ),  same for columns
  ```
  Nearest neighbour never invents new label values, so every cell keeps its id.

### 4.4 Background subtraction (`subtract_background`)

ImageJ's *rolling ball* rolls a ball under the intensity surface; what it can
reach is background. Here it is approximated by a **morphological grey
opening** with a flat disc:

```
erosion   (I ⊖ D)(p) = min_{q ∈ D} I(p + q)
dilation  (I ⊕ D)(p) = max_{q ∈ D} I(p + q)
opening   B = (I ⊖ D) ⊕ D
corrected I'(p) = max( I(p) − B(p), 0 )
```

`D` is a disc of radius `ROLLING_BALL_RADIUS` (50 px). The opening removes every
bright structure smaller than the disc (cells) and keeps slowly varying
background. To be fast it is computed on every 4th pixel (`I[::4, ::4]`) with a
correspondingly decimated disc (same physical radius), then bilinearly
interpolated back to full size. A flat disc instead of a paraboloid ball is a
minor difference for a background estimate.

### 4.5 Per-cell geometry: core and periphery

For each cell `C` the **Euclidean distance transform** gives each pixel its
distance to the nearest pixel outside the cell, normalised to 0–1:

```
d(p)  = min_{q ∉ C} ‖p − q‖
nd(p) = d(p) / max_{C} d               (0 at the membrane, 1 at the deepest point)

CORE       = { p ∈ C : nd(p) ≥ f }        f = CORE_RADIUS_FRACTION = 0.5
PERIPHERY  = { p ∈ C : nd(p) < f }
```

Normalised distance makes "core" mean the same thing in a small and a large
cell. If a cell has no core pixels it is classed `excluded_too_small`.

### 4.6 Per-cell features (`measure`, `extra_features`)

All on the background-subtracted image `I'`.

| feature | formula | meaning |
|---|---|---|
| `Area` | `|C|` (full-resolution pixels) | |
| `MeanWhole` | `mean_C(I')` | overall brightness — **the strongest single predictor** |
| `CoreMean`, `PeripheryMean` | `mean_CORE(I')`, `mean_PERIPHERY(I')` | |
| `CoreFillRatio` | `CoreMean / PeripheryMean` (if periphery empty or ≤ 0: `CoreMean / MeanWhole`) | ≈ 1: dye reached the centre; ≪ 1: dark centre |
| `CoreCV` | `SD_CORE(I') / CoreMean` (population SD); `999` if `CoreMean ≤ 0` | low = even fill; high = puncta among dark pixels. A dark core gets 999, not 0, so an *empty* cell cannot pass as "perfectly homogeneous" |
| `SatFraction` | `|{p ∈ C : I'(p) ≥ 250}| / |C|` | fraction near the top of the range — flags cells bright only because a saturated blob bleeds over them |
| `NucMean` / `CytoMean` | `mean` over `nd ≥ 0.75` / over `0.05 < nd < 0.5` | **nucleus proxy**: the deepest quarter of the cell. Shape-based estimate, *not* a nuclear stain |
| `NucCytoRatio` | `NucMean / CytoMean` | ≈ 1: dye in the nucleus; < 1: excluded from it |
| `PunctaCount` | connected regions with `I' ≥ P75_C(I') + 2·SD_C(I')`, ≥ 4 px each | number of discrete bright spots (intact particles) |
| `PunctaArea` | total puncta pixels / `|C|` | |
| `PunctaMeanSize` | mean puncta size in px | |
| `RadialSlope` | slope of a straight-line fit through the mean intensity of 5 shells (`nd` in [0,0.2), [0.2,0.4), …, [0.8,1.0]) against shell index 0…4, divided by `MeanWhole` | > 0: centre brighter than rim; < 0: rim brighter |
| `CentreEdgeRatio` | innermost shell mean / outermost shell mean | |
| `CentroidX/Y`, `Bx By Bw Bh` | centre of mass; bounding box | for overlays, crops, the curator |

Cells with `MeanWhole < MIN_CELL_MEAN` (5) are classed `excluded_negative`
before any of the shape features are computed.

The nuclear/puncta/radial features are **measured and stored but not used by
the current rule** — they exist so `cargo_compare.py` can test them against
labels without a second measurement run.

### 4.7 The classification rule (v3.1, `classify_row`, method `"brightness"`)

```
if MeanWhole < MIN_CELL_MEAN                     -> excluded_negative
if no core pixels                                -> excluded_too_small
if MeanWhole < T                                 -> not_released
if CoreCV  > CORE_CV_CAP (0.65)                  -> not_released
if SATURATION_GUARD > 0 and SatFraction ≥ guard  -> excluded_saturated   (off by default)
otherwise                                        -> released
```

In words: **a cell has released if it is bright enough AND evenly filled.**

`T` is the brightness threshold — **per image** by default (next section); the
global value 25.7 is used only when the image has too few cells for Otsu, or
when `AUTO_THRESHOLD = False`. The threshold actually used is written into every
summary CSV and log.

Other methods, kept for comparison only:

* `"texture"` (the v1 rule): `released ⇔ CoreFillRatio ≥ 0.65 AND CoreCV ≤ 0.60`
  (plus `released_highsignal` if `MeanWhole ≥ VERY_HIGH_SIGNAL_CUTOFF`).
* `"nuclear"`: `released ⇔ NucCytoRatio ≥ 0.6` (falls back to brightness when the
  ratio cannot be computed). Unvalidated on real labels.

`classify_row` is separate from the pixel measurement so the curator can
re-classify thousands of cells in ~1 ms whenever a threshold slider moves, and
so the batch output and the interactive tools can never drift apart.

### 4.8 Per-image threshold by Otsu's method (`suggest_threshold`)

Exposure differs between acquisitions. Fitted against hand labels, the best
brightness threshold ranged from **14.2 to 53.3** across four samples (3.7×).
One global threshold reproduced the labels on ~87 % of cells; per-sample
thresholds fitted to the labels reached ~95 %. Otsu picks a per-image split
*without* labels:

Take the `MeanWhole` values of all cells with signal (≥ `MIN_CELL_MEAN`), build a
256-bin histogram with bin centres `c_i` and counts `h_i`. For a split after bin
`k`:

```
w0(k) = Σ_{i≤k} h_i                 w1(k) = Σ_{i>k} h_i
m0(k) = Σ_{i≤k} h_i c_i / w0(k)     m1(k) = Σ_{i>k} h_i c_i / w1(k)

between-class variance  σ_B²(k) = w0(k) · w1(k) · (m0(k) − m1(k))²
T = c_k*   with  k* = argmax_k σ_B²(k)
```

Maximising the between-class variance is the same as minimising the spread
within the two groups — the split that best separates dim from bright cells.
Fewer than 20 cells with signal → no suggestion, the global value is kept.

### 4.9 Per-sample summary (`write_summary`)

```
Released          = cells whose class starts with "released"
WithSignal        = Released + NotReleased
PercentReleased_relative = 100 · Released / WithSignal     ← the main readout
PercentReleased_absolute = 100 · Released / TotalCellsInImage
PercentUptake            = 100 · (Total − Excluded_NoSignal) / Total
```

*Relative* answers "of the cells that took up cargo, how many released"; it is
the number for comparing release efficiency. *Absolute* also falls when fewer
cells take anything up.

### 4.10 Quality-control messages in the log

* **Dynamic range**: `ratio = P99(MeanWhole) / median(MeanWhole)`. Above 20 a
  few very bright objects dominate; blobs may bleed into neighbours.
* **Degenerate features**: median fill ratio > 0.95 *and* median CV < 0.10 means
  the "fluorescence" being measured is probably not a fluorescence image (this
  is exactly what measuring the mask against itself looks like; bug B2).
* Percentile lines `p10 … p90` for fill ratio and CV, to see the distributions
  behind the call.

---

## 5. How rules are evaluated against hand labels

### 5.1 Labels

The curation tool (`cargo_curate.py`) shows the whole field, zoomable; the user
labels cells with the keyboard (R released, N not released, X no uptake, S bad
segmentation, M a cell Cellpose missed). In **blind** mode the class colours
are hidden but outlines stay, so cells can be found without seeing the
software's answer. *Borderline review* walks through the 80 cells closest to
the threshold, uncoloured, and only then offers to reveal the call. Each label
is saved with its origin (`Source`): `manual_blind` (used for fitting and
training), `manual_review` (made with colours visible; recorded only),
`bad_segmentation` (merged/split cells, excluded from fitting because their
mean intensity belongs to no real cell), `manual_added` (missed cells,
counted only). File: `edited_results/<base>_curated.csv`; the automatic
`*_results.csv` is never modified.

**Display.** The *View* menu (key **V**) shows the calcein channel in green
(default), in green over the brightfield (cell borders and nucleus visible
next to the dye), or in grey; *Brightness* scales the display for dim cells.
In the green views released cells are outlined light blue instead of green.
Display settings never change a measurement.

*Fit threshold to my BLIND labels* tries every observed brightness value as
threshold, combined with every observed CV value (rounded to 0.01) as cap, and
keeps the pair with the highest agreement. Below 85 % it warns that brightness
and CV cannot capture what is being judged in that image. Only blind labels
(`Source = manual_blind`) are used for evaluation and for training: labels made
while seeing the software's answer lean towards it and would flatter whatever
rule was active at the time.

### 5.2 AUC — "how well does one feature separate the classes?"

The area under the ROC curve equals the probability that a randomly chosen
released cell has a higher feature value than a randomly chosen not-released
cell. Computed through ranks (the Mann–Whitney identity):

```
rank all n cells by the feature (1 = lowest)
R1  = sum of the ranks of the n1 released cells
AUC = ( R1 − n1(n1+1)/2 ) / ( n1 · n0 )
```

0.5 = useless, 1.0 = perfect separation, **< 0.5 = predictive in the opposite
direction**. Core CV scored 0.074 on 536 blind labels, i.e. 0.926 once flipped:
*low* CV means released.

### 5.3 Best single threshold and combined rules (`cargo_compare.py`)

For each feature every observed value is tried as a threshold and the accuracy
`(correct calls) / n` recorded; the best is reported. For two- and three-feature
AND-rules the same is done over a grid of value pairs/triples. The script then
recommends switching rules only if the best combination beats the current rule
by ≥ 1.5 percentage points — smaller gains are within noise for a few hundred
labels — and warns that a gain found this way must be confirmed on a sample
that was not used to find it.

### 5.4 The ceiling: ambiguous cells

Among the 536 blind labels there were **50 pairs of cells in the same image with
nearly identical brightness (< 2 grey levels apart) and CV (< 0.05 apart) but
opposite labels** — about 9 % of cells, close to the ~12 % the rule gets wrong.
No threshold on those two numbers can separate such pairs. Either other
information separates them (hence the extra features and the CNN), or the
labels themselves are uncertain there and no method can do better.

---

## 6. Statistics in the Figures tab

### 6.1 The unit of replication

Every test compares **groups of images** (samples), never groups of cells. Each
image holds ~2000 cells; treating them as independent replicates would give
tiny p-values that reflect cell count, not the experiment
(*pseudoreplication*). The GUI asks what the replicates are:

* **technical** — wells/fields from one experiment. Measures assay precision.
  The report says so and recommends reporting mean ± SD descriptively.
* **biological** — independent experiments (different days, passages). Only
  these support a claim that the treatment changes release.
* **unspecified** — descriptive only, no test.

`aggregate_replicates` averages technical replicates into one value per
biological replicate before any test.

### 6.2 Descriptives

`mean`, `SD` with `ddof = 1` (sample SD, divisor `n − 1`), `SEM = SD / √n`.
Every individual sample is plotted as a dot by default, so n and spread are
visible.

### 6.3 Tests (two groups a, b)

**Welch's t-test** (unequal variances; default for independent groups):
```
t  = (mean_a − mean_b) / √( s_a²/n_a + s_b²/n_b )
df = ( s_a²/n_a + s_b²/n_b )² / [ (s_a²/n_a)²/(n_a−1) + (s_b²/n_b)²/(n_b−1) ]   (Welch–Satterthwaite)
```

**Paired t-test** (same wells before and after — the stronger design):
```
d_i = a_i − b_i,   t = mean(d) / ( SD(d) / √n ),   df = n − 1
```

Two-sided p-value for both, from the Student t distribution via the regularised
incomplete beta function (computed in pure numpy with a continued fraction, so
no scipy is needed for these two):
```
p = I_x( df/2, 1/2 ),   x = df / (df + t²)
```

**Mann–Whitney U** (rank-based, no normality assumption):
```
U = R_a − n_a(n_a+1)/2
z = ( U − n_a n_b / 2 ) / √( n_a n_b (n_a+n_b+1) / 12 )
p = 2 · (1 − Φ(|z|))       (normal approximation; Φ via the Abramowitz–Stegun erf)
```
With n ≤ 3 per group a rank test cannot reach p < 0.05 even for perfect
separation; the report says so.

**Wilcoxon signed-rank** (paired, non-normal), **one-way ANOVA** and
**Kruskal–Wallis** (3+ groups) use scipy.

### 6.4 Automatic test choice (`recommend_test`)

* **Shapiro–Wilk** per group (n ≥ 3): p < 0.05 → "not normal".
* **Levene**: p < 0.05 → variances differ.
* Two groups: normal → Welch (or paired t); not normal → Mann–Whitney (or
  Wilcoxon). Three or more: ANOVA / Kruskal–Wallis.
* The reasoning is printed. With n < 8 normality tests have little power: a
  "pass" means "no evidence against", not "normal".
* Pairwise tests after ANOVA are **not** corrected for multiple comparisons; the
  report warns about this.

**n < 2** per group → no test. **n = 2** → a p-value is printed with the note
that it is essentially uninformative.

### 6.5 Interaction

Dragging bars only changes their **order**; the values and every statistic are
untouched (`cargo_interactive.py`). Bar heights cannot be edited by hand.

---

## 7. Sample naming from a plate layout

`cargo_labels.parse_mask` reads a table pasted from Excel (tab-separated):

```
Position | Fe-fum                 | Fe-fum - His           | ...
1        | Fe-fum + Histidin 5 uL | Fe-fum - Histidin 5 uL |
2        | Fe-fum + Histidin 3 uL | Fe-fum - Histidin 3 uL |
```

**Column-major** (default): data column *k* → well letter A, B, C …; the
Position column gives the number. So column 1 gives A1 (upper) and A2 (lower),
column 2 gives B1/B2. *Row-major* is available; the preview cross-checks the
resulting wells against the sample folders and warns when none match — the
typical sign of a wrong orientation.

Filename cleaning (`clean_for_filename`): `+` → `plus`, a free-standing `-` →
`minus` (kept apart because `Fe-fum + His` and `Fe-fum - His` are opposite
conditions and must not collapse into the same name), `@` → space, characters
illegal in Windows names (`<>:"/\|?*;`) → `-`, `5 uL` → `5uL`, whitespace →
`-`. The original label is kept unchanged in the `Label` column for figure text.

---

## 8. Provenance: run records and settings hashes

Every run writes `run_manifest_<date>_<time>.json` and `.txt`
(`cargo_version.py`) containing:

* `analysis_version` — bumped by hand whenever a change alters the numbers;
* `settings` — every setting in `TRACKED_SETTINGS` (all that affect results);
* `settings_hash` — `SHA-256("KEY=value;KEY=value;…" over sorted tracked settings)`,
  first 10 hex digits. **Same hash ⇔ same rules.**
* `code` — SHA-256 fingerprints of `cargo_measure.py`, `run_cargo_pipeline.py`,
  `cargo_labels.py`, **the git commit** and nearest tag (e.g.
  `v3.1-4-g5661605`), and a warning if the code had uncommitted edits;
* `environment` — Python, OS, machine, versions of numpy/scipy/cellpose/torch/
  tifffile/Pillow, GPU name;
* `inputs` — size and modification time of each brightfield and fluorescence
  file;
* `data` — parent folder, samples, completed/failed, the plate layout used.

`AnalysisVersion` and `SettingsHash` are also written into every summary CSV, so
a summary copied out of its folder is still traceable. The Figures tab warns
when samples in one chart come from **different settings hashes**.

Compare two runs:
```
python cargo_version.py run_manifest_A.json run_manifest_B.json
```

**result → manifest → git commit → exact code.** That chain is what makes a
number in a thesis reproducible.

---

## 9. The CNN classifier

`cargo_cnn/` · version 1.0 · optional — the rule-based pipeline is unchanged.

### 9.1 Why

The rule uses two numbers per cell (brightness, core CV). Section 5.4 shows that
for ~9 % of cells those two numbers are identical while the labels differ. A
person labelling looks at the **whole picture of the cell** — where the dye is,
whether the nucleus is filled, whether there are dots — and a threshold on two
numbers cannot do that. A convolutional neural network (CNN) learns directly
from the pixels which patterns go with which label.

The second problem is **background**. The brightness threshold is absolute, so
anything that shifts a cell's intensity without changing its biology — exposure,
uneven illumination, haze, a bright neighbour — moves cells across the
threshold. Per-image Otsu partly compensates for exposure but not for
background that varies *within* a field.

(A CNN is not an "LLM": a large language model processes text. For images the
right family is convolutional networks.)

### 9.2 What the network sees

Each labelled cell becomes one example: a square crop centred on the cell, side
= 1.4 × the longer side of its bounding box (+4 px), resized to **64 × 64**.
Four channels:

```
local background   b = median of crop pixels that belong to NO cell
                       (if those are < 5 % of the crop: b = 0, trust the rolling ball)
J = I' − b         (I' = rolling-ball-corrected image, exactly as in 4.4)

ch0  intensity  = J / 50                            absolute brightness kept
ch1  texture    = clip( J / max(mean_own(J), 1), −1, 4 ) / 2     pattern only
ch2  own mask   = 1 inside the cell being judged
ch3  other mask = 1 inside any neighbouring cell
```

* ch0 keeps brightness because it is genuinely informative (AUC 0.92).
* ch1 removes brightness and keeps only the shape of the signal — the CNN's
  version of core CV, fill ratio, puncta and radial profile at once.
* ch2/ch3 tell the network **which pixels it is being asked about**. Without
  them a bright neighbour inside the crop is indistinguishable from the cell
  itself.
* The local-median correction does what per-image normalisation could not:
  per-image *median* normalisation failed (AUC 0.870 → 0.50) because in a
  field where many cells released, the median **is** released cells. Here only
  pixels outside every cell are used, so released cells cannot raise it.

### 9.3 How background is made irrelevant: augmentation

During training each crop is changed at random every time it is shown — in ways
that **must not** change its label — so the only way to fit the labels is to
ignore those changes:

```
rotation by k·90°, k ∈ {0,1,2,3}; mirror with probability ½
exposure       I ← s · I,          s ~ U(0.8, 1.25)     (--exposure-aug)
background     I ← I + o + a·(cos θ · x + sin θ · y)
               o ~ U(−3, 8) grey levels,  a ~ U(0, 6),  θ ~ U(0, 2π),  x, y ∈ [−½, ½]
noise          I ← I + N(0, σ²),  σ ~ U(0, 1.5)
```

The augmentation happens **before** the local background correction, so the
network learns to cope with whatever the correction leaves behind (a tilt is
only partly removed by a median).

The exposure range is deliberately modest: a dim cell is partly dim *because*
it did not release, so exposure is not entirely label-free.

### 9.4 Architecture (`cargo_cnn/model.py`)

```
4×64×64 → [conv3×3–BN–ReLU ×2,  16] → maxpool → 32×32
        → [ … ,  32] → maxpool → 16×16
        → [ … ,  64] → maxpool →  8×8
        → [ … , 128]
        → concat(global average pool, global max pool) = 256 values
        → dropout 0.3 → linear → logit z;   P(released) = σ(z) = 1 / (1 + e^(−z))
```

~294 000 parameters. Small on purpose: with a few hundred labelled cells, a
large network (ResNet-50 has 25 million) would memorise the training cells.
Average pooling captures "how much signal overall", max pooling "is there a
strong local feature anywhere" (a punctum, a bright rim).

### 9.5 Training (`cargo_cnn/train.py`)

**Loss** — binary cross-entropy:
```
L = − [ y · log σ(z) + (1 − y) · log(1 − σ(z)) ]
```

**Balanced batches.** Cells are drawn for training with probability
`1 / (number of cells in their class)`, so every batch is about half released,
half not released, however unequal the labels are. Without this, a network
trained on e.g. 84 % "released" labels learns that answering "released" for
every cell already scores 84 % and stops there — in a test it caught 0 of 34
not-released cells. The rare class is shown many times, each time with a
different augmentation, which is what makes the repetition useful.

**Epoch size.** One epoch is at least 1024 drawn cells (16 batches). With a few
hundred labels a natural epoch is only 3–4 batches, and 60 such epochs were too
little training. The epoch setting now means the same amount of training
however many cells are labelled.

**GPU self-test.** Before training (and prediction) one forward/backward pass
is run on the GPU. With PyTorch-for-ROCm on Windows, AMD's MIOpen library can
fail to compile its batch-normalisation training kernel at run time
(`'type_traits' file not found` → `miopenStatusUnknownError`). The program then
switches MIOpen off (`torch.backends.cudnn.enabled = False`, which on ROCm
controls MIOpen) and uses PyTorch's own pre-built GPU kernels; only if that also
fails does it fall back to the CPU. The log says which happened.

**Optimiser** — AdamW, learning rate 1e-3, weight decay 1e-3 (pulls weights
towards zero; another guard against memorising), batch size 64, 60 epochs,
cosine learning-rate schedule:
```
lr(t) = lr₀ · ½ · (1 + cos(π t / T))
```

**Test-time augmentation** — at prediction the cell is shown in all 8
rotations/mirrors and the 8 probabilities averaged; a cell does not change
class when turned, so this only removes orientation noise.

**Decision** — `P(released) ≥ 0.5` → released (`--threshold`).

### 9.6 Validation — the part that decides whether to use it

**Leave-one-field-out cross-validation.** With F labelled fields the network is
trained F times, each time on F − 1 fields, and tested on the one left out. It
never sees a single cell of the image it is tested on.

*Why whole fields and not random cells:* cells in one image share exposure,
background and focus. With cells of the test image in the training set the
network could learn "this image is bright" instead of "this cell released",
and the accuracy would be inflated. (With fewer than 3 labelled fields this is
impossible; the script falls back to random 5-fold over cells and says loudly
that the number is optimistic.)

**Balanced accuracy decides.** Plain accuracy is misleading when one class
dominates: with 84 % "released" labels, "always released" scores 84 %. The
table and the verdict therefore use balanced accuracy — the mean of the hit
rate for released cells and the hit rate for not-released cells (guessing =
50 %) — and also print how many cells of each class were caught. When fewer
than 25 % of labels are in one class, the log says so and recommends labelling
more of that class.

On the same held-out cells the script compares the CNN against **the class the
pipeline actually assigned** (the `Class` column of `*_results.csv`, i.e. Otsu
threshold + CV cap as really run). It reports accuracy, balanced accuracy
(mean of sensitivity and specificity), AUC, and a verdict:

* CNN better by ≥ 2 points (balanced) → worth using, confirm on a newly labelled field;
* within ±2 points → keep the rule (simpler, explainable);
* worse → do not use it.

The rule's thresholds were fitted on these same labels, so the comparison gives
the rule a small home advantage. A CNN that still wins has earned it.

**Which mask belongs to which labels.** The curator names its file after the
mask it opened: `<base>_curated.csv` was made on `<base>_cp_masks.tif`, whose
measurements are `<base>_results.csv`. Training uses exactly this link first,
so labels stay attached to the right mask even when a later run (e.g. with
plate-layout names) left a second mask in the folder. Only label files that
do not follow this naming fall back to searching by well.

**Safety check.** Labels point at Cellpose label ids. If a field was
re-segmented after labelling (`--force`, or a renamed base after adding a
plate layout), the same id now marks a different cell. When a results file is
available, the area of every labelled cell is compared with the current mask;
if more than 10 % differ by more than 2 % the field is **refused**.

### 9.7 Using it

**From the GUI (easiest):** the **CNN** tab does all of this with buttons —
*Open curation tool*, *Train model*, *Classify ticked samples*, *Open training
report*. It runs everything in the same Python environment the window was
started in, which `Cargo Analysis.bat` chooses for its working GPU; the top of
the tab shows the device that will be used. Step-by-step: `GETTING_STARTED.md`.

**From the command line:**

```
# 1. train (needs the *_curated.csv files from the curator)
python -m cargo_cnn.train --data "D:\...\8_cond_pre_glucose"

#    if results were written to a separate --output folder, also give that folder:
python -m cargo_cnn.train --data "D:\...\8_cond_pre_glucose" --results "D:\...\analysis_out"

#    or list every field explicitly:
python -m cargo_cnn.train --pairs fields.csv
#    fields.csv (semicolons):
#    curated;fluorescence;mask;results
#    D:\...\A1_curated.csv;D:\...\A1\A1_fluorescence.tif;D:\...\A1\..._cp_masks.tif;D:\...\A1_..._results.csv

# 2. read models\cnn_<date>\training_report.txt — the verdict is at the end

# 3. predict (after the normal pipeline has made the masks)
python -m cargo_cnn.predict --model models\cnn_<date>\model.pt --parent "D:\...\pre_glucose" --samples all
```

Outputs per sample: `<base>_cnn.csv` (per cell: `P_released`, `Class`,
`RuleClass`, `Agree`), `<base>_cnn_summary.csv` (same columns as the normal
summary, `Method = cnn`, `SettingsHash` = model fingerprint,
`AgreementWithRule`), `<base>_cnn_overlay.png`; at parent level
`<parent>_cnn_batch_summary.csv`, which the Figures tab reads directly.

`cv_predictions.csv` in the model folder lists every held-out cell with its
label, CNN probability and rule call. Sorting by disagreement is the fastest
way to see *what kind* of cell each method gets wrong.

Timing (measured on a CPU-only test machine; the RX 7800 XT is much faster):
training with cross-validation on ~400 cells, 4 folds + final model, takes
~2.5 min at 30 epochs, so ~5 min at the default 60. Prediction on a full-size
synthetic field with 2000 cells, including loading and background subtraction
and all 8 TTA views, took 21 s.

### 9.8 Test on synthetic data

`tests/make_synthetic.py` builds fields with a known answer (evenly filled
"released" cells, punctate rim-weighted "not released" cells, no-uptake cells),
each field with its own exposure, tilted background and a haze blob, and 8 % of
labels deliberately flipped to imitate labelling disagreement. Result
(2026-11-14):

| | CNN | rule |
|---|---|---|
| leave-one-field-out, 4 fields, 382 labelled cells | **90.1 %** (AUC 0.89) | 82.7 % |
| fifth, unlabelled field scored against the true classes | **100 %** | 80.0 % |

With 8 % flipped labels, ~92 % is the best any method can score in
cross-validation. The rule's errors were dim but evenly filled cells falling
below an Otsu threshold pushed up by bright cells and background.

**This proves the code works and can learn a real pattern. It says nothing
about accuracy on real cells**, which are far more ambiguous. That number comes
only from training on your blind labels.

### 9.9 What the CNN cannot do

* It needs enough examples of BOTH classes. Tested with 84 % / 16 % labels
  (34 not-released cells in total), the rule stayed better even when scored
  against the true answers (89 % vs 81 % balanced). Each class needs on the
  order of 100+ blind labels before the CNN has a fair chance.
* It cannot exceed the consistency of the labels. If ~9 % of cells are truly
  ambiguous, ~90 % is near the ceiling.
* It learns the labeller's judgement, including any bias. More fields from
  more days help far more than a bigger network.
* It is less explainable than "brightness ≥ T and CV ≤ 0.65". For a thesis, the
  rule may remain the primary method with the CNN as a cross-check, unless the
  CNN is clearly better on held-out fields.
* A model is tied to the imaging conditions it was trained on (objective,
  exposure range, staining). New conditions → label a field and retrain.

---

## 10. History: how the pipeline got here

Numbers below come from the analyses run during development (recorded in the
code comments and `cargo_version.VERSION_HISTORY`). Data from the release
studies of 13.07.2026 and 17.07.2026.

### 10.1 Classification rule versions

| version | rule | evidence | agreement with labels |
|---|---|---|---|
| **1.0** | `CoreFillRatio ≥ 0.65 AND CoreCV ≤ 0.60` (ImageJ macro) | designed from the biology: released = centre as bright as rim, smooth | ~25 % — unusable |
| **2.0** | same maths, measured in numpy instead of the macro | — | (unchanged) |
| **2.1** | `MeanWhole ≥ T` | 195 hand labels, 4 samples: MeanWhole AUC 0.917; CoreFillRatio 0.451; CoreCV 0.453 (chance) | ~87 % global T; ~95 % per-sample T |
| **3.0** | + per-image Otsu threshold, recorded in every summary; saturation veto available | best T ranged 14.2–53.3 across samples; 6 of 50 bright false positives were blob bleed-over | |
| **3.1** | `MeanWhole ≥ T AND CoreCV ≤ 0.65` | 536 **blind** labels: CoreCV AUC 0.074 (= 0.926 inverted) | **88 %** (brightness alone 80 %) |
| CNN 1.0 | learned from pixels | leave-one-field-out vs the rule | to be measured on real labels |

Why CV went from useless (2.1) to useful (3.1): the first 195 labels were made
with the automatic colours visible, the 536 later ones blind. The earlier
"failure" of the feature was most likely label quality. Lesson: **only blind
labels are evidence.**

### 10.2 Experiments that were tried and rejected

* **Per-image median normalisation** (`MeanWhole / median`, robust z with MAD,
  ÷ P75, ÷ P90): AUC fell from 0.870 to ~0.50. In confluent fields the median is
  made of released cells, so dividing by it cancels the signal. → Never
  normalise by a statistic of the cells themselves. (The CNN's local correction
  uses only cell-free pixels for this reason.)
* **Better thresholds or noise correction for the remaining errors**: in sample
  A2, false positives had median brightness 58.4 and false negatives 58.1 —
  the same. The errors are not at a noisy edge; the information is not in the
  number. → New features, then the CNN.

### 10.3 Engineering milestones and the bugs behind them

| # | problem | cause | fix |
|---|---|---|---|
| B1 | many manual steps per sample | brightfield exported by `Export_Brightfield_For_Cellpose.ijm` (8-bit, 0.5×, JPEG q90), Cellpose run by hand in its GUI, macro run by hand | Python orchestrator (`run_cargo_pipeline.py`) runs Cellpose directly and then the measurement; later the JPEG step was dropped (every extra step was a place to corrupt the image) |
| B2 | **94.9 % "released"**, CoreFillRatio p10–p90 = 0.989–1.000, CV = 0 | the mask `.tif` sorted after the timestamped acquisitions and was picked as the **fluorescence** image — the mask was measured against itself | generated files excluded from the channel search; Python and macro both refuse when fluorescence = mask; degenerate-feature warning |
| B3 | 0 ROIs from the label map in ImageJ | 16-bit PNG opened by ImageJ as RGB, destroying label ids | masks saved as 16-bit TIFF |
| B4 | label ids above 255 merged | 8-bit save wraps ids (1670 cells → 1415 merged) | 16-bit |
| B5 | Fiji not found | launcher renamed between versions | try all known names, list folder contents on failure |
| B6 | `HeadlessException` | Fiji's ROI Manager is an AWT window and cannot exist headless | launch Fiji normally in batch mode |
| B7 | one sample 2.5 h instead of ~9 min | Java heap too small for ~2500 ROIs → continuous garbage collection | `--mem`, capped at half of physical RAM; a JVM asked for more than exists exits silently |
| B8 | macro died silently on 2400-cell images | a 6.6-million-iteration `getPixel()` loop exhausted the interpreter | bounding boxes computed in numpy and passed in; then the whole measurement moved to numpy (`cargo_measure.py`): **~80× faster (6 s vs 500 s)**, no interpreter to crash |
| B9 | GUI launcher used the CPU | `.bat` stopped at the first environment that imported everything (`base`, CPU-only torch) | every environment scored, GPU wins (`EnableDelayedExpansion` for variables inside loops) |
| B10 | AMD GPU unused | CUDA-only detection | ROCm via `torch.version.hip`, DirectML, MPS |
| B11 | `+His` and `-His` conditions overwrote each other | sign characters stripped from filenames | `plus` / `minus` |
| B12 | empty batch summary after adding a plate layout | batch collector rebuilt the old file names | actual base names passed through |
| B13 | *(2026-11-14)* run record listed the wrong input files | fingerprint block ran before `find_channels` for the current sample | moved below it |
| B14 | *(2026-11-14)* first "Cellpose cellprob" box had no effect | the setting was listed twice in the GUI | duplicate removed |
| B15 | *(2026-11-14)* GUI refused to run without the legacy macro | validation required the `.ijm` though numpy is the default | check removed; setup check marks it optional |
| B17 | *(2026-11-14)* launcher measured rocm312 as the GPU environment and still started `base` | the search activated every environment in turn inside one batch script; something in those repeated activations disturbed the script's own selection variables (path-only entries such as PyMOL were also not skipped, because `findstr /c:"\"` does not search for a backslash) | environments are tested with `conda run` in separate processes, nothing is activated during the search; the winner's `python.exe` is started by full path; path-only entries skipped by string comparison; choice remembered |
| B16 | *(2026-11-14)* curator could open the wrong mask | it took the first file containing `_cp_masks` — including `_cp_masks_bbox.csv`, and an older mask whenever its name sorted first | only image files; prefer the mask with a matching results file, then the newest; choice shown in the title line |

### 10.4 Tools added along the way

GUI with Run / Sample names / Settings / Figures / Setup check tabs; curation
tool with zoom, blind mode and threshold fitting; Figures tab with grouped bars,
assumption checks, replicate-aware statistics and drag-to-reorder; collage
builder; plate-layout naming; rule comparison script; provenance records; CNN
classifier; git repository.

---

## 11. Settings reference

Pipeline settings live at the top of `run_cargo_pipeline.py`; the GUI overrides
them through `_gui_config.json` (`apply_settings`). Your current values
(`config/gui_config.example.json`) are the defaults below unless noted.

| setting | default | effect | needs |
|---|---|---|---|
| `CLASSIFY_METHOD` | `brightness` | rule (4.7); `texture`, `nuclear` for comparison | re-classify |
| `BRIGHTNESS_THRESHOLD` | 25.7 | global fallback T | re-classify |
| `AUTO_THRESHOLD` | `True` | per-image Otsu T (4.8). *Not shown in the GUI* — always on from the GUI | re-classify |
| `CORE_CV_CAP` | 0.65 | released requires CoreCV ≤ this; 2.0 effectively disables | re-classify |
| `SATURATION_GUARD` | 0.0 (off) | SatFraction ≥ this → `excluded_saturated` | re-classify |
| `MIN_CELL_MEAN` | 5 | below = no uptake, excluded | **re-measure** |
| `CORE_RADIUS_FRACTION` | 0.5 | core/periphery split | **re-measure** |
| `ROLLING_BALL_RADIUS` | 50 px | background disc radius | **re-measure** |
| `FILL_RATIO_THRESHOLD` | 0.65 | legacy `texture` rule only | re-classify |
| `CORE_CV_THRESHOLD` | 0.60 | legacy `texture` rule only | re-classify |
| `VERY_HIGH_SIGNAL_CUTOFF_8BIT` | 200 | legacy only; auto-scaled for 16-bit | re-classify |
| `CELLPOSE_MODEL` | `cpsam` | | **re-segment** |
| `CELLPROB_THRESHOLD` | 0.0 | lower = more/fainter cells | **re-segment** (`--force`) |
| `FLOW_THRESHOLD` | 0.4 | higher = more irregular masks accepted | **re-segment** |
| `CELLPOSE_SCALE_FACTOR` | 0.5 | segmentation resolution | **re-segment** |
| `USE_GPU` | `True` | falls back to CPU | — |
| `USE_FIJI` | `False` | legacy ImageJ backend | — |
| `FIJI_MEMORY_MB` | 12288 | legacy only; capped at ½ RAM | — |
| `FIJI_DIR` | `C:\Users\cemin\Documents\Molbio\Fiji\Fiji` | legacy only | — |

*re-classify* = the curator applies it instantly; *re-measure* = pixels must be
measured again; *re-segment* = Cellpose must run again, which **changes cell
ids and invalidates existing labels for that field**.

CNN settings (`python -m cargo_cnn.train --help`): `--epochs 60`, `--batch 64`,
`--lr 1e-3`, `--weight-decay 1e-3`, `--width 16`, `--dropout 0.3`,
`--exposure-aug 1.25`, `--rolling-ball 50`, `--min-cell-mean 5`, `--seed 0`.
Everything used is stored inside `model.pt` and `model_info.json`.

---

## 12. Version control: how to work with this repository

### 12.1 What git does for you here

Git stores every saved state ("commit") of the code with a message saying what
changed and why. You can always see the history, compare versions, and go back.
Combined with the run records (section 8), every number you report points to
the exact code that produced it.

Two separate version numbers exist, on purpose:

* **Git commits** — every change, however small (typo, GUI layout, a fix).
* **`ANALYSIS_VERSION`** in `cargo_version.py` — bumped only when a change
  **alters the numbers** a run produces. Results with different analysis
  versions are not comparable.

When you bump `ANALYSIS_VERSION`, also add a line to `VERSION_HISTORY`, an
entry in `CHANGELOG.md`, and a git **tag** with the same number.

### 12.2 One-time setup on the lab PC

The click-only way, with **GitHub Desktop**, is in `GETTING_STARTED.md`. The
program finds GitHub Desktop's built-in git by itself, so run records get the
commit id and the *Get latest version from GitHub* button (Setup check tab)
works without installing anything else. With *Git for Windows* the same works
from a terminal:

```
cd C:\Users\cemin\Documents\Molbio\Master\Projektlabor\Pharma\Python
git clone https://github.com/<owner>/<repo>.git
git config --global user.name  "Your Name"
git config --global user.email "the e-mail of your GitHub account"
```

Run the analysis from the cloned folder from now on (`Cargo Analysis.bat` sits
inside it). Copy your old `cargo_gui_settings.json` into it to keep your GUI
settings — it is ignored by git because it holds local paths.

The files not yet in the repository (`collage_gui.py`, `collage_engine.py`,
`Start_Collage_Builder.bat`, the `.ijm` macro) can be copied in and committed
the same way as any change below. **Copy only the newest version of a file**:
an older copy of `cargo_measure.py` (without `core_cv_cap` and the extra
features) makes the pipeline crash, and an older `Cargo Analysis.bat` picks the
CPU-only `base` environment. `git diff` before committing shows immediately if a
file is going backwards.

### 12.3 Everyday workflow

```
git pull                         # 1. get the newest version (e.g. changes made with Claude)
                                 # 2. edit / run / test
git status                       # 3. what changed?
git diff                         #    exactly which lines?
git add cargo_measure.py         # 4. choose what goes into the commit
git commit -m "Short summary

Why the change was needed and what it affects."
git push                         # 5. upload to GitHub
```

In GitHub Desktop: the same steps are buttons (*Fetch/Pull*, tick files, write
the message, *Commit*, *Push*).

**Good commit messages** say *why*, not only *what*: "Raise CV cap to 0.70 —
536 blind labels give 89 % vs 88 %" is useful in a year; "update" is not.

### 12.4 Looking back

```
git log --oneline                # list of commits
git log -p cargo_measure.py      # every change to one file
git show v3.1                    # what the v3.1 tag points to
git diff v3.1 HEAD               # everything changed since v3.1
git checkout 5661605 -- cargo_measure.py    # bring back an old version of one file
```

To reproduce an old result: open its `run_manifest_*.txt`, read `git_commit`,
then `git checkout <commit>` (afterwards `git checkout main` to return).
If the manifest says **UNCOMMITTED EDITS**, the code was modified after that
commit and cannot be recovered exactly — so **commit before analysing data that
goes into a figure**.

### 12.5 Trying something risky: branches

```
git switch -c try-nuclear-rule   # a parallel line of development
... edit, commit ...
git switch main                  # main is untouched
git merge try-nuclear-rule       # adopt it, if it worked
```

### 12.6 Releases and tags

```
git tag -a v3.2 -m "Analysis 3.2: <what changed in the numbers>"
git push --tags
```

On GitHub a tag can be turned into a *Release*, where a trained `model.pt` used
for a figure can be attached. The `models/` folder itself is not committed:
weights are binary, and the reports contain per-cell data.

### 12.7 What never goes into git

Raw images, masks, results CSVs, curated labels, run records, the `models/`
folder, and the GUI's local settings files — all listed in `.gitignore`. Images are
hundreds of MB; GitHub rejects files > 100 MB and the repository would become
unusable. Data stays on the lab drive; the run record links data and code.

### 12.8 History of this repository

| tag / commit | content |
|---|---|
| `v3.1` | unchanged import of the working copy of 2026-11-14 |
| following commits | bug fixes B13–B16, git commit in run records, CNN module, synthetic test, documentation, curation tool |

`CHANGELOG.md` lists changes in plain language, including the pre-git history.

---

## 13. Known limitations and open questions

**Science**

1. **The 88 % agreement refers to the global threshold 25.7.** Production runs
   use the per-image Otsu threshold (AUTO_THRESHOLD = True, not switchable in
   the GUI). Agreement of *Otsu + CV cap* with the blind labels has not been
   reported separately. `python -m cargo_cnn.train` prints exactly this number
   (the "rule acc" column uses the class the pipeline actually assigned).
2. The **nucleus is a geometric proxy** (deepest 25 % of the cell), not a
   stained nucleus. A Hoechst/DAPI channel would make NucCytoRatio a real
   measurement.
3. The nuclear, puncta and radial features have **not yet been validated** on
   real blind labels (run `cargo_compare.py` after re-measuring curated
   samples).
4. Calcein is quenched by **Fe³⁺** as well as by self-quenching, so de-quenching
   may reflect two mechanisms. A titration of calcein with free Fe³⁺ would
   clarify what the brightness threshold measures.
5. `SatFraction` is computed **after** background subtraction, so "≥ 250" is a
   little below true sensor saturation. The guard is off by default.
6. **Re-segmenting changes cell ids.** Labels made on a field are only valid
   for the mask they were made on. Changing the plate layout changes the base
   name, so no mask is found and Cellpose re-runs — new ids. Keep labelled
   fields' masks.

**Code**

7. Some docstrings are older than the code they describe:
   `run_cargo_pipeline.py`'s header still mentions PNG masks and the Fiji macro
   as step 4; `classify_row`'s first paragraph describes v2.1. This document is
   the current description.
8. `cargo_measure.py` contains three functions not used by the pipeline
   (`measure_nucleus`, `measure_puncta`, `measure_radial`) whose definitions
   differ from the active `extra_features` (e.g. puncta threshold
   `median + 3·1.4826·MAD` vs `P75 + 2·SD`). The values in the results files
   come from `extra_features`.
9. The label count for v3.1 appears as both **523** and **536** in comments.
   536 is the total blind set; 523 probably the subset with finite features.
10. AUC in `cargo_compare.py` breaks ties by order instead of averaging ranks
    (the CNN's `auc` averages them). Negligible with continuous features.
11. Mann–Whitney uses the normal approximation without tie or continuity
    correction.
12. Not in this repository yet: `collage_gui.py`, `collage_engine.py`,
    `Start_Collage_Builder.bat`,
    `Cargo_Release_Classification_v3_headless.ijm`, `INSTALL.md`.
13. The curator's help text still quotes the 195-label result (v2.1); its
    classification uses the v3.1 rule (brightness + CV cap).

---

## 14. File reference

| file | role |
|---|---|
| `Cargo Analysis.bat` | Windows launcher: finds conda, picks the best (GPU) environment, starts the GUI |
| `cargo_gui.py` | the window: Run, Sample names, Settings, Figures, CNN, Setup check (incl. update button) |
| `run_cargo_pipeline.py` | orchestrator: channel search, brightfield prep, Cellpose, measurement, batch summary, run record; CLI |
| `cargo_measure.py` | per-cell measurement, classification rule, Otsu, CSV and overlay writing |
| `cargo_plot.py` | figures and statistics |
| `cargo_interactive.py` | click / drag / recolour / rename bars |
| `cargo_labels.py` | plate layout → sample names |
| `cargo_curate.py` | curation: zoom, blind labelling, borderline review, threshold fit |
| `cargo_compare.py` | which rule best matches the blind labels |
| `cargo_version.py` | analysis version, version history, run records, git stamp |
| `cargo_cnn/data.py` | crops, channels, augmentation, finding labelled fields |
| `cargo_cnn/model.py` | the network |
| `cargo_cnn/train.py` | cross-validation, final training, report |
| `cargo_cnn/predict.py` | classify new samples |
| `tests/make_synthetic.py` | test fields with a known answer |
| `config/gui_config.example.json` | example of the settings the GUI passes to the pipeline |
| `CHANGELOG.md` | changes in plain language |
| `GETTING_STARTED.md` | click-by-click setup, CNN workflow, updates, troubleshooting |
| `DOCUMENTATION.md` | this file |

---

## 15. Glossary

**AUC** — area under the ROC curve; probability that a random positive scores
above a random negative. 0.5 = chance.
**Augmentation** — random label-preserving changes to training examples.
**Balanced accuracy** — mean of sensitivity and specificity; not fooled by
class imbalance.
**Batch normalisation (BN)** — rescales activations inside the network;
stabilises training.
**Blind label** — a hand label made without seeing the software's call.
**Cellpose / cpsam** — deep-learning cell segmentation / its Segment-Anything-
based model.
**CNN** — convolutional neural network; learns image filters from examples.
**Cross-validation** — repeatedly holding part of the data out for testing.
**CV (coefficient of variation)** — SD / mean.
**Dropout** — randomly silencing units during training to prevent memorising.
**EDT** — Euclidean distance transform.
**Epoch** — one pass over all training examples.
**Grey opening** — erosion followed by dilation; removes bright objects smaller
than the structuring element.
**Logit** — the network's raw output; `σ(logit)` is a probability.
**Otsu's method** — threshold maximising the between-class variance.
**Pseudoreplication** — treating non-independent measurements as independent
replicates.
**ROCm** — AMD's GPU computing platform; PyTorch-for-ROCm exposes AMD GPUs via
`torch.cuda`.
**Settings hash** — short fingerprint of all result-affecting settings.
**TTA** — test-time augmentation; averaging predictions over transformed
copies.
**Weight decay** — penalty pulling network weights towards zero.
