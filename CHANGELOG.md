# Changelog

All notable changes, newest first. "Analysis version" changes alter the numbers
a run produces; results from different analysis versions are not comparable.
Details and evidence: `DOCUMENTATION.md`, sections 10 and 12.

## [Unreleased] — 2026-11-19

Analysis version unchanged (3.1): rule results are identical to before.

### Fixed
- **Cellpose on a CPU was ~10x too slow**: Cellpose 4 runs in bfloat16 by
  default, which most laptop processors only emulate (measured 3.0 s vs
  0.24 s per transformer block). On the CPU a ~1 s test now picks bfloat16
  only where it is native, otherwise float32. GPU runs are unchanged. Masks
  from CPU runs may differ marginally from earlier CPU runs (float32 is the
  more exact format).

### Added
- **Stitching** (tab + `stitch_bzx.py`): joins the tiles of a Keyence BZ-X
  (e.g. 4 x 4 per position) into one brightfield and one fluorescence image
  per position, for every position under a parent folder. Tile positions are
  measured from the brightfield (phase correlation, global least squares),
  overlaps feather-blended without seams; reads the `.gci` file for grid,
  channels and calibration. Positions with fewer than 4 tiles or missing
  tiles are skipped and flagged (`stitch_report.csv`). Brightfield vignetting
  corrected by default, fluorescence left as recorded. Output folder can be
  sent straight to the Run tab.
- Cell count: the **calibration is read from the image file** when present
  (stitched images); the entry is used otherwise.
- Cell count: **cell numbers** (counted cells 1..n in reading order, same on
  the figure and in the table) and positions in µm; **Excel workbook**
  `cell_count.xlsx` (summary, per-sample sheet with bar charts, one sheet per
  image); **batch bar charts** per sample folder (mean ± SD of images, dots
  for single images) as PNG/PDF.
- Cell count: **Speed** setting (segment at 1/2, 1/3 or 1/4 size; time grows
  with the pixel count, so 1/4 is ~4x faster than 1/2 - meant for laptops
  without GPU), a "still segmenting (x min)" line every 30 s, and a warning
  when cells become too small (< 15 px) at the chosen size.
- **Cell count** (tab + `cell_count.py`): brightfield-only images are
  segmented with Cellpose (same settings as the analysis) and counted; edge
  rule all / complete cells only / unbiased counting frame, minimum area,
  density per mm², cell size. Paper figures: smooth outlines just inside each
  cell, PNG at full resolution and PDF/SVG with vector outlines, scale bar,
  n label, optional filled-mask figure and cell numbers. Per-cell and summary
  CSV, 16-bit masks; masks are reused when only the figure changes.
- Fiji is shown as **optional** everywhere (the analysis runs in Python and
  never needed it); plain **ImageJ 1.x** (`ImageJ.exe`) is accepted as well,
  also by the old macro backend (started without Fiji-only options).
- **Runs on any Windows PC** (e.g. a laptop): the launcher finds conda in
  every usual place (PATH, registry, Anaconda / Miniconda / Miniforge /
  Mambaforge folders, `conda_path.txt` for unusual places) and also works
  **without conda**. `Setup on this PC.bat` installs everything once (conda
  environment `cargo`, or `.venv` with a normal Python, installing Python
  3.12 by winget if needed; CUDA PyTorch on NVIDIA, CPU otherwise); the
  launcher offers to run it. Fixes *"conda.bat not found"* on the ThinkPad.
- Environment check now **loads** every package instead of only looking it
  up, so an environment that is installed but crashes is reported as
  *installed but does not load: <reason>* and never started. When the reason
  is a missing DLL, setup offers to install the Microsoft Visual C++ runtime
  (winget). The program prints the real numpy error instead of
  *"numpy is required"*.
- CNN training reports **two tests**: same conditions/unseen cells (each field
  split in halves) and unseen condition (leave one field out), each with a
  **brightness-threshold yardstick** learned on the same training cells.
- **Early stopping** for CNN training, on a stopping set taken from the
  training fields (never the test field); best epoch by balanced accuracy;
  final model trained for the median chosen epoch. `--no-early-stop`,
  `--patience`, `--min-epochs`.
- CNN accepts sample folders without a brightfield image: a lone acquisition
  image is used if its pixels show it is fluorescence (green-only RGB, or dark
  background with bright objects); a lone brightfield is refused.
- Curator shows the calcein signal in **green** (default), **green on
  brightfield**, or grey (key **V**), with a display-only *Brightness* slider.
- **CNN tab in the GUI**: open the curation tool, train, classify, open the
  report, with the compute device shown. Runs in the environment the `.bat`
  picked, so the GPU is used without activating anything.
- **Update button** (Setup check tab): `git pull` with plain-language results;
  finds GitHub Desktop's built-in git when git is not on the PATH.
- `GETTING_STARTED.md`: click-by-click guide.
- Training finds results in a separate *Save results to* folder (`--results`),
  and attaches labels to the exact mask they were made on (`<base>_curated.csv`
  ↔ `<base>_cp_masks.tif`).
- **Git repository.** The working copy was imported unchanged and tagged `v3.1`;
  every later change is a separate commit.
- **CNN classifier** (`cargo_cnn/`, version 1.0, optional). Learns released /
  not released from blind labels; background-robust inputs; leave-one-field-out
  validation against the rule; prediction writes `*_cnn.csv`,
  `*_cnn_summary.csv`, an overlay and a batch summary for the Figures tab.
- **Curation tool** `cargo_curate.py` added to the repository.
- **Synthetic test data** (`tests/make_synthetic.py`) with known answers.
- **Run records store the git commit**, nearest tag, and a warning when the code
  had uncommitted edits.
- `DOCUMENTATION.md`, `README.md`, `CHANGELOG.md`, `requirements.txt`,
  `.gitignore` (keeps data and model weights out of git), `.gitattributes`
  (CRLF for the `.bat`), `config/gui_config.example.json`.

### Fixed
- Early stopping set aside a whole training field, which on real data removed
  a whole condition from training and gave erratic stopping; it now takes ~15 %
  of the cells of every training field.
- CNN training skips labelled cells below the uptake limit (brightness < 5),
  which the pipeline excludes and the CNN never classifies; they were
  inflating the not-released class (15 of 69 labels in the first real run).
- CNN training crashed on AMD + Windows (`miopenStatusUnknownError`: MIOpen
  could not compile its batch-norm kernel). A GPU self-test now switches MIOpen
  off and uses PyTorch's own GPU kernels, falling back to the CPU only if needed.
- CNN with unequal labels (e.g. 84 % released) learned to call every cell
  released. Training now uses balanced batches and a minimum epoch size; the
  verdict uses balanced accuracy and reports hits per class.
- Skipped training fields now list the .tif files found, so the reason is visible.
- Launcher started the CPU-only `base` environment although it had found
  `rocm312` with the GPU. Environments are now tested in separate processes,
  the GUI is started with the chosen environment's own Python, and the choice
  is remembered in `launcher_environment.txt`.
- Run record fingerprinted the **previous** sample's input files (first sample:
  none). Measurements were never affected.
- GUI: the Cellpose *cellprob* setting appeared twice; the first box did nothing.
- GUI refused to start a run when the legacy Fiji macro was missing, although
  the default numpy backend does not use it.
- Curator could open the bounding-box table or an older mask instead of the
  current one when a sample folder held several `_cp_masks` files.

## Pre-git history (reconstructed from the version record)

### Analysis 3.1
- Core CV restored as a co-criterion: `MeanWhole ≥ T AND CoreCV ≤ 0.65`.
  536 blind labels: CV AUC 0.926 (inverted); 88 % agreement vs 80 % for
  brightness alone.
- Nuclear/cytoplasmic ratio, puncta and radial-profile features measured and
  stored (not used by the rule). `cargo_compare.py` added to test them.
- Plate-layout naming (`cargo_labels.py`), provenance records
  (`cargo_version.py`), Figures tab with replicate-aware statistics and
  interactive bars.

### Analysis 3.0
- Per-image Otsu brightness threshold; threshold written into every summary.
- Optional saturation veto; dynamic-range warning.

### Analysis 2.1
- Classification by whole-cell brightness. 195 hand labels: MeanWhole AUC 0.917,
  fill ratio 0.451, CV 0.453.

### Analysis 2.0
- Measurement moved from the ImageJ macro to numpy (`cargo_measure.py`): same
  maths, ~80× faster, no silent interpreter crashes.
- Masks as 16-bit TIFF; generated files excluded from the channel search; GUI,
  launcher with GPU-environment scoring, AMD ROCm/DirectML detection.

### Analysis 1.0
- ImageJ macro: `CoreFillRatio ≥ 0.65 AND CoreCV ≤ 0.60`. ~25 % agreement with
  blind labels — replaced.
