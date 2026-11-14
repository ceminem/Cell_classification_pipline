# Changelog

All notable changes, newest first. "Analysis version" changes alter the numbers
a run produces; results from different analysis versions are not comparable.
Details and evidence: `DOCUMENTATION.md`, sections 10 and 12.

## [Unreleased] — 2026-11-14

Analysis version unchanged (3.1): rule results are identical to before.

### Added
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
