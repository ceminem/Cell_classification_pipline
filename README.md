# Cargo Release Analysis

Classifies HeLa cells by **calcein cargo release** from iron-fumarate MOF
nanoparticles, from paired brightfield + fluorescence microscopy images.

**Brightfield → Cellpose segmentation → per-cell measurement → released / not
released → per-well percentages → figures with honest statistics.**

* Rule (analysis 3.1): a cell has released if it is **bright enough** (per-image
  Otsu threshold) **and evenly filled** (core CV ≤ 0.65). 88 % agreement with 536
  blind hand labels.
* Optional **CNN** (`cargo_cnn/`) that learns the decision from blind labels and
  is trained to ignore background, exposure and neighbouring cells.
* **Cell count** for brightfield-only images: Cellpose segmentation, counts
  (incl. an unbiased counting frame for cells per mm²) and paper figures with
  vector cell outlines (`cell_count.py`, *Cell count* tab).
* Every run writes a record of settings, inputs, environment and the **git
  commit**, so each number can be traced to the code that produced it.

## Start

Double-click **`Cargo Analysis.bat`**. It finds the right environment by
itself (conda anywhere, or none). On a new computer it offers to install
everything once (`Setup on this PC.bat`). Then *Setup check* → *Run*.
Details, also for a laptop: `GETTING_STARTED.md`, Part 1b.

Command line:
```
python run_cargo_pipeline.py "D:\data\pre_glucose" --samples all
python -m cargo_cnn.train   --data "D:\data\Slide I"
python -m cargo_cnn.predict --model models\cnn_<date>\model.pt --parent "D:\data\pre_glucose"
python cell_count.py "D:\data\counting" --px-per-um 2.6428 --edge unbiased
```

## Read next

* **[GETTING_STARTED.md](GETTING_STARTED.md)**: click-by-click setup with GitHub
  Desktop, the CNN workflow in the GUI, updates. **Start here.**

* **[DOCUMENTATION.md](DOCUMENTATION.md)**: what is measured and why, every
  formula, the history, the CNN, settings, and how to use git with this project.
* **[CHANGELOG.md](CHANGELOG.md)**: what changed, when.

Raw images, masks, results and model weights are **not** stored here (see
`.gitignore`); they stay on the lab drive.
