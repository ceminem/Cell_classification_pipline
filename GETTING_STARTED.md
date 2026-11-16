# Getting started — no command line needed

Everything below is clicking. Your existing conda environment (`rocm312`, with
Cellpose and the GPU) is used as it is; you never activate it by hand.

---

## Part 1 — Put the program on your PC (once, ~10 minutes)

1. **Install GitHub Desktop**: <https://desktop.github.com> → install → sign in
   with your GitHub account (`ceminem`).
2. **Clone the repository**: *File → Clone repository… → URL* tab
   - URL: `https://github.com/ceminem/Cell_classification_pipline`
   - Local path: e.g. `C:\Users\cemin\Documents\Molbio\Cell_classification_pipline`
   - *Clone*.
3. **Open the folder**: *Repository → Show in Explorer*.
4. **Desktop shortcut** (optional): right-click `Cargo Analysis.bat` →
   *Send to → Desktop (create shortcut)*.
5. **Keep your old settings** (optional): copy `cargo_gui_settings.json` from
   your old script folder into this new folder. It remembers your data folders.
6. **Mark the starting version** (once): in GitHub Desktop open *History*,
   right-click the **oldest** commit (*Import cargo-release pipeline…*) →
   *Create Tag…* → `v3.1` → *Create* → then *Push origin* (top bar).

From now on, **only use this folder**. Delete or rename the old script folder
so an older copy of a file can never be started by accident.

---

## Part 1b — A second computer (laptop, e.g. the ThinkPad)

1. Install GitHub Desktop and clone the repository exactly as in Part 1
   (steps 1–4).
2. Double-click **`Cargo Analysis.bat`**.
   - **conda is found anywhere**: Anaconda, Miniconda, Miniforge or
     Mambaforge, for your user or for all users, on C: or D:, via the
     PATH, the registry or the usual folders.
   - **No environment with Cellpose yet?** It asks *Set it up now [Y,N]?* →
     press **Y**. `Setup on this PC.bat` then installs everything once
     (about 1–3 GB, 5–20 minutes, internet needed):
     - with conda: a new environment **`cargo`**;
     - without conda: a **`.venv`** folder inside the program folder, using
       Python 3.10–3.12; if no Python is installed it offers to install
       Python 3.12 for your user with *winget* (no admin rights).
   - Then the program starts. Later starts are quick.
3. Laptop graphics (Intel / AMD integrated) → the analysis runs on the
   **CPU**: it works and gives the same results, only slower (Cellpose
   several minutes per image instead of seconds). An NVIDIA card is used
   automatically (the setup installs the CUDA build of PyTorch for it).
4. The first Cellpose run downloads its model (~1 GB) once.
5. Settings that belong to one computer (data folders, Fiji path) are set in
   the program on each computer; they are not shared through GitHub.

**conda installed in an unusual place** and still not found: write the conda
folder (the one that contains `condabin`, e.g. `D:\Programs\miniconda3`) as
the only line of a text file **`conda_path.txt`** next to the `.bat`. To find
it: Start menu → *Anaconda Prompt* / *Miniforge Prompt* → type `where conda`.

---

## Part 2 — Start the program

Double-click **`Cargo Analysis.bat`** (or its desktop shortcut).

It finds conda, tests every environment (each in its own separate process,
which takes a little while the first time), and starts the program with the
Python of the environment that has a working GPU. It remembers that choice in
`launcher_environment.txt`, so later starts are quick. The black window shows
which:

```
 Environment : rocm312
 Compute     : AMD Radeon RX 7800 XT
```

Everything started from the program — Cellpose, the measurement, the curation
tool, CNN training and CNN classification — then runs in that environment, on
the GPU. Nothing extra to install: the CNN uses the PyTorch that Cellpose
already brought.

First time: open the **Setup check** tab → *Run setup check* → all green.

---

## Part 3 — Normal analysis (as before)

**Run** tab → *Browse…* the parent folder (the one holding A1, A2, …) → tick
samples → *Run analysis*.

---

## Part 3b — Counting cells in brightfield images (Cell count tab)

For images that have only brightfield (no fluorescence):

1. **Cell count** tab → *Add folder…* (or *Add images…*).
2. *Pixels per µm*: `2.6428` for the 20x objective (change for others).
3. *Speed*: on a laptop without GPU choose **Fastest: 1/4 size** (minutes
   instead of most of an hour per image); check the outlines once.
4. *Cells at the image edge*: *Count every cell* for the number in the
   field; *Counting frame* for cells per mm² (unbiased).
5. *Count cells*. The preview on the right shows each figure; check that the
   outlines fit before using the numbers.
6. *Open results folder* → `cell_count`:
   **`cell_count.xlsx`** (Summary, Samples with bar chart, one sheet per image
   with every cell: number, position, size), `figures/` (PNG and PDF for the
   paper; the PDF outlines are vector lines; with several images also
   `cell_count_bar_chart`), `masks/`, and the same tables as CSV.
   For a batch, put each sample's images in its own folder (A1, A2, … or a
   condition name) under one parent folder and add the parent folder: one bar
   per sample folder.

Changing only the figure (colour, line width, labels) is quick: the masks are
reused. Tick *segment again* after changing Cellpose settings.

---

## Part 4 — The CNN, step by step (CNN tab)

The line at the top must say **Compute device: GPU – AMD Radeon RX 7800 XT
(AMD ROCm)**. If it says CPU, the program was not started with the `.bat`.

### Step 1 — Label cells

*Open curation tool…* → choose a **sample** folder (e.g. `…\8_cond_pre_glucose\A1`)
that has already been analysed. In the curation window:

| key | does |
|---|---|
| **B** | blind mode on/off — **always label in blind mode** |
| **R** | cell under the mouse = released |
| **N** | = not released |
| **X** | = took nothing up (use this, not N, for empty/dark cells - N is for cells that took particles up but did not release) |
| **S** | = bad segmentation (two cells merged, or one cut in two) |
| **V** | view: green fluorescence → green on brightfield → grey (the *Brightness* slider in the top bar helps with dim cells; display only, the measurement is unchanged) |
| mouse wheel / middle-drag | zoom / move |

Then *Save labels*. Only labels made in blind mode are used.

How many: at least **3 fields from different images**, about **100+ cells
each**, both kinds well represented, **including the hard cells**. The
*Borderline review* button walks you through exactly those. Your existing ~536
blind labels on A1, A2, B1, B2 already count.

Do **not** re-run Cellpose (*Re-run Cellpose* / `--force`) on a field after
labelling it: new masks get new cell numbers and the labels would point at the
wrong cells. Training checks this and refuses such fields.

### Step 2 — Train

*Add folder…* → the parent folder, e.g. `…\8_cond_pre_glucose` (all subfolders
are searched for your labels; add several folders for several experiments).
If you used *Save results to* on the Run tab, also set *Results folder*.
Choose *normal* → **Train model**.

Takes a few minutes. At the end the log shows a table and a **VERDICT**:

```
  field                         n  CNN acc  rule acc
  8_cond_pre_glucose/A1       134    89.6%     87.3%
  ...
  same cells : CNN 89.1%  vs  rule 86.0%
  VERDICT: ...
```

There are two tests. **Test A** (same conditions, unseen cells) answers "does it
work on new images of what I labelled?". **Test B** (unseen condition) answers the
harder "does it work on a condition it has never seen?". Each compares the CNN
with a brightness threshold learned from the same cells (`bright.`) and, if a
Results folder is set, with your own rule. During training you'll see `stop-set bal.` per epoch: a training field set
aside to decide when to stop (a `*` marks a new best). The `[test field …
- shown only]` number is for your information and never used for any decision. The verdict tells you whether the CNN is better than the rule, equal
(keep the rule), or worse (don't use it). *Open training report* shows it
again later.

### Step 3 — Classify

On the **Run** tab: parent folder + tick samples (they must have been analysed
once — the CNN reuses the cell masks). Back on the CNN tab: pick the model →
**Classify ticked samples**.

Per sample, next to the normal results:
`…_cnn.csv` (every cell: probability, CNN call, rule call, agree yes/no),
`…_cnn_summary.csv`, `…_cnn_overlay.png`; at parent level
`…_cnn_batch_summary.csv` → load it in the **Figures** tab like any summary.

---

## Part 5 — Getting updates

When something new was pushed (e.g. after a session with Claude):

- **Setup check** tab → **Get latest version from GitHub** → restart the
  program. *or*
- GitHub Desktop → *Fetch origin* → *Pull origin*.

Your data, results, models and settings are never touched by an update.

If it says *"you edited some of these files yourself"*: GitHub Desktop shows
those files under *Changes* — either *Commit* them (keep your edit) or
right-click → *Discard changes*, then update again.

---

## Part 6 — When you change something yourself

GitHub Desktop lists every changed file with the exact lines. Write a short
summary of **why** (e.g. *"CV cap 0.70 — 89 % on blind labels"*) → *Commit to
main* → *Push origin*. That's your version history; every analysis run records
which commit it used (`run_manifest_*.txt`). **Commit before analysing data for
a figure**, so the result points to saved code.

---

## Troubleshooting

| you see | do |
|---|---|
| CNN tab says *CPU only* | start with `Cargo Analysis.bat`, not by double-clicking a `.py` |
| `.bat` picks the wrong environment | create `launcher_environment.txt` next to the `.bat` containing just `rocm312`; delete the file to let it search again. Don't edit the `.bat` itself (that blocks updates) |
| `.bat`: *No environment has everything…* | run `Setup on this PC.bat`, or `conda activate rocm312` then `pip install -r requirements.txt` |
| `.bat`: *conda.bat not found* (old launcher) | update (GitHub Desktop → *Pull origin*); the new launcher searches everywhere and works without conda |
| *installed but does not load: … DLL load failed …* | start `Cargo Analysis.bat` → **Y** → setup offers the Microsoft Visual C++ runtime → **Y** (asks for admin once). Or install <https://aka.ms/vs/17/release/vc_redist.x64.exe> by hand |
| *No Fiji launcher in …* (older version) | harmless: the analysis does not use Fiji. Current version accepts Fiji or ImageJ 1.x (`ImageJ.exe`), or no folder at all |
| conda installed but not found | put the conda folder into `conda_path.txt` (see Part 1b) |
| everything runs on *CPU only* on the laptop | expected without an NVIDIA/AMD-ROCm GPU; results are identical, only slower |
| training: *could not find the fluorescence image / mask* | that field was not analysed yet, or its files were moved — run the analysis on it |
| training: *…mask was probably re-segmented… SKIPPED* | the labels belong to an older mask; label that field again |
| training: *Only N labelled cells - too few* | label more cells (blind mode!) |
| update: *git is not installed* | use GitHub Desktop *Fetch origin → Pull origin* |
| training log: *GPU self-test failed … switching MIOpen off* | nothing to do - a known AMD-on-Windows problem; training continues on the GPU with PyTorch's own kernels |
| training: *Only N labelled cells were found in 1 field* | you added a single sample folder - add the folder that contains all labelled samples (e.g. `…\20261114`) |
| training: *skipped … no fluorescence image identified* | the log lists the .tif files in that folder; the original fluorescence image must sit next to the mask (name containing `fluor`) |
| training: *NOTE: only … are 'not released'* | label more cells of the rare kind (blind); the verdict uses balanced accuracy so it is not fooled meanwhile |

Full explanation of every step and formula: **DOCUMENTATION.md**.
