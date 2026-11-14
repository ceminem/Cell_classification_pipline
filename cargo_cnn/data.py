# -*- coding: utf-8 -*-
"""
cargo_cnn.data - turning images, masks and blind labels into training examples
=============================================================================

ONE EXAMPLE = ONE CELL
  For every labelled cell a square crop is cut out around it, a little larger
  than the cell so the network sees the immediate surroundings, and resized to
  64 x 64 pixels. The crop is turned into FOUR input channels:

    ch0  intensity     fluorescence minus the LOCAL background, / 50.
                       Keeps absolute brightness, which is the single most
                       predictive thing measured so far (AUC 0.92).
    ch1  texture       the same pixels divided by this cell's own mean.
                       Brightness removed, only the PATTERN left - even fill vs
                       puncta vs bright rim. This is what core CV approximates.
    ch2  this cell     1 inside the cell being judged, 0 elsewhere.
    ch3  other cells   1 inside neighbouring cells, 0 elsewhere.

  ch2/ch3 tell the network which pixels it is being asked about. Without them a
  bright neighbour in the crop is indistinguishable from the cell itself, and
  that is exactly the "background interferes" failure.

HOW BACKGROUND IS KEPT FROM INTERFERING - three layers
  1. Global: the same rolling-ball subtraction the pipeline uses
     (cargo_measure.subtract_background), so the CNN sees the numbers the rule
     sees.
  2. Local: inside each crop, the median of pixels belonging to NO cell is
     subtracted. This removes residual background that varies across the
     field (uneven illumination, out-of-focus haze, dye in the medium).
  3. Learned invariance: during training every crop is randomly given an extra
     background offset, a tilted background gradient, an exposure change and
     noise - none of which change its label. The network can only fit the
     labels by ignoring those things. This is the part a fixed threshold
     cannot do.

WHAT NEVER CHANGES A LABEL, AND WHAT DOES
  Rotations and mirror images of a cell are the same cell, so they are free
  extra examples. Exposure changes are NOT entirely label-free - a dim cell is
  partly dim BECAUSE it did not release - so the exposure range is kept modest
  (default 0.8-1.25x) and is a training option.
=============================================================================
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

# make the pipeline modules importable when run as "python -m cargo_cnn.x"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CROP = 64               # network input is CROP x CROP pixels
CONTEXT = 1.4           # crop side = CONTEXT * the cell's longer bbox side
INTENSITY_SCALE = 50.0  # ch0 = grey levels / this (threshold 25.7 -> ~0.5)
MIN_BG_FRACTION = 0.05  # need >=5% cell-free pixels to estimate local bg
N_CHANNELS = 4


# ───────────────────────────────────────────────────────────── loading ──

def load_fluorescence(path):
    """Read the fluorescence tif exactly like cargo_measure.run_sample()."""
    import tifffile
    raw = tifffile.imread(path)
    if raw.ndim == 3 and raw.shape[-1] in (3, 4):
        fl = raw[..., 1].astype(np.float32)          # calcein = green
    elif raw.ndim == 3:
        fl = raw.max(axis=0).astype(np.float32)
    else:
        fl = raw.astype(np.float32)
    return raw, fl


def load_labels(mask_path, shape):
    """Read a Cellpose label mask and upscale it to `shape` (nearest neighbour,
    which keeps label ids exact) - same arithmetic as the pipeline."""
    import tifffile
    if mask_path.lower().endswith((".tif", ".tiff")):
        lab = tifffile.imread(mask_path)
    else:
        from PIL import Image
        lab = np.array(Image.open(mask_path))
    if lab.ndim > 2:
        lab = lab[..., 0]
    if lab.shape != tuple(shape):
        fy = shape[0] / lab.shape[0]
        fx = shape[1] / lab.shape[1]
        yi = np.clip((np.arange(shape[0]) / fy).astype(np.int64), 0, lab.shape[0] - 1)
        xi = np.clip((np.arange(shape[1]) / fx).astype(np.int64), 0, lab.shape[1] - 1)
        lab = lab[yi[:, None], xi[None, :]]
    return lab.astype(np.int32)


def prepare_image(fluor_path, mask_path, rolling_ball_radius=50):
    """
    Load one field and return (fluor_raw, fluor_bgsub, labels, cell_table).

    cell_table maps label id -> dict(slice=(ys, xs), area, mean) where `mean`
    is the whole-cell mean on the background-subtracted image - identical to
    MeanWhole in *_results.csv, so the min_cell_mean exclusion is the same.
    """
    import cargo_measure
    from scipy import ndimage

    raw, fl = load_fluorescence(fluor_path)
    labels = load_labels(mask_path, fl.shape)
    fl, _ = cargo_measure.subtract_background(fl, rolling_ball_radius)

    objs = ndimage.find_objects(labels)
    ids = [i for i, s in enumerate(objs, start=1) if s is not None]
    if not ids:
        return raw, fl, labels, {}
    areas = ndimage.sum(np.ones_like(labels, dtype=np.float32), labels, ids)
    means = ndimage.mean(fl, labels, ids)
    table = {}
    for lb, a, m in zip(ids, areas, means):
        table[lb] = {"slice": objs[lb - 1], "area": int(a), "mean": float(m)}
    return raw, fl, labels, table


# ──────────────────────────────────────────────────────────── cropping ──

def _resize(arr, size, nearest=False):
    from PIL import Image
    if nearest:
        im = Image.fromarray(arr.astype(np.uint8) * 255)
        return np.array(im.resize((size, size), Image.NEAREST)) > 127
    im = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.array(im.resize((size, size), Image.BILINEAR), dtype=np.float32)


def extract_crop(fl, labels, lb, sl, size=CROP, context=CONTEXT):
    """
    Square crop around cell `lb`, resized to size x size.

    Returns (img, own, other):
      img   float32, background-subtracted intensities, NOT yet locally
            corrected (so augmentation can add background before correction)
      own   bool, this cell
      other bool, any other cell
    Pixels beyond the image border are filled with the local background level,
    so a cell at the edge does not get an artificial dark frame.
    """
    ys, xs = sl
    h, w = ys.stop - ys.start, xs.stop - xs.start
    side = int(np.ceil(context * max(h, w))) + 4
    cy = (ys.start + ys.stop) / 2.0
    cx = (xs.start + xs.stop) / 2.0
    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))

    H, W = fl.shape
    img = np.full((side, side), np.nan, np.float32)
    lab = np.zeros((side, side), np.int32)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(H, y0 + side), min(W, x0 + side)
    if sy1 > sy0 and sx1 > sx0:
        img[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = fl[sy0:sy1, sx0:sx1]
        lab[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = labels[sy0:sy1, sx0:sx1]

    own = lab == lb
    other = (lab > 0) & ~own
    inside = np.isfinite(img)
    bgpix = inside & (lab == 0)
    fill = float(np.median(img[bgpix])) if bgpix.sum() >= 10 else 0.0
    img[~inside] = fill

    return (_resize(img, size), _resize(own, size, nearest=True),
            _resize(other, size, nearest=True))


def make_channels(img, own, other):
    """
    The four network channels from one crop (see module docstring).

    Local background = median of cell-free pixels. In a confluent field there
    may be almost none; then the rolling-ball subtraction already applied is
    trusted and nothing more is removed.
    """
    bg_mask = ~own & ~other
    if bg_mask.mean() >= MIN_BG_FRACTION:
        bg = float(np.median(img[bg_mask]))
    else:
        bg = 0.0
    inten = img - bg
    cell_mean = float(inten[own].mean()) if own.any() else 1.0
    texture = np.clip(inten / max(cell_mean, 1.0), -1.0, 4.0) / 2.0
    return np.stack([inten / INTENSITY_SCALE, texture,
                     own.astype(np.float32), other.astype(np.float32)])


# ─────────────────────────────────────────────────────── augmentation ──

def augment(img, own, other, rng, exposure=(0.8, 1.25)):
    """
    Random changes that must NOT change the label. See module docstring.
    Operates on the raw crop, before local background correction.
    """
    k = int(rng.integers(4))
    img, own, other = (np.rot90(a, k) for a in (img, own, other))
    if rng.random() < 0.5:
        img, own, other = img[:, ::-1], own[:, ::-1], other[:, ::-1]
    img = np.ascontiguousarray(img, dtype=np.float32)
    own = np.ascontiguousarray(own)
    other = np.ascontiguousarray(other)

    img = img * rng.uniform(*exposure)                    # exposure
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) - 0.5
    ang = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(0.0, 6.0)                           # grey levels across crop
    img = img + rng.uniform(-3.0, 8.0) \
              + amp * (np.cos(ang) * xx + np.sin(ang) * yy)  # offset + gradient
    img = img + rng.normal(0.0, rng.uniform(0.0, 1.5), img.shape)  # noise
    return img.astype(np.float32), own, other


DIHEDRAL = [(k, f) for f in (False, True) for k in range(4)]


def dihedral(arr, k, flip):
    """One of the 8 rotations/mirrors, applied to a (C, H, W) array."""
    a = np.rot90(arr, k, axes=(-2, -1))
    if flip:
        a = a[..., ::-1]
    return np.ascontiguousarray(a)


# ─────────────────────────────────────────────── finding labelled data ──

def read_curated(path, include_nonblind=False):
    """{CellID: 1/0} from a curator export. Only blind labels by default: labels
    made with the automatic colours visible lean towards the software's answer
    and would teach the network to copy the rule."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if not include_nonblind and r.get("Source") != "manual_blind":
                continue
            lab = r.get("Label")
            if lab not in ("released", "not_released"):
                continue
            try:
                cid = int(float(r["CellID"]))
            except (KeyError, TypeError, ValueError):
                continue
            out[cid] = 1 if lab == "released" else 0      # later rows win
    return out


def read_results(path):
    """{CellID: row} from a *_results.csv, numeric fields as float."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            try:
                cid = int(float(r["CellID"]))
            except (KeyError, TypeError, ValueError):
                continue
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            out[cid] = row
    return out


def _find_mask(sample_dir, hint=""):
    cands = [p for p in glob.glob(os.path.join(sample_dir, "*_cp_masks.*"))
             if p.lower().endswith((".tif", ".tiff", ".png"))]
    if len(cands) == 1:
        return cands[0]
    # several masks (re-runs under different names): prefer one sharing the
    # curated file's well token
    for p in cands:
        if hint and hint in os.path.basename(p):
            return p
    tifs = [p for p in cands if p.lower().endswith((".tif", ".tiff"))]
    return tifs[0] if len(tifs) == 1 else None


def discover(roots, log=print):
    """
    Find (curated, fluorescence, mask, results) sets under the given folders.

    Expected layout - the pipeline default:
        <sample>/<acquisition tifs>  +  <sample>/<base>_cp_masks.tif
        <sample>/edited_results/<...>_results.csv  and  <...>_curated.csv

    If results were written to a separate --output folder, the raw images are
    not next to the curated file and discovery cannot pair them. Use a pairs
    file instead (see load_pairs_file).
    """
    import run_cargo_pipeline as rcp

    sets = []
    for root in roots:
        for dirpath, _d, files in os.walk(root):
            for fn in files:
                if not fn.endswith("_curated.csv"):
                    continue
                cpath = os.path.join(dirpath, fn)
                key = fn[:-len("_curated.csv")]
                sample_dir = (os.path.dirname(dirpath)
                              if os.path.basename(dirpath) == "edited_results"
                              else dirpath)
                try:
                    bf, fl = rcp.find_channels(sample_dir)
                except OSError:
                    bf, fl = None, None
                well = os.path.basename(sample_dir)
                res = [os.path.join(dirpath, f) for f in files
                       if f.endswith("_results.csv")]
                if len(res) > 1:
                    res = [r for r in res if well in os.path.basename(r)] or res[:1]
                # A sample folder can hold masks from several runs - e.g.
                # "8_cond_pre_glucose_A1_cp_masks.tif" from before the plate
                # layout was used and "A1_plus-His-..._cp_masks.tif" after. The
                # mask written alongside the results file is the right one.
                mask = None
                if res:
                    rb = os.path.basename(res[0])[:-len("_results.csv")]
                    for ext in (".tif", ".tiff", ".png"):
                        p = os.path.join(sample_dir, rb + "_cp_masks" + ext)
                        if os.path.isfile(p):
                            mask = p
                            break
                if mask is None:
                    mask = _find_mask(sample_dir, well)
                if not fl or not mask:
                    log(f"  ! {cpath}: could not find the "
                        f"{'fluorescence image' if not fl else 'mask'} in "
                        f"{sample_dir} - list it in a --pairs file instead")
                    continue
                sets.append({"key": f"{os.path.basename(os.path.dirname(sample_dir))}/{well}",
                             "curated": cpath, "fluorescence": fl, "mask": mask,
                             "results": res[0] if res else None})
    return sets


def load_pairs_file(path):
    """
    Semicolon CSV with a header:  curated;fluorescence;mask;results
    (results may be left empty). One line per labelled field.
    """
    sets = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, r in enumerate(csv.DictReader(fh, delimiter=";")):
            sets.append({"key": r.get("key") or os.path.basename(r["curated"]).replace("_curated.csv", ""),
                         "curated": r["curated"].strip(),
                         "fluorescence": r["fluorescence"].strip(),
                         "mask": r["mask"].strip(),
                         "results": (r.get("results") or "").strip() or None})
    return sets


# ─────────────────────────────────────────────────────── building sets ──

def build_examples(sets, rolling_ball_radius=50, include_nonblind=False, log=print):
    """
    Cut every labelled cell out of its image.

    Returns a list of dicts: group (which field), cell_id, y (0/1), img/own/
    other (the crop), mean_whole, rule (the pipeline's own class for this cell
    if a results file was found, else None).

    SAFETY CHECK: labels refer to Cellpose label ids. If the field was
    re-segmented after labelling (--force), the same id now points at a
    different cell and every label would be attached to the wrong crop. When a
    results file is available the cell areas are compared, and a field whose
    labelled cells no longer match is refused rather than silently used.
    """
    examples = []
    for s in sets:
        labels_by_id = read_curated(s["curated"], include_nonblind)
        if not labels_by_id:
            log(f"  {s['key']}: no usable blind labels, skipped")
            continue
        log(f"  {s['key']}: {len(labels_by_id)} labels  "
            f"({sum(labels_by_id.values())} released)")
        raw, fl, labels, table = prepare_image(
            s["fluorescence"], s["mask"], rolling_ball_radius)
        results = read_results(s.get("results"))

        if results:
            checked = bad = 0
            for cid in labels_by_id:
                if cid in table and cid in results and "Area" in results[cid]:
                    checked += 1
                    a0 = float(results[cid]["Area"])
                    if abs(table[cid]["area"] - a0) > 0.02 * max(a0, 1):
                        bad += 1
            if checked and bad / checked > 0.10:
                log(f"  ! {s['key']}: {bad}/{checked} labelled cells have a "
                    "different area in the current mask than when they were "
                    "measured. The mask was probably re-segmented after "
                    "labelling, so the labels point at the wrong cells. "
                    "SKIPPED.")
                continue

        missing = 0
        for cid, y in sorted(labels_by_id.items()):
            if cid not in table:
                missing += 1
                continue
            img, own, other = extract_crop(fl, labels, cid, table[cid]["slice"])
            if not own.any():
                missing += 1
                continue
            rule = None
            if cid in results:
                c = str(results[cid].get("Class", ""))
                rule = 1 if c.startswith("released") else (0 if c == "not_released" else None)
            examples.append({"group": s["key"], "cell_id": cid, "y": y,
                             "img": img, "own": own, "other": other,
                             "mean_whole": table[cid]["mean"], "rule": rule})
        if missing:
            log(f"    ({missing} labelled ids not found in the mask)")
        del raw, fl, labels
    return examples


class CellDataset:
    """torch-style dataset over build_examples() output."""

    def __init__(self, examples, train=False, seed=0, exposure=(0.8, 1.25)):
        self.ex = examples
        self.train = train
        self.exposure = exposure
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        import torch
        e = self.ex[i]
        img, own, other = e["img"], e["own"], e["other"]
        if self.train:
            img, own, other = augment(img, own, other, self.rng, self.exposure)
        x = make_channels(img, own, other)
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(float(e["y"]))
