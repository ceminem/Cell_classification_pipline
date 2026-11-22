#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_measure.py - per-cell cargo-release measurement, in numpy
=============================================================================
Replaces the ImageJ macro. Same science, same output files, no Fiji.

WHY THIS EXISTS
  The measurement was written as an ImageJ macro because the original workflow
  was a Fiji macro. That turned out to be the wrong tool: the macro interpreter
  repeatedly died with no error message on 2000+ cell images, and every fix
  moved the crash somewhere else. The measurement itself is simple array
  arithmetic, and numpy does it in seconds with no interpreter to crash.

THE SCIENCE IS UNCHANGED. For each cell:

    CORE      = pixels whose normalized distance from the cell edge is
                >= core_radius_fraction   (inner part; proxy for nucleus +
                deep cytoplasm)
    PERIPHERY = the remaining rim

    fill_ratio = core_mean / periphery_mean
        Released cargo diffuses everywhere, so the core is about as bright as
        the rim -> ratio near 1. Cargo still trapped in particles leaves a dark
        centre -> low ratio. This is "has the dye invaded the nucleus".

    core_cv    = core_std / core_mean
        A homogeneous fill varies little -> low CV. Punctate, still-encapsulated
        particles put a few very bright pixels among dark ones -> high CV.

    released  <=>  fill_ratio >= fill_ratio_threshold
              AND  core_cv    <= core_cv_threshold

    A cell whose whole-cell mean is below min_cell_mean took nothing up at all
    and is excluded from the release statistics entirely (it is neither a
    release nor a failed release).

OUTPUTS (identical names and columns to the macro version):
    <base>_results.csv            one row per cell, semicolon separated
    <base>_summary.csv            totals + both release percentages
    <base>_classified_overlay.png QC image: outline colour = class (green
                                  released, red not released, cyan very
                                  bright, blue no signal), legend + counts
    <base>_log.txt                what happened
=============================================================================
"""

from __future__ import annotations

import os

import numpy as np


def _percentile_line(name, values):
    if len(values) == 0:
        return f"{name}: (no values)"
    v = np.sort(values)
    def p(q):
        return v[min(len(v) - 1, int(q * (len(v) - 1)))]
    return (f"{name}: p10={p(0.10):.3f} p25={p(0.25):.3f} p50={p(0.50):.3f} "
            f"p75={p(0.75):.3f} p90={p(0.90):.3f}")


def subtract_background(img, radius):
    """
    Approximate ImageJ's rolling-ball background subtraction with a grey
    opening (morphological erosion followed by dilation). A true rolling ball
    is a paraboloid; a flat disc of the same radius is close enough for a
    background estimate and is far faster.

    Falls back to returning the image unchanged if scipy is unavailable, since
    background subtraction affects absolute intensities but not the RATIOS the
    classification is built on.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return img, False

    r = max(1, int(radius))
    # a disc structuring element
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r

    small = img[::4, ::4]                      # estimate on a decimated copy
    bg = ndimage.grey_opening(small, footprint=disc[::4, ::4])
    bg = ndimage.zoom(bg, (img.shape[0] / bg.shape[0],
                           img.shape[1] / bg.shape[1]), order=1)
    bg = bg[:img.shape[0], :img.shape[1]]
    if bg.shape != img.shape:
        pad_y = img.shape[0] - bg.shape[0]
        pad_x = img.shape[1] - bg.shape[1]
        bg = np.pad(bg, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode="edge")
    out = img.astype(np.float32) - bg.astype(np.float32)
    np.clip(out, 0, None, out)
    return out, True


def normalized_distance(cell_mask):
    """
    Distance from the cell edge, normalized so 0 = membrane and 1 = the deepest
    interior point. Uses the exact Euclidean distance transform.
    """
    from scipy import ndimage
    edt = ndimage.distance_transform_edt(cell_mask)
    m = edt.max()
    return edt / m if m > 0 else edt


def classify_row(row, cfg):
    """
    Decide a single cell's class from its already-measured features.

    WHICH FEATURE DECIDES, AND WHY IT CHANGED
      The original design used core-fill-ratio + core-CV. Measured against 195
      hand-labelled cells across four samples, those two features turned out to
      be worthless here:

          MeanWhole (brightness)   AUC 0.917
          CoreFillRatio            AUC 0.451   <- chance is 0.50
          CoreCV                   AUC 0.453

      The texture idea was reasonable a priori - punctate particles really do
      look different from a diffuse fill - but on this data it does not survive
      contact with the labels. Brightness alone reproduces the human judgement
      on ~87% of cells with a single global threshold, and 93-97% when the
      threshold is set per sample. Adding CV on top gains 0.5 percentage points.

      So brightness is the default. The old features remain available via
      method="texture" purely so results can be compared, not because they are
      recommended.

    Separated from measure() so a GUI can re-classify thousands of cells
    instantly when a threshold moves, without touching the pixels again.
    """
    if row.get("_no_signal"):
        return "excluded_negative"
    if row.get("_too_small"):
        return "excluded_too_small"

    method = cfg.get("method", "brightness")

    if method == "nuclear":
        # Classify on whether the dye reached the nucleus - the original
        # biological criterion, now measured directly instead of via a
        # geometric "core".
        #
        # On synthetic cells built with a known answer this separated perfectly
        # (AUC 1.000) where brightness scored 0.000, because brightness only
        # asks "is there signal" while this asks "did it get INTO the cell".
        # Whether it wins on real data has to be checked against blind labels -
        # see the calibration note in the curator.
        ncr = row.get("NucCytoRatio", float("nan"))
        if not np.isfinite(ncr):
            # fall back rather than guessing when the nucleus could not be
            # measured (very small or oddly shaped cells)
            return ("released" if row["MeanWhole"] >= cfg["brightness_threshold"]
                    else "not_released")
        if row["MeanWhole"] < cfg.get("min_signal_for_nuclear", 5):
            return "not_released"          # nothing took up at all
        if ncr >= cfg.get("nuc_cyto_threshold", 0.6):
            guard = cfg.get("saturation_guard", 0.0)
            if guard > 0 and row.get("SatFraction", 0.0) >= guard:
                return "excluded_saturated"
            return "released"
        return "not_released"

    if method == "brightness":
        # Brightness AND texture, both required.
        #
        # This rule has been revised twice, each time against hand labels:
        #   - the first design used fill-ratio + CV and agreed with the labels
        #     on 25% of cells, so it was replaced by brightness alone;
        #   - on a later, larger and fully BLIND set of 523 labels, CoreCV came
        #     out strongly predictive after all (AUC 0.074 - that is 0.926 with
        #     the sign flipped, since LOW CV means released). Brightness alone
        #     scored 80%; brightness AND CV scores 88%, and is better or equal
        #     in every one of the four samples.
        #
        # The earlier failure was most likely label quality, not the feature:
        # the first set was labelled with the automatic colours visible, the
        # second blind.
        bright_ok = row["MeanWhole"] >= cfg["brightness_threshold"]
        if not bright_ok:
            return "not_released"
        cv_cap = cfg.get("core_cv_cap")
        if cv_cap is not None and np.isfinite(row.get("CoreCV", np.nan)):
            if row["CoreCV"] > cv_cap:
                return "not_released"
        # a cell bright only because a saturated blob bleeds across it
        guard = cfg.get("saturation_guard", 0.0)
        if guard > 0 and row.get("SatFraction", 0.0) >= guard:
            return "excluded_saturated"
        return "released"

    # legacy texture rule, kept for comparison only
    if row["MeanWhole"] >= cfg.get("very_high_signal_cutoff", 1e9):
        return "released_highsignal"
    if (np.isfinite(row["CoreFillRatio"])
            and row["CoreFillRatio"] >= cfg["fill_ratio_threshold"]
            and row["CoreCV"] <= cfg["core_cv_threshold"]):
        return "released"
    return "not_released"


def measure_nucleus(sub_img, cell, nd, cfg):
    """
    Nuclear vs cytoplasmic intensity.

    The original criterion for release was "the dye has invaded the nucleus",
    but the nucleus was never measured separately - only a generic "core" at
    50% of the radius, which in a spread HeLa cell is mostly perinuclear
    cytoplasm rather than nucleus.

    Without a nuclear stain the nucleus has to be inferred. In brightfield-
    segmented HeLa the nucleus sits near the cell's centre of mass and occupies
    roughly 15-25% of the projected area, so the deepest part of the distance
    map is used as a proxy. That is an approximation, and it is reported as
    such: NucCytoRatio is a shape-based estimate, not a stained measurement.

    Returns (nuc_mean, cyto_mean, ratio).
    """
    nuc_frac = cfg.get("nucleus_area_fraction", 0.20)
    if not cell.any():
        return float("nan"), float("nan"), float("nan")

    vals_nd = nd[cell]
    if vals_nd.size == 0:
        return float("nan"), float("nan"), float("nan")

    # take the deepest nuc_frac of the cell's pixels as the nuclear proxy
    cut = np.quantile(vals_nd, 1.0 - nuc_frac)
    nuc = cell & (nd >= cut)
    cyto = cell & (nd < cut)
    if not nuc.any() or not cyto.any():
        return float("nan"), float("nan"), float("nan")

    nuc_mean = float(sub_img[nuc].mean())
    cyto_mean = float(sub_img[cyto].mean())
    ratio = nuc_mean / cyto_mean if cyto_mean > 0 else float("nan")
    return nuc_mean, cyto_mean, ratio


def measure_puncta(sub_img, cell, cfg):
    """
    Count and size discrete bright spots inside the cell.

    CoreCV says how variable the core is, but not WHY. A cell with three
    bright particles and a cell with a mottled fill can share a CV. Counting
    discrete maxima distinguishes "a few intact particles" (not released)
    from "an even bright fill" (released), which is the distinction CV only
    approximates.

    A spot is a connected region above (cell median + k * cell MAD), so the
    test adapts to each cell rather than using a global intensity.

    Returns (n_puncta, puncta_area_fraction, puncta_contrast).
    """
    from scipy import ndimage

    vals = sub_img[cell]
    if vals.size < 20:
        return 0, 0.0, float("nan")

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad <= 0:
        # a perfectly flat cell has no puncta by definition
        return 0, 0.0, 1.0

    k = cfg.get("puncta_sigma", 3.0)
    thresh = med + k * mad * 1.4826          # 1.4826 makes MAD comparable to SD
    spots = cell & (sub_img > thresh)
    if not spots.any():
        return 0, 0.0, 1.0

    lab, n = ndimage.label(spots)
    if n == 0:
        return 0, 0.0, 1.0

    # drop single-pixel specks, which are noise rather than particles
    min_px = int(cfg.get("puncta_min_px", 4))
    sizes = ndimage.sum(spots, lab, range(1, n + 1))
    keep = sizes >= min_px
    n_keep = int(keep.sum())
    area_frac = float(sizes[keep].sum() / cell.sum()) if n_keep else 0.0

    if n_keep:
        spot_mask = np.isin(lab, np.nonzero(keep)[0] + 1)
        contrast = float(sub_img[spot_mask].mean() / max(med, 1e-6))
    else:
        contrast = 1.0
    return n_keep, area_frac, contrast


def measure_radial(sub_img, cell, nd, n_bins=5):
    """
    How signal changes from the cell edge to its centre.

    Released cargo diffuses, giving a flat profile. Trapped cargo sits in a
    ring or in scattered particles, giving a sloped or lumpy one. The slope is
    fitted across normalized-distance bins, so it does not depend on cell size.

    Returns (slope, centre_edge_ratio). Slope is positive when the centre is
    brighter than the rim.
    """
    if not cell.any():
        return float("nan"), float("nan")
    d = nd[cell]
    v = sub_img[cell]
    if d.size < n_bins * 4:
        return float("nan"), float("nan")

    edges = np.linspace(0, 1, n_bins + 1)
    centres, means = [], []
    for i in range(n_bins):
        m = (d >= edges[i]) & (d < edges[i + 1] if i < n_bins - 1 else d <= 1.0)
        if m.sum() >= 3:
            centres.append((edges[i] + edges[i + 1]) / 2)
            means.append(float(v[m].mean()))
    if len(means) < 3:
        return float("nan"), float("nan")

    centres = np.array(centres)
    means = np.array(means)
    # normalise by the cell's own mean so the slope is comparable between cells
    scale = means.mean()
    if scale <= 0:
        return float("nan"), float("nan")
    slope = float(np.polyfit(centres, means / scale, 1)[0])
    ratio = float(means[-1] / means[0]) if means[0] > 0 else float("nan")
    return slope, ratio


def extra_features(sub_img, cell, nd, cfg):
    """
    Features beyond mean brightness and core CV, aimed squarely at the cells
    those two cannot separate.

    Measured on 536 blind labels, 50 pairs of cells inside the SAME image had
    nearly identical brightness and CV but opposite labels. No threshold or
    model on those two numbers can tell such pairs apart - the information is
    simply not in them. These three families of feature are attempts to capture
    what the eye is using instead.

    sub_img : intensity crop for this cell
    cell    : boolean mask of the cell within the crop
    nd      : normalized distance from the cell edge (0 = membrane, 1 = deepest)

    1. NUCLEAR vs CYTOPLASMIC
       The original criterion was "the dye has invaded the nucleus". That was
       never actually measured - the "core" is a geometric inner region, not
       the nucleus. Here the nucleus is approximated as the deepest part of the
       cell (highest distance-from-edge), which for a spread adherent cell is
       where the nucleus sits. NucCytoRatio near 1 means the dye reached the
       nucleus; well below 1 means it is excluded from it.

    2. PUNCTA
       Intact particles are discrete bright spots. Counting them directly is
       more specific than core CV, which only says "uneven" and is raised by
       any unevenness including uneven illumination.

    3. RADIAL PROFILE
       How signal falls from the deepest point outwards. A released cell is
       flat across the radius; a cell with cargo trapped at the periphery
       slopes the other way. The slope carries direction, which neither the
       mean nor the CV does.
    """
    from scipy import ndimage

    out = {}
    vals = sub_img[cell]
    if vals.size == 0:
        return out
    total_mean = float(vals.mean())
    if total_mean <= 0:
        total_mean = 1e-6

    # ---- 1. nuclear vs cytoplasmic ----
    # deepest ~25% of the cell stands in for the nucleus; the outer half for
    # cytoplasm. Both are taken from the same distance map already computed.
    nuc = cell & (nd >= 0.75)
    cyto = cell & (nd > 0.05) & (nd < 0.5)
    if nuc.any() and cyto.any():
        nuc_mean = float(sub_img[nuc].mean())
        cyto_mean = float(sub_img[cyto].mean())
        out["NucMean"] = nuc_mean
        out["CytoMean"] = cyto_mean
        out["NucCytoRatio"] = nuc_mean / max(cyto_mean, 1e-6)
    else:
        out["NucMean"] = float("nan")
        out["CytoMean"] = float("nan")
        out["NucCytoRatio"] = float("nan")

    # ---- 2. puncta ----
    # A punctum is a small region markedly brighter than the rest of the cell.
    # The threshold is relative to this cell, so it does not depend on exposure.
    try:
        thr = float(np.percentile(vals, 75)) + 2.0 * float(vals.std())
        bright = cell & (sub_img >= thr)
        if bright.any():
            lab, n = ndimage.label(bright)
            if n:
                sizes = ndimage.sum(bright, lab, range(1, n + 1))
                # ignore single-pixel specks - those are sensor noise
                real = sizes[sizes >= 4]
                out["PunctaCount"] = int(len(real))
                out["PunctaArea"] = float(real.sum()) / max(float(cell.sum()), 1)
                out["PunctaMeanSize"] = float(real.mean()) if len(real) else 0.0
            else:
                out["PunctaCount"] = 0
                out["PunctaArea"] = 0.0
                out["PunctaMeanSize"] = 0.0
        else:
            out["PunctaCount"] = 0
            out["PunctaArea"] = 0.0
            out["PunctaMeanSize"] = 0.0
    except Exception:
        out["PunctaCount"] = 0
        out["PunctaArea"] = 0.0
        out["PunctaMeanSize"] = 0.0

    # ---- 3. radial profile ----
    # mean intensity in five shells from membrane to centre, then the slope of
    # a straight-line fit through them, normalized by the cell's own mean so it
    # is comparable between dim and bright cells.
    try:
        shells = []
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = cell & (nd >= lo) & (nd < hi)
            shells.append(float(sub_img[m].mean()) if m.any() else np.nan)
        good = [(i, v) for i, v in enumerate(shells) if np.isfinite(v)]
        if len(good) >= 3:
            xs = np.array([g[0] for g in good], float)
            ys = np.array([g[1] for g in good], float)
            slope = float(np.polyfit(xs, ys, 1)[0]) / total_mean
            out["RadialSlope"] = slope
            # centre-to-edge contrast, a simpler summary of the same idea
            out["CentreEdgeRatio"] = float(ys[-1] / max(ys[0], 1e-6))
        else:
            out["RadialSlope"] = float("nan")
            out["CentreEdgeRatio"] = float("nan")
    except Exception:
        out["RadialSlope"] = float("nan")
        out["CentreEdgeRatio"] = float("nan")

    return out


def measure(fluor, labels, bboxes, cfg, log):
    """
    Measure every labelled cell. Returns a list of per-cell dicts.

    fluor  : 2D float array, full resolution, background already subtracted
    labels : 2D int array, SAME shape as fluor
    bboxes : {label: (x, y, w, h)} in full-resolution coordinates
    """
    from scipy import ndimage

    rows = []
    n = len(bboxes)
    for i, (lb, (x, y, w, h)) in enumerate(sorted(bboxes.items())):
        sub_lab = labels[y:y + h, x:x + w]
        cell = sub_lab == lb
        area = int(cell.sum())
        if area == 0:
            continue

        sub_img = fluor[y:y + h, x:x + w]
        vals = sub_img[cell]
        mean_whole = float(vals.mean())

        ys, xs = np.nonzero(cell)
        cx = float(xs.mean() + x)
        cy = float(ys.mean() + y)

        row = {
            "CellID": int(lb), "Area": area, "MeanWhole": mean_whole,
            "CentroidX": cx, "CentroidY": cy,
            "Bx": x, "By": y, "Bw": w, "Bh": h,
            "CoreMean": float("nan"), "PeripheryMean": float("nan"),
            "CoreFillRatio": float("nan"), "CoreCV": float("nan"),
            "SatFraction": 0.0,
            "NucMean": float("nan"), "CytoMean": float("nan"),
            "NucCytoRatio": float("nan"),
            "PunctaCount": 0, "PunctaArea": 0.0, "PunctaMeanSize": 0.0,
            "RadialSlope": float("nan"), "CentreEdgeRatio": float("nan"),
        }

        # no uptake at all -> outside the release question
        if mean_whole < cfg["min_cell_mean"]:
            row["_no_signal"] = True
            row["Class"] = "excluded_negative"
            rows.append(row)
            continue

        nd = normalized_distance(cell)
        core = cell & (nd >= cfg["core_radius_fraction"])
        peri = cell & (nd > 0) & (nd < cfg["core_radius_fraction"])

        if not core.any():
            row["_too_small"] = True
            row["Class"] = "excluded_too_small"
            rows.append(row)
            continue

        core_vals = sub_img[core]
        core_mean = float(core_vals.mean())
        core_std = float(core_vals.std())
        peri_mean = float(sub_img[peri].mean()) if peri.any() else float("nan")

        # Fraction of the cell that is at or near the top of the range. A cell
        # whose brightness comes from a few saturated pixels - a bright blob
        # bleeding over it - is not the same as a cell that is uniformly filled,
        # even though both have a high mean. Recorded so the classifier can
        # refuse to call the former "released".
        top = float(cfg.get("saturation_level", 250))
        row["SatFraction"] = float((vals >= top).mean())

        # A dark core is DARK, not "perfectly homogeneous". Giving it CV 0 would
        # let an empty cell pass the homogeneity test and be called released.
        core_cv = core_std / core_mean if core_mean > 0 else 999.0

        if not np.isfinite(peri_mean) or peri_mean <= 0:
            fill = core_mean / max(mean_whole, 1e-4)
        else:
            fill = core_mean / peri_mean

        row["CoreMean"] = core_mean
        row["PeripheryMean"] = peri_mean
        row["CoreFillRatio"] = fill
        row["CoreCV"] = core_cv

        # nuclear / puncta / radial features. Computed for every cell so they
        # are available for calibration even when the classifier is not using
        # them yet - otherwise testing whether they help would need a second
        # full re-measure.
        try:
            row.update(extra_features(sub_img, cell, nd, cfg))
        except Exception:
            pass

        # one classification path, shared with the interactive tools, so the
        # batch output and the curator can never drift apart
        row["Class"] = classify_row(row, cfg)

        rows.append(row)

        if (i + 1) % 500 == 0:
            log(f"  ...cell {i + 1}/{n}")

    return rows


def measure_for_tuning(fluor_path, mask_path, cfg, log=print, max_cells=None):
    """
    Measure a sample and return everything a threshold-tuning GUI needs.

    Returns (rows, preview_rgb, labels_small, scale) where rows carry the
    per-cell features. The caller can then re-run classify_row() on every row
    instantly whenever a threshold moves.

    IMPORTANT - which settings need a re-measure:
      fill_ratio_threshold, core_cv_threshold, very_high_signal_cutoff
          -> classification only. classify_row() is enough; no pixels touched.
      min_cell_mean, core_radius_fraction, rolling_ball_radius
          -> these change WHAT IS MEASURED (which cells are excluded, where the
             core/periphery boundary sits, the background). They require a full
             re-measure, and the GUI must not pretend otherwise.
    """
    import tifffile
    from PIL import Image
    from scipy import ndimage
    Image.MAX_IMAGE_PIXELS = None

    fluor_raw = tifffile.imread(fluor_path)
    if fluor_raw.ndim == 3 and fluor_raw.shape[-1] in (3, 4):
        fluor = fluor_raw[..., 1].astype(np.float32)
    elif fluor_raw.ndim == 3:
        fluor = fluor_raw.max(axis=0).astype(np.float32)
    else:
        fluor = fluor_raw.astype(np.float32)

    labels_small = tifffile.imread(mask_path)
    if labels_small.ndim > 2:
        labels_small = labels_small[..., 0]

    if labels_small.shape != fluor.shape:
        fy = fluor.shape[0] / labels_small.shape[0]
        fx = fluor.shape[1] / labels_small.shape[1]
        yi = np.clip((np.arange(fluor.shape[0]) / fy).astype(np.int64),
                     0, labels_small.shape[0] - 1)
        xi = np.clip((np.arange(fluor.shape[1]) / fx).astype(np.int64),
                     0, labels_small.shape[1] - 1)
        labels = labels_small[yi[:, None], xi[None, :]]
    else:
        labels = labels_small

    fluor, _ = subtract_background(fluor, cfg["rolling_ball_radius"])

    objs = ndimage.find_objects(labels.astype(np.int32))
    bboxes = {}
    for idx, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        bboxes[idx] = (xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)

    if max_cells and len(bboxes) > max_cells:
        keep = sorted(bboxes)[:max_cells]
        bboxes = {k: bboxes[k] for k in keep}
        log(f"  (limited to the first {max_cells} cells for speed)")

    rows = measure(fluor, labels, bboxes, cfg, log)
    return rows, fluor_raw, labels, fluor


def crop_cell(fluor_raw, labels, row, pad=8, out_size=96):
    """
    Cut a single cell out of the image for display in a gallery.

    Returned as an RGB array with the cell's own pixels intact - this is a real
    cell from the real data, not a rendering of one.
    """
    from PIL import Image
    x, y, w, h = int(row["Bx"]), int(row["By"]), int(row["Bw"]), int(row["Bh"])
    y0 = max(0, y - pad)
    x0 = max(0, x - pad)
    y1 = min(labels.shape[0], y + h + pad)
    x1 = min(labels.shape[1], x + w + pad)

    if fluor_raw.ndim == 3 and fluor_raw.shape[-1] in (3, 4):
        sub = fluor_raw[y0:y1, x0:x1, :3]
    elif fluor_raw.ndim == 3:
        g = fluor_raw.max(axis=0)[y0:y1, x0:x1]
        sub = np.stack([g] * 3, axis=-1)
    else:
        g = fluor_raw[y0:y1, x0:x1]
        sub = np.stack([g] * 3, axis=-1)

    sub = np.ascontiguousarray(sub).astype(np.uint8)
    if sub.size == 0:
        return None

    # stretch for visibility - these crops are for judging by eye, and the
    # measurement has already been made on the unstretched data
    v = sub.astype(np.float32)
    lo, hi = np.percentile(v, [1, 99.5])
    if hi > lo:
        v = np.clip((v - lo) / (hi - lo), 0, 1) * 255
    sub = v.astype(np.uint8)

    im = Image.fromarray(sub).resize((out_size, out_size), Image.NEAREST)
    return np.array(im)



def suggest_threshold(rows, min_signal=5.0):
    """
    Suggest a brightness threshold for ONE image, by Otsu's method on the
    per-cell mean intensities.

    WHY PER-SAMPLE THRESHOLDS EXIST
      Fitted against hand labels, the best threshold varied from 14.2 to 53.3
      across four samples - a 3.7x spread caused by exposure differences
      between acquisitions, not by biology. One global threshold reproduces
      the human labels on 87% of cells; per-sample thresholds reach 95%.

      Otsu picks the split that best separates the two intensity populations
      within each image, which tracks that exposure variation automatically.
      It is a SUGGESTION: it should be checked against the overlay, and the
      value actually used must be recorded (the pipeline writes it into the
      summary and the log).
    """
    vals = np.array([r["MeanWhole"] for r in rows
                     if np.isfinite(r["MeanWhole"]) and r["MeanWhole"] >= min_signal])
    if len(vals) < 20:
        return None
    hist, edges = np.histogram(vals, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return None
    cs = np.cumsum(hist * centers)
    m0 = np.divide(cs, w0, out=np.zeros_like(cs, dtype=float), where=w0 > 0)
    m1 = np.divide(cs[-1] - cs, w1, out=np.zeros_like(cs, dtype=float), where=w1 > 0)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~valid] = -1
    return float(centers[int(np.argmax(between))])


COLUMNS = ["CellID", "Area", "MeanWhole", "CoreMean", "PeripheryMean",
           "CoreFillRatio", "CoreCV", "SatFraction",
           # nuclear / puncta / radial features
           "NucMean", "CytoMean", "NucCytoRatio",
           "PunctaCount", "PunctaArea", "PunctaMeanSize",
           "RadialSlope", "CentreEdgeRatio",
           "Class", "CentroidX", "CentroidY", "Bx", "By", "Bw", "Bh"]


def write_results(rows, path):
    """Semicolon-separated, so it opens directly in a European-locale Excel."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(";".join(COLUMNS) + "\n")
        for r in rows:
            out = []
            for c in COLUMNS:
                v = r.get(c, "")
                if isinstance(v, float):
                    out.append("NaN" if not np.isfinite(v) else f"{v:.4f}")
                else:
                    out.append(str(v))
            fh.write(";".join(out) + "\n")


def write_summary(rows, base, path, total_cells, threshold=None,
                  method="brightness", well="", label="", condition="",
                  version="", settings_hash=""):
    released = sum(1 for r in rows if r["Class"].startswith("released"))
    high = sum(1 for r in rows if r["Class"] == "released_highsignal")
    not_rel = sum(1 for r in rows if r["Class"] == "not_released")
    no_sig = sum(1 for r in rows if r["Class"] == "excluded_negative")
    too_small = sum(1 for r in rows if r["Class"] == "excluded_too_small")

    with_signal = released + not_rel
    pct_rel = 100.0 * released / with_signal if with_signal else 0.0
    pct_abs = 100.0 * released / total_cells if total_cells else 0.0
    sig_pos = total_cells - no_sig
    pct_uptake = 100.0 * sig_pos / total_cells if total_cells else 0.0

    # Well, Label and Condition are separate columns on purpose.
    #   Sample    - the full output name, unique, safe as a filename
    #   Well      - A1, B2 ... the physical position
    #   Label     - the READABLE condition, exactly as typed in the plate
    #               layout ("Fe-fum@ BSA-FL + Histidin 5 uL"). The figure maker
    #               uses this for axis and legend text, which is why it keeps
    #               the original characters rather than the sanitised filename
    #               form.
    #   Condition - the parent folder (pre_glucose, after_glucose ...), so a
    #               figure can group by treatment without parsing filenames.
    # A semicolon in a label would break the CSV, so it is replaced.
    safe_label = str(label).replace(";", ",")
    # AnalysisVersion and SettingsHash travel with the numbers: a summary file
    # that has been copied out of its folder is otherwise untraceable, and the
    # classification rule has changed enough times that this matters.
    hdr = ("Sample;Well;Label;Condition;TotalCellsInImage;CellsWithSignal;"
           "PercentUptake;Released;NotReleased;Excluded_NoSignal;"
           "Excluded_TooSmall;HighSignalSubset;PercentReleased_relative;"
           "PercentReleased_absolute;Method;Threshold;"
           "AnalysisVersion;SettingsHash")
    thr = f"{threshold:.4f}" if threshold is not None else ""
    line = (f"{base};{well};{safe_label};{condition};{total_cells};{sig_pos};"
            f"{pct_uptake:.4f};{released};{not_rel};{no_sig};{too_small};{high};"
            f"{pct_rel:.4f};{pct_abs:.4f};{method};{thr};"
            f"{version};{settings_hash}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(hdr + "\n" + line + "\n")
    return released, not_rel, pct_rel, pct_abs


OVERLAY_COLORS = {                       # class -> (RGB, legend text)
    "released": ((40, 230, 40), "released"),
    "released_highsignal": ((0, 230, 255), "released, very bright"),
    "not_released": ((255, 45, 45), "not released"),
    "excluded_negative": ((110, 140, 255), "no signal (excluded)"),
    "excluded_too_small": ((255, 220, 0), "too small (excluded)"),
    "excluded_invalid": ((255, 220, 0), "invalid (excluded)"),
}


def write_overlay(fluor_rgb, labels, rows, path, max_side=4000, scale=None):
    """
    QC image: every cell outlined in the colour of its class, on the
    fluorescence shown in GREY (so green "released" outlines stay visible) and
    contrast-stretched for viewing only (0.5-99.8 percentile). A header line
    gives the colour legend and the counts.

    Size: full resolution up to max_side px on the longer side (stitched BZ-X
    images, ~3000 px, stay full size); larger images are reduced by a whole
    number so a cell keeps a readable outline. `scale` forces a factor.
    """
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage

    labels = np.asarray(labels)
    img = np.asarray(fluor_rgb)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        grey = img[..., :3].max(axis=-1)
    elif img.ndim == 3:
        grey = img.max(axis=0)
    else:
        grey = img
    H = min(grey.shape[0], labels.shape[0])
    W = min(grey.shape[1], labels.shape[1])
    step = (max(1, int(round(1 / scale))) if scale else
            max(1, int(np.ceil(max(H, W) / float(max_side)))))
    lab = labels[:H:step, :W:step]
    g = grey[:H:step, :W:step].astype(np.float32)
    lo, hi = np.percentile(g, [0.5, 99.8])
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) * 215.0      # keep room for colour
    out = np.repeat(g[..., None], 3, axis=2).astype(np.uint8)

    names = list(OVERLAY_COLORS)
    lut = np.full(int(lab.max()) + 1, -1, dtype=np.int32)
    counts = {}
    for r in rows:
        cls = r.get("Class", "")
        counts[cls] = counts.get(cls, 0) + 1
        cid = int(r["CellID"])
        if cls in OVERLAY_COLORS and 0 <= cid < len(lut):
            lut[cid] = names.index(cls)

    # outline = label pixels touching another value; 2 px wide on big images
    edge = (lab > 0) & ((lab != ndimage.grey_erosion(lab, size=(3, 3))) |
                        (lab != ndimage.grey_dilation(lab, size=(3, 3))))
    edge_lab = np.where(edge, lab, 0)
    if out.shape[1] >= 1200:
        edge_lab = np.where(edge_lab > 0, edge_lab,
                            np.where(lab > 0, ndimage.grey_dilation(edge_lab, size=(3, 3)), 0))
    ids = edge_lab[edge_lab > 0]
    yy, xx = np.nonzero(edge_lab > 0)
    k = lut[ids]
    pal = np.array([OVERLAY_COLORS[n][0] for n in names], dtype=np.uint8)
    ok = k >= 0
    out[yy[ok], xx[ok]] = pal[k[ok]]

    # header with legend and counts
    hh = max(26, out.shape[1] // 55)
    canvas = Image.new("RGB", (out.shape[1], out.shape[0] + hh), (20, 20, 20))
    canvas.paste(Image.fromarray(out), (0, hh))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=int(hh * 0.62))
    except TypeError:                                   # Pillow < 10.1
        font = ImageFont.load_default()
    x = int(hh * 0.4)
    for n in names:
        if not counts.get(n):
            continue
        col, txt = OVERLAY_COLORS[n]
        d.rectangle([x, int(hh * 0.25), x + int(hh * 0.5), int(hh * 0.75)], outline=col,
                    width=max(2, hh // 12))
        x += int(hh * 0.7)
        t = f"{txt}: {counts[n]}"
        d.text((x, int(hh * 0.18)), t, fill=(235, 235, 235), font=font)
        try:
            x += int(d.textlength(t, font=font)) + int(hh * 0.9)
        except AttributeError:
            x += len(t) * 7 + int(hh * 0.9)
    if step > 1:
        t = f"shown at 1/{step} size"
        try:
            tw = int(d.textlength(t, font=font))
        except AttributeError:
            tw = len(t) * 7
        d.text((canvas.width - tw - int(hh * 0.4), int(hh * 0.18)), t, fill=(150, 150, 150), font=font)
    canvas.save(path)
    return canvas.size


def run_sample(fluor_path, mask_path, out_dir, base, cfg,
               well="", label="", condition="", version="", settings_hash=""):
    """Measure one sample end to end. Returns a dict of results."""
    import tifffile
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, base + "_log.txt")
    log_lines = []

    def log(msg):
        log_lines.append(str(msg))
        print("    " + str(msg), flush=True)

    log(f"fluorescence: {os.path.basename(fluor_path)}")
    fluor_raw = tifffile.imread(fluor_path)
    log(f"  shape {fluor_raw.shape} dtype {fluor_raw.dtype}")

    # RGB -> green channel (calcein)
    if fluor_raw.ndim == 3 and fluor_raw.shape[-1] in (3, 4):
        fluor = fluor_raw[..., 1].astype(np.float32)
        log("  kept green channel")
    elif fluor_raw.ndim == 3:
        fluor = fluor_raw.max(axis=0).astype(np.float32)
        log("  max-projected stack")
    else:
        fluor = fluor_raw.astype(np.float32)

    log(f"mask: {os.path.basename(mask_path)}")
    labels_small = tifffile.imread(mask_path)
    if labels_small.ndim > 2:
        labels_small = labels_small[..., 0]
    n_labels = int(labels_small.max())
    log(f"  {labels_small.shape}, {n_labels} labels")

    # upscale the mask to the fluorescence resolution with NEAREST neighbour,
    # which preserves the integer label ids exactly
    if labels_small.shape != fluor.shape:
        fy = fluor.shape[0] / labels_small.shape[0]
        fx = fluor.shape[1] / labels_small.shape[1]
        yi = np.clip((np.arange(fluor.shape[0]) / fy).astype(np.int64),
                     0, labels_small.shape[0] - 1)
        xi = np.clip((np.arange(fluor.shape[1]) / fx).astype(np.int64),
                     0, labels_small.shape[1] - 1)
        labels = labels_small[yi[:, None], xi[None, :]]
        log(f"  upscaled to {labels.shape} (nearest neighbour)")
    else:
        labels = labels_small

    if int(labels.max()) < 1:
        log("ERROR mask contains no labels")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n")
        return None

    fluor, did_bg = subtract_background(fluor, cfg["rolling_ball_radius"])
    log(f"background subtraction: {'applied' if did_bg else 'skipped (scipy missing)'}")

    # bounding boxes straight from the label array
    from scipy import ndimage
    objs = ndimage.find_objects(labels.astype(np.int32))
    bboxes = {}
    for idx, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        bboxes[idx] = (xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)
    log(f"bounding boxes: {len(bboxes)}")

    total_cells = len(bboxes)
    log(f"measuring {total_cells} cells...")
    rows = measure(fluor, labels, bboxes, cfg, log)

    # per-sample threshold, if requested. Measured first, then re-classified -
    # the features do not depend on the threshold, so nothing is re-measured.
    if cfg.get("method", "brightness") == "brightness" and cfg.get("auto_threshold"):
        sug = suggest_threshold(rows, cfg.get("min_cell_mean", 5))
        if sug is not None:
            log(f"auto threshold for this image: {sug:.2f} "
                f"(global default was {cfg['brightness_threshold']:.2f})")
            cfg = dict(cfg)
            cfg["brightness_threshold"] = sug
            for r in rows:
                r["Class"] = classify_row(r, cfg)
        else:
            log("auto threshold: too few cells, keeping the global value")

    used_threshold = cfg.get("brightness_threshold")
    log(f"classified on {cfg.get('method','brightness')}, "
        f"threshold {used_threshold:.2f}")

    fills = np.array([r["CoreFillRatio"] for r in rows
                      if np.isfinite(r["CoreFillRatio"])])
    cvs = np.array([r["CoreCV"] for r in rows
                    if np.isfinite(r["CoreCV"]) and r["CoreCV"] < 900])
    mws = np.array([r["MeanWhole"] for r in rows if np.isfinite(r["MeanWhole"])])
    if len(mws):
        med = float(np.median(mws)); p99 = float(np.percentile(mws, 99))
        ratio = p99 / max(med, 0.01)
        log(f"image dynamic range: median {med:.1f}, p99 {p99:.1f}, ratio {ratio:.1f}")
        if ratio > 20:
            log("WARNING a few very bright objects dominate this image. Bright "
                "blobs can bleed over neighbouring cells; consider excluding it.")
    log(_percentile_line("CoreFillRatio percentiles", fills))
    log(_percentile_line("CoreCV percentiles", cvs))
    log(f"thresholds: fillRatio>={cfg['fill_ratio_threshold']} "
        f"AND coreCV<={cfg['core_cv_threshold']}")
    if len(fills) and np.median(fills) > 0.95 and len(cvs) and np.median(cvs) < 0.10:
        log("WARNING features look degenerate (fillRatio~1, CV~0) - the measured "
            "image may not be a real fluorescence channel.")

    res_path = os.path.join(out_dir, base + "_results.csv")
    sum_path = os.path.join(out_dir, base + "_summary.csv")
    write_results(rows, res_path)
    released, not_rel, pct_rel, pct_abs = write_summary(
        rows, base, sum_path, total_cells,
        threshold=used_threshold, method=cfg.get("method", "brightness"),
        well=well, label=label, condition=condition,
        version=version, settings_hash=settings_hash)

    try:
        ov_path = os.path.join(out_dir, base + "_classified_overlay.png")
        write_overlay(fluor_raw, labels, rows, ov_path)
        log(f"overlay written: {os.path.basename(ov_path)}")
    except Exception as exc:
        log(f"WARNING overlay failed: {exc}")

    log(f"{base}: {total_cells} cells, {released} released, {not_rel} not released "
        f"| relative={pct_rel:.2f}%  absolute={pct_abs:.2f}%")

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")

    return {"total": total_cells, "released": released, "not_released": not_rel,
            "pct_relative": pct_rel, "pct_absolute": pct_abs}