# -*- coding: utf-8 -*-
"""
make_synthetic.py - fake microscopy fields with a KNOWN answer, for testing
=============================================================================
Builds a folder tree shaped exactly like real data:

    <out>/synth_slide/pre_glucose/A1/A1_brightfield.tif
                                    /A1_fluorescence.tif      (RGB, calcein green)
                                    /pre_glucose_A1_cp_masks.tif
                                    /edited_results/          (filled by the pipeline)

Cells are ellipses. "released" cells are filled evenly (dye everywhere,
nucleus included); "not released" cells are dim with a few bright puncta
near the rim; some cells took nothing up. Each field gets its own exposure
and its own uneven background (offset + tilt + a bright haze blob) - the
nuisance the CNN is meant to ignore.

This is NOT a substitute for real labels. It only proves the code runs and
can learn a pattern that is genuinely there.

    python tests/make_synthetic.py <out_folder> [--fields 5] [--noise 0.08]
=============================================================================
"""

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_field(rng, size=1100, n_cells=110, exposure=1.0, bg_level=6.0):
    H = W = size
    fl = np.zeros((H, W), np.float32)
    lab = np.zeros((H, W), np.uint16)
    truth = {}
    yy, xx = np.mgrid[0:H, 0:W]
    lb = 0
    tries = 0
    while lb < n_cells and tries < n_cells * 30:
        tries += 1
        cy, cx = rng.uniform(50, H - 50), rng.uniform(50, W - 50)
        a, b = rng.uniform(26, 42), rng.uniform(22, 36)
        th = rng.uniform(0, np.pi)
        y0, y1 = int(max(0, cy - 45)), int(min(H, cy + 46))
        x0, x1 = int(max(0, cx - 45)), int(min(W, cx + 46))
        sy, sx = yy[y0:y1, x0:x1] - cy, xx[y0:y1, x0:x1] - cx
        u = (sx * np.cos(th) + sy * np.sin(th)) / a
        v = (-sx * np.sin(th) + sy * np.cos(th)) / b
        r = np.sqrt(u * u + v * v)
        cell = r <= 1.0
        if (lab[y0:y1, x0:x1][cell] > 0).any():
            continue
        lb += 1
        lab[y0:y1, x0:x1][cell] = lb
        kind = rng.choice(["released", "not_released", "none"], p=[0.42, 0.45, 0.13])
        sub = fl[y0:y1, x0:x1]
        if kind == "released":
            level = rng.uniform(28, 90)
            sub[cell] += level * (0.85 + 0.15 * (1 - r[cell])) \
                + rng.normal(0, level * 0.08, cell.sum())
        elif kind == "not_released":
            sub[cell] += rng.uniform(3, 14) * (0.4 + 0.6 * r[cell])  # rim-weighted
            for _ in range(rng.integers(3, 9)):
                ang = rng.uniform(0, 2 * np.pi)
                rr = rng.uniform(0.45, 0.9)
                py = cy + rr * b * np.sin(ang)
                px = cx + rr * a * np.cos(ang)
                d2 = (yy[y0:y1, x0:x1] - py) ** 2 + (xx[y0:y1, x0:x1] - px) ** 2
                sub += cell * rng.uniform(60, 180) * np.exp(-d2 / (2 * rng.uniform(1.5, 3.0) ** 2))
        else:
            sub[cell] += rng.uniform(0, 2.5)
        truth[lb] = kind

    # uneven background: offset + tilt + haze blob
    g = bg_level + rng.uniform(-1, 1) * 6 * (xx / W - 0.5) + rng.uniform(-1, 1) * 6 * (yy / H - 0.5)
    hy, hx = rng.uniform(0, H), rng.uniform(0, W)
    g = g + rng.uniform(5, 20) * np.exp(-((yy - hy) ** 2 + (xx - hx) ** 2) / (2 * (size / 4) ** 2))
    img = (fl + g) * exposure + rng.normal(0, 1.2, (H, W))
    img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 1] = img
    return rgb, lab, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fields", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.08,
                    help="fraction of labels flipped, imitating labelling disagreement")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import tifffile
    import cargo_measure as cm

    rng = np.random.default_rng(a.seed)
    parent = os.path.join(a.out, "synth_slide", "pre_glucose")
    wells = [f"{L}{n}" for L in "ABCD" for n in (1, 2)][:a.fields]
    cfg = {"method": "brightness", "brightness_threshold": 25.7, "core_cv_cap": 0.65,
           "saturation_guard": 0.0, "auto_threshold": True, "min_cell_mean": 5,
           "core_radius_fraction": 0.5, "fill_ratio_threshold": 0.65,
           "core_cv_threshold": 0.6, "very_high_signal_cutoff": 1e9,
           "rolling_ball_radius": 50}
    for i, w in enumerate(wells):
        d = os.path.join(parent, w)
        os.makedirs(d, exist_ok=True)
        rgb, lab, truth = make_field(rng, exposure=rng.uniform(0.6, 1.5),
                                     bg_level=rng.uniform(2, 15))
        tifffile.imwrite(os.path.join(d, f"{w}_brightfield.tif"), rgb[..., 1])
        tifffile.imwrite(os.path.join(d, f"{w}_fluorescence.tif"), rgb)
        base = f"pre_glucose_{w}"
        mask = os.path.join(d, f"{base}_cp_masks.tif")
        tifffile.imwrite(mask, lab)
        out = os.path.join(d, "edited_results")
        cm.run_sample(os.path.join(d, f"{w}_fluorescence.tif"), mask, out, base, cfg,
                      well=w, condition="pre_glucose")
        # the last field is left unlabelled - it is the "new experiment". Its
        # true classes go to *_truth.csv so a prediction can be scored.
        if i == len(wells) - 1:
            with open(os.path.join(out, f"{w}_truth.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                wr = csv.writer(fh, delimiter=";")
                wr.writerow(["CellID", "Truth"])
                wr.writerows(sorted(truth.items()))
            print(f"{w}: {len(truth)} cells (unlabelled, truth file only)")
            continue
        with open(os.path.join(out, f"{w}_curated.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh, delimiter=";")
            wr.writerow(["CellID", "Label", "Source", "Truth"])
            for cid, k in truth.items():
                if k == "none":
                    continue
                lbl = k
                if rng.random() < a.noise:
                    lbl = "released" if k == "not_released" else "not_released"
                wr.writerow([cid, lbl, "manual_blind", k])
        print(f"{w}: {len(truth)} cells")
    print("done ->", parent)


if __name__ == "__main__":
    main()
