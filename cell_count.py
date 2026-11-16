# -*- coding: utf-8 -*-
"""
cell_count.py - count cells in BRIGHTFIELD images with Cellpose and make
publication-ready figures of the segmentation.
=============================================================================
No fluorescence channel is needed. Give it folders and/or image files:

    python cell_count.py "D:\\data\\counting" --output "D:\\results" --px-per-um 2.6428

What it finds
    Every image (.tif .tiff .png .jpg .jpeg) in the given folders and their
    subfolders (two levels deep). In a folder that names its channels
    (A1_brightfield.tif / A1_fluorescence.tif), only the brightfield is used;
    fluorescence, overlays and files this program wrote are skipped.

What it writes (into <output>/cell_count/)
    cell_count_summary.csv           one row per image: counts, coverage,
                                     cell size, density per mm2
    cells/<image>_cells.csv          one row per cell: area, diameter,
                                     centre, edge contact, counted or not
    masks/<image>_cp_masks.tif       Cellpose label mask, full resolution,
                                     16-bit (opens in ImageJ / Fiji / Cellpose)
    figures/<image>_outlines.png     image + cell outlines, full resolution
    figures/<image>_outlines.pdf     same, outlines as VECTOR lines (sharp at
                                     any zoom, editable in Illustrator /
                                     Inkscape / Affinity), image embedded
    figures/<image>_outlines.svg     optional (--svg)
    figures/<image>_mask.png         optional (--filled): every cell filled in
                                     its own colour, like the Cellpose GUI

The figures are made for papers: smooth outlines drawn just inside each cell
(neighbouring cells stay separate), uniform line width, optional scale bar and
"n = ..." label, edge cells or rejected objects shown dimmed.

Counting
    n_objects        everything Cellpose segmented
    n_touching_edge  cut off by the image border
    n_too_small      below --min-area-um2 (debris)
    n_counted        the number to report, depending on --edge:
        all       every object (minus too small)                  [default]
        exclude   only cells completely inside the image
        unbiased  counting frame (Gundersen): cells touching the
                  top or right edge count, bottom or left do not.
                  This is the unbiased estimate of cells per area.
    density_per_mm2  n_unbiased / field area, when --px-per-um is given

Cellpose settings (model cpsam, flow / cellprob threshold, downscale for
speed, GPU) are the same as for the release analysis and come from the
Settings tab via --config.
=============================================================================
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
# files written by this program or by the release pipeline
GENERATED = ("_cp_masks", "_classified_overlay", "_brightfield_for_cellpose",
             "_bbox", "_outlines", "_mask", "_overlay", "_cells", "_cnn")
FLUOR_WORDS = ("fluor", "calcein", "_fl.", "_fl_", "gfp", "dapi", "hoechst")
BF_WORDS = ("brightfield", "bright_field", "_bf", "phase", "_dic", "transmitted")

COLORS = {
    "yellow": "#FFD400", "cyan": "#00E5FF", "magenta": "#FF2BD6",
    "green": "#39FF14", "red": "#FF3030", "white": "#FFFFFF",
    "black": "#000000", "orange": "#FF8C00", "blue": "#2F7BFF",
}
DIM = "#9AA4AD"       # edge cells / rejected objects when they are not counted


# =========================================================== finding images ==
def _is_generated(name):
    stem = os.path.splitext(name)[0].lower()
    return any(t in stem for t in GENERATED)


def _images_in(folder):
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    imgs = [n for n in names if n.lower().endswith(IMG_EXT) and not _is_generated(n)]
    named_bf = [n for n in imgs if any(w in n.lower() for w in BF_WORDS)]
    if named_bf:                       # channels are named: take brightfield only
        return [os.path.join(folder, n) for n in named_bf]
    return [os.path.join(folder, n) for n in imgs
            if not any(w in n.lower() for w in FLUOR_WORDS)]


def find_images(inputs, skip_dirs=()):
    """Image files from files and folders (folders searched two levels deep)."""
    skip = {os.path.normcase(os.path.abspath(d)) for d in skip_dirs if d}
    found, seen = [], set()

    def add(p, root):
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            found.append((p, root))

    for item in inputs:
        item = item.strip().strip('"')
        if os.path.isfile(item):
            add(item, os.path.dirname(item))
            continue
        if not os.path.isdir(item):
            print(f"WARNING: not found, skipped: {item}")
            continue
        stack = [(item, 0)]
        while stack:
            d, depth = stack.pop(0)
            if os.path.normcase(os.path.abspath(d)) in skip:
                continue
            for p in _images_in(d):
                add(p, item)
            if depth < 2:
                try:
                    subs = sorted(e for e in os.listdir(d)
                                  if os.path.isdir(os.path.join(d, e))
                                  and e not in ("cell_count", "edited_results",
                                                "figures", "masks", "cells"))
                except OSError:
                    subs = []
                stack += [(os.path.join(d, s), depth + 1) for s in subs]
    return found


def base_name(path, root):
    """A readable, unique name: the path below the input folder, joined by _."""
    rel = os.path.relpath(os.path.splitext(path)[0], root)
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    stem = parts[-1] if parts else os.path.splitext(os.path.basename(path))[0]
    # drop folder names the file name already contains: A1/A1_brightfield -> A1_brightfield
    keep = [p for p in parts[:-1] if p.lower() not in stem.lower()]
    name = "_".join(keep + [stem])
    for ch in '<>:"|?*':
        name = name.replace(ch, "-")
    return name


# ============================================================ image loading ==
def load_gray(path):
    """Any microscopy image -> 2D float32 (RGB averaged, stacks max-projected)."""
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        arr = tifffile.imread(path)
    else:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im)
    arr = np.asarray(arr)
    if arr.ndim > 2:
        big = sorted(sorted(range(arr.ndim), key=lambda i: arr.shape[i])[-2:])
        for ax in sorted([i for i in range(arr.ndim) if i not in big], reverse=True):
            if arr.shape[ax] <= 4:
                take = arr
                if arr.shape[ax] == 4:            # drop alpha
                    take = np.take(arr, [0, 1, 2], axis=ax)
                arr = take.mean(axis=ax)
            else:
                arr = arr.max(axis=ax)
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 2 or min(arr.shape) < 32:
        raise ValueError(f"could not read {os.path.basename(path)} as a 2D image "
                         f"(shape {arr.shape})")
    return arr


def to_8bit(arr, lo_pct=0.5, hi_pct=99.5):
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


# ============================================================== segmentation ==
def segment(gray8, scale, holder):
    """Cellpose on the (optionally downscaled) image, through the release
    pipeline's own run_cellpose: same model, device choice and thresholds.
    Returns the label mask at segmentation resolution."""
    import tempfile
    import tifffile
    from PIL import Image
    import run_cargo_pipeline as rcp

    img = gray8
    if scale != 1.0:
        im = Image.fromarray(gray8)
        img = np.array(im.resize((max(1, round(im.width * scale)),
                                  max(1, round(im.height * scale))), Image.BILINEAR))
    tmp = os.path.join(tempfile.gettempdir(), f"cellcount_{os.getpid()}_cp_masks.tif")
    try:
        rcp.run_cellpose(img, tmp, holder)
        masks = tifffile.imread(tmp)
    finally:
        for p in (tmp, os.path.splitext(tmp)[0] + "_bbox.csv"):
            try:
                os.remove(p)
            except OSError:
                pass
    return masks.astype(np.int32)


def upscale_labels(masks, shape):
    """Nearest-neighbour resize of a label image to the full image size."""
    if masks.shape == tuple(shape):
        return masks
    H, W = shape
    ri = np.minimum((np.arange(H) * masks.shape[0] / H).astype(np.int64), masks.shape[0] - 1)
    ci = np.minimum((np.arange(W) * masks.shape[1] / W).astype(np.int64), masks.shape[1] - 1)
    return masks[ri[:, None], ci[None, :]]


# =================================================================== geometry ==
def cell_table(masks_full, px_per_um, min_area_um2, edge_mode):
    """Per-cell measurements on the full-resolution mask."""
    from scipy import ndimage
    H, W = masks_full.shape
    n = int(masks_full.max())
    objs = ndimage.find_objects(masks_full)
    labels = np.arange(1, n + 1)
    area = ndimage.sum_labels(np.ones_like(masks_full, dtype=np.float32),
                              masks_full, labels) if n else np.zeros(0)
    com = ndimage.center_of_mass(np.ones_like(masks_full, dtype=np.float32),
                                 masks_full, labels) if n else []
    um2 = (1.0 / px_per_um) ** 2 if px_per_um else None
    rows = []
    for i, lab in enumerate(labels):
        sl = objs[i]
        if sl is None or area[i] == 0:
            continue
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        edges = [e for e, hit in (("top", r0 == 0), ("bottom", r1 == H),
                                  ("left", c0 == 0), ("right", c1 == W)) if hit]
        a_px = float(area[i])
        a_um = a_px * um2 if um2 else None
        too_small = bool(min_area_um2) and (
            (a_um if a_um is not None else a_px) < min_area_um2)
        unbiased = not ("bottom" in edges or "left" in edges)
        if too_small:
            counted, why = False, "too small"
        elif edge_mode == "exclude" and edges:
            counted, why = False, "touches edge"
        elif edge_mode == "unbiased" and not unbiased:
            counted, why = False, "bottom/left edge"
        else:
            counted, why = True, ""
        eq_d = 2 * math.sqrt(a_px / math.pi)
        rows.append({
            "label": int(lab), "area_px": round(a_px, 1),
            "area_um2": round(a_um, 2) if a_um is not None else "",
            "eq_diameter_px": round(eq_d, 2),
            "eq_diameter_um": round(eq_d / px_per_um, 2) if px_per_um else "",
            "centre_x": round(com[i][1], 1), "centre_y": round(com[i][0], 1),
            "touches_edge": "+".join(edges), "too_small": int(too_small),
            "unbiased_frame": int(unbiased and not too_small),
            "counted": int(counted), "not_counted_because": why,
        })
    return rows


def outlines(masks, scale_to_full, smooth=True):
    """Smooth closed outlines {label: (N,2) array of x,y in FULL-res pixels}.

    Contours run through the centres of each cell's boundary pixels, i.e. just
    inside the cell, so neighbouring cells keep a small gap. A short circular
    moving average removes the pixel staircase (2x stronger when the mask was
    segmented at half resolution), then Douglas-Peucker keeps the file small.
    """
    import cv2
    from scipy import ndimage
    out = {}
    objs = ndimage.find_objects(masks)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        lab = i + 1
        r0, c0 = sl[0].start, sl[1].start
        sub = np.pad((masks[sl] == lab).astype(np.uint8), 1)
        found = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = found[0] if len(found) == 2 else found[1]
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        if smooth and len(c) >= 5:
            # circular moving average over 5 (or 3) boundary points
            k = 2 if len(c) >= 15 else 1
            w = 2 * k + 1
            pad = np.concatenate([c[-k:], c, c[:k]])
            c = np.column_stack([np.convolve(pad[:, j], np.ones(w) / w, mode="valid")
                                 for j in (0, 1)])
        if len(c) >= 8:
            c = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), 0.25, True)\
                   .reshape(-1, 2).astype(np.float64)
        # back to image coordinates (undo pad), pixel centres, then full res
        c[:, 0] += c0 - 1
        c[:, 1] += r0 - 1
        c = (c + 0.5) * scale_to_full - 0.5
        if len(c) >= 3:
            out[lab] = c
    return out


# ==================================================================== figures ==
def nice_bar_um(width_um):
    """A round scale-bar length of ~15-20 % of the image width."""
    target = width_um * 0.18
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if v >= target * 0.6:
            return v
    return 5000


def draw_figure(gray8, polys, rows, path_base, opt, filled=False):
    """Render image + outlines with matplotlib.

    PNG at the image's own pixel size (1 image pixel = 1 output pixel);
    PDF / SVG with vector outlines over the embedded image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe

    H, W = gray8.shape
    color = COLORS.get(opt.color, opt.color)
    counted = {r["label"]: r["counted"] for r in rows}
    lw_px = float(opt.line_width)

    def render(dpi, img, vector):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        # "none": the PNG maps image pixels 1:1, and PDF/SVG embed the image
        # at its own resolution instead of resampling it to the page
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="none",
                  extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        lw = lw_px * 72.0 / dpi                  # line width in points
        on = [p for l, p in polys.items() if counted.get(l, 1)]
        off = [p for l, p in polys.items() if not counted.get(l, 1)]
        if filled:
            rng = np.random.default_rng(7)
            hues = (np.arange(len(polys)) * 0.61803398875 + rng.random()) % 1.0
            import colorsys
            fc = [colorsys.hsv_to_rgb(h, 0.75, 1.0) + (0.45,) for h in hues]
            ax.add_collection(PolyCollection(list(polys.values()), facecolors=fc,
                                             edgecolors="none"))
        if off and opt.show_uncounted:
            ax.add_collection(LineCollection([np.vstack([p, p[:1]]) for p in off],
                                             colors=DIM, linewidths=lw * 0.8,
                                             linestyles=(0, (2, 1.5)), alpha=0.9))
        ax.add_collection(LineCollection([np.vstack([p, p[:1]]) for p in on],
                                         colors=color, linewidths=lw,
                                         joinstyle="round", capstyle="round"))
        fs = max(6.0, W / dpi * 72 * 0.022)   # text ~2.2 % of the width
        if opt.numbers:
            lookup = {r["label"]: r for r in rows}
            for l in polys:
                r = lookup.get(l)
                if r:
                    ax.text(r["centre_x"], r["centre_y"], str(l), color=color,
                            fontsize=fs * 0.35, ha="center", va="center")
        halo = [pe.withStroke(linewidth=fs * 0.18, foreground="black")]
        if opt.count_label:
            n = sum(1 for r in rows if r["counted"])
            ax.text(W * 0.025, H * 0.04, f"n = {n}", color="white", fontsize=fs,
                    fontweight="bold", ha="left", va="top", path_effects=halo)
        if opt.scale_bar and opt.px_per_um:
            um = nice_bar_um(W / opt.px_per_um)
            L = um * opt.px_per_um
            th = max(H * 0.012, 3)
            x1, y1 = W * 0.975, H * 0.955
            ax.add_patch(Rectangle((x1 - L, y1 - th), L, th, facecolor="white",
                                   edgecolor="black", linewidth=fs * 0.05))
            ax.text(x1 - L / 2, y1 - th * 1.6, f"{um:g} µm", color="white",
                    fontsize=fs * 0.8, ha="center", va="bottom",
                    fontweight="bold", path_effects=halo)
        return fig

    written = []
    # raster at native resolution
    fig = render(100, gray8, vector=False)
    p = path_base + ".png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    written.append(p)
    # vector: embed the image at <= opt.vector_max_px on its long side
    if opt.pdf or opt.svg:
        img = gray8
        f = max(H, W) / float(opt.vector_max_px)
        if f > 1:
            from PIL import Image
            img = np.array(Image.fromarray(gray8).resize(
                (max(1, round(W / f)), max(1, round(H / f))), Image.LANCZOS))
        # page 18 cm wide (a full-width journal figure); line width and text
        # scale with it, so the PDF looks exactly like the PNG
        fig = render(W / (18 / 2.54), img, vector=True)
        for ext, on in ((".pdf", opt.pdf), (".svg", opt.svg)):
            if on:
                fig.savefig(path_base + ext)
                written.append(path_base + ext)
        plt.close(fig)
    return written


# ======================================================================= main ==
def main(argv=None):
    ap = argparse.ArgumentParser(description="Count cells in brightfield images "
                                             "with Cellpose; paper-ready outline figures.")
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--output", help="results go to <output>/cell_count "
                                     "(default: next to the first input)")
    ap.add_argument("--px-per-um", type=float, default=0.0,
                    help="pixels per micrometre (scale bar, areas in um2, density)")
    ap.add_argument("--edge", choices=("all", "exclude", "unbiased"), default="all")
    ap.add_argument("--min-area-um2", type=float, default=0.0,
                    help="objects smaller than this are debris (um2, or px when "
                         "no --px-per-um)")
    ap.add_argument("--color", default="yellow")
    ap.add_argument("--line-width", type=float, default=3.0,
                    help="outline width in image pixels")
    ap.add_argument("--numbers", action="store_true", help="write each cell's number")
    ap.add_argument("--no-count-label", dest="count_label", action="store_false")
    ap.add_argument("--no-scale-bar", dest="scale_bar", action="store_false")
    ap.add_argument("--hide-uncounted", dest="show_uncounted", action="store_false",
                    help="do not draw edge cells / debris (dimmed dashed by default)")
    ap.add_argument("--filled", action="store_true",
                    help="also a figure with every cell filled in its own colour")
    ap.add_argument("--svg", action="store_true", help="also SVG")
    ap.add_argument("--no-pdf", dest="pdf", action="store_false")
    ap.add_argument("--vector-max-px", type=int, default=3000,
                    help="image resolution inside PDF/SVG (long side)")
    ap.add_argument("--force", action="store_true",
                    help="segment again even if a mask exists")
    ap.add_argument("--config", help="JSON settings from the GUI (Cellpose thresholds etc.)")
    opt = ap.parse_args(argv)

    import run_cargo_pipeline as rcp
    if opt.config:
        try:
            with open(opt.config, encoding="utf-8") as fh:
                rcp.apply_settings(json.load(fh))
        except Exception as exc:
            print(f"WARNING: could not read {opt.config}: {exc}")
    scale = float(rcp.CELLPOSE_SCALE_FACTOR or 1.0)

    first = opt.inputs[0].strip().strip('"')
    out_parent = opt.output or (first if os.path.isdir(first) else os.path.dirname(first))
    out = os.path.join(os.path.abspath(out_parent), "cell_count")
    dirs = {k: os.path.join(out, k) for k in ("figures", "masks", "cells")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    images = find_images(opt.inputs, skip_dirs=[out])
    if not images:
        raise SystemExit("ERROR: no images found. Give image files or folders that "
                         "contain .tif / .png / .jpg brightfield images.")
    unit = "um2" if opt.px_per_um else "px"
    print(f"Images     : {len(images)}")
    print(f"Results    : {out}")
    print(f"Cellpose   : {rcp.CELLPOSE_MODEL}, cellprob {rcp.CELLPROB_THRESHOLD}, "
          f"flow {rcp.FLOW_THRESHOLD}, downscale {scale}")
    print(f"Counting   : edge = {opt.edge}"
          + (f", min area {opt.min_area_um2:g} {unit}" if opt.min_area_um2 else ""))
    print(f"Pixel size : {opt.px_per_um:g} px/um" if opt.px_per_um
          else "Pixel size : not given - no scale bar, sizes in pixels")
    print()

    holder = {"model": None}
    summary, failed = [], []
    names_used = set()
    t_all = time.time()
    for i, (path, root) in enumerate(images, 1):
        t0 = time.time()
        name = base_name(path, root)
        while name in names_used:
            name += "_2"
        names_used.add(name)
        print(f"[{i}/{len(images)}] {name}")
        try:
            raw = load_gray(path)
            gray8 = to_8bit(raw)
            H, W = gray8.shape
            mpath = os.path.join(dirs["masks"], name + "_cp_masks.tif")
            import tifffile
            if os.path.isfile(mpath) and not opt.force:
                masks = tifffile.imread(mpath).astype(np.int32)
                print("    mask exists - reused (use --force to segment again)")
            else:
                print(f"    Cellpose on {W}x{H} px (x{scale:g}) ...")
                masks = segment(gray8, scale, holder)
            seg_scale = W / float(masks.shape[1])        # mask px -> image px
            masks_full = upscale_labels(masks, (H, W))
            if not (os.path.isfile(mpath) and not opt.force):
                tifffile.imwrite(mpath, masks_full.astype(np.uint16), compression="zlib")
            rows = cell_table(masks_full, opt.px_per_um, opt.min_area_um2, opt.edge)
            polys = outlines(masks, seg_scale)
            figs = draw_figure(gray8, polys, rows,
                               os.path.join(dirs["figures"], name + "_outlines"), opt)
            if opt.filled:
                o2 = argparse.Namespace(**vars(opt))
                o2.pdf = o2.svg = False
                figs += draw_figure(gray8, polys, rows,
                                    os.path.join(dirs["figures"], name + "_mask"), o2,
                                    filled=True)

            with open(os.path.join(dirs["cells"], name + "_cells.csv"), "w",
                      newline="", encoding="utf-8") as fh:
                cols = ["label", "area_px", "area_um2", "eq_diameter_px", "eq_diameter_um",
                        "centre_x", "centre_y", "touches_edge", "too_small",
                        "unbiased_frame", "counted", "not_counted_because"]
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)

            interior = [r for r in rows if not r["touches_edge"] and not r["too_small"]]
            n_obj = len(rows)
            n_edge = sum(1 for r in rows if r["touches_edge"])
            n_small = sum(r["too_small"] for r in rows)
            n_count = sum(r["counted"] for r in rows)
            n_unb = sum(r["unbiased_frame"] for r in rows)
            med_a = float(np.median([r["area_px"] for r in interior])) if interior else float("nan")
            area_mm2 = (H * W) / (opt.px_per_um ** 2) / 1e6 if opt.px_per_um else None
            row = {
                "image": name, "file": path, "width_px": W, "height_px": H,
                "n_objects": n_obj, "n_touching_edge": n_edge, "n_too_small": n_small,
                "edge_rule": opt.edge, "n_counted": n_count, "n_unbiased_frame": n_unb,
                "coverage_pct": round(100.0 * float((masks_full > 0).mean()), 1),
                "median_area_px": round(med_a, 1) if interior else "",
                "median_area_um2": round(med_a / opt.px_per_um ** 2, 1)
                                   if (interior and opt.px_per_um) else "",
                "median_diameter_um": round(2 * math.sqrt(med_a / math.pi) / opt.px_per_um, 1)
                                      if (interior and opt.px_per_um) else "",
                "field_area_mm2": round(area_mm2, 4) if area_mm2 else "",
                "density_per_mm2": round(n_unb / area_mm2, 1) if area_mm2 else "",
                "px_per_um": opt.px_per_um or "",
                "model": rcp.CELLPOSE_MODEL, "cellprob_threshold": rcp.CELLPROB_THRESHOLD,
                "flow_threshold": rcp.FLOW_THRESHOLD, "downscale": scale,
            }
            summary.append(row)
            extra = (f", {row['density_per_mm2']} cells/mm2" if area_mm2 else "")
            print(f"    {n_count} counted  ({n_obj} objects, {n_edge} at the edge"
                  + (f", {n_small} too small" if n_small else "") + f"){extra}"
                  f"   [{time.time() - t0:.0f} s]")
            print("    figure: " + os.path.basename(figs[0]))
        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}")
            traceback.print_exc(file=sys.stdout)
            failed.append(name)

    if summary:
        sp = os.path.join(out, "cell_count_summary.csv")
        with open(sp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print()
        print("=" * 60)
        print(f"{len(summary)} image(s) counted in {time.time() - t_all:.0f} s"
              + (f", {len(failed)} failed: {', '.join(failed)}" if failed else ""))
        print(f"Summary : {sp}")
        print(f"Figures : {dirs['figures']}")
    else:
        raise SystemExit("ERROR: no image could be counted - see the messages above.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
