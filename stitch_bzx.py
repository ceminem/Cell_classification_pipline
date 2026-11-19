# -*- coding: utf-8 -*-
"""
stitch_bzx.py - stitch Keyence BZ-X tile images (brightfield + fluorescence)
=============================================================================
Replaces the BZ-X Analyzer "image stitching" step, so the classification
pipeline can run without the Keyence licence:

    python stitch_bzx.py "D:\\data\\experiment" --output "D:\\data\\experiment_stitched"

What it does, for every XY position found below the parent folder:

  1. finds the tiles   <name>_XY01_00001_CH2.tif, ..._CH4.tif, ... (any depth)
  2. reads the BZ-X group file (<name>_XY01.gci) when present: grid size, the
     row / column of every tile, channel names (GFP, brightfield = "Relief"),
     pseudo colours and the calibration (nm per pixel). Without it the grid is
     inferred (square-ish, BZ-X snake order) and checked by the registration.
  3. aligns the tiles from their content: phase correlation in the overlap of
     every pair of neighbours (nominal overlap 30 %, searched +-12 % of the tile),
     scored by normalised cross-correlation, then one global least-squares fit
     of all positions. Unreliable pairs fall back to the nominal step. The
     positions are found on the BRIGHTFIELD (structure everywhere) and used for
     every channel, so brightfield and fluorescence stay pixel-aligned.
  4. blends the overlaps with linear feathering (no seams). Vignetting
     (darker tile edges) is corrected for the BRIGHTFIELD by default - it only
     evens the background for segmentation - and optionally for fluorescence
     (OFF by default, so intensities stay as recorded, like BZ-X output).
  5. writes, per position, into the output folder:
        XY01/XY01_brightfield.tif      RGB 8-bit, grey (like BZ-X output)
        XY01/XY01_fluorescence.tif     RGB 8-bit, in its pseudo colour
        (more fluorescence channels: XY01_fluorescence_CH1_DAPI.tif, ...)
     The calibration is stored in the TIFF resolution tags (Fiji shows um).
     This layout is exactly what the classification pipeline reads: choose the
     output folder as parent folder on the Run tab.
  6. positions with fewer than --min-tiles tiles (default 4), or with tiles
     missing from the grid, are SKIPPED and listed in stitch_report.csv and at
     the end of the log.

CPU only (numpy FFT): about 1-3 s per position, the same on any PC; several
positions are stitched in parallel (--workers).
=============================================================================
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import struct
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

TILE_RE = re.compile(r"^(?P<prefix>.*?)_?(?P<xy>XY\d+)?_(?P<idx>\d{5})_(?P<ch>CH\d|Overlay)"
                     r"\.tiff?$", re.IGNORECASE)
COLORS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
          "white": (1, 1, 1), "gray": (1, 1, 1), "grey": (1, 1, 1),
          "cyan": (0, 1, 1), "magenta": (1, 0, 1), "yellow": (1, 1, 0)}
BF_OBSERVATIONS = ("relief", "brightfield", "bright field", "phase", "phasecontrast",
                   "dic", "transmitted", "oblique")


# ============================================================ .gci group file ==
def _d(bits):
    """BZ-X stores doubles as the int64 of their bits."""
    try:
        return struct.unpack("<d", struct.pack("<q", int(bits)))[0]
    except (ValueError, struct.error):
        try:
            return float(bits)
        except ValueError:
            return None


def read_gci(path):
    """Grid, tile positions, channels and calibration from a BZ-X .gci file
    (a ZIP of XML property files plus a binary file list)."""
    info = {"path": path, "rows": None, "cols": None, "tiles": {}, "channels": {},
            "nm_per_px": None, "lens": ""}
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return None
    names = set(z.namelist())

    def props(p):
        p = "GroupFileProperty/" + p + "/properties.xml" if p else "GroupFileProperty/properties.xml"
        if p not in names:
            return {}
        s = z.read(p).decode("utf-8", "replace")
        return {k: v.strip() for k, _t, v in
                re.findall(r'<(\w+) Type="([^"]+)"(?: /)?>([^<]*)', s)}

    j = props("ImageJoint")
    if j.get("Enabled", "").lower() == "true":
        info["rows"], info["cols"] = int(j.get("Row", 0)) or None, int(j.get("Column", 0)) or None
    cal = props("Image").get("Calibration")
    if cal:
        v = _d(cal)
        if v and 1 < v < 1e5:
            info["nm_per_px"] = v
    info["lens"] = props("Lens").get("LensName", "")
    for ch in range(8):
        base = props(f"Channel{ch}")
        if not base:
            continue
        sh = props(f"Channel{ch}/Shooting")
        par = props(f"Channel{ch}/Shooting/Parameter")
        info["channels"][f"CH{ch + 1}"] = {
            "comment": base.get("Comment", ""), "shot": base.get("IsShot", "") == "True",
            "observation": sh.get("Observation", ""), "color": par.get("PseudoColor", ""),
        }
    # binary file list: per entry  channel(str) 4*int32(?, ?, row, col) file(str) double
    fl = "GroupFileProperty/ImageList/FileList"
    if fl in names:
        b = z.read(fl)
        try:
            n = struct.unpack_from("<i", b, 0)[0]
            o = 4

            def rstr(o):
                L, sh_ = 0, 0
                while True:
                    c = b[o]
                    o += 1
                    L |= (c & 0x7F) << sh_
                    sh_ += 7
                    if c < 0x80:
                        break
                return b[o:o + L].decode("utf-8", "replace"), o + L
            for _ in range(n):
                _ch, o = rstr(o)
                _a, _b, row, col = struct.unpack_from("<4i", b, o)
                o += 16
                fn, o = rstr(o)
                o += 8
                m = TILE_RE.match(fn)
                if m:
                    info["tiles"][int(m.group("idx"))] = (row, col)
        except (struct.error, IndexError):
            pass
    return info


# =========================================================== finding tiles ==
def find_groups(parent):
    """{group key: {"dir", "name", "tiles": {channel: {index: path}}}}"""
    groups = {}
    for dirpath, _dirs, files in os.walk(parent):
        for f in files:
            m = TILE_RE.match(f)
            if not m:
                continue
            prefix = (m.group("prefix") or "").strip(" _")
            xy = (m.group("xy") or "").upper()
            key = (os.path.normcase(dirpath), prefix, xy)
            g = groups.setdefault(key, {"dir": dirpath, "prefix": prefix, "xy": xy,
                                        "tiles": {}})
            ch = m.group("ch").upper() if m.group("ch").lower() != "overlay" else "Overlay"
            g["tiles"].setdefault(ch, {})[int(m.group("idx"))] = os.path.join(dirpath, f)
    # readable unique names: XY01, or prefix_XY01 when the same XY occurs twice
    by_xy = {}
    for k, g in groups.items():
        by_xy.setdefault(g["xy"] or g["prefix"], []).append(k)
    for base, keys in by_xy.items():
        for k in keys:
            g = groups[k]
            name = base or "tiles"
            if len(keys) > 1:
                rel = os.path.relpath(g["dir"], parent).replace("\\", "/").strip("./")
                name = "_".join(p for p in (rel.replace("/", "_"), g["prefix"], g["xy"]) if p)
            g["name"] = re.sub(r'[<>:"/\\|?*]', "-", name) or "tiles"
    return groups


def find_gci(g):
    """The group file of a tile group: <prefix>_<XY>.gci, else any .gci in the
    folder whose file list names these tiles."""
    cands = []
    if g["xy"]:
        cands += glob.glob(os.path.join(glob.escape(g["dir"]), f"*{g['xy']}*.gci"))
    cands += glob.glob(os.path.join(glob.escape(g["dir"]), "*.gci"))
    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        info = read_gci(p)
        if not info:
            continue
        if info["tiles"] or info["rows"]:
            if not info["tiles"] or any(i in info["tiles"] for i in
                                        next(iter(g["tiles"].values())).keys()):
                return info
    return None


# ============================================================ tile images ==
def load_tile(path):
    """Tile as float32 2D plus its colour: BZ-X tiles are RGB with the signal in
    the pseudo-colour channel(s); grey = all three equal."""
    import tifffile
    a = tifffile.imread(path)
    color = (1, 1, 1)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3]
        m = a.reshape(-1, 3).max(axis=0).astype(float)
        if m.max() > 0:
            rel = m / m.max()
            if not (rel > 0.5).all():
                color = tuple(int(x > 0.5) for x in rel)
        g = a.max(axis=-1).astype(np.float32)
    else:
        g = np.squeeze(a).astype(np.float32)
    return g, color


def channel_roles(g, gci):
    """Decide which channel is brightfield and which are fluorescence."""
    roles = {}
    for ch, tiles in g["tiles"].items():
        if ch == "Overlay":
            continue
        meta = (gci or {}).get("channels", {}).get(ch, {})
        obs = meta.get("observation", "").lower()
        if obs:
            bf = any(w in obs for w in BF_OBSERVATIONS)
        else:   # no metadata: brightfield is grey with a bright background
            img, color = load_tile(next(iter(tiles.values())))
            bf = color == (1, 1, 1) and float(np.median(img)) > 30
        roles[ch] = {"bf": bf, "comment": meta.get("comment", ""),
                     "color": COLORS.get(meta.get("color", "").lower())}
    return roles


# ============================================================== registration ==
def _prep(img):
    """High-pass + contrast normalisation: removes vignetting / exposure so the
    correlation sees structure, not illumination."""
    from scipy import ndimage
    f = img - ndimage.gaussian_filter(img, 12)
    s = f.std()
    return f / s if s > 0 else f


def _hann2(h, w):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _phase_corr(a, b, max_dy, max_dx):
    """Shift d (dy, dx) with a(x) ~ b(x - d), searched within +-max."""
    win = _hann2(*a.shape)
    A = np.fft.rfft2(a * win)
    B = np.fft.rfft2(b * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, a.shape)
    h, w = a.shape
    dy = np.fft.fftfreq(h, 1 / h).astype(int)
    dx = np.fft.fftfreq(w, 1 / w).astype(int)
    mask = (np.abs(dy)[:, None] <= max_dy) & (np.abs(dx)[None, :] <= max_dx)
    r = np.where(mask, r, -np.inf)
    i, j = np.unravel_index(np.argmax(r), r.shape)
    return int(dy[i]), int(dx[j])


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / d if d > 0 else 0.0


def pair_offset(t1, t2, direction, overlap, search):
    """Offset (oy, ox) of tile t2 relative to t1 (t2 right of / below t1).

    Only the expected overlap strip (+ search margin) is correlated. Returns
    (oy, ox, ncc); ncc is measured on the actually overlapping pixels."""
    H, W = t1.shape
    if direction == "h":
        nom = W * (1 - overlap)
        s = int(min(W, round(W - nom + search * W)))
        a, b = t1[:, W - s:], t2[:, :s]
        dy, dx = _phase_corr(a, b, int(search * H), int(search * W))
        # a(x) = t1[x + W - s];  a(x) ~ b(x - dx) = t2[x - dx]  ->  t2 col = t1 col - (W - s + dx)
        oy, ox = dy, W - s + dx
    else:
        nom = H * (1 - overlap)
        s = int(min(H, round(H - nom + search * H)))
        a, b = t1[H - s:, :], t2[:s, :]
        dy, dx = _phase_corr(a, b, int(search * H), int(search * W))
        oy, ox = H - s + dy, dx
    # NCC on the real overlap of t1 and t2 at (oy, ox)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(H, oy + H), min(W, ox + W)
    if y1 - y0 < 16 or x1 - x0 < 16:
        return oy, ox, 0.0
    ncc = _ncc(t1[y0:y1, x0:x1], t2[y0 - oy:y1 - oy, x0 - ox:x1 - ox])
    return oy, ox, ncc


def solve_positions(grid, offsets, nominal, min_ncc):
    """Global least squares: positions p_i with p_j - p_i = o_ij for all
    measured pairs (weight = ncc), nominal offsets with a tiny weight for
    unreliable pairs so the grid stays connected. One round of outlier
    removal (pairs disagreeing > 6 px with the solution)."""
    ids = sorted(grid)
    index = {t: k for k, t in enumerate(ids)}
    n = len(ids)

    def solve(use):
        rows, rhs_y, rhs_x, wts = [], [], [], []
        for (i, j), (oy, ox, c) in offsets.items():
            good = use.get((i, j), False)
            ny, nx = nominal[(i, j)]
            w = c if good else 0.02
            ty, tx = (oy, ox) if good else (ny, nx)
            r = np.zeros(n)
            r[index[i]], r[index[j]] = -1, 1
            rows.append(r * w)
            rhs_y.append(ty * w)
            rhs_x.append(tx * w)
        anchor = np.zeros(n)
        anchor[0] = 1
        A = np.vstack(rows + [anchor * 10])
        py = np.linalg.lstsq(A, np.array(rhs_y + [0]), rcond=None)[0]
        px = np.linalg.lstsq(A, np.array(rhs_x + [0]), rcond=None)[0]
        return py, px

    use = {k: v[2] >= min_ncc for k, v in offsets.items()}
    py, px = solve(use)
    changed = False
    for (i, j), (oy, ox, c) in offsets.items():
        if use[(i, j)]:
            ry = py[index[j]] - py[index[i]]
            rx = px[index[j]] - px[index[i]]
            if abs(ry - oy) > 6 or abs(rx - ox) > 6:
                use[(i, j)] = False
                changed = True
    if changed:
        py, px = solve(use)
    pos = {t: (int(round(py[index[t]])), int(round(px[index[t]]))) for t in ids}
    return pos, use


# ================================================================ blending ==
def _feather(h, w, ramp):
    y = np.minimum(np.arange(h) + 1, h - np.arange(h)).astype(np.float32)
    x = np.minimum(np.arange(w) + 1, w - np.arange(w)).astype(np.float32)
    return (np.minimum(y[:, None] / ramp, 1) * np.minimum(x[None, :] / ramp, 1)) + 1e-4


def flat_field(tiles):
    """Illumination profile from all tiles of a channel (median of smoothed
    tiles), normalised to its bright centre (99th percentile = 1): corrected
    pixels keep the intensity they would have in the middle of a tile."""
    from scipy import ndimage
    st = np.median(np.stack([ndimage.gaussian_filter(t, 25) for t in tiles]), axis=0)
    top = float(np.percentile(st, 99))
    st = np.maximum(st, top * 0.05)
    return st / top


def compose(tiles, pos, shape, ramp, trim, rows_cols):
    H, W = shape
    ys = [p[0] for p in pos.values()]
    xs = [p[1] for p in pos.values()]
    oy, ox = min(ys), min(xs)
    CH, CW = max(ys) - oy + H, max(xs) - ox + W
    acc = np.zeros((CH, CW), np.float32)
    wsum = np.zeros((CH, CW), np.float32)
    wt = _feather(H, W, ramp)
    for t, img in tiles.items():
        y, x = pos[t][0] - oy, pos[t][1] - ox
        acc[y:y + H, x:x + W] += img * wt
        wsum[y:y + H, x:x + W] += wt
    out = acc / np.maximum(wsum, 1e-6)
    out[wsum < 1e-3] = 0
    if trim and rows_cols:
        grid, R, C = rows_cols
        inv = {rc: t for t, rc in grid.items()}
        left = max(pos[inv[(r, 0)]][1] for r in range(R) if (r, 0) in inv) - ox
        right = min(pos[inv[(r, C - 1)]][1] for r in range(R) if (r, C - 1) in inv) - ox + W
        top = max(pos[inv[(0, c)]][0] for c in range(C) if (0, c) in inv) - oy
        bot = min(pos[inv[(R - 1, c)]][0] for c in range(C) if (R - 1, c) in inv) - oy + H
        if right - left > W and bot - top > H:
            out = out[top:bot, left:right]
    return out


# ================================================================ one group ==
def infer_grid(n, rows=None, cols=None):
    """BZ-X default: snake (row 0 left->right, row 1 right->left, ...)."""
    if not (rows and cols):
        r = int(math.floor(math.sqrt(n)))
        while r > 1 and n % r:
            r -= 1
        rows, cols = r, n // r
    grid = {}
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        grid[i + 1] = (r, c if r % 2 == 0 else cols - 1 - c)
    return grid, rows, cols


def stitch_group(g, opt):
    """Stitch one XY position. Returns a report row (dict)."""
    import tifffile
    t0 = time.time()
    rep = {"position": g["name"], "folder": g["dir"], "status": "", "note": "",
           "tiles": "", "grid": "", "layout_from": "", "pairs_ok": "", "median_ncc": "",
           "max_shift_px": "", "size_px": "", "um_per_px": "", "files": "", "seconds": ""}
    chans = {c: t for c, t in g["tiles"].items() if c != "Overlay"}
    rep["tiles"] = " ".join(f"{c}:{len(t)}" for c, t in sorted(chans.items()))
    if not chans:
        rep.update(status="SKIPPED", note="only overlay tiles")
        return rep
    n = max(len(t) for t in chans.values())
    n_grid = max(max(t) for t in chans.values())   # highest tile number = grid size
    if n < opt.min_tiles:
        rep.update(status="SKIPPED", note=f"only {n} tile(s) (minimum {opt.min_tiles})")
        return rep

    gci = find_gci(g)
    if gci and gci["tiles"]:
        grid = dict(gci["tiles"])
        R = max(r for r, _ in grid.values()) + 1
        C = max(c for _, c in grid.values()) + 1
        rep["layout_from"] = os.path.basename(gci["path"])
    else:
        grid, R, C = infer_grid(n_grid, *(gci["rows"], gci["cols"]) if gci else (None, None))
        rep["layout_from"] = "inferred (snake)" + (" + .gci grid" if gci else "")
    rep["grid"] = f"{R}x{C}"
    roles = channel_roles(g, gci)
    present = set().union(*[set(t) for t in chans.values()])
    missing = [i for i in grid if i not in present]
    if missing:
        rep.update(status="SKIPPED", note=f"{len(missing)} of {len(grid)} tiles missing "
                                          f"(e.g. {missing[:4]})")
        return rep

    # ---- registration on the brightfield (else the first fluorescence) ----
    reg_ch = next((c for c in sorted(chans) if roles[c]["bf"]), sorted(chans)[0])
    ref = {}
    shape = None
    for i in grid:
        p = chans[reg_ch].get(i)
        if p:
            img, _ = load_tile(p)
            shape = shape or img.shape
            ref[i] = _prep(img)
    H, W = shape
    inv = {rc: i for i, rc in grid.items()}
    offsets, nominal = {}, {}
    for (r, c), i in inv.items():
        for (dr, dc, d) in ((0, 1, "h"), (1, 0, "v")):
            j = inv.get((r + dr, c + dc))
            if j is None or i not in ref or j not in ref:
                continue
            nominal[(i, j)] = (0, round(W * (1 - opt.overlap))) if d == "h" else \
                              (round(H * (1 - opt.overlap)), 0)
            offsets[(i, j)] = pair_offset(ref[i], ref[j], d, opt.overlap, opt.search)
    pos, use = solve_positions(grid, offsets, nominal, opt.min_ncc)
    nccs = [v[2] for v in offsets.values()]
    nom_pos = {i: (rc[0] * round(H * (1 - opt.overlap)), rc[1] * round(W * (1 - opt.overlap)))
               for i, rc in grid.items()}
    p0 = pos[min(grid)]
    rep["pairs_ok"] = f"{sum(use.values())}/{len(use)}"
    rep["median_ncc"] = round(float(np.median(nccs)), 3) if nccs else ""
    rep["max_shift_px"] = int(max(max(abs(pos[i][0] - p0[0] - nom_pos[i][0]),
                                      abs(pos[i][1] - p0[1] - nom_pos[i][1])) for i in grid))
    if sum(use.values()) < max(1, len(use) // 2):
        rep["note"] = (f"only {sum(use.values())} of {len(use)} overlaps matched - "
                       "nominal positions used for the rest, check the seams")

    # ---- every channel with the same positions ----
    out_dir = os.path.join(opt.output, g["name"])
    os.makedirs(out_dir, exist_ok=True)
    nm = (gci or {}).get("nm_per_px") or (opt.nm_per_px or None)
    res_kw = {}
    if nm:
        px_per_cm = 1e7 / nm
        res_kw = dict(resolution=(px_per_cm, px_per_cm), resolutionunit="CENTIMETER")
        rep["um_per_px"] = round(nm / 1000.0, 5)
    fluo = [c for c in sorted(chans) if not roles[c]["bf"]]
    files = []
    stitched_rgb = {}
    for ch in sorted(chans):
        if any(i not in chans[ch] for i in grid):
            rep["note"] = (rep["note"] + "; " if rep["note"] else "") + \
                f"{ch} incomplete ({len(chans[ch])}/{len(grid)} tiles) - not stitched"
            continue
        tiles, color = {}, None
        for i in grid:
            img, col = load_tile(chans[ch][i])
            tiles[i] = img
            color = color or col
        if opt.flat_field_bf if roles[ch]["bf"] else opt.flat_field:
            ff = flat_field(list(tiles.values()))
            tiles = {i: t / ff for i, t in tiles.items()}
        img = compose(tiles, pos, (H, W), ramp=max(8, int(min(H, W) * opt.overlap / 2)),
                      trim=opt.trim, rows_cols=(grid, R, C))
        img8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
        if roles[ch]["bf"]:
            rgb_col, tag = (1, 1, 1), "brightfield"
        else:
            rgb_col = roles[ch]["color"] or color or (0, 1, 0)
            tag = "fluorescence" if len(fluo) == 1 else \
                  "fluorescence_" + ch + ("_" + roles[ch]["comment"] if roles[ch]["comment"] else "")
        rgb = np.zeros(img8.shape + (3,), np.uint8)
        for k in range(3):
            if rgb_col[k]:
                rgb[..., k] = img8
        stitched_rgb[ch] = (rgb, roles[ch]["bf"])
        path = os.path.join(out_dir, f"{g['name']}_{tag}.tif")
        tifffile.imwrite(path, rgb, photometric="rgb", compression="zlib",
                         software="stitch_bzx (cargo pipeline)", **res_kw)
        files.append(os.path.basename(path))
    if opt.overlay and len(stitched_rgb) > 1:
        acc = np.zeros_like(next(iter(stitched_rgb.values()))[0], dtype=np.float32)
        for rgb, bf in stitched_rgb.values():
            acc += rgb.astype(np.float32) * (0.6 if bf else 1.0)
        path = os.path.join(out_dir, f"{g['name']}_overlay.tif")
        tifffile.imwrite(path, np.clip(acc, 0, 255).astype(np.uint8), photometric="rgb",
                         compression="zlib", **res_kw)
        files.append(os.path.basename(path))
    shp = next(iter(stitched_rgb.values()))[0].shape
    rep["_positions"] = {i: (pos[i][0] - p0[0], pos[i][1] - p0[1]) for i in grid}
    rep.update(status="OK", size_px=f"{shp[1]}x{shp[0]}", files=" ".join(files),
               seconds=round(time.time() - t0, 1))
    return rep


def _run_one(args):
    g, opt = args
    try:
        return stitch_group(g, opt)
    except Exception as exc:   # one bad position must not stop the batch
        import traceback
        return {"position": g.get("name", "?"), "folder": g.get("dir", ""),
                "status": "FAILED", "note": f"{type(exc).__name__}: {exc}",
                "_tb": traceback.format_exc()}


# ===================================================================== main ==
def main(argv=None):
    ap = argparse.ArgumentParser(description="Stitch BZ-X tiles (brightfield + fluorescence).")
    ap.add_argument("parent", help="folder with the BZ-X tiles (searched with subfolders)")
    ap.add_argument("--output", help="results folder (default: <parent>_stitched)")
    ap.add_argument("--min-tiles", type=int, default=4,
                    help="skip and flag positions with fewer tiles (default 4)")
    ap.add_argument("--overlap", type=float, default=30.0,
                    help="nominal tile overlap in %% (BZ-X default 30)")
    ap.add_argument("--search", type=float, default=12.0,
                    help="search range around the nominal position, %% of a tile")
    ap.add_argument("--min-ncc", type=float, default=0.15,
                    help="minimum correlation for a measured overlap to be trusted")
    ap.add_argument("--flat-field", action="store_true",
                    help="correct vignetting of the FLUORESCENCE (changes intensities near "
                         "tile edges; off = like BZ-X)")
    ap.add_argument("--no-flat-field-bf", dest="flat_field_bf", action="store_false",
                    help="do not correct vignetting of the brightfield (on by default: it "
                         "only evens the background for segmentation)")
    ap.add_argument("--no-trim", dest="trim", action="store_false",
                    help="keep the ragged border instead of the fully covered rectangle")
    ap.add_argument("--overlay", action="store_true", help="also write a colour overlay")
    ap.add_argument("--nm-per-px", type=float, default=0.0,
                    help="calibration when there is no .gci file (nm per pixel)")
    ap.add_argument("--workers", type=int, default=0,
                    help="positions stitched in parallel (default: CPU cores - 1)")
    ap.add_argument("--force", action="store_true", help="redo positions already stitched")
    opt = ap.parse_args(argv)
    opt.overlap /= 100.0
    opt.search /= 100.0

    parent = os.path.abspath(opt.parent.strip().strip('"').rstrip("\\/"))
    if not os.path.isdir(parent):
        raise SystemExit(f"ERROR: not a folder: {parent}")
    opt.output = os.path.abspath(opt.output or parent + "_stitched")
    os.makedirs(opt.output, exist_ok=True)

    groups = find_groups(parent)
    groups = {k: g for k, g in groups.items()
              if not os.path.normcase(g["dir"]).startswith(os.path.normcase(opt.output))}
    if not groups:
        raise SystemExit("ERROR: no BZ-X tiles found (names like ..._XY01_00001_CH2.tif).")
    todo = sorted(groups.values(), key=lambda g: g["name"])
    if not opt.force:
        done = [g for g in todo if glob.glob(os.path.join(glob.escape(opt.output),
                                                          glob.escape(g["name"]), "*_*.tif"))]
        if done:
            print(f"{len(done)} position(s) already stitched - kept (use --force to redo)")
        todo = [g for g in todo if g not in done]
    print(f"Parent   : {parent}")
    print(f"Output   : {opt.output}")
    print(f"Positions: {len(groups)} found, {len(todo)} to stitch "
          f"(min. {opt.min_tiles} tiles, overlap {opt.overlap * 100:g} %"
          + (", vignetting corrected: brightfield" if opt.flat_field_bf else "")
          + (" + fluorescence" if opt.flat_field and opt.flat_field_bf else
             ", vignetting corrected: fluorescence" if opt.flat_field else "") + ")")
    print()

    workers = opt.workers or max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, max(1, len(todo)))
    reports = []
    t_all = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, (g, opt)): g for g in todo}
            for k, f in enumerate(as_completed(futs), 1):
                reports.append(_print_row(f.result(), k, len(todo)))
    else:
        for k, g in enumerate(todo, 1):
            reports.append(_print_row(_run_one((g, opt)), k, len(todo)))

    cols = ["position", "status", "note", "tiles", "grid", "layout_from", "pairs_ok",
            "median_ncc", "max_shift_px", "size_px", "um_per_px", "files", "seconds", "folder"]
    rp = os.path.join(opt.output, "stitch_report.csv")
    old = []
    if os.path.isfile(rp) and not opt.force:
        with open(rp, encoding="utf-8") as fh:
            new_names = {r["position"] for r in reports}
            old = [r for r in csv.DictReader(fh) if r.get("position") not in new_names]
    with open(rp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(old + reports, key=lambda r: r["position"]):
            w.writerow(r)

    ok = [r for r in reports if r["status"] == "OK"]
    flagged = [r for r in reports if r["status"] != "OK"]
    warn = [r for r in ok if r.get("note")]
    print()
    print("=" * 60)
    print(f"{len(ok)} stitched, {len(flagged)} not stitched in {time.time() - t_all:.0f} s")
    if flagged:
        print("FLAGGED - not stitched:")
        for r in flagged:
            print(f"   {r['position']}: {r['status']} - {r['note']}")
    if warn:
        print("CHECK the seams of:")
        for r in warn:
            print(f"   {r['position']}: {r['note']}")
    print(f"Report : {rp}")
    print(f"Output : {opt.output}   (use it as parent folder on the Run tab)")
    return 0


def _print_row(r, k, n):
    if r["status"] == "OK":
        print(f"[{k}/{n}] {r['position']}: OK  {r['grid']} tiles, {r['size_px']} px, "
              f"{r['pairs_ok']} overlaps matched (median r {r['median_ncc']}), "
              f"{r['seconds']} s" + (f"  NOTE: {r['note']}" if r.get("note") else ""), flush=True)
    else:
        print(f"[{k}/{n}] {r['position']}: {r['status']} - {r['note']}", flush=True)
        if r.get("_tb"):
            print(r["_tb"], flush=True)
    return r


if __name__ == "__main__":
    sys.exit(main())
