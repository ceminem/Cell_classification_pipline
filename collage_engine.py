#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collage_engine.py - figure assembly for microscopy panels
=============================================================================
A refactor of Collage_maker_automatic_iteration.py into an importable module,
so the same code serves both the GUI and any script.

WHAT CHANGED FROM THE ORIGINAL
  - all module-level CONFIG constants became a CollageConfig dataclass, so
    settings can be changed at runtime instead of by editing the file
  - added BLOCK LAYOUT: rows can be split into several blocks placed side by
    side (e.g. 8 rows as 4+4 columns) instead of one tall stack
  - added output format conversion (PNG / TIFF / JPEG / WEBP) independent of
    the input format
  - added render_preview() which builds a downscaled version quickly, for a
    live GUI preview without waiting on full-resolution assembly
  - added per-row and per-image overrides so a GUI can reorder, relabel or
    exclude individual items

The drawing logic itself - fit_cover / fit_contain, scale-bar maths, auto font
sizing, native cell sizing - is preserved from the original.
=============================================================================
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# files the analysis pipeline writes next to the raw images: never a panel
GENERATED_MARKERS = ("_cp_masks", "_brightfield_for_cellpose", "_classified_overlay",
                     "_cnn_overlay", "_bbox", "_seg.npy")

CANDIDATE_BOLD_FONTS = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Output formats the GUI offers. Value = (Pillow format, extension, has_alpha)
OUTPUT_FORMATS = {
    "PNG":  ("PNG",  ".png",  True),
    "TIFF": ("TIFF", ".tif",  False),
    "JPEG": ("JPEG", ".jpg",  False),
    "WEBP": ("WEBP", ".webp", True),
}


@dataclass
class CollageConfig:
    """Every knob the original script exposed, now settable at runtime."""

    # --- input ---
    main_folder: str = ""
    subfolder_order: Optional[list] = None
    label_overrides: dict = field(default_factory=dict)

    # --- layout ---
    # blocks_across: how many side-by-side blocks the rows are split into.
    # 1 = the original single tall stack. 2 with 8 rows = two columns of 4.
    blocks_across: int = 1
    block_gap: int = 80           # horizontal space between blocks (px)
    repeat_column_labels: bool = True   # draw column headers atop every block

    cell_w: object = "auto"       # int or "auto" (native source resolution)
    cell_h: object = "auto"
    fit_mode: str = "cover"       # "cover" (crop to fill) or "contain" (no crop)

    grid_gap: object = "auto"     # int or "auto"
    grid_gap_scale: float = 0.02
    outer_pad: int = 20

    # skip files written by the pipeline (masks, overlays) when scanning
    exclude_generated: bool = True
    # one linear contrast stretch per COLUMN (same for every row, so rows stay
    # comparable): makes dim fluorescence visible without per-image tricks
    column_contrast: bool = False
    contrast_low_pct: float = 0.1
    contrast_high_pct: float = 99.8

    # --- labels ---
    column_labels: list = field(default_factory=lambda: ["Brightfield", "Calcein", "Merge"])
    column_label_color: tuple = (255, 255, 255, 255)
    column_label_font_size: object = "auto"
    column_label_font_scale: float = 0.09
    column_label_margin: int = 12

    row_label_w: object = "auto"
    row_label_w_padding: int = 30
    row_label_font_size: object = "auto"
    row_label_font_scale: float = 0.09
    row_label_color: tuple = (20, 20, 20, 255)
    row_label_align: str = "right"

    # --- scale bar ---
    scale_bar_enabled: bool = True
    scale_bar_column: int = -1        # -1 = last column
    scale_bar_rows: object = "all"    # "all" or list of row indices
    scale_bar_length_um: float = 20
    scale_bar_pixels_per_um: float = 2.6428
    scale_bar_height_px: object = "auto"
    scale_bar_height_scale: float = 0.015
    scale_bar_margin: int = 20
    scale_bar_color: tuple = (255, 255, 255, 255)
    scale_bar_label: Optional[str] = None
    scale_bar_label_color: tuple = (255, 255, 255, 255)
    scale_bar_label_gap: int = 6
    scale_bar_label_font_size: object = "auto"
    scale_bar_label_font_scale: float = 0.06

    # --- output ---
    transparent_background: bool = True
    output_format: str = "PNG"
    jpeg_quality: int = 95
    dpi: tuple = (300, 300)
    font_path_override: Optional[str] = None

    @property
    def scale_bar_um_per_pixel(self):
        return 1.0 / self.scale_bar_pixels_per_um

    @property
    def background_color(self):
        if self.transparent_background and OUTPUT_FORMATS[self.output_format][2]:
            return (255, 255, 255, 0)
        return (255, 255, 255, 255)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ── helpers (preserved from the original) ──────────────────────────────────

def natural_key(s):
    """Sort key so 'A2' < 'A10' instead of 'A10' < 'A2'."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def discover_rows(cfg: CollageConfig):
    """Scan the main folder for subfolders and turn each into a row dict."""
    if not os.path.isdir(cfg.main_folder):
        raise FileNotFoundError(f"Folder does not exist: {cfg.main_folder}")

    all_subfolders = [n for n in os.listdir(cfg.main_folder)
                      if os.path.isdir(os.path.join(cfg.main_folder, n))]

    if cfg.subfolder_order:
        names = [n for n in cfg.subfolder_order if n in all_subfolders]
    else:
        names = sorted(all_subfolders, key=natural_key)

    rows = []
    for name in names:
        folder = os.path.join(cfg.main_folder, name)
        images = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(IMAGE_EXTENSIONS)
             and not (cfg.exclude_generated and
                      any(m in f.lower() for m in GENERATED_MARKERS))],
            key=lambda p: natural_key(os.path.basename(p)),
        )
        if not images:
            continue
        rows.append({
            "name": name,
            "label": cfg.label_overrides.get(name, name),
            "images": images,
        })

    if not rows:
        raise ValueError("No subfolders containing images were found.")
    return rows


def file_px_per_um(path):
    """Calibration stored in an image file (px per um), or 0.

    TIFF resolution in pixels per cm (stitch_bzx.py, Fiji, most microscope
    software) or ImageJ metadata with unit micron. The 72-dpi default many
    programs write - BZ-X stitched images too - is not a calibration and is
    ignored. Pillow only, so the collage still runs on a bare Python."""
    if not path.lower().endswith((".tif", ".tiff")):
        return 0.0
    try:
        with Image.open(path) as im:
            tags = getattr(im, "tag_v2", {}) or {}
            xr = tags.get(282)
            unit = tags.get(296)
            desc = str(tags.get(270, ""))
        if not xr:
            return 0.0
        v = float(xr)
        if re.search(r"unit=(micron|um|\u00b5m)", desc):
            px = v
        elif unit is not None and int(unit) == 3:             # centimetre
            px = v / 1e4
        else:
            return 0.0
        return px if 0.05 <= px <= 50 else 0.0
    except Exception:
        return 0.0


_LIMIT_CACHE = {}


def _percentile_from_hist(hist, pct):
    total = sum(hist)
    if not total:
        return 0
    want = total * pct / 100.0
    run = 0
    for i, n in enumerate(hist):
        run += n
        if run >= want:
            return i
    return len(hist) - 1


def column_luts(rows, cfg):
    """Per column a lookup table per RGB band: the column's low/high percentile
    (over ALL its images together) stretched to 0..255. Identical for every row,
    so brightness differences between conditions stay as recorded."""
    ncol = max((len(r["images"]) for r in rows), default=0)
    luts = []
    for c in range(ncol):
        paths = tuple(r["images"][c] for r in rows if c < len(r["images"]))
        key = (paths, tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths),
               cfg.contrast_low_pct, cfg.contrast_high_pct)
        if key not in _LIMIT_CACHE:
            hist = [[0] * 256 for _ in range(3)]
            for p in paths:
                try:
                    with Image.open(p) as im:
                        im.draft("RGB", (1000, 1000))
                        im = im.convert("RGB")
                        if max(im.size) > 1200:
                            im = im.reduce(max(1, max(im.size) // 1000))
                        h = im.histogram()
                    for b in range(3):
                        hb = h[b * 256:(b + 1) * 256]
                        hist[b] = [x + y for x, y in zip(hist[b], hb)]
                except OSError:
                    continue
            lims = []
            for b in range(3):
                lo = _percentile_from_hist(hist[b], cfg.contrast_low_pct)
                hi = _percentile_from_hist(hist[b], cfg.contrast_high_pct)
                lims.append((lo, hi))
            _LIMIT_CACHE[key] = lims
        lut = []
        for lo, hi in _LIMIT_CACHE[key]:
            if hi <= lo or hi == 0:
                lut += list(range(256))                    # empty band: unchanged
            else:
                lut += [max(0, min(255, round((v - lo) * 255.0 / (hi - lo))))
                        for v in range(256)]
        luts.append(lut)
    return luts


def load_bold_font(size, override=None):
    candidates = ([override] if override else []) + CANDIDATE_BOLD_FONTS
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def fit_contain(img, max_w, max_h):
    """Resize to fit within the cell, preserving aspect, no crop, no upscale."""
    sw, sh = img.size
    scale = min(max_w / sw, max_h / sh, 1.0)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    return img.copy() if (nw, nh) == (sw, sh) else img.resize((nw, nh), Image.LANCZOS)


def fit_cover(img, tw, th):
    """Resize + centre-crop so the image completely fills the cell."""
    sw, sh = img.size
    scale = max(tw / sw, th / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def compute_display_scale(sw, sh, tw, th, mode):
    """Effective resize factor, needed to convert a physical scale-bar length
    calibrated on the SOURCE image into output pixels."""
    if mode == "cover":
        return max(tw / sw, th / sh)
    return min(tw / sw, th / sh, 1.0)


def text_size(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def native_cell_size(paths):
    """Largest (w, h) among a row's images, so detail is preserved."""
    mw = mh = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
                mw, mh = max(mw, w), max(mh, h)
        except (FileNotFoundError, OSError):
            continue
    return (mw, mh) if mw and mh else (300, 300)


def draw_scale_bar(draw, cfg, cx, cy, cw, ch, bar_len, bar_h, font, label):
    """Filled bar in the bottom-right of a cell, optional label above it."""
    bar_len = max(1, min(bar_len, cw - 2 * cfg.scale_bar_margin))
    right = cx + cw - cfg.scale_bar_margin
    left = right - bar_len
    bottom = cy + ch - cfg.scale_bar_margin
    top = bottom - bar_h
    draw.rectangle([left, top, right, bottom], fill=cfg.scale_bar_color)
    if label:
        tw, th = text_size(draw, label, font)
        # centred over the bar, but never past the cell edge (a label wider
        # than a short bar used to run out of the image)
        lx = left + (bar_len - tw) / 2
        lx = max(cx + cfg.scale_bar_margin, min(lx, cx + cw - cfg.scale_bar_margin - tw))
        draw.text((lx, top - th - cfg.scale_bar_label_gap),
                  label, font=font, fill=cfg.scale_bar_label_color)


# ── layout ─────────────────────────────────────────────────────────────────

def split_into_blocks(rows, blocks_across):
    """
    Split rows into side-by-side blocks.

    With 8 rows and blocks_across=2 you get two blocks of 4, drawn next to each
    other - the "4 under each other and the other 4 beside it" layout. Rows are
    dealt in order, so block 1 holds rows 1-4 and block 2 holds rows 5-8.
    """
    n = max(1, int(blocks_across))
    if n == 1:
        return [rows]
    per = -(-len(rows) // n)          # ceiling division
    return [rows[i:i + per] for i in range(0, len(rows), per)] or [rows]


def build_row_info(rows, cfg, scale=1.0):
    """Resolve per-row cell size and column count. scale<1 shrinks for preview."""
    info = []
    for r in rows:
        cw, ch = cfg.cell_w, cfg.cell_h
        if cw == "auto" or ch == "auto":
            aw, ah = native_cell_size(r["images"])
            cw = aw if cw == "auto" else cw
            ch = ah if ch == "auto" else ch
        cw = max(1, int(round(float(cw) * scale)))
        ch = max(1, int(round(float(ch) * scale)))
        info.append({"label": r["label"], "images": r["images"],
                     "cell_w": cw, "cell_h": ch, "cols": len(r["images"])})
    return info


def render(cfg: CollageConfig, scale=1.0, rows=None, progress=None):
    """
    Assemble the figure and return it as a PIL image.

    scale < 1 renders a smaller version - used for the live preview so the GUI
    stays responsive instead of assembling full-resolution images on every
    keystroke.
    """
    rows = rows if rows is not None else discover_rows(cfg)
    blocks = split_into_blocks(rows, cfg.blocks_across)
    luts = column_luts(rows, cfg) if cfg.column_contrast else None

    pad = max(1, int(round(cfg.outer_pad * scale)))
    block_gap = max(1, int(round(cfg.block_gap * scale)))

    font_cache = {}

    def get_font(size):
        size = max(int(size), 8)
        if size not in font_cache:
            font_cache[size] = load_bold_font(size, cfg.font_path_override)
        return font_cache[size]

    # --- resolve each block's geometry ---
    block_layouts = []
    for rows_in_block in blocks:
        ri = build_row_info(rows_in_block, cfg, scale)
        if not ri:
            continue

        avg_w = sum(r["cell_w"] for r in ri) / len(ri)
        avg_h = sum(r["cell_h"] for r in ri) / len(ri)
        min_gap = max(2, int(round(10 * scale)))
        col_gap = (max(round(avg_w * cfg.grid_gap_scale), min_gap)
                   if cfg.grid_gap == "auto" else max(1, int(round(float(cfg.grid_gap) * scale))))
        row_gap = (max(round(avg_h * cfg.grid_gap_scale), min_gap)
                   if cfg.grid_gap == "auto" else max(1, int(round(float(cfg.grid_gap) * scale))))

        for r in ri:
            fs = (cfg.row_label_font_size if cfg.row_label_font_size != "auto"
                  else r["cell_h"] * cfg.row_label_font_scale)
            r["row_font"] = get_font(fs)
            r["bar_h"] = (int(cfg.scale_bar_height_px * scale)
                          if cfg.scale_bar_height_px != "auto"
                          else max(2, round(r["cell_h"] * cfg.scale_bar_height_scale)))
            bfs = (cfg.scale_bar_label_font_size if cfg.scale_bar_label_font_size != "auto"
                   else r["cell_h"] * cfg.scale_bar_label_font_scale)
            r["bar_font"] = get_font(bfs)

        # row label column width
        if cfg.row_label_w == "auto":
            probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
            longest = max((text_size(probe, r["label"], r["row_font"])[0] for r in ri), default=0)
            label_w = longest + int(round(cfg.row_label_w_padding * scale)) * 2
        else:
            label_w = int(round(float(cfg.row_label_w) * scale))

        grid_w = max(r["cols"] * r["cell_w"] + max(r["cols"] - 1, 0) * col_gap for r in ri)
        block_w = label_w + grid_w
        block_h = sum(r["cell_h"] for r in ri) + row_gap * (len(ri) - 1)

        block_layouts.append({"rows": ri, "label_w": label_w, "col_gap": col_gap,
                              "row_gap": row_gap, "w": block_w, "h": block_h})

    if not block_layouts:
        raise ValueError("Nothing to render.")

    total_w = pad * 2 + sum(b["w"] for b in block_layouts) + block_gap * (len(block_layouts) - 1)
    total_h = pad * 2 + max(b["h"] for b in block_layouts)

    canvas = Image.new("RGBA", (total_w, total_h), cfg.background_color)
    draw = ImageDraw.Draw(canvas)

    col_font = get_font(cfg.column_label_font_size if cfg.column_label_font_size != "auto"
                        else block_layouts[0]["rows"][0]["cell_h"] * cfg.column_label_font_scale)
    col_margin = max(2, int(round(cfg.column_label_margin * scale)))

    total_cells = sum(sum(r["cols"] for r in b["rows"]) for b in block_layouts)
    done = 0

    bx = pad
    for b_idx, block in enumerate(block_layouts):
        y = pad
        for row_idx, r in enumerate(block["rows"]):
            cw, ch = r["cell_w"], r["cell_h"]

            # global row index, so scale_bar_rows refers to the original order
            global_row = sum(len(bb["rows"]) for bb in block_layouts[:b_idx]) + row_idx
            wants_bar = cfg.scale_bar_enabled and (
                cfg.scale_bar_rows == "all" or global_row in (cfg.scale_bar_rows or []))
            target_col = (r["cols"] + cfg.scale_bar_column
                          if cfg.scale_bar_column < 0 else cfg.scale_bar_column)

            # row label
            tw, th = text_size(draw, r["label"], r["row_font"])
            x0 = bx
            x1 = bx + block["label_w"] - max(2, int(round(12 * scale)))
            if cfg.row_label_align == "right":
                lx = x1 - tw
            elif cfg.row_label_align == "center":
                lx = x0 + (block["label_w"] - tw) // 2
            else:
                lx = x0
            draw.text((lx, y + (ch - th) // 2), r["label"],
                      font=r["row_font"], fill=cfg.row_label_color)

            x = bx + block["label_w"]
            for col_idx, fpath in enumerate(r["images"]):
                sw = sh = None
                try:
                    img = Image.open(fpath)
                    # Record the TRUE source size before any draft downscale -
                    # the scale-bar maths is calibrated against the original
                    # pixel size and would be wrong if taken after drafting.
                    sw, sh = img.size
                    if scale < 1.0:
                        # draft() decodes JPEGs directly at a reduced size,
                        # which is far cheaper than full decode + resize. It is
                        # a no-op for formats that do not support it, so this is
                        # always safe. Only used for previews.
                        img.draft("RGB", (max(1, cw), max(1, ch)))
                    img = img.convert("RGB") if img.mode != "RGB" else img
                    if luts and col_idx < len(luts):
                        img = img.point(luts[col_idx])
                    fitted = (fit_cover(img, cw, ch) if cfg.fit_mode == "cover"
                              else fit_contain(img, cw, ch)).convert("RGBA")
                    if cfg.fit_mode == "contain":
                        ox = x + (cw - fitted.width) // 2
                        oy = y + (ch - fitted.height) // 2
                    else:
                        ox, oy = x, y
                    canvas.paste(fitted, (ox, oy), fitted)
                except (FileNotFoundError, OSError):
                    draw.rectangle([x, y, x + cw, y + ch], fill=(60, 60, 60, 255))

                # column headers: first row of the first block, or of every
                # block when repeat_column_labels is on
                show_headers = row_idx == 0 and (cfg.repeat_column_labels or b_idx == 0)
                if cfg.column_labels and show_headers and col_idx < len(cfg.column_labels):
                    draw.text((x + col_margin, y + col_margin),
                              cfg.column_labels[col_idx],
                              font=col_font, fill=cfg.column_label_color)

                if wants_bar and col_idx == target_col and sw:
                    ds = compute_display_scale(sw, sh, cw, ch, cfg.fit_mode)
                    bar_len = round((cfg.scale_bar_length_um / cfg.scale_bar_um_per_pixel) * ds)
                    draw_scale_bar(draw, cfg, x, y, cw, ch, bar_len,
                                   r["bar_h"], r["bar_font"], cfg.scale_bar_label)

                x += cw + block["col_gap"]
                done += 1
                if progress and total_cells:
                    progress(done / total_cells)

            y += ch + block["row_gap"]
        bx += block["w"] + block_gap

    return canvas


def render_preview(cfg: CollageConfig, max_px=900, rows=None):
    """Small, fast render for the GUI preview. Returns (image, scale_used)."""
    probe_rows = rows if rows is not None else discover_rows(cfg)
    ri = build_row_info(probe_rows, cfg, 1.0)
    if not ri:
        raise ValueError("Nothing to preview.")
    blocks = split_into_blocks(probe_rows, cfg.blocks_across)
    per_block = -(-len(ri) // max(1, len(blocks)))
    est_w = sum(max(r["cols"] * r["cell_w"] for r in ri) for _ in blocks)
    est_h = sum(r["cell_h"] for r in ri[:per_block])
    scale = min(1.0, max_px / max(est_w, est_h, 1))
    scale = max(scale, 0.02)
    return render(cfg, scale=scale, rows=probe_rows), scale


def save(canvas, path, cfg: CollageConfig):
    """Write the canvas in the configured format, handling alpha correctly."""
    fmt, ext, has_alpha = OUTPUT_FORMATS[cfg.output_format]
    if not path.lower().endswith(ext):
        path = os.path.splitext(path)[0] + ext

    if has_alpha and cfg.transparent_background:
        if fmt == "WEBP":
            canvas.save(path, format=fmt, lossless=True)
        else:
            canvas.save(path, format=fmt, dpi=cfg.dpi)
    else:
        # flatten onto white - JPEG and TIFF do not carry usable alpha
        flat = Image.new("RGB", canvas.size, (255, 255, 255))
        flat.paste(canvas, mask=canvas.split()[3])
        if fmt == "JPEG":
            flat.save(path, format=fmt, quality=cfg.jpeg_quality,
                      subsampling=0, dpi=cfg.dpi)
        elif fmt == "TIFF":
            flat.save(path, format=fmt, compression="tiff_lzw", dpi=cfg.dpi)
        else:
            flat.save(path, format=fmt, dpi=cfg.dpi)
    return path


def make_collage(cfg: CollageConfig, out_path=None, progress=None):
    """Full-resolution render + save. Returns the path written."""
    canvas = render(cfg, scale=1.0, progress=progress)
    if not out_path:
        ext = OUTPUT_FORMATS[cfg.output_format][1]
        out_path = os.path.join(cfg.main_folder, "collage" + ext)
    return save(canvas, out_path, cfg), canvas.size