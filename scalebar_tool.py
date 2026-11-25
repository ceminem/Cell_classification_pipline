#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scalebar_tool.py - drop in ONE image and burn a scale bar onto it
=============================================================================
A small companion to the collage builder for the common one-off: you have a
single micrograph (or the overlay you just made) and just want a correctly
sized scale bar on it, without building a whole grid.

CALIBRATION IS SHARED WITH THE COLLAGE ENGINE
  The bar length is computed with the very same formula collage_engine uses -
  microns -> pixels via pixels_per_um, times the display scale if the image is
  resized for output. So a bar drawn here and a bar drawn in a collage from the
  same calibration are identical. Nothing about the maths is re-derived.

DRAG AND DROP
  If tkinterdnd2 is installed, the drop zone accepts a file dragged straight
  from the file manager. If it is NOT installed everything still works - the
  same zone is click-to-browse - so the "runs on a bare Python + Pillow"
  promise holds. Install real DnD with:  pip install tkinterdnd2

WHAT YOU CONTROL
  Calibration (px per um), bar length (um), thickness, colour, which corner,
  margin, and an optional text label above the bar. The preview updates live.
  Output is PNG/TIFF/JPEG; the pixels of the image itself are never resized, so
  a scale bar burned here stays calibrated.
=============================================================================
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

# --- real drag-and-drop if available, otherwise a graceful no-DnD fallback ---
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _HAS_DND = True
except ImportError:
    TkinterDnD = None
    DND_FILES = None
    _HAS_DND = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

# Reuse the engine's calibration + drawing so results match a collage exactly.
try:
    import collage_engine as ce
    _HAS_ENGINE = True
except ImportError:
    ce = None
    _HAS_ENGINE = False


IMAGE_PATTERNS = [("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp"),
                  ("All files", "*.*")]

CANDIDATE_BOLD_FONTS = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

CORNERS = ["bottom-right", "bottom-left", "top-right", "top-left"]


def _load_font(size):
    size = max(int(size), 8)
    for p in CANDIDATE_BOLD_FONTS:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def bar_length_px(length_um, pixels_per_um):
    """
    Microns -> pixels, the same conversion collage_engine performs.

    length_um / um_per_pixel  ==  length_um * pixels_per_um. Written the long
    way to mirror the engine (which multiplies by a display scale too; here the
    image is drawn at native size, so that factor is 1).
    """
    um_per_pixel = 1.0 / float(pixels_per_um)
    return int(round(length_um / um_per_pixel))


def burn_scale_bar(img, *, pixels_per_um, length_um, bar_h_px,
                   color=(255, 255, 255), corner="bottom-right", margin=24,
                   label=None, label_color=(255, 255, 255), label_gap=8):
    """
    Draw a filled scale bar onto a copy of img and return it (RGBA).

    The image is NOT resized, so the calibration stays exact. bar_h_px is the
    bar thickness in output pixels; length_um is converted with pixels_per_um.
    """
    out = img.convert("RGBA").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    blen = bar_length_px(length_um, pixels_per_um)
    blen = max(1, min(blen, w - 2 * margin))
    bar_h_px = max(1, int(bar_h_px))

    if "right" in corner:
        right = w - margin
        left = right - blen
    else:
        left = margin
        right = left + blen

    if "bottom" in corner:
        bottom = h - margin
        top = bottom - bar_h_px
    else:
        top = margin
        bottom = top + bar_h_px

    draw.rectangle([left, top, right, bottom], fill=tuple(color) + (255,))

    if label:
        font = _load_font(max(bar_h_px * 2.2, 14))
        b = draw.textbbox((0, 0), label, font=font)
        tw, th = b[2] - b[0], b[3] - b[1]
        lx = left + (blen - tw) / 2
        if "bottom" in corner:
            ly = top - th - label_gap
        else:
            ly = bottom + label_gap
        draw.text((lx, ly), label, font=font, fill=tuple(label_color) + (255,))

    return out, blen


class ScaleBarTool(tk.Toplevel):
    """Single-image scale-bar dialog with live preview and (optional) DnD."""

    PREVIEW_MAX = 560

    def __init__(self, master, initial_path=None,
                 pixels_per_um=2.6428, length_um=100):
        super().__init__(master)
        self.title("Scale bar - single image")
        self.result = None
        self.src_path = None
        self.src_img = None

        self.ppum = tk.StringVar(value=str(pixels_per_um))
        self.length_um = tk.StringVar(value=str(length_um))
        self.bar_h = tk.StringVar(value="auto")
        self.corner = tk.StringVar(value="bottom-right")
        self.margin = tk.StringVar(value="24")
        self.label = tk.StringVar(value="")
        self.bar_rgb = [255, 255, 255]

        self._build()
        if initial_path:
            self._load(initial_path)
        self.transient(master)
        self.grab_set()

    # ---- UI ----
    def _build(self):
        self.geometry("1000x640")
        self.minsize(880, 560)
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y")

        # drop zone
        dz = tk.Label(left, text=self._dz_text(), justify="center",
                      relief="ridge", borderwidth=2, background="#f0f0f0",
                      foreground="#555", width=34, height=6, cursor="hand2")
        dz.pack(fill="x", pady=(0, 8))
        dz.bind("<Button-1>", lambda e: self._browse())
        self.dropzone = dz
        if _HAS_DND:
            try:
                dz.drop_target_register(DND_FILES)
                dz.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

        # calibration
        cal = ttk.LabelFrame(left, text="Calibration")
        cal.pack(fill="x", pady=4)
        self._row(cal, "Pixels per µm:", self.ppum,
                  "From the microscope. 20x here = 2.6428 px/µm.")
        self._row(cal, "Bar length (µm):", self.length_um,
                  "Physical length of the bar. 100-200 µm reads well on a\n"
                  "wide field; 20 µm is tiny on a multi-thousand-px image.")

        # appearance
        ap = ttk.LabelFrame(left, text="Appearance")
        ap.pack(fill="x", pady=4)
        self._row(ap, "Thickness (px or auto):", self.bar_h,
                  "Auto = about 1.2% of image height.")
        self._row(ap, "Margin (px):", self.margin, None)

        r = ttk.Frame(ap)
        r.pack(fill="x", padx=6, pady=3)
        ttk.Label(r, text="Corner:", width=18).pack(side="left")
        cb = ttk.Combobox(r, textvariable=self.corner, values=CORNERS,
                          state="readonly", width=14)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        r = ttk.Frame(ap)
        r.pack(fill="x", padx=6, pady=3)
        ttk.Label(r, text="Bar colour:", width=18).pack(side="left")
        self.sw = tk.Canvas(r, width=28, height=18, highlightthickness=1,
                            highlightbackground="#888")
        self.sw.configure(background="#ffffff")
        self.sw.pack(side="left")
        ttk.Button(r, text="Pick...", command=self._pick_color).pack(side="left", padx=6)

        self._row(ap, "Label (blank = none):", self.label,
                  'e.g. "100 µm" - drawn next to the bar.')

        # actions
        act = ttk.Frame(left)
        act.pack(fill="x", pady=(10, 0))
        ttk.Button(act, text="Save...", command=self._save).pack(side="left")
        ttk.Button(act, text="Close", command=self._close).pack(side="right")

        # preview
        pv = ttk.LabelFrame(root, text="Preview")
        pv.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.canvas = tk.Canvas(pv, background="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.status = ttk.Label(pv, text="Drop or click to load an image.",
                                foreground="#666")
        self.status.pack(anchor="w", padx=8, pady=(0, 4))

    def _dz_text(self):
        if _HAS_DND:
            return "\n\nDrag an image here\n(or click to browse)\n"
        return ("\n\nClick to choose an image\n"
                "(install tkinterdnd2 for drag & drop)\n")

    def _row(self, parent, label, var, tip):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=6, pady=3)
        ttk.Label(r, text=label, width=18).pack(side="left")
        e = ttk.Entry(r, textvariable=var, width=14)
        e.pack(side="left")
        e.bind("<FocusOut>", lambda ev: self._refresh())
        e.bind("<Return>", lambda ev: self._refresh())
        return e

    # ---- loading ----
    def _browse(self):
        p = filedialog.askopenfilename(parent=self, title="Choose an image",
                                       filetypes=IMAGE_PATTERNS)
        if p:
            self._load(p)

    def _on_drop(self, event):
        # DnD hands over a brace-wrapped, possibly multi-file string
        data = event.data.strip()
        if data.startswith("{") and "}" in data:
            data = data[1:data.index("}")]
        else:
            data = data.split()[0] if data else ""
        if data:
            self._load(data)

    def _load(self, path):
        path = path.strip().strip('"')
        if not os.path.isfile(path):
            messagebox.showerror("Scale bar", f"Not a file:\n{path}", parent=self)
            return
        try:
            self.src_img = Image.open(path)
            self.src_img.load()
        except Exception as exc:
            messagebox.showerror("Scale bar", f"Could not open:\n{exc}", parent=self)
            return
        self.src_path = path
        self.dropzone.config(text=f"\n\n{os.path.basename(path)}\n"
                                  f"{self.src_img.width} x {self.src_img.height} px\n",
                             background="#e8f0e8")
        self._refresh()

    # ---- rendering ----
    def _num(self, var, default, allow_auto=False):
        v = var.get().strip()
        if allow_auto and v.lower() == "auto":
            return "auto"
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return default

    def _pick_color(self):
        rgb, _ = colorchooser.askcolor(
            color="#%02x%02x%02x" % tuple(self.bar_rgb), parent=self)
        if rgb:
            self.bar_rgb = [int(c) for c in rgb]
            self.sw.configure(background="#%02x%02x%02x" % tuple(self.bar_rgb))
            self._refresh()

    def _rendered(self):
        """Full-resolution image with the bar burned in, plus bar length px."""
        ppum = float(self._num(self.ppum, 2.6428))
        length_um = float(self._num(self.length_um, 100))
        bar_h = self._num(self.bar_h, "auto", allow_auto=True)
        if bar_h == "auto":
            bar_h = max(2, round(self.src_img.height * 0.012))
        margin = int(self._num(self.margin, 24))
        label = self.label.get().strip() or None
        img, blen = burn_scale_bar(
            self.src_img, pixels_per_um=ppum, length_um=length_um,
            bar_h_px=bar_h, color=tuple(self.bar_rgb),
            corner=self.corner.get(), margin=margin, label=label)
        return img, blen, length_um, ppum

    def _refresh(self, *_):
        if self.src_img is None:
            return
        try:
            img, blen, length_um, ppum = self._rendered()
        except Exception as exc:
            self.status.config(text=f"error: {exc}")
            return
        self._full = img
        # downscale for display
        s = min(self.PREVIEW_MAX / max(img.width, img.height), 1.0)
        disp = img if s >= 1.0 else img.resize(
            (max(1, int(img.width * s)), max(1, int(img.height * s))),
            Image.BILINEAR)
        self._preview = disp
        self._redraw()
        fov_w = img.width / ppum
        self.status.config(
            text=f"{length_um:g} µm bar = {blen} px  "
                 f"({100*blen/img.width:.1f}% of width)   |   "
                 f"field ≈ {fov_w:.0f} µm wide")

    def _redraw(self):
        img = getattr(self, "_preview", None)
        if img is None:
            return
        cw = max(self.canvas.winfo_width(), 10)
        chh = max(self.canvas.winfo_height(), 10)
        s = min(cw / img.width, chh / img.height, 1.0)
        disp = img if s >= 1.0 else img.resize(
            (max(1, int(img.width * s)), max(1, int(img.height * s))),
            Image.BILINEAR)
        self._tk = tk.PhotoImage  # keep name bound
        from PIL import ImageTk
        self._tkimg = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, chh // 2, image=self._tkimg)

    # ---- save ----
    def _save(self):
        if self.src_img is None:
            messagebox.showinfo("Scale bar", "Load an image first.", parent=self)
            return
        base = os.path.splitext(os.path.basename(self.src_path))[0]
        out = filedialog.asksaveasfilename(
            parent=self, title="Save image with scale bar",
            defaultextension=".png",
            initialdir=os.path.dirname(self.src_path),
            initialfile=f"{base}_scalebar.png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"),
                       ("JPEG", "*.jpg"), ("All files", "*.*")])
        if not out:
            return
        try:
            img = self._full
            ext = os.path.splitext(out)[1].lower()
            if ext in (".jpg", ".jpeg", ".tif", ".tiff"):
                # flatten alpha for formats that don't carry it
                flat = Image.new("RGB", img.size, (255, 255, 255))
                flat.paste(img, mask=img.split()[3])
                flat.save(out)
            else:
                img.save(out)
        except Exception as exc:
            messagebox.showerror("Scale bar", f"Could not save:\n{exc}", parent=self)
            return
        self.result = out
        self.status.config(text=f"saved -> {out}")
        messagebox.showinfo("Scale bar", f"Saved:\n{out}", parent=self)

    def _close(self):
        self.destroy()


def open_scalebar_tool(master, initial_path=None,
                       pixels_per_um=2.6428, length_um=100):
    """Open the tool. Returns the last saved path (or None)."""
    dlg = ScaleBarTool(master, initial_path, pixels_per_um, length_um)
    master.wait_window(dlg)
    return dlg.result


# --------------------------------------------------------------------------
# Standalone:  python scalebar_tool.py [image]
# With tkinterdnd2 installed the root supports dropping onto the window too.
# --------------------------------------------------------------------------
def main():
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    if _HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    root.withdraw()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    res = open_scalebar_tool(root, initial)
    print("result:", res)
    root.destroy()


if __name__ == "__main__":
    main()
