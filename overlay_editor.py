#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
overlay_editor.py - interactive two-channel overlay with a live preview
=============================================================================
A Tk dialog for merging two images into a colour overlay, with per-channel
brightness / contrast (black & white point), gamma and colour - all updating
the preview in real time.

WHY NOT DO THE LIVE PART IN FIJI
  Every Fiji call is a fresh subprocess with a ~20-40 s cold start, so it
  cannot drive a live preview. The interactive compositing here is done with
  NumPy + Pillow, which is instant. Fiji is still available for the FINAL
  render if you want its exact Merge Channels output - the dialog can either
  save the NumPy overlay directly (fast, identical to what you see) or hand
  the adjusted channels to fiji_merge for a Fiji-produced file.

WHAT EACH CONTROL DOES  (per channel, A and B independently)
  Black / White point : the input intensity range that maps to 0..full. This
                        is the same operation as ImageJ's Brightness/Contrast
                        "Minimum" and "Maximum" sliders - it stretches the
                        display range. Narrowing it raises contrast.
  Brightness          : a linear gain applied after the range stretch.
  Gamma               : non-linear mid-tone adjustment (<1 brightens mids,
                        >1 darkens them).
  Colour              : which additive colour the channel is tinted with.
  Visible             : toggle the channel in/out of the overlay.

The overlay is additive (screen-like): each channel is tinted and the tints
are summed then clipped, which is how fluorescence overlays combine. Green +
magenta co-localisation therefore trends toward white.
=============================================================================
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, colorchooser
except ImportError:
    sys.exit("ERROR: tkinter missing. On Linux: sudo apt install python3-tk")

from PIL import Image, ImageTk


# Additive tint colours, as 0..1 RGB weights.
NAMED_COLORS = {
    "red":     (1.0, 0.0, 0.0),
    "green":   (0.0, 1.0, 0.0),
    "blue":    (0.0, 0.0, 1.0),
    "cyan":    (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow":  (1.0, 1.0, 0.0),
    "grey":    (1.0, 1.0, 1.0),
    "gray":    (1.0, 1.0, 1.0),
}


def load_gray(path):
    """
    Load an image as a float32 array normalised to 0..1, preserving the true
    dynamic range of the source.

    16-bit microscopy images are scaled by their actual bit usage, not a flat
    /65535, so a 12-bit camera image (max ~4095) does not come out nearly
    black. RGB inputs are reduced to luminance.
    """
    im = Image.open(path)
    # keep the original mode's depth; convert colour to a single channel
    if im.mode in ("RGB", "RGBA"):
        arr = np.asarray(im.convert("L"), dtype=np.float32)
        return arr / 255.0
    if im.mode == "I;16" or im.mode == "I":
        arr = np.asarray(im, dtype=np.float32)
    else:
        arr = np.asarray(im.convert("F"), dtype=np.float32)

    mx = float(arr.max()) if arr.size else 1.0
    if mx <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    # scale by the smallest power-of-two range that contains the data, so an
    # 8/12/16-bit image each maps sensibly to 0..1
    for depth in (255.0, 4095.0, 65535.0):
        if mx <= depth:
            return np.clip(arr / depth, 0.0, 1.0)
    return np.clip(arr / mx, 0.0, 1.0)


def apply_channel(arr, black, white, brightness, gamma):
    """
    Apply display-range stretch, gamma and brightness to a 0..1 array.

    black/white are 0..1 input cut points (black<white). The range between
    them is stretched to 0..1, gamma reshapes the mid-tones, brightness is a
    final linear gain. Result is clipped to 0..1.
    """
    black = float(black)
    white = float(white)
    if white <= black:
        white = black + 1e-4
    out = (arr - black) / (white - black)
    out = np.clip(out, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        out = np.power(out, gamma, dtype=np.float32)
    out = out * float(brightness)
    return np.clip(out, 0.0, 1.0)


def composite(channels):
    """
    Additively combine tinted channels into an RGB uint8 image.

    channels : list of dicts, each with
        "data"   -> processed 0..1 array (already through apply_channel)
        "color"  -> 0..1 RGB weight tuple
        "visible"-> bool
    Arrays must share a shape. Returns an HxWx3 uint8 array.
    """
    vis = [c for c in channels if c.get("visible", True)]
    if not vis:
        h, w = channels[0]["data"].shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    h, w = vis[0]["data"].shape
    acc = np.zeros((h, w, 3), dtype=np.float32)
    for c in vis:
        d = c["data"]
        cr, cg, cb = c["color"]
        acc[..., 0] += d * cr
        acc[..., 1] += d * cg
        acc[..., 2] += d * cb
    acc = np.clip(acc, 0.0, 1.0)
    return (acc * 255.0 + 0.5).astype(np.uint8)


def match_shapes(a, b):
    """
    If the two channel arrays differ in size, resize the second to the first
    with nearest sampling (so a mask or lower-res channel still overlays).
    Returns (a, b) both at a's shape.
    """
    if a.shape == b.shape:
        return a, b
    ph, pw = a.shape
    im = Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8))
    im = im.resize((pw, ph), Image.NEAREST)
    return a, np.asarray(im, dtype=np.float32) / 255.0


class OverlayEditor(tk.Toplevel):
    """
    Modal-ish overlay editor. Construct with two image paths; on save it writes
    the file and stores the path in self.result (None if cancelled).
    """

    PREVIEW_MAX = 520   # longest preview edge, px - keeps sliders instant

    def __init__(self, master, path_a, path_b,
                 color_a="green", color_b="magenta", fiji_merge_module=None):
        super().__init__(master)
        self.title("Overlay editor")
        self.result = None
        self._fiji = fiji_merge_module
        self.path_a = path_a
        self.path_b = path_b

        # load once at full res, keep a downscaled copy for live preview
        raw_a = load_gray(path_a)
        raw_b = load_gray(path_b)
        raw_a, raw_b = match_shapes(raw_a, raw_b)
        self.full_a, self.full_b = raw_a, raw_b
        self.prev_a = self._downscale(raw_a)
        self.prev_b = self._downscale(raw_b)

        # per-channel state
        self.ch = [
            self._new_channel_state(color_a, os.path.basename(path_a)),
            self._new_channel_state(color_b, os.path.basename(path_b)),
        ]

        self._build()
        self._update_preview()
        self.transient(master)
        self.grab_set()

    # ---- state ----
    def _new_channel_state(self, color, name):
        color = color.lower()
        rgb = NAMED_COLORS.get(color, (1.0, 1.0, 1.0))
        return {
            "name": name,
            "black": tk.DoubleVar(value=0.0),
            "white": tk.DoubleVar(value=1.0),
            "bright": tk.DoubleVar(value=1.0),
            "gamma": tk.DoubleVar(value=1.0),
            "color": color,
            "color_rgb": rgb,
            "visible": tk.BooleanVar(value=True),
        }

    def _downscale(self, arr):
        h, w = arr.shape
        s = min(self.PREVIEW_MAX / max(h, w), 1.0)
        if s >= 1.0:
            return arr
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

    # ---- UI ----
    def _build(self):
        self.geometry("980x620")
        self.minsize(880, 560)

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=8, pady=8)

        controls = ttk.Frame(root)
        controls.pack(side="left", fill="y")

        for idx, c in enumerate(self.ch):
            self._channel_box(controls, idx, c)

        # output row
        out = ttk.LabelFrame(controls, text="Output")
        out.pack(fill="x", pady=(8, 0))
        self.use_fiji = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(
            out, text="Final render through Fiji (slower, exact Merge Channels)",
            variable=self.use_fiji)
        cb.pack(anchor="w", padx=6, pady=3)
        if self._fiji is None:
            cb.state(["disabled"])
        row = ttk.Frame(out)
        row.pack(fill="x", padx=6, pady=6)
        ttk.Button(row, text="Save overlay...", command=self._save).pack(side="left")
        ttk.Button(row, text="Reset", command=self._reset).pack(side="left", padx=6)
        ttk.Button(row, text="Cancel", command=self._cancel).pack(side="right")

        # preview pane
        pv = ttk.LabelFrame(root, text="Live overlay")
        pv.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.canvas = tk.Canvas(pv, background="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.status = ttk.Label(pv, text="", foreground="#666")
        self.status.pack(anchor="w", padx=8, pady=(0, 4))

    def _channel_box(self, parent, idx, c):
        letter = "A" if idx == 0 else "B"
        box = ttk.LabelFrame(parent, text=f"Channel {letter}  -  {c['name'][:28]}")
        box.pack(fill="x", pady=4)

        top = ttk.Frame(box)
        top.pack(fill="x", padx=6, pady=3)
        ttk.Checkbutton(top, text="Visible", variable=c["visible"],
                        command=self._update_preview).pack(side="left")
        self._color_swatch(top, c)

        self._slider(box, c, "black", "Black point", 0.0, 1.0)
        self._slider(box, c, "white", "White point", 0.0, 1.0)
        self._slider(box, c, "bright", "Brightness", 0.0, 3.0)
        self._slider(box, c, "gamma", "Gamma", 0.2, 3.0)

        ttk.Button(box, text="Auto contrast",
                   command=lambda i=idx: self._auto(i)).pack(anchor="w",
                                                             padx=6, pady=(0, 4))

    def _color_swatch(self, parent, c):
        sw = tk.Canvas(parent, width=26, height=16, highlightthickness=1,
                       highlightbackground="#888")
        rgb255 = tuple(int(v * 255) for v in c["color_rgb"])
        sw.configure(background="#%02x%02x%02x" % rgb255)
        sw.pack(side="left", padx=(10, 4))

        def pick():
            initial = "#%02x%02x%02x" % tuple(int(v * 255) for v in c["color_rgb"])
            rgb, _ = colorchooser.askcolor(color=initial, parent=self)
            if rgb:
                c["color_rgb"] = tuple(v / 255.0 for v in rgb)
                sw.configure(background="#%02x%02x%02x" % tuple(int(v) for v in rgb))
                self._update_preview()

        ttk.Button(parent, text="Colour...", command=pick).pack(side="left")

    def _slider(self, parent, c, key, label, lo, hi):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text=label, width=12).pack(side="left")
        val = ttk.Label(r, text=f"{c[key].get():.2f}", width=5)
        val.pack(side="right")

        def on_move(_=None):
            val.config(text=f"{c[key].get():.2f}")
            self._update_preview()

        s = ttk.Scale(r, from_=lo, to=hi, variable=c[key],
                      orient="horizontal", command=on_move)
        s.pack(side="left", fill="x", expand=True, padx=4)

    # ---- processing ----
    def _auto(self, idx):
        """Set black/white to the 0.35 / 99.65 percentiles of the channel."""
        arr = self.full_a if idx == 0 else self.full_b
        lo = float(np.percentile(arr, 0.35))
        hi = float(np.percentile(arr, 99.65))
        if hi <= lo:
            hi = lo + 1e-3
        self.ch[idx]["black"].set(round(lo, 4))
        self.ch[idx]["white"].set(round(hi, 4))
        self.ch[idx]["bright"].set(1.0)
        self.ch[idx]["gamma"].set(1.0)
        self._update_preview()

    def _processed(self, preview=True):
        """Return channel dicts ready for composite(), at preview or full res."""
        a = self.prev_a if preview else self.full_a
        b = self.prev_b if preview else self.full_b
        out = []
        for arr, c in zip((a, b), self.ch):
            data = apply_channel(arr, c["black"].get(), c["white"].get(),
                                 c["bright"].get(), c["gamma"].get())
            out.append({"data": data, "color": c["color_rgb"],
                        "visible": c["visible"].get()})
        return out

    def _update_preview(self, *_):
        rgb = composite(self._processed(preview=True))
        self._preview_pil = Image.fromarray(rgb, "RGB")
        self._redraw()
        self.status.config(text=f"preview {rgb.shape[1]} x {rgb.shape[0]} px  -  "
                                f"full {self.full_a.shape[1]} x {self.full_a.shape[0]} px")

    def _redraw(self):
        img = getattr(self, "_preview_pil", None)
        if img is None:
            return
        cw = max(self.canvas.winfo_width(), 10)
        chh = max(self.canvas.winfo_height(), 10)
        s = min(cw / img.width, chh / img.height, 1.0)
        disp = img if s >= 1.0 else img.resize(
            (max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, chh // 2, image=self._tk_img)

    def _reset(self):
        for c in self.ch:
            c["black"].set(0.0)
            c["white"].set(1.0)
            c["bright"].set(1.0)
            c["gamma"].set(1.0)
            c["visible"].set(True)
        self._update_preview()

    # ---- save ----
    def _save(self):
        out = filedialog.asksaveasfilename(
            parent=self, title="Save overlay as",
            defaultextension=".png",
            initialdir=os.path.dirname(self.path_a),
            initialfile="overlay.png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("All files", "*.*")])
        if not out:
            return

        if self.use_fiji.get() and self._fiji is not None:
            self._save_via_fiji(out)
            return

        # direct NumPy save - identical to the live preview, full resolution
        rgb = composite(self._processed(preview=False))
        try:
            Image.fromarray(rgb, "RGB").save(out)
        except Exception as exc:
            messagebox.showerror("Overlay editor", f"Could not save:\n{exc}",
                                 parent=self)
            return
        self.result = out
        messagebox.showinfo("Overlay editor", f"Saved:\n{out}", parent=self)
        self.destroy()

    def _save_via_fiji(self, out):
        """
        Write the range-adjusted channels to temp 8-bit PNGs, then let Fiji's
        Merge Channels build the final overlay from those. This keeps the
        adjustments you dialled in while giving a Fiji-produced file.
        """
        import tempfile
        proc = self._processed(preview=False)
        tmp_paths = []
        try:
            for tag, c in zip(("A", "B"), proc):
                # bake each channel to greyscale intensity for Fiji to tint
                g = (np.clip(c["data"], 0, 1) * 255 + 0.5).astype(np.uint8)
                tf = tempfile.NamedTemporaryFile(suffix=f"_{tag}.png", delete=False)
                tf.close()
                Image.fromarray(g, "L").save(tf.name)
                tmp_paths.append(tf.name)

            # map each channel's chosen colour back to the nearest Fiji slot name
            ca = self._nearest_named(self.ch[0]["color_rgb"])
            cb = self._nearest_named(self.ch[1]["color_rgb"])
            self._fiji.merge_two(tmp_paths[0], tmp_paths[1], out,
                                 color_a=ca, color_b=cb)
            self.result = out
            messagebox.showinfo("Overlay editor",
                                f"Saved (via Fiji):\n{out}", parent=self)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Overlay editor",
                                 f"Fiji render failed:\n{exc}", parent=self)
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @staticmethod
    def _nearest_named(rgb):
        """Closest Merge Channels colour name to an arbitrary RGB weight."""
        best, bestd = "grey", 1e9
        for name, ref in NAMED_COLORS.items():
            if name == "gray":
                continue
            d = sum((a - b) ** 2 for a, b in zip(rgb, ref))
            if d < bestd:
                best, bestd = name, d
        return best

    def _cancel(self):
        self.result = None
        self.destroy()


def edit_overlay(master, path_a, path_b, color_a="green", color_b="magenta",
                 fiji_merge_module=None):
    """
    Open the editor and block until it closes. Returns the saved path or None.
    """
    dlg = OverlayEditor(master, path_a, path_b, color_a, color_b,
                        fiji_merge_module)
    master.wait_window(dlg)
    return dlg.result


# --------------------------------------------------------------------------
# Standalone test:  python overlay_editor.py a.tif b.tif
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python overlay_editor.py imageA imageB")
    root = tk.Tk()
    root.withdraw()
    try:
        import fiji_merge as _fm
    except ImportError:
        _fm = None
    res = edit_overlay(root, sys.argv[1], sys.argv[2], fiji_merge_module=_fm)
    print("result:", res)
    root.destroy()
