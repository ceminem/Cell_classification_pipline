#!/usr/bin/env python3
"""
roi_editor.py - draw the area in which cells are counted.

Opened from the Cell count tab ("Define counting area...") after a count, or
on its own:

    python roi_editor.py  <folder>/cell_count

Shows each counted image with its cell outlines. Drag with the left mouse
button to draw the counting area:

    Rectangle  drag a box
    Polygon    click the corners, double-click (or Enter) to close
    Freehand   hold the button and trace the area (lasso)

Several shapes on one image are joined into one area. Mouse wheel zooms,
right or middle button drags the view. The count of cells inside the area is
updated while you draw (a cell belongs to the area when its centre lies
inside - an unbiased rule, because a cell cut by the border is counted on one
side only).

"Save and recount" writes <cell_count>/rois.json and runs the count again with
the saved masks (no new segmentation): the CSV, Excel file and figures then
count only the area, and the density is per drawn area. An image without
shapes is counted whole.
"""

import csv
import json
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

ROI_FILE = "rois.json"
ROI_COLOR = "#00E5FF"
DOT_IN = "#00E5FF"
TOOLS = ("Rectangle", "Polygon", "Freehand")


# ---------------------------------------------------------------- helpers --
def shape_points(sh):
    """Shape -> list of (x, y); same as cell_count.shape_points."""
    if sh.get("type") == "rect":
        x0, x1 = sorted((float(sh["x0"]), float(sh["x1"])))
        y0, y1 = sorted((float(sh["y0"]), float(sh["y1"])))
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(float(x), float(y)) for x, y in sh.get("points", [])]


def inside(px, py, poly):
    """Even-odd point-in-polygon test for many points at once."""
    poly = np.asarray(poly, float)
    res = np.zeros(len(px), bool)
    if len(poly) < 3 or not len(px):
        return res
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for a, b, c, d in zip(x1, y1, x2, y2):
        cond = (b > py) != (d > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = a + (py - b) * (c - a) / (d - b)
        res ^= cond & (px < xint)
    return res


def polygon_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def read_rois(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d = d.get("images", d) if isinstance(d, dict) else {}
        return {k: list(v) for k, v in d.items() if isinstance(v, list)}
    except (OSError, ValueError):
        return {}


def write_rois(path, rois):
    data = {"version": 1,
            "note": "Counting areas for cell_count.py, in full-resolution image "
                    "pixels. Images without shapes are counted whole.",
            "images": {k: v for k, v in rois.items() if v}}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def read_summary(count_dir):
    """Images of a finished count: list of dicts (image, file, px_per_um,
    edge_rule) in the order they were counted."""
    p = os.path.join(count_dir, "cell_count_summary.csv")
    out = []
    try:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("image"):
                    out.append(r)
    except OSError:
        pass
    return out


def read_cells(count_dir, name):
    """Centres of the cells of one image and whether each could count at all
    (not debris, not excluded by the edge rule)."""
    p = os.path.join(count_dir, "cells", name + "_cells.csv")
    xs, ys, ok = [], [], []
    try:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    xs.append(float(r["centre_x_px"]))
                    ys.append(float(r["centre_y_px"]))
                except (KeyError, ValueError):
                    continue
                ok.append((str(r.get("too_small", "0")) in ("0", "", "False"),
                           r.get("touches_edge", "")))
    except OSError:
        pass
    return np.array(xs), np.array(ys), ok


# ----------------------------------------------------------------- editor --
class RoiEditor:
    def __init__(self, master, count_dir, on_recount=None, start=None):
        """count_dir: the cell_count folder. on_recount(rois_path) is called by
        "Save and recount"; without it that button only saves."""
        self.count_dir = os.path.abspath(count_dir)
        self.on_recount = on_recount
        self.images = read_summary(self.count_dir)
        if not self.images:
            messagebox.showinfo(
                "Counting area",
                "No finished count in\n" + self.count_dir +
                "\n\nCount the cells first; the counting area is drawn on the "
                "result.", parent=master)
            self.win = None
            return
        self.roi_path = os.path.join(self.count_dir, ROI_FILE)
        self.rois = read_rois(self.roi_path)
        self.dirty = False

        self.win = tk.Toplevel(master)
        self.win.title("Counting area - " + self.count_dir)
        self.win.geometry("1180x820")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self._build()
        names = [r["image"] for r in self.images]
        idx = names.index(start) if start in names else 0
        self.win.after(50, lambda: self.load(idx))

    # ---- layout ----
    def _build(self):
        w = self.win
        top = ttk.Frame(w)
        top.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Button(top, text="<", width=3, command=lambda: self.step(-1)).pack(side="left")
        self.img_var = tk.StringVar()
        self.img_cb = ttk.Combobox(top, textvariable=self.img_var, state="readonly",
                                   width=46, values=[r["image"] for r in self.images])
        self.img_cb.pack(side="left", padx=2)
        self.img_cb.bind("<<ComboboxSelected>>",
                         lambda e: self.load(self.img_cb.current()))
        ttk.Button(top, text=">", width=3, command=lambda: self.step(+1)).pack(side="left")

        ttk.Label(top, text="   Draw:").pack(side="left")
        self.tool_var = tk.StringVar(value=TOOLS[0])
        for t in TOOLS:
            ttk.Radiobutton(top, text=t, value=t, variable=self.tool_var,
                            command=self.cancel_draw).pack(side="left", padx=2)
        ttk.Button(top, text="Undo last", command=self.undo).pack(side="left", padx=(12, 2))
        ttk.Button(top, text="Clear image", command=self.clear).pack(side="left", padx=2)
        ttk.Button(top, text="Use on all images",
                   command=self.copy_to_all).pack(side="left", padx=2)
        ttk.Button(top, text="Fit", command=self.fit).pack(side="left", padx=(12, 2))

        self.canvas = tk.Canvas(w, background="#1b1b1b", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=6, pady=2)

        bot = ttk.Frame(w)
        bot.pack(fill="x", padx=6, pady=(2, 6))
        self.info = ttk.Label(bot, text="", font=("Segoe UI", 10, "bold"))
        self.info.pack(side="left")
        ttk.Button(bot, text="Close", command=self.close).pack(side="right")
        ttk.Button(bot, text="Save and recount" if self.on_recount else "Save",
                   command=self.save_recount).pack(side="right", padx=4)
        ttk.Button(bot, text="Save", command=self.save).pack(side="right")
        self.hint = ttk.Label(
            w, foreground="#666", font=("Segoe UI", 8),
            text="Left button: draw   |   mouse wheel: zoom   |   right / middle "
                 "button: move the view   |   Polygon: click corners, double-click "
                 "or Enter to close, Esc cancels   |   cyan dots = cells counted "
                 "in the area")
        self.hint.pack(anchor="w", padx=8, pady=(0, 4))

        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Double-Button-1>", self.on_double)
        c.bind("<Motion>", self.on_motion)
        for b in ("2", "3"):
            c.bind(f"<ButtonPress-{b}>", self.pan_start)
            c.bind(f"<B{b}-Motion>", self.pan_move)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.25))
        c.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 0.8))
        w.bind("<Return>", lambda e: self.finish_polygon())
        w.bind("<Escape>", lambda e: self.cancel_draw())
        w.bind("<Control-z>", lambda e: self.undo())

        self.full = self.prev = None
        self.prev_f = 1.0
        self.z, self.ox, self.oy = 1.0, 0.0, 0.0
        self.drawing = None          # points (image coords) of the shape in progress
        self.press = None
        self._photo = None
        self._redraw_pending = False

    # ---- image loading ----
    def step(self, d):
        i = self.idx + d
        if 0 <= i < len(self.images):
            self.load(i)

    def load(self, i):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        self.cancel_draw()
        self.idx = i
        r = self.images[i]
        self.name = r["image"]
        self.img_cb.current(i)
        fig = os.path.join(self.count_dir, "figures", self.name + "_outlines.png")
        img = None
        if os.path.isfile(fig):
            try:
                img = Image.open(fig).convert("RGB")
            except OSError:
                img = None
        if img is None:
            try:
                import cell_count as cc
                img = Image.fromarray(cc.to_8bit(cc.load_gray(r["file"]))).convert("RGB")
            except Exception as exc:
                messagebox.showerror("Counting area",
                                     f"Could not open the image of {self.name}:\n{exc}",
                                     parent=self.win)
                return
        try:
            W, H = int(float(r.get("width_px") or 0)), int(float(r.get("height_px") or 0))
        except ValueError:
            W = H = 0
        if W and H and img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        self.full = img
        self.W, self.H = img.size
        f = max(self.W, self.H) / 2400.0
        self.prev_f = max(1.0, f)
        self.prev = (img.resize((max(1, round(self.W / self.prev_f)),
                                 max(1, round(self.H / self.prev_f))), Image.BILINEAR)
                     if f > 1 else img)
        try:
            self.px_um = float(r.get("px_per_um") or 0)
        except ValueError:
            self.px_um = 0.0
        self.edge_rule = r.get("edge_rule", "all")
        self.cx, self.cy, ok = read_cells(self.count_dir, self.name)
        self.cell_ok = np.array([
            good and not (self.edge_rule == "exclude" and edge)
            for good, edge in ok], bool) if ok else np.zeros(0, bool)
        self.fit()

    # ---- view ----
    def fit(self):
        if self.full is None:
            return
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        self.z = min(cw / self.W, ch / self.H)
        self.ox = -(cw / self.z - self.W) / 2
        self.oy = -(ch / self.z - self.H) / 2
        self.redraw()

    def to_img(self, x, y):
        return self.ox + x / self.z, self.oy + y / self.z

    def to_can(self, x, y):
        return (x - self.ox) * self.z, (y - self.oy) * self.z

    def zoom_at(self, x, y, k):
        if self.full is None:
            return
        ix, iy = self.to_img(x, y)
        cw = max(50, self.canvas.winfo_width())
        zmin = 0.5 * min(cw / self.W, max(50, self.canvas.winfo_height()) / self.H)
        self.z = min(16.0, max(zmin, self.z * k))
        self.ox, self.oy = ix - x / self.z, iy - y / self.z
        self.redraw()

    def on_wheel(self, e):
        self.zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8)

    def pan_start(self, e):
        self._pan = (e.x, e.y, self.ox, self.oy)

    def pan_move(self, e):
        x0, y0, ox, oy = self._pan
        self.ox = ox - (e.x - x0) / self.z
        self.oy = oy - (e.y - y0) / self.z
        self.redraw()

    def redraw(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.win.after_idle(self._redraw)

    def _redraw(self):
        from PIL import Image, ImageTk
        self._redraw_pending = False
        c = self.canvas
        c.delete("all")
        if self.full is None:
            return
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        # visible part of the image
        x0, y0 = max(0.0, self.ox), max(0.0, self.oy)
        x1 = min(float(self.W), self.ox + cw / self.z)
        y1 = min(float(self.H), self.oy + ch / self.z)
        if x1 > x0 and y1 > y0:
            # zoomed out: crop from the small copy (fast); zoomed in: full res
            src, f = ((self.prev, self.prev_f) if self.z * self.prev_f <= 1.0
                      else (self.full, 1.0))
            box = (int(x0 / f), int(y0 / f),
                   max(int(x0 / f) + 1, int(np.ceil(x1 / f))),
                   max(int(y0 / f) + 1, int(np.ceil(y1 / f))))
            crop = src.crop(box)
            dw = max(1, round((box[2] - box[0]) * f * self.z))
            dh = max(1, round((box[3] - box[1]) * f * self.z))
            crop = crop.resize((dw, dh), Image.NEAREST if self.z * f > 2
                               else Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(crop)
            px, py = self.to_can(box[0] * f, box[1] * f)
            c.create_image(px, py, image=self._photo, anchor="nw")
        self._draw_overlay()

    def _draw_overlay(self):
        c = self.canvas
        c.delete("ov")
        shapes = self.rois.get(self.name, [])
        inside_all = self._inside_mask(shapes)
        # cells counted in the area
        if shapes and len(self.cx):
            r = max(2.0, min(5.0, 6 * self.z))
            sel = np.nonzero(inside_all & self.cell_ok)[0]
            if len(sel) <= 6000:
                for i in sel:
                    x, y = self.to_can(self.cx[i], self.cy[i])
                    c.create_oval(x - r, y - r, x + r, y + r, fill=DOT_IN,
                                  outline="black", tags="ov")
        for sh in shapes:
            pts = [v for p in shape_points(sh) for v in self.to_can(*p)]
            if len(pts) >= 6:
                c.create_polygon(*pts, outline="black", fill="", width=5, tags="ov")
                c.create_polygon(*pts, outline=ROI_COLOR, fill="", width=2,
                                 dash=(8, 4), tags="ov")
        if self.drawing:
            pts = [v for p in self.drawing for v in self.to_can(*p)]
            if len(pts) >= 4:
                c.create_line(*pts, fill="#ffd400", width=2, tags=("ov", "rubber"))
        self._update_info(shapes, inside_all)

    def _inside_mask(self, shapes):
        m = np.zeros(len(self.cx), bool)
        for sh in shapes:
            m |= inside(self.cx, self.cy, shape_points(sh))
        return m

    def _update_info(self, shapes, inside_all):
        if not shapes:
            n = int(self.cell_ok.sum())
            txt = f"{self.name}:  whole image  -  {n} cells"
        else:
            n = int((inside_all & self.cell_ok).sum())
            # area of the union, measured on a coarse raster (shapes may overlap)
            area_px = self._union_area(shapes)
            txt = f"{self.name}:  {n} cells in the drawn area ({len(shapes)} shape" \
                  f"{'s' if len(shapes) > 1 else ''})"
            if self.px_um:
                mm2 = area_px / self.px_um ** 2 / 1e6
                txt += f",  {mm2:.3f} mm²,  {n / mm2:.0f} cells/mm²" if mm2 else ""
            else:
                txt += f",  {area_px:.0f} px²"
        if self.dirty:
            txt += "    (not saved)"
        self.info.config(text=txt)

    def _union_area(self, shapes):
        # rasterised like cell_count.roi_mask, so the numbers here match the
        # recount (exactly for images up to 2000 px, within ~0.5 % above)
        from PIL import Image, ImageDraw
        f = max(1.0, max(self.W, self.H) / 2000.0)
        im = Image.new("1", (max(1, round(self.W / f)), max(1, round(self.H / f))), 0)
        d = ImageDraw.Draw(im)
        for sh in shapes:
            p = [(x / f, y / f) for x, y in shape_points(sh)]
            if len(p) >= 3:
                d.polygon(p, fill=1, outline=1)
        return float(np.array(im).sum()) * f * f

    # ---- drawing ----
    def clamp(self, x, y):
        return (min(max(x, 0.0), self.W - 1.0), min(max(y, 0.0), self.H - 1.0))

    def on_press(self, e):
        if self.full is None:
            return
        p = self.clamp(*self.to_img(e.x, e.y))
        t = self.tool_var.get()
        if t == "Polygon":
            if self.drawing is None:
                self.drawing = [p]
            # close by clicking near the first corner
            elif len(self.drawing) >= 3 and \
                    np.hypot(*np.subtract(self.to_can(*self.drawing[0]),
                                          (e.x, e.y))) < 8:
                self.finish_polygon()
                return
            else:
                self.drawing.append(p)
            self._draw_overlay()
        else:
            self.press = p
            self.drawing = [p]

    def on_drag(self, e):
        if self.press is None:
            return
        p = self.clamp(*self.to_img(e.x, e.y))
        t = self.tool_var.get()
        if t == "Rectangle":
            (x0, y0), (x1, y1) = self.press, p
            self.drawing = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        else:                                   # Freehand
            last = self.to_can(*self.drawing[-1])
            if np.hypot(last[0] - e.x, last[1] - e.y) >= 3:
                self.drawing.append(p)
        self._draw_overlay()

    def on_release(self, e):
        if self.press is None:
            return
        t = self.tool_var.get()
        p = self.clamp(*self.to_img(e.x, e.y))
        shape = None
        if t == "Rectangle":
            (x0, y0) = self.press
            if abs(p[0] - x0) * self.z > 4 and abs(p[1] - y0) * self.z > 4:
                shape = {"type": "rect", "x0": round(min(x0, p[0]), 1),
                         "y0": round(min(y0, p[1]), 1),
                         "x1": round(max(x0, p[0]), 1),
                         "y1": round(max(y0, p[1]), 1)}
        elif t == "Freehand" and len(self.drawing) >= 3:
            pts = simplify(self.drawing, 0.75 / self.z)
            if len(pts) >= 3 and polygon_area(pts) * self.z ** 2 > 30:
                shape = {"type": "polygon",
                         "points": [[round(x, 1), round(y, 1)] for x, y in pts]}
        self.press = None
        self.drawing = None
        if shape:
            self.add(shape)
        else:
            self._draw_overlay()

    def on_double(self, e):
        if self.tool_var.get() == "Polygon":
            self.finish_polygon()

    def on_motion(self, e):
        if self.tool_var.get() == "Polygon" and self.drawing:
            self.canvas.delete("rubber")
            pts = [v for p in self.drawing for v in self.to_can(*p)] + [e.x, e.y]
            self.canvas.create_line(*pts, fill="#ffd400", width=2,
                                    tags=("ov", "rubber"))

    def finish_polygon(self):
        pts = self.drawing or []
        # a double-click also delivers a single click: drop a duplicate corner
        dedup = []
        for p in pts:
            if not dedup or np.hypot(p[0] - dedup[-1][0], p[1] - dedup[-1][1]) * self.z > 2:
                dedup.append(p)
        self.drawing = None
        if len(dedup) >= 3:
            self.add({"type": "polygon",
                      "points": [[round(x, 1), round(y, 1)] for x, y in dedup]})
        else:
            self._draw_overlay()

    def cancel_draw(self):
        self.drawing = None
        self.press = None
        if getattr(self, "full", None) is not None:
            self._draw_overlay()

    def add(self, shape):
        self.rois.setdefault(self.name, []).append(shape)
        self.dirty = True
        self._draw_overlay()

    def undo(self):
        if self.drawing:
            self.cancel_draw()
            return
        if self.rois.get(self.name):
            self.rois[self.name].pop()
            self.dirty = True
            self._draw_overlay()

    def clear(self):
        if self.rois.get(self.name):
            self.rois[self.name] = []
            self.dirty = True
            self._draw_overlay()

    def copy_to_all(self):
        shapes = self.rois.get(self.name)
        if not shapes:
            messagebox.showinfo("Counting area", "Draw an area on this image first.",
                                parent=self.win)
            return
        same = [r["image"] for r in self.images
                if r["image"] != self.name and
                (r.get("width_px"), r.get("height_px")) ==
                (self.images[self.idx].get("width_px"),
                 self.images[self.idx].get("height_px"))]
        if not same:
            messagebox.showinfo("Counting area",
                                "No other image has the same size.", parent=self.win)
            return
        if not messagebox.askyesno(
                "Counting area",
                f"Use this area on {len(same)} other image(s) of the same size?\n"
                "Their own areas are replaced.", parent=self.win):
            return
        for n in same:
            self.rois[n] = json.loads(json.dumps(shapes))
        self.dirty = True
        self._draw_overlay()

    # ---- saving ----
    def save(self):
        try:
            write_rois(self.roi_path, self.rois)
        except OSError as exc:
            messagebox.showerror("Counting area", f"Could not save:\n{exc}",
                                 parent=self.win)
            return False
        self.dirty = False
        self._draw_overlay()
        return True

    def save_recount(self):
        if not self.save():
            return
        if self.on_recount:
            self.on_recount(self.roi_path)
            self.win.destroy()
        else:
            messagebox.showinfo(
                "Counting area",
                f"Saved to\n{self.roi_path}\n\nCount again (the saved masks are "
                "reused, so it is quick) to apply the areas.", parent=self.win)

    def close(self):
        if self.dirty and messagebox.askyesno(
                "Counting area", "Save the counting areas before closing?",
                parent=self.win):
            if not self.save():
                return
        self.win.destroy()


def simplify(pts, tol):
    """Ramer-Douglas-Peucker: fewer points for a freehand line, same shape."""
    pts = [tuple(p) for p in pts]
    if len(pts) < 4:
        return pts

    def rdp(a, b):
        (x1, y1), (x2, y2) = pts[a], pts[b]
        dx, dy = x2 - x1, y2 - y1
        n = np.hypot(dx, dy)
        best, idx = 0.0, None
        for i in range(a + 1, b):
            x0, y0 = pts[i]
            # distance to the chord; to the point when the ends coincide
            # (a closed lasso starts and ends at the same place)
            d = (abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / n if n > 1e-9
                 else np.hypot(x0 - x1, y0 - y1))
            if d > best:
                best, idx = d, i
        if idx is not None and best > tol:
            return rdp(a, idx)[:-1] + rdp(idx, b)
        return [pts[a], pts[b]]

    sys.setrecursionlimit(max(1000, len(pts) * 2 + 100))
    out = rdp(0, len(pts) - 1)
    return out[:-1] if len(out) > 3 and out[0] == out[-1] else out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = sys.argv[1]
    if os.path.isdir(os.path.join(d, "cell_count")):
        d = os.path.join(d, "cell_count")
    root = tk.Tk()
    root.withdraw()
    ed = RoiEditor(root, d)
    if ed.win is None:
        return 1
    ed.win.bind("<Destroy>", lambda e: e.widget is ed.win and root.after(0, root.destroy))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
