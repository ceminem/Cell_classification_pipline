#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_curate.py - inspect, correct and calibrate a measured sample
=============================================================================
CONTROLS
  Mouse wheel        zoom in / out centred on the pointer
  Middle-drag        pan
  Left-click a cell  cycle its label
  R / N / X          label the cell under the pointer released / not / excluded
  S                  flag a SEGMENTATION error (merged or split cell)
  M                  mark a cell Cellpose missed entirely
  Delete             remove your label from the cell under the pointer
  B                  toggle blind mode
  F11                fullscreen        0  reset zoom       arrow keys  pan

BLIND MODE
  Cell OUTLINES stay visible - you need to see where cells are to judge them.
  Only the class COLOUR is hidden. Your own labels stay visible so you can see
  what you have already done. Labels made blind are the only ones used to fit
  a threshold: tuning while looking at the classifier's answer means agreeing
  with yourself, which is not evidence.

TWO KINDS OF ERROR, KEPT SEPARATE
  A wrong RELEASE call and a wrong SEGMENTATION are different problems. A
  merged pair of cells has a meaningless mean intensity, so labelling it
  "released" would poison the threshold fit. Press S to flag those instead;
  they are excluded from fitting and recorded separately.
=============================================================================
"""

import csv
import os
import sys
import threading
import traceback

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    sys.exit("ERROR: tkinter missing. On Linux: sudo apt install python3-tk")

import numpy as np

try:
    from PIL import Image, ImageTk, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    sys.exit("ERROR: Pillow required.  python -m pip install pillow")

import cargo_measure as cm

APP = "Cargo Curation"

CLASS_COLORS = {
    "released":            (0, 230, 0),
    "released_highsignal": (0, 230, 230),
    "not_released":        (235, 0, 0),
    "excluded_negative":   (130, 130, 130),
    "excluded_saturated":  (255, 140, 0),
    "excluded_too_small":  (225, 225, 0),
    "excluded_invalid":    (225, 225, 0),
    "bad_segmentation":    (255, 0, 255),
}
OUTLINE_NEUTRAL = (185, 185, 185)
CYCLE = ["released", "not_released", "excluded_negative"]


class Curate:
    def __init__(self, root, folder=None):
        self.root = root
        root.title(APP)
        root.geometry("1500x950")

        self.rows = []
        self.by_id = {}
        self.fluor_raw = None
        self.labels = None
        self.base = ""
        self.out_dir = ""
        self.manual = {}
        self.added = []
        self.blind = tk.BooleanVar(value=False)
        self.reveal = tk.BooleanVar(value=False)

        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self._drag = None
        self.fullscreen = False
        self.borderline = []
        self.border_idx = -1
        self.cls_by_id = {}
        self._mouse = (0, 0)

        self._build()
        self._bind()
        if folder:
            self._load(folder)

    # ---------------------------------------------------------------- UI ---
    def _build(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Button(top, text="Open sample...", command=self._pick).pack(side="left")
        self.lbl = ttk.Label(top, text="no sample loaded", foreground="#666")
        self.lbl.pack(side="left", padx=10)
        ttk.Button(top, text="Save labels", command=self._save).pack(side="right")
        ttk.Button(top, text="Fullscreen (F11)", command=self._fs).pack(side="right", padx=6)

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=5)
        paned.add(right, weight=2)
        # kept so fullscreen can drop the side panel and give the image the
        # whole screen
        self._paned = paned
        self._right = right

        self.canvas = tk.Canvas(left, background="#101010", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.status = ttk.Label(left, text="", foreground="#555")
        self.status.pack(anchor="w")

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.t_thr = ttk.Frame(nb)
        self.t_gt = ttk.Frame(nb)
        self.t_key = ttk.Frame(nb)
        nb.add(self.t_thr, text=" Threshold ")
        nb.add(self.t_gt, text=" Ground truth ")
        nb.add(self.t_key, text=" Keys ")
        self._build_thr()
        self._build_gt()
        self._build_keys()

    def _build_thr(self):
        f = self.t_thr
        ttk.Label(f, text="Classification", font=("Segoe UI", 9, "bold"))\
            .pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Label(f, foreground="#555", justify="left", font=("Segoe UI", 8),
                  wraplength=300,
                  text="Fitted against 195 of your hand-labelled cells, whole-cell "
                       "brightness separates released from not-released far better "
                       "(AUC 0.92) than fill ratio or CV (AUC 0.45 - chance). "
                       "Brightness is therefore the decision; the saturation guard "
                       "below vetoes cells that are bright only because a blob "
                       "bleeds over them.").pack(anchor="w", padx=8)

        self.bright_var = tk.DoubleVar(value=25.7)
        self.cvcap_var = tk.DoubleVar(value=0.65)
        self.sat_var = tk.DoubleVar(value=0.0)
        self._slider(f, "Brightness threshold", self.bright_var, 0, 150,
                     "Whole-cell mean at or above this = released.")
        self._slider(f, "Core CV cap", self.cvcap_var, 0, 2.0,
                     "A released cell must ALSO be smooth: low CV = even fill,\n"
                     "high CV = bright puncta among dark pixels. Set 2.0 to\n"
                     "effectively disable and use brightness alone.")
        self._slider(f, "Saturation veto", self.sat_var, 0, 0.5,
                     "If this fraction of a cell is at the top of the range, it is\n"
                     "flagged saturated instead of released. 0 disables the veto.")

        ttk.Button(f, text="Suggest threshold for this image (Otsu)",
                   command=self._suggest).pack(anchor="w", padx=8, pady=6)
        self.count_lbl = ttk.Label(f, text="", font=("Segoe UI", 10, "bold"))
        self.count_lbl.pack(anchor="w", padx=8, pady=8)
        self.hist_lbl = ttk.Label(f, text="", font=("Consolas", 8), justify="left")
        self.hist_lbl.pack(anchor="w", padx=8)

    def _build_gt(self):
        f = self.t_gt
        ttk.Checkbutton(f, text="BLIND MODE (B) - hide classification,\nkeep outlines",
                        variable=self.blind, command=self._redraw)\
            .pack(anchor="w", padx=8, pady=6)

        ttk.Separator(f).pack(fill="x", pady=4)
        ttk.Label(f, text="Borderline review", font=("Segoe UI", 9, "bold"))\
            .pack(anchor="w", padx=8)
        ttk.Label(f, foreground="#555", justify="left", font=("Segoe UI", 8),
                  wraplength=300,
                  text="Walks through the cells closest to the threshold - where the "
                       "classifier is least sure. Each is centred and outlined but "
                       "NOT coloured, so you judge it first. Then reveal.")\
            .pack(anchor="w", padx=8)
        br = ttk.Frame(f)
        br.pack(fill="x", padx=8, pady=4)
        ttk.Button(br, text="Start", command=self._start_border).pack(side="left")
        ttk.Button(br, text="< Prev", command=lambda: self._step(-1)).pack(side="left", padx=3)
        ttk.Button(br, text="Next >", command=lambda: self._step(1)).pack(side="left")
        ttk.Checkbutton(f, text="Reveal classification", variable=self.reveal,
                        command=self._redraw).pack(anchor="w", padx=8)
        self.border_lbl = ttk.Label(f, text="", font=("Consolas", 8), justify="left")
        self.border_lbl.pack(anchor="w", padx=8, pady=4)

        ttk.Separator(f).pack(fill="x", pady=6)
        self.gt_lbl = ttk.Label(f, text="", justify="left")
        self.gt_lbl.pack(anchor="w", padx=8, pady=4)
        ttk.Button(f, text="Fit threshold to my BLIND labels",
                   command=self._fit).pack(anchor="w", padx=8, pady=6)
        self.fit_lbl = ttk.Label(f, text="", justify="left", font=("Consolas", 8))
        self.fit_lbl.pack(anchor="w", padx=8)

    def _build_keys(self):
        f = self.t_key
        t = tk.Text(f, wrap="word", font=("Segoe UI", 9), relief="flat",
                    background="#f7f7f7")
        t.pack(fill="both", expand=True, padx=6, pady=6)
        t.insert("1.0", """MOUSE
  wheel          zoom at the pointer
  middle-drag    pan
  left-click     cycle released -> not released -> excluded

KEYS  (act on the cell under the pointer)
  R   released
  N   not released
  X   excluded - took nothing up
  S   bad segmentation (merged or split cell)
  M   mark a cell Cellpose missed here
  Del remove your label

  B   blind mode      F11 fullscreen
  0   reset zoom      arrows  pan

WHY SEGMENTATION ERRORS ARE FLAGGED SEPARATELY
  Two cells merged into one label have a mean intensity that belongs to
  neither. If you labelled that "released" it would be fed into the threshold
  fit as though it were a real measurement, and it would drag the threshold
  the wrong way. Press S instead: those cells are recorded, excluded from the
  fit, and counted so you know how much of the image the segmentation got
  wrong.

WHY BLIND MODE KEEPS OUTLINES
  You need to see the cells. What is hidden is the software's opinion, so
  your judgement is independent evidence rather than an echo.

SAVED FILE
  <base>_curated.csv, with Source recording how each label was made:
    manual_blind      judged with the answer hidden  (used for fitting)
    manual_review     corrected with the answer shown (recorded only)
    manual_added      a cell Cellpose missed
    bad_segmentation  a segmentation error, not a release call
  Your automatic *_results.csv is never modified.
""")
        t.config(state="disabled")

    def _slider(self, parent, label, var, lo, hi, tip):
        box = ttk.Frame(parent)
        box.pack(fill="x", padx=8, pady=5)
        head = ttk.Frame(box)
        head.pack(fill="x")
        ttk.Label(head, text=label, font=("Segoe UI", 9, "bold")).pack(side="left")
        val = ttk.Label(head, text=f"{var.get():.2f}")
        val.pack(side="right")
        ttk.Scale(box, from_=lo, to=hi, variable=var, orient="horizontal",
                  command=lambda v: (val.config(text=f"{float(v):.2f}"),
                                     self._reclassify())).pack(fill="x", pady=2)
        ttk.Label(box, text=tip, foreground="#666", justify="left",
                  font=("Segoe UI", 8)).pack(anchor="w")

    # ------------------------------------------------------------- keys ---
    def _bind(self):
        c = self.canvas
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Button-4>", lambda e: self._wheel(e, 1))
        c.bind("<Button-5>", lambda e: self._wheel(e, -1))
        c.bind("<Button-1>", self._click)
        c.bind("<Button-2>", self._pan_start)
        c.bind("<B2-Motion>", self._pan_move)
        c.bind("<ButtonRelease-2>", lambda e: setattr(self, "_drag", None))
        c.bind("<Motion>", lambda e: setattr(self, "_mouse", (e.x, e.y)))
        r = self.root
        r.bind("<F11>", lambda e: self._fs())
        r.bind("<Escape>", lambda e: self._fs(False))
        for key, cls in (("r", "released"), ("n", "not_released"),
                         ("x", "excluded_negative"), ("s", "bad_segmentation")):
            r.bind(key, lambda e, c=cls: self._key(c))
            r.bind(key.upper(), lambda e, c=cls: self._key(c))
        r.bind("m", lambda e: self._add())
        r.bind("M", lambda e: self._add())
        r.bind("b", lambda e: (self.blind.set(not self.blind.get()), self._redraw()))
        r.bind("B", lambda e: (self.blind.set(not self.blind.get()), self._redraw()))
        r.bind("0", lambda e: self._fit_view())
        r.bind("<Delete>", lambda e: self._key(None))
        for k, dx, dy in (("<Left>", 80, 0), ("<Right>", -80, 0),
                          ("<Up>", 0, 80), ("<Down>", 0, -80)):
            r.bind(k, lambda e, a=dx, b=dy: self._nudge(a, b))

    def _fs(self, val=None):
        """
        Fullscreen the IMAGE, not just the window.

        Toggling the window attribute alone left the control panel taking a
        third of the screen, so the picture barely grew. Here the side panel is
        removed from the layout and the canvas is refitted, so fullscreen
        actually gives the image the whole display - which is the point when
        judging faint cells.
        """
        self.fullscreen = (not self.fullscreen) if val is None else val
        self.root.attributes("-fullscreen", self.fullscreen)
        try:
            if self.fullscreen:
                self._paned.forget(self._right)
            else:
                self._paned.add(self._right, weight=2)
        except (AttributeError, tk.TclError):
            pass
        # let the window settle before measuring the canvas, or the refit uses
        # the old size and the image ends up the wrong scale
        self.root.after(80, self._fit_view)

    def _nudge(self, dx, dy):
        self.pan[0] += dx
        self.pan[1] += dy
        self._redraw()

    # ------------------------------------------------------------ load ---
    def _pick(self):
        d = filedialog.askdirectory(title="Choose a SAMPLE folder (e.g. .../A1)")
        if d:
            self._load(d)

    def _load(self, folder):
        try:
            names = os.listdir(folder)
        except OSError as exc:
            messagebox.showerror(APP, str(exc))
            return

        # Only IMAGE masks. "_cp_masks" also matches the bounding-box table
        # "<base>_cp_masks_bbox.csv", which used to be picked whenever it
        # listed first.
        masks = [f for f in names
                 if "_cp_masks" in f.lower()
                 and f.lower().endswith((".tif", ".tiff", ".png"))]
        tifs = [f for f in sorted(names)
                if f.lower().endswith((".tif", ".tiff"))
                and not any(t in f.lower()
                            for t in ("_cp_masks", "_overlay", "_bbox"))]
        if not masks or not tifs:
            messagebox.showerror(
                APP, "That folder needs the acquisition tifs and a *_cp_masks file.\n"
                     "Run the analysis on it first.")
            return

        mask_name = self._choose_mask(folder, masks)
        # same channel identification as the pipeline, so the curator always
        # measures the image the pipeline measured
        fluor = None
        try:
            import run_cargo_pipeline as rcp
            _bf, fluor = rcp.find_channels(folder)
        except Exception:
            fluor = None
        if not fluor:
            fluor = os.path.join(folder, next((f for f in tifs if "fluor" in f.lower()),
                                              tifs[-1]))
        mask = os.path.join(folder, mask_name)
        self.base = os.path.splitext(mask_name)[0].replace("_cp_masks", "")
        self.out_dir = os.path.join(folder, "edited_results")

        self.lbl.config(text="measuring, please wait...")
        self.root.update_idletasks()
        # read tk vars on the MAIN thread - tkinter is not thread-safe
        cfg = {"method": "brightness",
               "brightness_threshold": float(self.bright_var.get()),
               "saturation_guard": float(self.sat_var.get()),
               "min_cell_mean": 5, "core_radius_fraction": 0.5,
               "fill_ratio_threshold": 0.65, "core_cv_threshold": 0.60,
               "very_high_signal_cutoff": 1e9, "rolling_ball_radius": 50}

        def work():
            try:
                rows, raw, labels, _ = cm.measure_for_tuning(
                    fluor, mask, cfg, log=lambda m: None)
                self.root.after(0, lambda: self._loaded(rows, raw, labels))
            except Exception:
                tb = traceback.format_exc()
                self.root.after(0, lambda: messagebox.showerror(APP, tb.splitlines()[-1]))

        threading.Thread(target=work, daemon=True).start()

    def _choose_mask(self, folder, masks):
        """
        Pick the mask to curate when a folder holds more than one.

        This happens after a plate layout is added: the old run left
        "8_cond_pre_glucose_A1_cp_masks.tif", the new one
        "A1_plus-His-..._cp_masks.tif". Taking the first in the listing (the
        old behaviour) silently opened the OLD mask, so labels were attached to
        cell ids that the current results no longer use.

        Preference: a mask whose results file exists in edited_results (the
        newest such), then the newest mask by modification time, .tif before
        .png. The choice is shown in the title bar.
        """
        if len(masks) == 1:
            return masks[0]
        res_dir = os.path.join(folder, "edited_results")

        def key(m):
            base = os.path.splitext(m)[0].replace("_cp_masks", "")
            has_results = os.path.isfile(os.path.join(res_dir, base + "_results.csv"))
            is_tif = m.lower().endswith((".tif", ".tiff"))
            try:
                mtime = os.path.getmtime(os.path.join(folder, m))
            except OSError:
                mtime = 0.0
            return (has_results, is_tif, mtime)

        chosen = max(masks, key=key)
        self._mask_note = (f"   (folder has {len(masks)} masks - using "
                           f"{chosen}; the others are older runs)")
        return chosen

    def _loaded(self, rows, raw, labels):
        self.rows = rows
        self.by_id = {r["CellID"]: r for r in rows}
        self.fluor_raw = raw
        self.labels = labels
        self.lbl.config(text=f"{self.base}  -  {len(rows)} cells"
                             + getattr(self, "_mask_note", ""))
        self._mask_note = ""
        self._read_existing()
        self._prepare()
        self._fit_view()
        self._suggest(quiet=True)
        self._reclassify()

    def _read_existing(self):
        p = os.path.join(self.out_dir, self.base + "_curated.csv")
        if not os.path.isfile(p):
            return
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh, delimiter=";"):
                    if r["Source"] == "manual_added":
                        self.added.append({"CellID": int(r["CellID"]),
                                           "X": float(r["CentroidX"]),
                                           "Y": float(r["CentroidY"]),
                                           "Label": r["Label"]})
                    else:
                        self.manual[int(r["CellID"])] = {"label": r["Label"],
                                                         "source": r["Source"]}
        except (OSError, ValueError, KeyError):
            pass

    def _prepare(self):
        raw = self.fluor_raw
        g = (raw[..., 1] if (raw.ndim == 3 and raw.shape[-1] in (3, 4))
             else raw.max(axis=0) if raw.ndim == 3 else raw)
        step = max(1, int(max(g.shape) / 2200))
        self.vs = 1.0 / step
        s = g[::step, ::step].astype(np.float32)
        lo, hi = np.percentile(s, [1, 99.5])
        if hi > lo:
            s = np.clip((s - lo) / (hi - lo), 0, 1) * 255
        self.base_rgb = np.stack([s.astype(np.uint8)] * 3, axis=-1)
        small = self.labels[::step, ::step]
        from scipy import ndimage
        er = ndimage.grey_erosion(small, size=(3, 3))
        em = (small > 0) & (small != er)
        self.edge_yx = np.nonzero(em)
        self.edge_ids = small[em]

    # -------------------------------------------------------- classify ---
    def _cfg(self):
        cap = float(self.cvcap_var.get())
        return {"method": "brightness",
                "brightness_threshold": float(self.bright_var.get()),
                "core_cv_cap": None if cap >= 1.99 else cap,
                "saturation_guard": float(self.sat_var.get())}

    def _suggest(self, quiet=False):
        if not self.rows:
            return
        t = cm.suggest_threshold(self.rows)
        if t is None:
            if not quiet:
                messagebox.showinfo(APP, "Too few cells to suggest a threshold.")
            return
        self.bright_var.set(round(t, 2))
        self._reclassify()
        if not quiet:
            messagebox.showinfo(
                APP, f"Suggested threshold for this image: {t:.2f}\n\n"
                     "This is Otsu's split of the per-cell intensities. Check it "
                     "against the overlay before accepting it - it is a starting "
                     "point, not an answer.")

    def _reclassify(self, *_):
        if not self.rows:
            return
        cfg = self._cfg()
        rel = nrel = badseg = 0
        self.cls_by_id = {}
        for r in self.rows:
            man = self.manual.get(r["CellID"], {}).get("label")
            c = man or cm.classify_row(r, cfg)
            self.cls_by_id[r["CellID"]] = c
            if c == "bad_segmentation":
                badseg += 1
            elif c.startswith("released"):
                rel += 1
            elif c == "not_released":
                nrel += 1
        total = len(self.rows) + len(self.added)
        ws = rel + nrel
        self.count_lbl.config(
            text=f"released {rel}    not released {nrel}\n"
                 f"relative {100*rel/ws if ws else 0:.1f}%    "
                 f"absolute {100*rel/total if total else 0:.1f}%"
                 + (f"\n+{len(self.added)} hand-marked" if self.added else "")
                 + (f"\n{badseg} flagged bad segmentation" if badseg else ""))
        mws = np.array([r["MeanWhole"] for r in self.rows
                        if np.isfinite(r["MeanWhole"])])
        if len(mws):
            self.hist_lbl.config(
                text="cell brightness: p10=%.1f p50=%.1f p90=%.1f p99=%.1f"
                     % tuple(np.percentile(mws, [10, 50, 90, 99])))
        self._gt_counts()
        self._redraw()

    # ------------------------------------------------------------ view ---
    def _fit_view(self):
        if self.fluor_raw is None:
            return
        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        h, w = self.base_rgb.shape[:2]
        self.zoom = min(cw / w, ch / h)
        self.pan = [(cw - w * self.zoom) / 2, (ch - h * self.zoom) / 2]
        self._redraw()

    def _wheel(self, ev, direction=None):
        if self.fluor_raw is None:
            return
        d = direction if direction is not None else (1 if ev.delta > 0 else -1)
        f = 1.25 if d > 0 else 1 / 1.25
        # keep whatever is under the cursor fixed while zooming
        bx = (ev.x - self.pan[0]) / self.zoom
        by = (ev.y - self.pan[1]) / self.zoom
        self.zoom = max(0.05, min(60.0, self.zoom * f))
        self.pan[0] = ev.x - bx * self.zoom
        self.pan[1] = ev.y - by * self.zoom
        self._redraw()

    def _pan_start(self, ev):
        self._drag = (ev.x, ev.y, self.pan[0], self.pan[1])

    def _pan_move(self, ev):
        if self._drag:
            x0, y0, p0, p1 = self._drag
            self.pan = [p0 + ev.x - x0, p1 + ev.y - y0]
            self._redraw()

    def _redraw(self):
        if self.fluor_raw is None:
            return
        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        h, w = self.base_rgb.shape[:2]

        # render only the visible crop, so zooming stays fast on huge images
        x0 = max(0, int(-self.pan[0] / self.zoom))
        y0 = max(0, int(-self.pan[1] / self.zoom))
        x1 = min(w, int((cw - self.pan[0]) / self.zoom) + 2)
        y1 = min(h, int((ch - self.pan[1]) / self.zoom) + 2)
        if x1 <= x0 or y1 <= y0:
            self.canvas.delete("all")
            return

        sub = self.base_rgb[y0:y1, x0:x1].copy()
        yy, xx = self.edge_yx
        m = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        if m.any():
            vy, vx, vid = yy[m] - y0, xx[m] - x0, self.edge_ids[m]
            if self.blind.get() and not self.reveal.get():
                sub[vy, vx] = OUTLINE_NEUTRAL     # outlines, no answer
            else:
                for cls, col in CLASS_COLORS.items():
                    sel = np.fromiter(
                        (self.cls_by_id.get(int(i), "") == cls for i in vid),
                        dtype=bool, count=len(vid))
                    if sel.any():
                        sub[vy[sel], vx[sel]] = col

        img = Image.fromarray(sub)
        dw = max(1, int(sub.shape[1] * self.zoom))
        dh = max(1, int(sub.shape[0] * self.zoom))
        img = img.resize((dw, dh), Image.NEAREST if self.zoom >= 1 else Image.BILINEAR)
        d = ImageDraw.Draw(img)

        # your own labels are always visible, including in blind mode
        for cid, info in self.manual.items():
            r = self.by_id.get(cid)
            if r is None:
                continue
            px = (r["CentroidX"] * self.vs - x0) * self.zoom
            py = (r["CentroidY"] * self.vs - y0) * self.zoom
            if -20 < px < dw + 20 and -20 < py < dh + 20:
                col = CLASS_COLORS.get(info["label"], (255, 255, 255))
                d.ellipse([px - 5, py - 5, px + 5, py + 5],
                          fill=col, outline=(255, 255, 255))
        for a in self.added:
            px = (a["X"] * self.vs - x0) * self.zoom
            py = (a["Y"] * self.vs - y0) * self.zoom
            if -20 < px < dw + 20 and -20 < py < dh + 20:
                d.line([px - 9, py, px + 9, py], fill=(255, 0, 255), width=2)
                d.line([px, py - 9, px, py + 9], fill=(255, 0, 255), width=2)

        if 0 <= self.border_idx < len(self.borderline):
            r = self.borderline[self.border_idx]
            px = (r["CentroidX"] * self.vs - x0) * self.zoom
            py = (r["CentroidY"] * self.vs - y0) * self.zoom
            rr = max(16, 26 * self.zoom)
            d.ellipse([px - rr, py - rr, px + rr, py + rr],
                      outline=(255, 255, 0), width=3)

        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.pan[0] + x0 * self.zoom,
                                 self.pan[1] + y0 * self.zoom,
                                 image=self.tkimg, anchor="nw")
        self.status.config(
            text=f"zoom {self.zoom*100:.0f}%   "
                 f"{'BLIND - outlines only' if (self.blind.get() and not self.reveal.get()) else 'showing classification'}"
                 "   |  wheel zoom · middle-drag pan · R/N/X/S label · M missed · B blind")

    # ---------------------------------------------------------- picking ---
    def _at(self, ex, ey):
        if self.labels is None:
            return None, 0.0, 0.0
        fx = ((ex - self.pan[0]) / self.zoom) / self.vs
        fy = ((ey - self.pan[1]) / self.zoom) / self.vs
        if not (0 <= fy < self.labels.shape[0] and 0 <= fx < self.labels.shape[1]):
            return None, fx, fy
        return int(self.labels[int(fy), int(fx)]), fx, fy

    def _click(self, ev):
        self.canvas.focus_set()
        cid, _, _ = self._at(ev.x, ev.y)
        if not cid:
            return
        cur = (self.manual.get(cid, {}).get("label")
               or self.cls_by_id.get(cid, "not_released"))
        nxt = CYCLE[(CYCLE.index(cur) + 1) % len(CYCLE)] if cur in CYCLE else CYCLE[0]
        self._set(cid, nxt)

    def _key(self, cls):
        cid, _, _ = self._at(*self._mouse)
        if not cid:
            return
        if cls is None:
            self.manual.pop(cid, None)
            self._reclassify()
        else:
            self._set(cid, cls)

    def _set(self, cid, cls):
        src = ("bad_segmentation" if cls == "bad_segmentation"
               else "manual_blind" if self.blind.get() else "manual_review")
        self.manual[cid] = {"label": cls, "source": src}
        self._reclassify()

    def _add(self):
        _, fx, fy = self._at(*self._mouse)
        for a in list(self.added):
            if abs(a["X"] - fx) < 40 and abs(a["Y"] - fy) < 40:
                self.added.remove(a)
                self._reclassify()
                return
        self.added.append({"CellID": -(len(self.added) + 1),
                           "X": fx, "Y": fy, "Label": "counted_only"})
        self._reclassify()

    # ------------------------------------------------------ borderline ---
    def _start_border(self):
        if not self.rows:
            return
        t = float(self.bright_var.get())
        cand = [r for r in self.rows
                if not r.get("_no_signal") and np.isfinite(r["MeanWhole"])]
        cand.sort(key=lambda r: abs(r["MeanWhole"] - t))
        self.borderline = cand[:80]
        self.border_idx = 0
        self.blind.set(True)
        self.reveal.set(False)
        self._show_border()

    def _step(self, d):
        if not self.borderline:
            self._start_border()
            return
        self.border_idx = max(0, min(len(self.borderline) - 1, self.border_idx + d))
        self.reveal.set(False)
        self._show_border()

    def _show_border(self):
        r = self.borderline[self.border_idx]
        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        self.zoom = max(self.zoom, 4.0)
        self.pan[0] = cw / 2 - r["CentroidX"] * self.vs * self.zoom
        self.pan[1] = ch / 2 - r["CentroidY"] * self.vs * self.zoom
        mine = self.manual.get(r["CellID"], {}).get("label", "-")
        self.border_lbl.config(
            text=f"{self.border_idx+1} of {len(self.borderline)}\n"
                 f"cell {r['CellID']}\n"
                 f"brightness {r['MeanWhole']:.1f}  (threshold {self.bright_var.get():.1f})\n"
                 f"your label: {mine}")
        self._redraw()

    # ------------------------------------------------------------- fit ---
    def _gt_counts(self):
        b = sum(1 for v in self.manual.values() if v["source"] == "manual_blind")
        rv = sum(1 for v in self.manual.values() if v["source"] == "manual_review")
        bs = sum(1 for v in self.manual.values() if v["source"] == "bad_segmentation")
        self.gt_lbl.config(text=f"blind labels     : {b}   (used for fitting)\n"
                                f"review edits     : {rv}\n"
                                f"bad segmentation : {bs}   (excluded from fitting)\n"
                                f"hand-marked      : {len(self.added)}")

    def _fit(self):
        pts = []
        for cid, info in self.manual.items():
            if info["source"] != "manual_blind":
                continue
            if info["label"] not in ("released", "not_released"):
                continue          # excluded and bad-segmentation cells cannot fit a threshold
            r = self.by_id.get(cid)
            if r is None or not np.isfinite(r["MeanWhole"]):
                continue
            pts.append((r["MeanWhole"], 1 if info["label"] == "released" else 0))
        if len(pts) < 8:
            messagebox.showinfo(
                APP, f"Only {len(pts)} usable blind labels.\n\n"
                     "Label at least 8 - ideally 30+, including borderline cells - "
                     "before fitting. Fewer than that and the threshold just "
                     "memorises a handful of cells.")
            return
        v = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        if y.sum() in (0, len(y)):
            messagebox.showinfo(APP, "Your blind labels are all one class.\n"
                                     "Label some of both kinds.")
            return
        cvs = np.array([self.by_id[c]["CoreCV"] for c, i in self.manual.items()
                        if i["source"] == "manual_blind"
                        and i["label"] in ("released", "not_released")
                        and self.by_id.get(c) is not None
                        and np.isfinite(self.by_id[c]["MeanWhole"])])
        best = None
        cand_cv = np.unique(np.round(cvs[np.isfinite(cvs)], 2)) if len(cvs) else []
        for t in np.unique(v):
            for c in (list(cand_cv) + [None]):
                pred = (v >= t)
                if c is not None and len(cvs) == len(v):
                    pred = pred & (cvs <= c)
                a = (pred.astype(int) == y).mean()
                if best is None or a > best[0]:
                    best = (a, t, c)
        acc, t, cbest = best
        self.bright_var.set(round(float(t), 2))
        if cbest is not None:
            self.cvcap_var.set(round(float(cbest), 2))
        self._reclassify()
        wrong = int(round((1 - acc) * len(pts)))
        self.fit_lbl.config(text=f"fitted on {len(pts)} blind labels\n"
                                 f"brightness >= {t:.2f}\n"
                                 f"agrees on {len(pts)-wrong}/{len(pts)} ({acc*100:.0f}%)")
        if acc < 0.85:
            messagebox.showwarning(
                APP, f"The best threshold still disagrees with you on {wrong} of "
                     f"{len(pts)} cells.\n\nThat suggests brightness alone cannot "
                     "capture what you are seeing in this image - worth checking "
                     "whether those cells share something (saturation, overlap).")

    # ------------------------------------------------------------ save ---
    def _save(self):
        if not self.rows:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        p = os.path.join(self.out_dir, self.base + "_curated.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["CellID", "Source", "Label", "AutoLabel", "MeanWhole",
                        "SatFraction", "CoreFillRatio", "CoreCV",
                        "CentroidX", "CentroidY"])
            cfg = self._cfg()
            for cid, info in sorted(self.manual.items()):
                r = self.by_id.get(cid)
                if r is None:
                    continue
                w.writerow([cid, info["source"], info["label"],
                            cm.classify_row(r, cfg),
                            f"{r['MeanWhole']:.2f}",
                            f"{r.get('SatFraction', 0):.4f}",
                            f"{r['CoreFillRatio']:.4f}", f"{r['CoreCV']:.4f}",
                            f"{r['CentroidX']:.1f}", f"{r['CentroidY']:.1f}"])
            for a in self.added:
                w.writerow([a["CellID"], "manual_added", a["Label"], "", "", "",
                            "", "", f"{a['X']:.1f}", f"{a['Y']:.1f}"])

        note = os.path.join(self.out_dir, self.base + "_curation_note.txt")
        with open(note, "w", encoding="utf-8") as fh:
            fh.write(f"sample: {self.base}\n"
                     f"brightness threshold in force: {self.bright_var.get():.2f}\n"
                     f"saturation veto: {self.sat_var.get():.3f}\n"
                     f"blind labels: {sum(1 for v in self.manual.values() if v['source']=='manual_blind')}\n"
                     f"review edits: {sum(1 for v in self.manual.values() if v['source']=='manual_review')}\n"
                     f"bad segmentation: {sum(1 for v in self.manual.values() if v['source']=='bad_segmentation')}\n"
                     f"hand-marked missed cells: {len(self.added)}\n\n"
                     "The automatic *_results.csv was not modified. Report these\n"
                     "labels as manual curation.\n")
        messagebox.showinfo(APP, f"Saved {len(self.manual)} labels and "
                                 f"{len(self.added)} marks:\n{p}")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    Curate(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()