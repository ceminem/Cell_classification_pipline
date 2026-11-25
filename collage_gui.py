#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collage_gui.py - figure builder front end
=============================================================================
Assembles microscopy panels into a publication figure, with a live preview.

TABS
  Source   - pick the folder, choose and reorder rows, edit row labels
  Layout   - side-by-side blocks, cell size, fit mode, gaps
  Labels   - column headers, row label styling
  Scale bar- calibration, length, which column/rows, appearance
  Output   - format conversion (TIFF in, JPEG/PNG/WEBP out), quality, DPI

The preview re-renders at a reduced scale so it stays responsive; the saved
figure is always assembled at full resolution.

Runs on any machine with Python and Pillow - no Fiji, no Cellpose, no conda.
Office PCs included. In the analysis pipeline it is the "Collage" tab of
cargo_gui.py (CollageGUI(root, parent=<frame>)); standalone it is started with
"Start Collage Builder.bat" or python collage_gui.py.  (The optional two-image overlay merge does call Fiji,
but only when you use that button - everything else is pure Pillow.)
=============================================================================
"""

import os
import queue
import subprocess
import sys
import threading
import traceback

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, colorchooser
except ImportError:
    sys.exit("ERROR: tkinter missing. On Linux: sudo apt install python3-tk")

try:
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("ERROR: Pillow is required.  python -m pip install pillow")

import collage_engine as ce

try:
    import cargo_labels
    LABELS_OK = True
except ImportError:
    cargo_labels = None
    LABELS_OK = False

try:
    import fiji_merge
    FIJI_OK = True
except ImportError:
    fiji_merge = None
    FIJI_OK = False

try:
    import overlay_editor
    EDITOR_OK = True
except ImportError:
    overlay_editor = None
    EDITOR_OK = False

try:
    import scalebar_tool
    SCALEBAR_OK = True
except ImportError:
    scalebar_tool = None
    SCALEBAR_OK = False

APP_TITLE = "Collage Builder"
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "collage_gui_settings.json")


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, font=("Segoe UI", 8)).pack(ipadx=4, ipady=2)

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class CollageGUI:
    def __init__(self, root, parent=None, get_mask_text=None):
        """root: the Tk root. parent: a frame to build into (a tab of the
        pipeline window); None = own window. get_mask_text: callable returning
        the pasted plate layout of the pipeline's Sample names tab."""
        self.root = root
        self.container = parent if parent is not None else root
        self.embedded = parent is not None
        self.get_mask_text = get_mask_text
        if not self.embedded:
            root.title(APP_TITLE)
            root.geometry("1280x860")
            root.minsize(1050, 700)

        self.rows = []              # discovered rows, in display order
        self.row_enabled = {}       # name -> BooleanVar
        self.row_labels = {}        # name -> StringVar
        self.preview_img = None
        self.render_queue = queue.Queue()
        self.rendering = False
        self.pending = False

        # default overlay channel colours for the Fiji merge dialog
        self._merge_color_a = "green"
        self._merge_color_b = "magenta"

        self._build()
        self._poll()

    # ---------------------------------------------------------------- UI ---
    def _build(self):
        paned = ttk.PanedWindow(self.container, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=4)
        # the settings ask for more width than they need (long button rows);
        # start with ~45 % for them so the preview is not squeezed away
        state = {"done": False}

        def first_split(_e=None):
            w = paned.winfo_width()
            if state["done"] or w < 200:
                return
            state["done"] = True
            try:
                paned.sashpos(0, int(w * 0.45))
            except tk.TclError:
                pass
        paned.bind("<Configure>", first_split, add="+")

        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)
        self.t_src = ttk.Frame(nb)
        self.t_lay = ttk.Frame(nb)
        self.t_lab = ttk.Frame(nb)
        self.t_bar = ttk.Frame(nb)
        self.t_out = ttk.Frame(nb)
        for t, n in [(self.t_src, "Source"), (self.t_lay, "Layout"),
                     (self.t_lab, "Labels"), (self.t_bar, "Scale bar"),
                     (self.t_out, "Output")]:
            nb.add(t, text=f" {n} ")

        self._build_source()
        self._build_layout()
        self._build_labels()
        self._build_bar()
        self._build_output()
        self._build_preview(right)

    def _build_source(self):
        f = self.t_src
        box = ttk.LabelFrame(f, text="Image folder")
        box.pack(fill="x", padx=6, pady=6)
        r = ttk.Frame(box)
        r.pack(fill="x", padx=6, pady=6)
        self.folder_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.folder_var).pack(side="left", fill="x",
                                                        expand=True, padx=(0, 4))
        ttk.Button(r, text="Browse...", command=self._pick_folder).pack(side="left")
        ttk.Label(box, text="Each subfolder becomes one row; images inside it "
                            "become the columns.",
                  foreground="#666").pack(anchor="w", padx=8, pady=(0, 6))

        lb = ttk.LabelFrame(f, text="Plate layout  (optional - paste from Excel "
                                    "to label rows automatically)")
        lb.pack(fill="x", padx=6, pady=6)
        self.mask_text = tk.Text(lb, height=5, wrap="none", font=("Consolas", 8))
        self.mask_text.pack(fill="x", padx=6, pady=4)
        rr = ttk.Frame(lb)
        rr.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(rr, text="Apply to rows", command=self._apply_mask).pack(side="left")
        ttk.Button(rr, text="Example", command=self._mask_example).pack(side="left", padx=4)
        if self.get_mask_text is not None:
            b = ttk.Button(rr, text="From Sample names tab", command=self._mask_from_pipeline)
            b.pack(side="left", padx=4)
            ToolTip(b, "Copies the layout pasted on the Sample names tab of the\n"
                       "analysis into this box and labels the rows with it.")
        ttk.Button(rr, text="Overlay 2 images...",
                   command=self._merge_overlay).pack(side="left", padx=4)
        ttk.Button(rr, text="Scale bar (1 image)...",
                   command=self._single_scalebar).pack(side="left", padx=4)
        ttk.Label(rr, foreground="#666", font=("Segoe UI", 8),
                  text="Each column = a well letter, each row = a position "
                       "(A1/A2, then B1/B2, ...)").pack(side="left", padx=8)

        rb = ttk.LabelFrame(f, text="Rows  (untick to exclude, edit the label text, "
                                    "use the arrows to reorder)")
        rb.pack(fill="both", expand=True, padx=6, pady=6)

        bar = ttk.Frame(rb)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Rescan", command=self._scan).pack(side="left")
        ttk.Button(bar, text="Move up", command=lambda: self._move(-1)).pack(side="left", padx=4)
        ttk.Button(bar, text="Move down", command=lambda: self._move(1)).pack(side="left")
        ttk.Button(bar, text="All", command=lambda: self._toggle_all(True)).pack(side="right")
        ttk.Button(bar, text="None", command=lambda: self._toggle_all(False)).pack(side="right", padx=4)

        canvas = tk.Canvas(rb, highlightthickness=0)
        sb = ttk.Scrollbar(rb, orient="vertical", command=canvas.yview)
        self.row_holder = ttk.Frame(canvas)
        self.row_holder.bind("<Configure>",
                             lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.row_holder, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        sb.pack(side="right", fill="y", pady=4)
        self.row_canvas = canvas
        self.sel_row = tk.StringVar(value="")

    def _build_layout(self):
        f = self.t_lay
        b = ttk.LabelFrame(f, text="Arrangement")
        b.pack(fill="x", padx=6, pady=6)

        r = ttk.Frame(b)
        r.pack(fill="x", padx=8, pady=6)
        lb = ttk.Label(r, text="Blocks side by side:", width=24)
        lb.pack(side="left")
        self.blocks_var = tk.StringVar(value="1")
        cb = ttk.Combobox(r, textvariable=self.blocks_var, width=6,
                          values=["1", "2", "3", "4"], state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._request_preview())
        tip = ("1 = one tall stack (all rows under each other).\n"
               "2 = rows split into two blocks placed side by side -\n"
               "    e.g. 8 rows become 4 under each other, and the other 4 beside them.")
        ToolTip(lb, tip)
        ToolTip(cb, tip)

        self._num_row(b, "Gap between blocks (px):", "block_gap", "80")
        self._check(b, "Repeat column headers on every block", "repeat_headers", True)

        c = ttk.LabelFrame(f, text="Cells")
        c.pack(fill="x", padx=6, pady=6)
        r = ttk.Frame(c)
        r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="Fit mode:", width=24).pack(side="left")
        self.fit_var = tk.StringVar(value="cover")
        fc = ttk.Combobox(r, textvariable=self.fit_var, width=10,
                          values=["cover", "contain"], state="readonly")
        fc.pack(side="left")
        fc.bind("<<ComboboxSelected>>", lambda e: self._request_preview())
        ToolTip(fc, "cover  = crop to fill the cell completely (no margins)\n"
                    "contain = never crop, may leave margins")

        k = ttk.LabelFrame(f, text="Contrast")
        k.pack(fill="x", padx=6, pady=6)
        self._check(k, "Same contrast stretch for each column", "col_contrast", False)
        ttk.Label(k, foreground="#666", font=("Segoe UI", 8), justify="left", wraplength=420,
                  text="Stretches every column once, from its darkest 0.1 % to its "
                       "brightest 0.2 % over ALL rows - the same setting for every "
                       "condition, so brighter stays brighter. Makes dim fluorescence "
                       "visible; the image files are not changed.")\
            .pack(anchor="w", padx=10, pady=(0, 6))
        self._check(k, "Skip files written by the analysis (masks, overlays)",
                    "skip_generated", True).configure(command=self._scan)

        self._num_row(c, "Cell width (px or auto):", "cell_w", "auto")
        self._num_row(c, "Cell height (px or auto):", "cell_h", "auto")
        self._num_row(c, "Grid gap (px or auto):", "grid_gap", "auto")
        self._num_row(c, "Outer padding (px):", "outer_pad", "20")

    def _build_labels(self):
        f = self.t_lab
        b = ttk.LabelFrame(f, text="Column headers")
        b.pack(fill="x", padx=6, pady=6)
        r = ttk.Frame(b)
        r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="Headers (comma separated):").pack(anchor="w")
        self.collabels_var = tk.StringVar(value="Brightfield, Calcein, Merge")
        e = ttk.Entry(r, textvariable=self.collabels_var)
        e.pack(fill="x", pady=3)
        e.bind("<FocusOut>", lambda ev: self._request_preview())
        e.bind("<Return>", lambda ev: self._request_preview())

        self._num_row(b, "Header font scale:", "col_font_scale", "0.09")
        self._num_row(b, "Header margin (px):", "col_margin", "12")
        self._color_row(b, "Header colour:", "col_color", (255, 255, 255))

        c = ttk.LabelFrame(f, text="Row labels")
        c.pack(fill="x", padx=6, pady=6)
        r = ttk.Frame(c)
        r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="Align:", width=24).pack(side="left")
        self.align_var = tk.StringVar(value="right")
        ac = ttk.Combobox(r, textvariable=self.align_var, width=10,
                          values=["right", "center", "left"], state="readonly")
        ac.pack(side="left")
        ac.bind("<<ComboboxSelected>>", lambda e: self._request_preview())

        self._num_row(c, "Label column width:", "row_label_w", "auto")
        self._num_row(c, "Row font scale:", "row_font_scale", "0.09")
        self._color_row(c, "Row label colour:", "row_color", (20, 20, 20))

    def _build_bar(self):
        f = self.t_bar
        b = ttk.LabelFrame(f, text="Scale bar")
        b.pack(fill="x", padx=6, pady=6)

        self._check(b, "Draw a scale bar", "bar_on", True)

        r = ttk.Frame(b)
        r.pack(fill="x", padx=8, pady=6)
        lb = ttk.Label(r, text="Calibration (px per µm):", width=24)
        lb.pack(side="left")
        self.ppum_var = tk.StringVar(value="2.6428")
        e = ttk.Entry(r, textvariable=self.ppum_var, width=12)
        e.pack(side="left")
        e.bind("<FocusOut>", lambda ev: self._request_preview())
        tip = ("How many pixels of your SOURCE image equal one micron, at the\n"
               "objective used for acquisition. Get it from the microscope\n"
               "software or a calibration slide. The bar length is converted\n"
               "automatically even after the image is resized or cropped.")
        ToolTip(lb, tip)
        ToolTip(e, tip)

        self._num_row(b, "Bar length (µm):", "bar_um", "20")
        self._num_row(b, "Bar column (-1 = last):", "bar_col", "-1")
        self._num_row(b, "Bar thickness (px or auto):", "bar_h", "auto")
        self._num_row(b, "Margin from corner (px):", "bar_margin", "20")
        self._color_row(b, "Bar colour:", "bar_color", (255, 255, 255))

        r = ttk.Frame(b)
        r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="Bar label (blank = none):", width=24).pack(side="left")
        self.barlabel_var = tk.StringVar(value="")
        e = ttk.Entry(r, textvariable=self.barlabel_var, width=16)
        e.pack(side="left")
        e.bind("<FocusOut>", lambda ev: self._request_preview())
        ToolTip(e, 'e.g. "20 µm" - drawn just above the bar.')

        r = ttk.Frame(b)
        r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="Rows (all, or 0,2,3):", width=24).pack(side="left")
        self.barrows_var = tk.StringVar(value="all")
        e = ttk.Entry(r, textvariable=self.barrows_var, width=16)
        e.pack(side="left")
        e.bind("<FocusOut>", lambda ev: self._request_preview())

    def _build_output(self):
        f = self.t_out
        b = ttk.LabelFrame(f, text="File format")
        b.pack(fill="x", padx=6, pady=6)

        r = ttk.Frame(b)
        r.pack(fill="x", padx=8, pady=6)
        lb = ttk.Label(r, text="Save as:", width=24)
        lb.pack(side="left")
        self.fmt_var = tk.StringVar(value="PNG")
        fc = ttk.Combobox(r, textvariable=self.fmt_var, width=10,
                          values=list(ce.OUTPUT_FORMATS), state="readonly")
        fc.pack(side="left")
        fc.bind("<<ComboboxSelected>>", lambda e: self._fmt_changed())
        ToolTip(lb, "Input format does not constrain the output - feed it TIFFs\n"
                    "and save JPEG, or any other combination.\n"
                    "PNG / WEBP keep transparency. TIFF / JPEG flatten onto white.")

        self._num_row(b, "JPEG quality (1-100):", "jpeg_q", "95")
        self._num_row(b, "DPI:", "dpi", "300")
        self.transparent_chk = self._check(b, "Transparent background "
                                              "(PNG and WEBP only)", "transparent", True)

        note = ttk.Label(b, text="", foreground="#b06000", justify="left")
        note.pack(anchor="w", padx=10, pady=(0, 6))
        self.fmt_note = note
        self._fmt_changed()

        a = ttk.Frame(f)
        a.pack(fill="x", padx=6, pady=10)
        ttk.Button(a, text="Save figure...", command=self._save).pack(side="left")
        ttk.Button(a, text="Batch: every subfolder",
                   command=self._batch).pack(side="left", padx=6)
        ToolTip(a, "Batch treats each immediate subfolder of the chosen folder\n"
                   "as its own figure and writes one file per subfolder.")

    def _build_preview(self, parent):
        b = ttk.LabelFrame(parent, text="Preview")
        b.pack(fill="both", expand=True)

        bar = ttk.Frame(b)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Refresh", command=self._request_preview).pack(side="left")
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Auto-refresh", variable=self.auto_var).pack(side="left", padx=8)
        self.prev_status = ttk.Label(bar, text="")
        self.prev_status.pack(side="right")

        self.canvas = tk.Canvas(b, background="#3c3c3c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        self.info = ttk.Label(b, text="Choose a folder to begin.", foreground="#666")
        self.info.pack(anchor="w", padx=8, pady=(0, 6))

    # -------------------------------------------------------- small widgets ---
    def _num_row(self, parent, label, key, default):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text=label, width=24).pack(side="left")
        var = tk.StringVar(value=default)
        e = ttk.Entry(r, textvariable=var, width=12)
        e.pack(side="left")
        e.bind("<FocusOut>", lambda ev: self._request_preview())
        e.bind("<Return>", lambda ev: self._request_preview())
        setattr(self, key + "_var", var)
        return var

    def _check(self, parent, label, key, default):
        var = tk.BooleanVar(value=default)
        c = ttk.Checkbutton(parent, text=label, variable=var,
                            command=self._request_preview)
        c.pack(anchor="w", padx=10, pady=3)
        setattr(self, key + "_var", var)
        return c

    def _color_row(self, parent, label, key, default):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text=label, width=24).pack(side="left")
        sw = tk.Canvas(r, width=28, height=18, highlightthickness=1,
                       highlightbackground="#888")
        sw.pack(side="left")
        sw.configure(background="#%02x%02x%02x" % default)
        setattr(self, key + "_rgb", list(default))
        setattr(self, key + "_swatch", sw)

        def pick():
            rgb, _ = colorchooser.askcolor(color="#%02x%02x%02x" % tuple(getattr(self, key + "_rgb")))
            if rgb:
                rgb = [int(c) for c in rgb]
                setattr(self, key + "_rgb", rgb)
                sw.configure(background="#%02x%02x%02x" % tuple(rgb))
                self._request_preview()

        ttk.Button(r, text="Pick...", command=pick).pack(side="left", padx=6)

    # -------------------------------------------------------------- actions ---
    def _pick_folder(self):
        d = filedialog.askdirectory(title="Choose the folder containing your image subfolders")
        if d:
            self.folder_var.set(os.path.normpath(d))
            self._scan()

    def _scan(self):
        for w in self.row_holder.winfo_children():
            w.destroy()
        self.row_enabled.clear()
        self.row_labels.clear()

        folder = self.folder_var.get().strip()
        if not os.path.isdir(folder):
            ttk.Label(self.row_holder, text="Not a folder.",
                      foreground="#c00000").pack(anchor="w")
            return
        try:
            cfg = ce.CollageConfig(main_folder=folder,
                                   exclude_generated=self.skip_generated_var.get())
            self.rows = ce.discover_rows(cfg)
        except (ValueError, FileNotFoundError) as exc:
            ttk.Label(self.row_holder, text=str(exc), foreground="#c00000",
                      wraplength=380).pack(anchor="w")
            self.rows = []
            return

        for r in self.rows:
            self._row_widget(r)
        # calibration from the images (stitched files carry it), so the scale
        # bar is right without typing the number
        px = 0.0
        for r in self.rows:
            for p in r["images"]:
                px = ce.file_px_per_um(p)
                if px:
                    break
            if px:
                break
        if px:
            self.ppum_var.set(f"{px:.4f}")
            self.info.config(text=f"Calibration read from the images: {px:.4f} px/µm "
                                  "(Scale bar tab).")
        self._request_preview()

    def open_folder(self, folder, column_contrast=None):
        """Show a folder (e.g. stitched images) - used by the pipeline's
        hand-over buttons."""
        self.folder_var.set(os.path.normpath(folder))
        if column_contrast is not None:
            self.col_contrast_var.set(bool(column_contrast))
        self._scan()

    def _mask_from_pipeline(self):
        try:
            txt = self.get_mask_text() or ""
        except Exception:
            txt = ""
        if not txt.strip():
            messagebox.showinfo(APP_TITLE, "Nothing is pasted on the Sample names tab.")
            return
        self.mask_text.delete("1.0", "end")
        self.mask_text.insert("1.0", txt.strip() + "\n")
        self._apply_mask()

    def _row_widget(self, r):
        name = r["name"]
        fr = ttk.Frame(self.row_holder)
        fr.pack(fill="x", pady=1)
        en = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr, variable=en, command=self._request_preview).pack(side="left")
        rb = ttk.Radiobutton(fr, text=name, value=name, variable=self.sel_row, width=12)
        rb.pack(side="left")
        lv = tk.StringVar(value=r["label"])
        e = ttk.Entry(fr, textvariable=lv)
        e.pack(side="left", fill="x", expand=True, padx=4)
        e.bind("<FocusOut>", lambda ev: self._request_preview())
        e.bind("<Return>", lambda ev: self._request_preview())
        ttk.Label(fr, text=f"{len(r['images'])} img",
                  foreground="#666", width=7).pack(side="left")
        self.row_enabled[name] = en
        self.row_labels[name] = lv

    def _mask_example(self):
        if LABELS_OK:
            self.mask_text.delete("1.0", "end")
            self.mask_text.insert("1.0", cargo_labels.EXAMPLE)

    def _apply_mask(self):
        """
        Replace each row's label with the real condition from the plate layout.

        Matching is on the WELL id found in the folder name, so a folder called
        A1 or pre_glucose_A1 both resolve to well A1. Rows whose well is not in
        the layout keep the label they already had rather than being blanked.
        """
        if not LABELS_OK:
            messagebox.showerror(APP_TITLE, "cargo_labels.py is missing.")
            return
        txt = self.mask_text.get("1.0", "end")
        if not txt.strip():
            return
        m, _ = cargo_labels.parse_mask(txt)
        if not m:
            messagebox.showerror(APP_TITLE, "No wells found in that layout.")
            return
        import re
        hits = 0
        for r in self.rows:
            mm = re.search(r"([A-H]\d{1,2})(?!.*[A-H]\d)", r["name"])
            well = mm.group(1) if mm else r["name"]
            if well in m:
                self.row_labels[r["name"]].set(m[well])
                hits += 1
        self._request_preview()
        messagebox.showinfo(APP_TITLE,
                            f"Labelled {hits} of {len(self.rows)} rows from the layout."
                            + ("" if hits == len(self.rows) else
                               "\n\nRows without a matching well kept their old label."))

    # ------------------------------------------------------- Fiji overlay ---
    def _merge_overlay(self):
        """
        Pick two images and open the interactive overlay editor, where per-
        channel contrast, brightness, gamma and colour update a live preview.
        Saving writes the overlay (directly, or through Fiji if that option is
        ticked in the editor). Produces a single merged file you can drop into
        a subfolder as a 'Merge' column.

        If overlay_editor.py is missing it falls back to the old one-shot Fiji
        merge with the default green/magenta pairing.
        """
        a = filedialog.askopenfilename(
            title="First image (channel A)",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        if not a:
            return
        b = filedialog.askopenfilename(
            title="Second image (channel B)",
            initialdir=os.path.dirname(a),
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        if not b:
            return

        if EDITOR_OK:
            try:
                res = overlay_editor.edit_overlay(
                    self.root, a, b,
                    color_a=self._merge_color_a, color_b=self._merge_color_b,
                    fiji_merge_module=(fiji_merge if FIJI_OK else None))
            except Exception as exc:
                messagebox.showerror(APP_TITLE, f"Editor failed:\n{exc}")
                return
            if res:
                self.info.config(text=f"Overlay written -> {res}")
                self._offer_open_folder(res)
            return

        # ---- fallback: no editor module, one-shot Fiji merge ----
        if not FIJI_OK:
            messagebox.showerror(
                APP_TITLE,
                "Neither overlay_editor.py nor fiji_merge.py is available.")
            return
        out = filedialog.asksaveasfilename(
            title="Save overlay as", defaultextension=".png",
            initialdir=os.path.dirname(a), initialfile="overlay.png",
            filetypes=[("PNG overlay", "*.png"), ("All files", "*.*")])
        if not out:
            return
        self.info.config(text="Merging in Fiji... (first run can take a while)")
        self.root.update_idletasks()
        ca, cb = self._merge_color_a, self._merge_color_b

        def worker():
            try:
                path = fiji_merge.merge_two(a, b, out, color_a=ca, color_b=cb)
                self.root.after(0, lambda: self._merged_ok(path))
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: (
                    self.info.config(text="Merge failed."),
                    messagebox.showerror(APP_TITLE, msg)))

        threading.Thread(target=worker, daemon=True).start()

    def _offer_open_folder(self, path):
        if messagebox.askyesno(APP_TITLE,
                               f"Overlay saved:\n{path}\n\nOpen its folder?"):
            folder = os.path.dirname(path)
            try:
                if os.name == "nt":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])
            except Exception:
                pass

    def _single_scalebar(self):
        """
        Open the single-image scale-bar tool: drop in (or browse to) one image
        and burn a correctly calibrated bar onto it. Uses the same calibration
        the collage engine uses, and seeds the tool with the calibration and
        bar length currently set on the Scale bar tab.
        """
        if not SCALEBAR_OK:
            messagebox.showerror(
                APP_TITLE,
                "scalebar_tool.py is missing - add it next to this script.")
            return
        try:
            ppum = float(self.ppum_var.get())
        except (ValueError, AttributeError):
            ppum = 2.6428
        try:
            length_um = float(self.bar_um_var.get())
        except (ValueError, AttributeError):
            length_um = 100.0
        try:
            res = scalebar_tool.open_scalebar_tool(
                self.root, pixels_per_um=ppum, length_um=length_um)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Scale-bar tool failed:\n{exc}")
            return
        if res:
            self.info.config(text=f"Scale-bar image written -> {res}")
            self._offer_open_folder(res)



    def _move(self, delta):
        sel = self.sel_row.get()
        if not sel:
            messagebox.showinfo(APP_TITLE, "Select a row first (click its name).")
            return
        names = [r["name"] for r in self.rows]
        if sel not in names:
            return
        i = names.index(sel)
        j = i + delta
        if not (0 <= j < len(self.rows)):
            return
        # capture edits before rebuilding the widgets
        labels = {n: v.get() for n, v in self.row_labels.items()}
        enabled = {n: v.get() for n, v in self.row_enabled.items()}
        self.rows[i], self.rows[j] = self.rows[j], self.rows[i]
        for w in self.row_holder.winfo_children():
            w.destroy()
        self.row_enabled.clear()
        self.row_labels.clear()
        for r in self.rows:
            self._row_widget(r)
            self.row_labels[r["name"]].set(labels.get(r["name"], r["label"]))
            self.row_enabled[r["name"]].set(enabled.get(r["name"], True))
        self._request_preview()

    def _toggle_all(self, state):
        for v in self.row_enabled.values():
            v.set(state)
        self._request_preview()

    def _fmt_changed(self):
        fmt = self.fmt_var.get()
        has_alpha = ce.OUTPUT_FORMATS[fmt][2]
        if has_alpha:
            self.fmt_note.config(text="")
        else:
            self.fmt_note.config(
                text=f"{fmt} cannot store transparency - the background will be "
                     "flattened onto white.")
        self._request_preview()

    # ---------------------------------------------------------- config build ---
    def _num(self, var, default, allow_auto=True):
        v = var.get().strip()
        if allow_auto and v.lower() == "auto":
            return "auto"
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return default

    def _cfg(self):
        folder = self.folder_var.get().strip()
        labels = {n: v.get() for n, v in self.row_labels.items()}
        col = [s.strip() for s in self.collabels_var.get().split(",") if s.strip()]

        rows_sel = self.barrows_var.get().strip().lower()
        if rows_sel in ("", "all"):
            bar_rows = "all"
        else:
            try:
                bar_rows = [int(x) for x in rows_sel.replace(";", ",").split(",") if x.strip()]
            except ValueError:
                bar_rows = "all"

        cfg = ce.CollageConfig(
            main_folder=folder,
            label_overrides=labels,
            blocks_across=int(self.blocks_var.get()),
            block_gap=self._num(self.block_gap_var, 80, False),
            repeat_column_labels=self.repeat_headers_var.get(),
            cell_w=self._num(self.cell_w_var, "auto"),
            cell_h=self._num(self.cell_h_var, "auto"),
            fit_mode=self.fit_var.get(),
            grid_gap=self._num(self.grid_gap_var, "auto"),
            outer_pad=self._num(self.outer_pad_var, 20, False),
            column_labels=col,
            column_label_color=tuple(self.col_color_rgb) + (255,),
            column_label_font_scale=float(self._num(self.col_font_scale_var, 0.09, False)),
            column_label_margin=self._num(self.col_margin_var, 12, False),
            row_label_w=self._num(self.row_label_w_var, "auto"),
            row_label_font_scale=float(self._num(self.row_font_scale_var, 0.09, False)),
            row_label_color=tuple(self.row_color_rgb) + (255,),
            row_label_align=self.align_var.get(),
            scale_bar_enabled=self.bar_on_var.get(),
            scale_bar_column=int(self._num(self.bar_col_var, -1, False)),
            scale_bar_rows=bar_rows,
            scale_bar_length_um=float(self._num(self.bar_um_var, 20, False)),
            scale_bar_pixels_per_um=float(self._num(self.ppum_var, 2.6428, False)),
            scale_bar_height_px=self._num(self.bar_h_var, "auto"),
            scale_bar_margin=self._num(self.bar_margin_var, 20, False),
            scale_bar_color=tuple(self.bar_color_rgb) + (255,),
            scale_bar_label=(self.barlabel_var.get().strip() or None),
            exclude_generated=self.skip_generated_var.get(),
            column_contrast=self.col_contrast_var.get(),
            transparent_background=self.transparent_var.get(),
            output_format=self.fmt_var.get(),
            jpeg_quality=int(self._num(self.jpeg_q_var, 95, False)),
            dpi=(int(self._num(self.dpi_var, 300, False)),) * 2,
        )
        rows = [r for r in self.rows if self.row_enabled.get(r["name"], tk.BooleanVar(value=True)).get()]
        for r in rows:
            r["label"] = labels.get(r["name"], r["label"])
        return cfg, rows

    # -------------------------------------------------------------- preview ---
    def _request_preview(self, *_):
        # Settings callbacks can fire while the UI is still being constructed
        # (a Combobox setting its initial value, for instance), before the
        # preview pane exists. Nothing to refresh yet in that case.
        if not hasattr(self, "auto_var") or not self.auto_var.get() or not self.rows:
            return
        if self.rendering:
            self.pending = True
            return
        self._start_preview()

    def _start_preview(self):
        try:
            cfg, rows = self._cfg()
        except Exception:
            return
        if not rows:
            self.info.config(text="No rows selected.")
            self.canvas.delete("all")
            return
        self.rendering = True
        self.prev_status.config(text="rendering...")
        # read the canvas size HERE: Tk must not be called from the worker thread
        size = (max(self.canvas.winfo_width(), 400), max(self.canvas.winfo_height(), 300))
        threading.Thread(target=self._preview_worker, args=(cfg, rows, size),
                         daemon=True).start()

    def _preview_worker(self, cfg, rows, size=(900, 700)):
        try:
            img, scale = ce.render_preview(cfg, max_px=max(size), rows=rows)
            self.render_queue.put(("ok", img, scale, cfg, rows))
        except Exception:
            self.render_queue.put(("err", traceback.format_exc(), None, None, None))

    def _poll(self):
        try:
            while True:
                kind, a, b, c, d = self.render_queue.get_nowait()
                self.rendering = False
                self.prev_status.config(text="")
                if kind == "ok":
                    self.preview_src = a
                    self.preview_scale = b
                    self._redraw()
                    est_w = int(a.width / max(b, 1e-6))
                    est_h = int(a.height / max(b, 1e-6))
                    self.info.config(
                        text=f"Preview at {b*100:.0f}%  -  full size will be about "
                             f"{est_w} x {est_h} px")
                else:
                    self.info.config(text="Preview failed - see message")
                    messagebox.showerror(APP_TITLE, a.splitlines()[-1])
                if self.pending:
                    self.pending = False
                    self._start_preview()
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _redraw(self):
        img = getattr(self, "preview_src", None)
        if img is None:
            return
        cw = max(self.canvas.winfo_width(), 10)
        chh = max(self.canvas.winfo_height(), 10)
        s = min(cw / img.width, chh / img.height, 1.0)
        disp = img if s >= 1.0 else img.resize(
            (max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        # checkerboard so transparency is visible rather than guessed at
        bg = Image.new("RGB", disp.size, (200, 200, 200))
        for yy in range(0, disp.height, 16):
            for xx in range(0, disp.width, 16):
                if (xx // 16 + yy // 16) % 2 == 0:
                    bg.paste((170, 170, 170), (xx, yy, min(xx + 16, disp.width),
                                               min(yy + 16, disp.height)))
        bg.paste(disp, (0, 0), disp if disp.mode == "RGBA" else None)
        self.preview_img = ImageTk.PhotoImage(bg)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, chh // 2, image=self.preview_img)

    # ----------------------------------------------------------------- save ---
    def _save(self):
        if not self.rows:
            messagebox.showerror(APP_TITLE, "Choose a folder with images first.")
            return
        cfg, rows = self._cfg()
        if not rows:
            messagebox.showerror(APP_TITLE, "No rows are ticked.")
            return
        ext = ce.OUTPUT_FORMATS[cfg.output_format][1]
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            initialdir=cfg.main_folder,
            initialfile="collage" + ext,
            filetypes=[(cfg.output_format, "*" + ext), ("All files", "*.*")])
        if not path:
            return
        self.info.config(text="Rendering at full resolution...")
        self.root.update_idletasks()
        threading.Thread(target=self._save_worker, args=(cfg, rows, path),
                         daemon=True).start()

    def _save_worker(self, cfg, rows, path):
        try:
            canvas = ce.render(cfg, scale=1.0, rows=rows)
            out = ce.save(canvas, path, cfg)
            self.root.after(0, lambda: self._saved(out, canvas.size))
        except Exception:
            tb = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror(APP_TITLE, tb.splitlines()[-1]))

    def _saved(self, path, size):
        self.info.config(text=f"Saved {size[0]} x {size[1]} px  ->  {path}")
        messagebox.showinfo(APP_TITLE, f"Saved:\n{path}\n\n{size[0]} x {size[1]} px")

    def _batch(self):
        parent = filedialog.askdirectory(
            title="Choose the folder whose SUBFOLDERS each become one figure")
        if not parent:
            return
        subs = sorted([d for d in os.listdir(parent)
                       if os.path.isdir(os.path.join(parent, d))], key=ce.natural_key)
        if not subs:
            messagebox.showerror(APP_TITLE, "No subfolders found.")
            return
        if not messagebox.askyesno(
                APP_TITLE,
                f"Build one figure for each of these {len(subs)} folders?\n\n"
                + ", ".join(subs[:12]) + ("..." if len(subs) > 12 else "")):
            return
        cfg, _ = self._cfg()
        threading.Thread(target=self._batch_worker, args=(cfg, parent, subs),
                         daemon=True).start()

    def _batch_worker(self, cfg, parent, subs):
        ok, bad = [], []
        for s in subs:
            try:
                sub_cfg = ce.CollageConfig.from_dict(cfg.to_dict())
                sub_cfg.main_folder = os.path.join(parent, s)
                rows = ce.discover_rows(sub_cfg)
                canvas = ce.render(sub_cfg, scale=1.0, rows=rows)
                ext = ce.OUTPUT_FORMATS[sub_cfg.output_format][1]
                out = ce.save(canvas, os.path.join(parent, s, "collage" + ext), sub_cfg)
                ok.append(os.path.basename(out))
                self.root.after(0, lambda s=s: self.info.config(text=f"batch: {s} done"))
            except Exception as exc:
                bad.append(f"{s}: {exc}")
        msg = f"Finished {len(ok)} of {len(subs)}."
        if bad:
            msg += "\n\nFailed:\n" + "\n".join(bad[:10])
        self.root.after(0, lambda: messagebox.showinfo(APP_TITLE, msg))


def main():
    # A TkinterDnD root enables OS drag-and-drop in the scale-bar tool. If the
    # library is absent we fall back to a plain Tk root - everything still works,
    # the drop zone just becomes click-to-browse.
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    CollageGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()