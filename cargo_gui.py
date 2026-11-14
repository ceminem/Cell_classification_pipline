#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_gui.py - graphical front-end for the cargo-release pipeline
=============================================================================
One window that:
  - locates Fiji and the Cellpose environment (and checks they actually work)
  - lets you pick the parent folder and tick which samples to run
  - exposes every threshold without editing any source file
  - runs the pipeline and streams the log live
  - opens the results folder when it finishes

WHAT THIS IS NOT: it does not bundle Fiji, Cellpose, Python or conda. Those are
multi-gigabyte installs with their own GPU drivers and Java runtimes; no single
downloadable .exe can legally or practically contain them. Run setup_check to
verify a machine, and INSTALL.md to set one up from scratch.

LAUNCH:  python cargo_gui.py
         (or double-click Start_Cargo_GUI.bat on Windows)

tkinter ships with Python, so there is nothing extra to install for the GUI.
=============================================================================
"""

import json
import os
import queue
import subprocess
import sys
import threading
import traceback

try:
    import tkinter as tk
    from tkinter import (filedialog, messagebox, ttk, scrolledtext,
                         colorchooser, simpledialog)
except ImportError:
    sys.exit(
        "ERROR: tkinter is not available.\n"
        "On Windows/macOS it ships with Python. On Linux install python3-tk:\n"
        "    sudo apt install python3-tk"
    )

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is required.")

# The figures tab is optional: matplotlib is not needed to run an analysis, so
# a missing install degrades to a message in that tab rather than stopping the
# whole program from opening.
try:
    import cargo_plot as cp
    PLOT_OK = cp.HAVE_MPL
except ImportError:
    cp = None
    PLOT_OK = False

try:
    import cargo_labels
    LABELS_OK = True
except ImportError:
    cargo_labels = None
    LABELS_OK = False

try:
    import cargo_interactive as ci
    INTERACT_OK = True
except ImportError:
    ci = None
    INTERACT_OK = False

APP_TITLE = "Cargo Release Analysis"
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cargo_gui_settings.json")
PIPELINE_SCRIPT = "run_cargo_pipeline.py"
MACRO_FILE = "Cargo_Release_Classification_v3_headless.ijm"

# Settings shown in the Settings tab: (key, label, default, tooltip)
TUNABLES = [
    ("fill_ratio_threshold", "Fill ratio threshold", "0.65",
     "coreMean / peripheryMean must be >= this to call a cell released.\n"
     "High = cargo spread into the cell centre. CALIBRATE against cells you\n"
     "would score by eye."),
    ("core_cv_cap", "Core CV cap", "0.65",
     "A released cell must also be SMOOTH. Low CV = even fill; high CV =\n"
     "bright puncta among dark pixels. Fitted on 523 blind labels:\n"
     "brightness alone 80%, brightness AND this cap 88%.\n"
     "Set 2.0 to disable and use brightness only."),
    ("brightness_threshold", "Brightness threshold", "25.7",
     "Whole-cell mean at or above this counts as released, provided the\n"
     "CV cap above is also satisfied."),
    ("cellprob_threshold", "Cellpose cellprob", "0.0",
     "MAIN KNOB FOR MISSED CELLS. Lower accepts fainter cells:\n"
     "  0.0 default,  -1.0 more cells,  -2.0 aggressive.\n"
     "Change this and re-run with 'force' ticked, or the old mask is reused."),
    ("core_cv_threshold", "Legacy CV threshold", "0.60",
     "coreStdDev / coreMean must be <= this to call a cell released.\n"
     "Low = smooth homogeneous fill. High = punctate NPs still trapped."),
    ("min_cell_mean", "Min cell mean intensity", "5",
     "Whole-cell mean below this = no uptake at all; the cell is excluded\n"
     "from the release statistics entirely."),
    ("core_radius_fraction", "Core radius fraction", "0.5",
     "Where to split each cell. 0.5 = inner 50% of the radius is the CORE,\n"
     "the outer rim is the PERIPHERY."),
    ("rolling_ball_radius", "Rolling ball radius (px)", "50",
     "Background subtraction radius applied to the fluorescence channel."),
    ("cellpose_scale_factor", "Cellpose downscale", "0.5",
     "0.5 = segment at half resolution (~4x faster). 1.0 = full resolution.\n"
     "The mask is scaled back up before measurement either way."),
    ("flow_threshold", "Cellpose flow threshold", "0.4",
     "Cellpose default is 0.4. Raise to accept more cells."),
    ("cellprob_threshold", "Cellpose cellprob threshold", "0.0",
     "Cellpose default is 0.0. LOWER (e.g. -2) finds more/fainter cells."),
]

# How much RAM the machine has -> how much heap Fiji is given (about half).
# Offered as a dropdown because "how much RAM does this PC have" is a question
# people can answer, whereas "what Java heap size" is not.
RAM_PRESETS = [
    ("4 GB  (older office PC)", 2048),
    ("8 GB  (typical office PC)", 4096),
    ("16 GB (workstation)", 8192),
    ("32 GB (analysis PC)", 12288),
    ("64 GB or more", 24576),
]


class ToolTip:
    """Minimal hover tooltip - tkinter has none built in."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left",
                 background="#ffffe0", relief="solid", borderwidth=1,
                 font=("Segoe UI", 8)).pack(ipadx=4, ipady=2)

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class CargoGUI:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("880x720")
        root.minsize(760, 600)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.proc = None
        self.log_queue = queue.Queue()
        self.sample_vars = {}

        self._build_ui()
        self._load_settings()
        self._poll_log()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_run = ttk.Frame(nb)
        self.tab_settings = ttk.Frame(nb)
        self.tab_labels = ttk.Frame(nb)
        self.tab_plot = ttk.Frame(nb)
        self.tab_setup = ttk.Frame(nb)
        nb.add(self.tab_run, text="  Run  ")
        nb.add(self.tab_labels, text="  Sample names  ")
        nb.add(self.tab_settings, text="  Settings  ")
        nb.add(self.tab_plot, text="  Figures  ")
        nb.add(self.tab_setup, text="  Setup check  ")

        self._build_run_tab()
        self._build_settings_tab()
        self._build_labels_tab()
        self._build_plot_tab()
        self._build_setup_tab()

    def _build_run_tab(self):
        f = self.tab_run

        # --- data folder ---
        box = ttk.LabelFrame(f, text="Data")
        box.pack(fill="x", padx=6, pady=6)

        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Parent folder:", width=14).pack(side="left")
        self.parent_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.parent_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Browse...", command=self._pick_parent).pack(side="left")

        hint = ttk.Label(
            box,
            text="The folder that CONTAINS the sample subfolders (A1, A2, B1 ...), "
                 "not a sample folder itself.",
            foreground="#666")
        hint.pack(anchor="w", padx=10, pady=(0, 4))

        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Save results to:", width=14).pack(side="left")
        self.output_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Browse...", command=self._pick_output).pack(side="left")
        ttk.Button(row, text="Clear",
                   command=lambda: self.output_var.set("")).pack(side="left", padx=2)
        ttk.Label(box, foreground="#666", justify="left", wraplength=470,
                  text="Leave blank to write results next to each sample, in "
                       "<sample>/edited_results/. Choose a folder and everything is "
                       "collected there instead, inside a subfolder named after the "
                       "data location.").pack(anchor="w", padx=10, pady=(0, 6))

        # --- samples ---
        sbox = ttk.LabelFrame(f, text="Samples")
        sbox.pack(fill="both", expand=False, padx=6, pady=6)

        btnrow = ttk.Frame(sbox)
        btnrow.pack(fill="x", padx=6, pady=4)
        ttk.Button(btnrow, text="Refresh", command=self._scan_samples).pack(side="left")
        ttk.Button(btnrow, text="Select all",
                   command=lambda: self._set_all(True)).pack(side="left", padx=4)
        ttk.Button(btnrow, text="Select none",
                   command=lambda: self._set_all(False)).pack(side="left")

        self.sample_frame = ttk.Frame(sbox)
        self.sample_frame.pack(fill="x", padx=10, pady=4)
        self.sample_hint = ttk.Label(self.sample_frame,
                                     text="Choose a parent folder to list samples.",
                                     foreground="#666")
        self.sample_hint.pack(anchor="w")

        # --- options + run ---
        obox = ttk.Frame(f)
        obox.pack(fill="x", padx=6, pady=4)
        self.force_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(obox, text="Re-run Cellpose even if a mask exists (--force)",
                             variable=self.force_var)
        cb.pack(side="left")
        ToolTip(cb, "Tick this after changing any Cellpose setting,\n"
                    "otherwise the existing mask is reused unchanged.")

        self.run_btn = ttk.Button(obox, text="Run analysis", command=self._run)
        self.run_btn.pack(side="right")
        self.stop_btn = ttk.Button(obox, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="right", padx=6)

        # --- log ---
        lbox = ttk.LabelFrame(f, text="Log")
        lbox.pack(fill="both", expand=True, padx=6, pady=6)
        self.log = scrolledtext.ScrolledText(lbox, height=16, wrap="word",
                                             font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log.tag_config("err", foreground="#c00000")
        self.log.tag_config("ok", foreground="#008000")
        self.log.tag_config("warn", foreground="#b06000")

        bottom = ttk.Frame(f)
        bottom.pack(fill="x", padx=6, pady=(0, 6))
        self.status = ttk.Label(bottom, text="Idle")
        self.status.pack(side="left")
        ttk.Button(bottom, text="Open results folder",
                   command=self._open_results).pack(side="right")

    def _build_settings_tab(self):
        f = self.tab_settings

        pbox = ttk.LabelFrame(f, text="Program locations")
        pbox.pack(fill="x", padx=6, pady=6)

        row = ttk.Frame(pbox)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Fiji folder:", width=14).pack(side="left")
        self.fiji_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.fiji_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Browse...", command=self._pick_fiji).pack(side="left")

        ttk.Label(pbox,
                  text="The folder containing fiji-windows-x64.exe (or ImageJ-win64.exe).",
                  foreground="#666").pack(anchor="w", padx=10, pady=(0, 4))

        tbox = ttk.LabelFrame(f, text="Analysis parameters  (hover each for an explanation)")
        tbox.pack(fill="both", expand=True, padx=6, pady=6)

        self.tune_vars = {}
        for i, (key, label, default, tip) in enumerate(TUNABLES):
            r = ttk.Frame(tbox)
            r.pack(fill="x", padx=8, pady=3)
            lb = ttk.Label(r, text=label, width=28)
            lb.pack(side="left")
            var = tk.StringVar(value=default)
            ent = ttk.Entry(r, textvariable=var, width=12)
            ent.pack(side="left")
            self.tune_vars[key] = var
            ToolTip(lb, tip)
            ToolTip(ent, tip)

        self.gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tbox, text="Use GPU if available",
                        variable=self.gpu_var).pack(anchor="w", padx=8, pady=6)

        # RAM picker -> Fiji heap. Asking for the machine's RAM is answerable;
        # asking for a Java heap size is not.
        r = ttk.Frame(tbox)
        r.pack(fill="x", padx=8, pady=3)
        lb = ttk.Label(r, text="This computer has", width=28)
        lb.pack(side="left")
        self.ram_var = tk.StringVar(value=RAM_PRESETS[3][0])
        cb = ttk.Combobox(r, textvariable=self.ram_var, width=26, state="readonly",
                          values=[p[0] for p in RAM_PRESETS])
        cb.pack(side="left")
        tip = ("Sets how much memory Fiji is allowed to use (about half your RAM).\n"
               "Too little makes large images crawl; more than the machine has\n"
               "stops Fiji starting at all. Pick your actual RAM and forget it.")
        ToolTip(lb, tip)
        ToolTip(cb, tip)

        warn = ttk.Label(
            tbox,
            text="The two threshold values decide every released / not-released call.\n"
                 "They are starting estimates, NOT validated against your data. Check the\n"
                 "percentile lines in the log and the overlay before trusting any number.",
            foreground="#b06000", justify="left")
        warn.pack(anchor="w", padx=8, pady=(4, 6))

        brow = ttk.Frame(f)
        brow.pack(fill="x", padx=6, pady=6)
        ttk.Button(brow, text="Save settings", command=self._save_settings).pack(side="left")
        ttk.Button(brow, text="Reset to defaults",
                   command=self._reset_settings).pack(side="left", padx=6)

    def _build_labels_tab(self):
        """
        Paste a plate layout from Excel so results are named by condition.

        The preview is the point: a mislabelled plate silently mislabels every
        downstream file and figure, and that is very hard to notice later. The
        table below shows exactly what each well will be called BEFORE anything
        is written.
        """
        f = self.tab_labels

        if not LABELS_OK:
            ttk.Label(f, foreground="#c00000",
                      text="cargo_labels.py is missing from this folder.")\
                .pack(anchor="w", padx=14, pady=14)
            return

        ttk.Label(f, justify="left", wraplength=760, foreground="#444",
                  text="Copy the layout table out of Excel and paste it here. "
                       "Each COLUMN becomes a well letter and each row a position, "
                       "so the first column is A1 (top) and A2 (below), the next "
                       "is B1/B2, and so on - matching how the plate is filled.")\
            .pack(anchor="w", padx=10, pady=(10, 4))

        r = ttk.Frame(f)
        r.pack(fill="x", padx=10, pady=4)
        ttk.Label(r, text="Layout:").pack(side="left")
        self.lbl_orient = tk.StringVar(value="column-major")
        ocb = ttk.Combobox(r, textvariable=self.lbl_orient, state="readonly",
                           width=26,
                           values=["column-major", "row-major"])
        ocb.pack(side="left", padx=6)
        ocb.bind("<<ComboboxSelected>>", lambda e: self._labels_preview())
        ToolTip(ocb,
                "column-major - each COLUMN of the table is one well letter.\n"
                "               This matches your iron-fumarate mask.\n"
                "row-major    - each ROW is a well letter instead.\n"
                "Check the preview: getting this backwards would mislabel "
                "every sample.")

        ttk.Button(r, text="Paste example", command=self._labels_example)\
            .pack(side="left", padx=6)
        ttk.Button(r, text="Load from file...", command=self._labels_load)\
            .pack(side="left")
        ttk.Button(r, text="Clear",
                   command=lambda: (self.lbl_text.delete("1.0", "end"),
                                    self._labels_preview())).pack(side="left", padx=6)

        self.lbl_text = scrolledtext.ScrolledText(f, height=8, wrap="none",
                                                  font=("Consolas", 9))
        self.lbl_text.pack(fill="x", expand=False, padx=10, pady=6)
        self.lbl_text.bind("<KeyRelease>", lambda e: self._labels_preview())
        self.lbl_text.bind("<<Paste>>", lambda e: self.root.after(50, self._labels_preview))

        ttk.Label(f, text="Preview - this is how results will be named:",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10)
        self.lbl_prev = scrolledtext.ScrolledText(f, height=14, wrap="none",
                                                  font=("Consolas", 8))
        self.lbl_prev.pack(fill="both", expand=True, padx=10, pady=6)

        r = ttk.Frame(f)
        r.pack(fill="x", padx=10, pady=(0, 10))
        self.use_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(r, text="Use these names for analysis output",
                        variable=self.use_labels).pack(side="left")
        ttk.Button(r, text="Save layout...", command=self._labels_save).pack(side="right")

    def _labels_example(self):
        self.lbl_text.delete("1.0", "end")
        self.lbl_text.insert("1.0", cargo_labels.EXAMPLE)
        self._labels_preview()

    def _labels_load(self):
        p = filedialog.askopenfilename(
            title="Open a saved layout",
            filetypes=[("Text/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")])
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as fh:
                self.lbl_text.delete("1.0", "end")
                self.lbl_text.insert("1.0", fh.read())
            self._labels_preview()
        except OSError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _labels_save(self):
        txt = self.lbl_text.get("1.0", "end").strip()
        if not txt:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="plate_layout.txt",
            filetypes=[("Text", "*.txt")])
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(txt + "\n")
            messagebox.showinfo(APP_TITLE, f"Saved:\n{p}")
        except OSError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _label_map(self, verbose=False):
        """
        The parsed plate layout, or {} if it cannot be used.

        Each failure path reports itself when verbose: an empty mapping used to
        be indistinguishable from "no layout pasted", which meant a run would
        quietly fall back to well ids with nothing to explain why.
        """
        why = None
        m = {}
        if not LABELS_OK:
            why = "cargo_labels.py is not importable from this folder"
        elif not hasattr(self, "lbl_text"):
            why = "the Sample names tab was not built"
        elif hasattr(self, "use_labels") and not self.use_labels.get():
            why = ("the 'Use these names for analysis output' box on the "
                   "Sample names tab is unticked")
        else:
            txt = self.lbl_text.get("1.0", "end")
            if not txt.strip():
                why = "no layout has been pasted on the Sample names tab"
            else:
                try:
                    m, _ = cargo_labels.parse_mask(txt, self.lbl_orient.get())
                    if not m:
                        why = ("the pasted layout produced no wells - check the "
                               "orientation setting")
                except Exception as exc:
                    why = f"the layout could not be parsed ({exc})"
        if verbose:
            self._label_why = why
        return m

    def _labels_preview(self, *_):
        if not hasattr(self, "lbl_prev"):
            return
        self.lbl_prev.delete("1.0", "end")
        txt = self.lbl_text.get("1.0", "end")
        if not txt.strip():
            self.lbl_prev.insert("1.0", "Paste a layout above to see the preview.")
            return
        try:
            m, notes = cargo_labels.parse_mask(txt, self.lbl_orient.get())
        except Exception as exc:
            self.lbl_prev.insert("1.0", f"Could not read that layout: {exc}")
            return
        if not m:
            self.lbl_prev.insert("1.0",
                                 "No wells were found. Check the orientation, or "
                                 "that the table was pasted with its columns intact.")
            return

        suffix = os.path.basename(self.parent_var.get().strip().rstrip("\\/")) \
            if hasattr(self, "parent_var") else ""
        out = [cargo_labels.preview(m, suffix)]

        names = [cargo_labels.build_name(w, l, suffix) for w, l in m.items()]
        if len(set(names)) != len(names):
            out.append("")
            out.append("WARNING: two or more wells produce the SAME name. They "
                       "would overwrite each other - make the labels distinct.")

        # Cross-check against the folders actually selected on the Run tab.
        # A layout pasted with the wrong orientation still parses cleanly - it
        # just maps to the wrong wells - so the only way to catch it is to see
        # whether the wells it produces exist in the data.
        have = set(getattr(self, "sample_vars", {}) or {})
        if have:
            mapped = set(m)
            missing = sorted(have - mapped)
            extra = sorted(mapped - have)
            out.append("")
            if missing:
                out.append(f"WARNING: no label for {', '.join(missing)} - these "
                           "samples will keep their well id.")
            if extra:
                out.append(f"Note: the layout also covers {', '.join(extra[:8])}"
                           + (" ..." if len(extra) > 8 else "")
                           + " which are not among the selected samples.")
            if not missing and not extra:
                out.append(f"All {len(have)} selected samples have a label.")
            if missing and len(missing) == len(have):
                out.append("NONE of the selected samples matched. The "
                           "orientation setting is probably wrong - try the "
                           "other one and watch this line.")

        if notes:
            out.append("")
            out.append("Column headers read as: " + "; ".join(notes))
        self.lbl_prev.insert("1.0", "\n".join(out))

    def _build_plot_tab(self):
        """
        Figures tab: load batch summaries, assign samples to groups, style the
        chart, run a test, export.

        The layout deliberately puts REPLICATE STRUCTURE next to the test
        selector. Choosing a test without saying what the replicates are is
        how pseudoreplication happens, and the numbers here come from ~2000
        cells per image - it would be very easy to produce a p-value that
        reflects the cell count rather than the experiment.
        """
        f = self.tab_plot

        if not PLOT_OK:
            ttk.Label(f, justify="left", foreground="#c00000",
                      text="matplotlib is not installed, so figures are unavailable.\n\n"
                           "Install it with:\n    pip install matplotlib\n\n"
                           "then reopen this window.").pack(anchor="w", padx=14, pady=14)
            return

        outer = ttk.PanedWindow(f, orient="horizontal")
        outer.pack(fill="both", expand=True, padx=4, pady=4)
        ctrl = ttk.Frame(outer)
        view = ttk.Frame(outer)
        outer.add(ctrl, weight=2)
        outer.add(view, weight=3)

        # ---------------- data ----------------
        db = ttk.LabelFrame(ctrl, text="1. Data")
        db.pack(fill="x", padx=4, pady=4)
        r = ttk.Frame(db)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Button(r, text="Add summary CSV...",
                   command=self._plot_add_files).pack(side="left")
        ttk.Button(r, text="Clear", command=self._plot_clear).pack(side="left", padx=4)
        self.plot_files_lbl = ttk.Label(db, text="no files loaded",
                                        foreground="#666", wraplength=330,
                                        justify="left")
        self.plot_files_lbl.pack(anchor="w", padx=8, pady=(0, 6))

        r = ttk.Frame(db)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Value:", width=8).pack(side="left")
        self.plot_value = tk.StringVar(value=cp.VALUE_COLUMNS[0][0])
        self.plot_value_cb = ttk.Combobox(
            r, textvariable=self.plot_value, state="readonly", width=34,
            values=[d for _, d in cp.VALUE_COLUMNS])
        self.plot_value_cb.current(0)
        self.plot_value_cb.pack(side="left")
        self.plot_value_cb.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())

        # ---------------- groups ----------------
        gb = ttk.LabelFrame(ctrl, text="2. Groups  (one bar per group)")
        gb.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Label(gb, foreground="#666", justify="left", font=("Segoe UI", 8),
                  wraplength=330,
                  text="Type a group name against each sample. Samples sharing a "
                       "name become one bar. Leave blank to exclude a sample.")\
            .pack(anchor="w", padx=8, pady=(2, 4))

        canv = tk.Canvas(gb, highlightthickness=0, height=200)
        sb = ttk.Scrollbar(gb, orient="vertical", command=canv.yview)
        self.plot_rows_frame = ttk.Frame(canv)
        self.plot_rows_frame.bind(
            "<Configure>", lambda e: canv.configure(scrollregion=canv.bbox("all")))
        canv.create_window((0, 0), window=self.plot_rows_frame, anchor="nw")
        canv.configure(yscrollcommand=sb.set)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        sb.pack(side="right", fill="y", pady=4)

        auto = ttk.Frame(gb)
        auto.pack(fill="x", padx=6, pady=4)
        ttk.Button(auto, text="Auto: by file",
                   command=self._plot_autogroup).pack(side="left")
        ttk.Button(auto, text="Auto: by well",
                   command=self._plot_autogroup_well).pack(side="left", padx=4)
        ttk.Button(auto, text="Auto: by label",
                   command=self._plot_autogroup_label).pack(side="left")
        ttk.Button(auto, text="Apply", command=self._plot_refresh).pack(side="right")

        r = ttk.Frame(gb)
        r.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(r, text="Layout:", width=8).pack(side="left")
        self.plot_layout = tk.StringVar(value="simple")
        lcb = ttk.Combobox(r, textvariable=self.plot_layout, state="readonly",
                           width=28, values=["simple", "grouped (cluster by well)"])
        lcb.pack(side="left")
        lcb.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())
        ToolTip(lcb,
                "simple  - one bar per group.\n"
                "grouped - each well becomes a cluster and each condition a bar\n"
                "          inside it, so A1_pre, A1_after and A1_1h sit together.\n"
                "          Use 'Auto: by well' to fill the names in.")

        # ---------------- replicates + stats ----------------
        sb2 = ttk.LabelFrame(ctrl, text="3. Replicates and statistics")
        sb2.pack(fill="x", padx=4, pady=4)

        r = ttk.Frame(sb2)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Replicates:", width=11).pack(side="left")
        self.rep_type = tk.StringVar(value="technical")
        rcb = ttk.Combobox(r, textvariable=self.rep_type, state="readonly", width=22,
                           values=["technical", "biological", "unspecified"])
        rcb.pack(side="left")
        rcb.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())
        ToolTip(rcb,
                "technical   - wells or fields from ONE experiment. Measures assay\n"
                "              precision, not whether the biology reproduces.\n"
                "biological  - independent experiments, separate days/passages.\n"
                "              Only these support a claim about the effect.\n"
                "unspecified - descriptive only, no test reported.")
        ttk.Label(sb2, foreground="#a06000", justify="left", font=("Segoe UI", 8),
                  wraplength=330,
                  text="This choice changes what the report says the numbers mean. "
                       "It does not change the arithmetic - it changes the claim "
                       "the arithmetic can support.")\
            .pack(anchor="w", padx=8)

        r = ttk.Frame(sb2)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Test:", width=8).pack(side="left")
        self.plot_test = tk.StringVar(value="auto (recommended)")
        tcb = ttk.Combobox(r, textvariable=self.plot_test, state="readonly", width=22,
                           values=["auto (recommended)", "welch", "paired",
                                   "wilcoxon", "mannwhitney", "anova",
                                   "kruskal", "none"])
        tcb.pack(side="left")
        tcb.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())
        ToolTip(tcb, "welch       - two independent groups, unequal variance\n"
                     "paired      - same wells before and after (stronger design)\n"
                     "mannwhitney - rank based; almost no power at small n\n"
                     "none        - no test, descriptive only")

        r = ttk.Frame(sb2)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Compare:", width=8).pack(side="left")
        self.plot_comp = tk.StringVar(value="all pairs")
        ccb = ttk.Combobox(r, textvariable=self.plot_comp, state="readonly", width=22,
                           values=["all pairs", "first vs each", "none"])
        ccb.pack(side="left")
        ccb.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())

        # ---------------- style ----------------
        yb = ttk.LabelFrame(ctrl, text="4. Style")
        yb.pack(fill="x", padx=4, pady=4)
        grid = ttk.Frame(yb)
        grid.pack(fill="x", padx=6, pady=4)

        def entry(row, col, label, var, width=12):
            ttk.Label(grid, text=label).grid(row=row, column=col * 2, sticky="w",
                                             padx=(0, 4), pady=2)
            e = ttk.Entry(grid, textvariable=var, width=width)
            e.grid(row=row, column=col * 2 + 1, sticky="w", pady=2)
            e.bind("<Return>", lambda ev: self._plot_refresh())
            e.bind("<FocusOut>", lambda ev: self._plot_refresh())

        self.p_title = tk.StringVar(value="")
        self.p_ylabel = tk.StringVar(value="% released")
        self.p_ymax = tk.StringVar(value="")
        self.p_w = tk.StringVar(value="6.0")
        self.p_h = tk.StringVar(value="4.5")
        self.p_dpi = tk.StringVar(value="300")
        self.p_xlabel = tk.StringVar(value="")
        self.p_ymin = tk.StringVar(value="0")
        self.p_ystep = tk.StringVar(value="")
        self.p_xrot = tk.StringVar(value="0")
        self.p_tick = tk.StringVar(value="10")
        self.p_lab = tk.StringVar(value="11")
        entry(0, 0, "Title", self.p_title, 18)
        entry(1, 0, "Y label", self.p_ylabel, 18)
        entry(2, 0, "X label", self.p_xlabel, 18)
        entry(3, 0, "Y min", self.p_ymin, 8)
        entry(3, 1, "Y max", self.p_ymax, 8)
        entry(4, 0, "Y step", self.p_ystep, 8)
        entry(4, 1, "X rotate", self.p_xrot, 8)
        entry(5, 0, "Tick size", self.p_tick, 8)
        entry(5, 1, "Label size", self.p_lab, 8)
        entry(6, 0, "Width in", self.p_w, 8)
        entry(6, 1, "Height in", self.p_h, 8)
        entry(7, 0, "DPI", self.p_dpi, 8)

        r = ttk.Frame(yb)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Chart:").pack(side="left")
        self.p_ctype = tk.StringVar(value="bar")
        cc = ttk.Combobox(r, textvariable=self.p_ctype, state="readonly",
                          width=12, values=cp.CHART_TYPES)
        cc.pack(side="left", padx=4)
        cc.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())
        ToolTip(cc, "bar        - mean with error bar\n"
                    "box        - median, quartiles, whiskers\n"
                    "dot        - every sample, with a mean line\n"
                    "bar + dots - both\n"
                    "violin     - distribution shape (needs n>=2)\n"
                    "line       - means joined, for ordered conditions")

        r = ttk.Frame(yb)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Palette:").pack(side="left")
        self.p_pal = tk.StringVar(value="Grey / Blue")
        pcb = ttk.Combobox(r, textvariable=self.p_pal, state="readonly", width=16,
                           values=list(cp.PALETTES))
        pcb.pack(side="left", padx=4)
        pcb.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())

        ttk.Label(r, text="Error:").pack(side="left", padx=(8, 0))
        self.p_err = tk.StringVar(value="SD")
        ecb = ttk.Combobox(r, textvariable=self.p_err, state="readonly", width=6,
                           values=["SD", "SEM"])
        ecb.pack(side="left", padx=4)
        ecb.bind("<<ComboboxSelected>>", lambda e: self._plot_refresh())

        r = ttk.Frame(yb)
        r.pack(fill="x", padx=6, pady=2)
        self.p_points = tk.BooleanVar(value=True)
        self.p_stars = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="Show individual samples", variable=self.p_points,
                        command=self._plot_refresh).pack(side="left")
        ttk.Checkbutton(r, text="Stars instead of p", variable=self.p_stars,
                        command=self._plot_refresh).pack(side="left", padx=8)
        ToolTip(r, "Showing every sample point is honest practice: the reader\n"
                   "sees n and the spread rather than a bar that hides them.")

        r = ttk.Frame(ctrl)
        r.pack(fill="x", padx=6, pady=8)
        ttk.Button(r, text="Refresh figure", command=self._plot_refresh).pack(side="left")
        ttk.Button(r, text="Save figure...", command=self._plot_save).pack(side="left", padx=6)
        ttk.Button(r, text="Save stats...", command=self._plot_save_stats).pack(side="left")

        # ---------------- preview + report ----------------
        self.plot_holder = ttk.Frame(view)
        self.plot_holder.pack(fill="both", expand=True)
        self.plot_sel_lbl = ttk.Label(view, text="", foreground="#0173b2",
                                      font=("Segoe UI", 8))
        self.plot_sel_lbl.pack(anchor="w", padx=6)
        ttk.Label(view, foreground="#666", font=("Segoe UI", 8), justify="left",
                  text="Click a bar to select - drag sideways to reorder - "
                       "right-click for colour - double-click to rename. "
                       "Reordering never changes a value.")\
            .pack(anchor="w", padx=6, pady=(0, 2))
        ttk.Label(view, text="Statistics", font=("Segoe UI", 9, "bold"))\
            .pack(anchor="w", padx=6)
        self.plot_report = scrolledtext.ScrolledText(
            view, height=13, wrap="word", font=("Consolas", 8))
        self.plot_report.pack(fill="both", expand=False, padx=4, pady=4)

        self.plot_rows_data = []
        self.plot_group_vars = {}
        self.plot_canvas = None
        self.plot_fig = None

    # ---------------------------------------------------- figures logic ---
    def _plot_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Choose batch_summary.csv or *_summary.csv files",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not paths:
            return
        new = cp.read_summaries(paths)
        if not new:
            messagebox.showerror(APP_TITLE,
                                 "No sample rows found in those files.\n"
                                 "Expected a semicolon CSV with a Sample column.")
            return
        self.plot_rows_data.extend(new)
        self._plot_build_rows()
        self._plot_refresh()

    def _plot_clear(self):
        self.plot_rows_data = []
        self.plot_group_vars = {}
        self._plot_build_rows()
        self._plot_refresh()

    def _plot_build_rows(self):
        for w in self.plot_rows_frame.winfo_children():
            w.destroy()
        self.plot_group_vars = {}
        if not self.plot_rows_data:
            self.plot_files_lbl.config(text="no files loaded")
            return
        srcs = sorted({r["SourceFile"] for r in self.plot_rows_data})
        self.plot_files_lbl.config(
            text=f"{len(self.plot_rows_data)} samples from: " + ", ".join(srcs))
        for i, r in enumerate(self.plot_rows_data):
            fr = ttk.Frame(self.plot_rows_frame)
            fr.pack(fill="x", pady=1)
            ttk.Label(fr, text=r["Sample"], width=26).pack(side="left")
            v = tk.StringVar(value="")
            e = ttk.Entry(fr, textvariable=v, width=16)
            e.pack(side="left", padx=4)
            e.bind("<Return>", lambda ev: self._plot_refresh())
            ttk.Label(fr, text=r["SourceFile"][:18], foreground="#888",
                      font=("Segoe UI", 7)).pack(side="left")
            self.plot_group_vars[i] = v

    def _plot_autogroup_well(self):
        """
        Fill group names with the WELL (A1, B2, ...) so grouped layout clusters
        each well and shows its conditions side by side. The condition comes
        from the source file, which is where the well/condition split lives in
        these filenames.
        """
        import re
        for i, r in enumerate(self.plot_rows_data):
            # prefer the recorded Well column; parsing the name is only a
            # fallback for data analysed before that column existed
            well = (r.get("Well") or "").strip()
            if not well:
                m = re.search(r"([A-H]\d{1,2})(?!.*[A-H]\d)", r["Sample"])
                well = m.group(1) if m else r["Sample"]
            if i in self.plot_group_vars:
                self.plot_group_vars[i].set(well)
        self.plot_layout.set("grouped (cluster by well)")
        self._plot_refresh()

    def _plot_autogroup_label(self):
        """
        Group by the CONDITION recorded in the summary, so bars are named
        "Fe-fum + Histidin 5 uL" rather than "A1". Falls back to the well when
        a row has no label, which happens for data analysed before a plate
        layout was supplied.
        """
        missing = 0
        for i, r in enumerate(self.plot_rows_data):
            lab = (r.get("Label") or "").strip()
            if not lab:
                lab = (r.get("Well") or r["Sample"]).strip()
                missing += 1
            if i in self.plot_group_vars:
                self.plot_group_vars[i].set(lab[:30])
        self._plot_refresh()
        if missing:
            messagebox.showinfo(
                APP_TITLE,
                f"{missing} of {len(self.plot_rows_data)} samples had no Label "
                "column.\n\nThose were analysed before a plate layout was "
                "pasted in. Re-run them with the layout in place to get the "
                "condition names.")

    def _plot_series_name(self, row):
        """The condition label for a bar inside a cluster."""
        # the Condition column (pre_glucose / after_glucose) is the natural
        # series when clustering by well; fall back to the file name
        cond = (row.get("Condition") or "").strip()
        if cond:
            return cond[:26]
        name = row["SourceFile"]
        for junk in ("_batch_summary.csv", "_summary.csv", ".csv"):
            name = name.replace(junk, "")
        return name[:26]

    def _plot_autogroup(self):
        """Group by source file - the usual case is one summary per condition."""
        for i, r in enumerate(self.plot_rows_data):
            name = r["SourceFile"].replace("_batch_summary.csv", "")
            name = name.replace("_summary.csv", "").replace(".csv", "")
            if i in self.plot_group_vars:
                self.plot_group_vars[i].set(name[:24])
        self._plot_refresh()

    def _plot_groups(self):
        col = dict(cp.VALUE_COLUMNS)
        key = next((k for k, d in cp.VALUE_COLUMNS
                    if d == self.plot_value_cb.get()), cp.VALUE_COLUMNS[0][0])
        buckets = {}
        order = []
        for i, r in enumerate(self.plot_rows_data):
            g = self.plot_group_vars.get(i)
            if g is None:
                continue
            name = g.get().strip()
            if not name:
                continue
            val = r.get(key)
            if not isinstance(val, float) or not np.isfinite(val):
                continue
            if name not in buckets:
                buckets[name] = []
                order.append(name)
            buckets[name].append((val, r))
        # a drag sets an explicit order; keep it until the groups change
        forced = getattr(self, "_forced_order", None)
        if forced:
            known = [n for n in forced if n in buckets]
            order = known + [n for n in order if n not in known]

        if self.plot_layout.get().startswith("grouped"):
            out = []
            for n in order:
                series = {}
                for val, row in buckets[n]:
                    s_name = self._plot_series_name(row)
                    series.setdefault(s_name, []).append(val)
                out.append((n, series))
            return out, key
        return [(n, [v for v, _r in buckets[n]]) for n in order], key

    def _plot_opts(self, groups):
        n = len(groups)
        comps = []
        mode = self.plot_comp.get()
        # No test unless the replicate type can support one, and never in
        # grouped mode where the comparison would be ambiguous.
        allow = (self.plot_test.get() != "none"
                 and self.rep_type.get() != "unspecified"
                 and not self.plot_layout.get().startswith("grouped"))
        if allow and n >= 2:
            if mode == "all pairs":
                comps = [(i, j) for i in range(n) for j in range(i + 1, n)]
            elif mode == "first vs each":
                comps = [(0, j) for j in range(1, n)]
        def num(var, default):
            try:
                return float(var.get())
            except (ValueError, TypeError):
                return default
        return {
            "title": self.p_title.get().strip(),
            "ylabel": self.p_ylabel.get().strip() or "%",
            "xlabel": self.p_xlabel.get().strip(),
            "ymin": self.p_ymin.get().strip(),
            "ystep": self.p_ystep.get().strip(),
            "xrot": num(self.p_xrot, 0),
            "tick_size": num(self.p_tick, 10),
            "label_size": num(self.p_lab, 11),
            "ymax": self.p_ymax.get().strip(),
            "width": num(self.p_w, 6.0),
            "height": num(self.p_h, 4.5),
            "dpi": 110,
            "chart_type": self.p_ctype.get(),
            "colors": getattr(self, "_group_colors", {}),
            "palette": self.p_pal.get(),
            "error": self.p_err.get(),
            "show_points": self.p_points.get(),
            "star_style": "stars" if self.p_stars.get() else "p",
            "comparisons": comps,
            "test": self._resolved_test(groups),
            "replicate_type": self.rep_type.get(),
            "grouped_mode": self.plot_layout.get().startswith("grouped"),
        }

    def _plot_reordered(self, groups):
        """
        A bar was dragged to a new position.

        Only the drawing order changed. The group's values travel with it, so
        every mean, error bar and p-value is identical - the figure is
        rearranged, not edited. The group-name boxes are rewritten to match so
        the panel and the chart never disagree.
        """
        order = [g[0] for g in groups]
        # _forced_order is honoured by _plot_groups(), so the new order
        # survives the redraw. Nothing else is touched.
        self._forced_order = order
        self._sel_idx = None
        self._plot_refresh()

    def _plot_selected(self, index, action):
        if action == "heartbeat":
            # a click reached the chart; show it so a dead canvas is
            # distinguishable from a bar that simply was not hit
            self._click_count = getattr(self, "_click_count", 0) + 1
            self._plot_status(f"click {self._click_count} received - "
                              "click directly ON a bar to select it")
            return
        self._sel_idx = index
        if index is None:
            self._plot_status(f"")
            return
        groups, _ = self._plot_groups()
        if index >= len(groups):
            return
        name = groups[index][0]

        if action == "select":
            vals = groups[index][1]
            if isinstance(vals, dict):
                return
            v = np.asarray(vals, float)
            self._plot_status(
                f"{name}:  n={len(v)}  mean={v.mean():.2f}  "
                f"SD={v.std(ddof=1) if len(v) > 1 else 0:.2f}   "
                "(drag to reorder, right-click for colour, double-click to rename)")
        elif action == "rename":
            new = simpledialog.askstring(APP_TITLE, f"New name for '{name}':",
                                         initialvalue=name, parent=self.root)
            if new and new.strip():
                for i, r in enumerate(self.plot_rows_data):
                    if i in self.plot_group_vars and \
                            self.plot_group_vars[i].get().strip() == name:
                        self.plot_group_vars[i].set(new.strip())
                if not hasattr(self, "_group_colors"):
                    self._group_colors = {}
                if name in self._group_colors:
                    self._group_colors[new.strip()] = self._group_colors.pop(name)
                self._plot_refresh()
        elif action == "color":
            rgb, hx = colorchooser.askcolor(parent=self.root,
                                            title=f"Colour for {name}")
            if hx:
                if not hasattr(self, "_group_colors"):
                    self._group_colors = {}
                self._group_colors[name] = hx
                self._plot_refresh()

    def _plot_status(self, text):
        if hasattr(self, "plot_sel_lbl"):
            self.plot_sel_lbl.config(text=text)

    def _resolved_test(self, groups):
        """
        Turn "auto" into a concrete test, using the assumption checks.

        The chosen test and the reasoning are both shown in the report - an
        automatically selected test the reader cannot inspect would be worse
        than one picked by hand.
        """
        t = self.plot_test.get()
        if not t.startswith("auto"):
            return t
        vals = [g[1] for g in groups if not isinstance(g[1], dict)]
        if not vals:
            return "none"
        paired = self.plot_test.get() == "paired"
        name, why, _a = cp.recommend_test(vals, paired=paired)
        self._auto_reason = why
        return name

    def _plot_refresh(self, *_):
        if not PLOT_OK or not hasattr(self, "plot_holder"):
            return
        groups, key = self._plot_groups()
        self.plot_report.delete("1.0", "end")
        if not groups:
            self.plot_report.insert("1.0",
                                    "Load a summary CSV and give at least one "
                                    "sample a group name.")
            for w in self.plot_holder.winfo_children():
                w.destroy()
            return
        opts = self._plot_opts(groups)
        try:
            fig = cp.make_figure(groups, opts)
        except Exception as exc:
            self.plot_report.insert("1.0", f"Could not draw the figure: {exc}")
            return
        for w in self.plot_holder.winfo_children():
            w.destroy()
        canvas = cp.FigureCanvasTkAgg(fig, master=self.plot_holder)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)

        # A Tk canvas only receives mouse events reliably once it can take
        # focus. Without this the matplotlib callbacks are connected but never
        # fire, so clicking and dragging appear to do nothing at all.
        try:
            widget.config(takefocus=1)
            widget.bind("<Enter>", lambda e: widget.focus_set())
            widget.bind("<Button-1>", lambda e: widget.focus_set(), add="+")
        except tk.TclError:
            pass

        self.plot_fig = fig
        self.plot_canvas = canvas

        # direct manipulation: drag to reorder, click to select, right-click
        # to recolour, double-click to rename. Reordering is presentational
        # only - the values and every statistic stay exactly as measured.
        if INTERACT_OK and not opts.get("grouped_mode"):
            old = getattr(self, "_inter", None)
            if old is not None:
                old.disconnect()
            self._plot_groups_cache = list(groups)
            self._inter = ci.InteractiveChart(
                canvas, fig.axes[0], self._plot_groups_cache, opts,
                on_change=self._plot_reordered,
                on_select=self._plot_selected)
            if getattr(self, "_sel_idx", None) is not None:
                ci.highlight_selected(fig.axes[0], self._sel_idx, len(groups))
                canvas.draw_idle()
        elif not INTERACT_OK:
            self._plot_status(
                "cargo_interactive.py is not in this folder - clicking and "
                "dragging are unavailable. Copy it next to cargo_gui.py.")
        elif opts.get("grouped_mode"):
            self._plot_status(
                "Grouped layout: dragging is available in 'simple' layout only.")

        self.plot_report.insert("1.0", cp.stats_report(groups, opts))

    def _plot_save(self):
        if self.plot_fig is None:
            messagebox.showinfo(APP_TITLE, "Nothing to save yet.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile="figure.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                       ("TIFF", "*.tif")])
        if not p:
            return
        try:
            dpi = int(float(self.p_dpi.get()))
        except (ValueError, TypeError):
            dpi = 300
        try:
            self.plot_fig.savefig(p, dpi=dpi, bbox_inches="tight")
            messagebox.showinfo(APP_TITLE, f"Saved at {dpi} dpi:\n{p}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _plot_save_stats(self):
        txt = self.plot_report.get("1.0", "end").strip()
        if not txt:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="statistics.txt",
            filetypes=[("Text", "*.txt")])
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(txt + "\n")
            messagebox.showinfo(APP_TITLE, f"Saved:\n{p}")
        except OSError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _build_setup_tab(self):
        f = self.tab_setup
        ttk.Label(f,
                  text="Checks that everything the pipeline needs is present and working.\n"
                       "Run this first on any new machine.",
                  justify="left").pack(anchor="w", padx=10, pady=8)
        ttk.Button(f, text="Run setup check", command=self._setup_check).pack(anchor="w", padx=10)
        self.setup_log = scrolledtext.ScrolledText(f, height=24, wrap="word",
                                                   font=("Consolas", 9))
        self.setup_log.pack(fill="both", expand=True, padx=10, pady=8)
        self.setup_log.tag_config("err", foreground="#c00000")
        self.setup_log.tag_config("ok", foreground="#008000")
        self.setup_log.tag_config("warn", foreground="#b06000")

    # ------------------------------------------------------------ actions ---
    def _pick_output(self):
        d = filedialog.askdirectory(title="Where should the results be saved?")
        if d:
            self.output_var.set(os.path.normpath(d))

    def _pick_parent(self):
        d = filedialog.askdirectory(title="Choose the parent folder (contains A1, A2, ...)")
        if d:
            self.parent_var.set(os.path.normpath(d))
            self._scan_samples()

    def _pick_fiji(self):
        d = filedialog.askdirectory(title="Choose the Fiji folder")
        if d:
            self.fiji_var.set(os.path.normpath(d))

    def _scan_samples(self):
        for w in self.sample_frame.winfo_children():
            w.destroy()
        self.sample_vars.clear()

        parent = self.parent_var.get().strip()
        if not os.path.isdir(parent):
            ttk.Label(self.sample_frame, text="Not a folder.",
                      foreground="#c00000").pack(anchor="w")
            return

        subs = sorted(d for d in os.listdir(parent)
                      if os.path.isdir(os.path.join(parent, d)))
        # only offer folders that actually contain acquisition tifs
        valid = []
        for s in subs:
            try:
                tifs = [f for f in os.listdir(os.path.join(parent, s))
                        if f.lower().endswith((".tif", ".tiff"))
                        and not any(t in f.lower() for t in
                                    ("_cp_masks", "_classified_overlay",
                                     "_brightfield_for_cellpose"))]
            except OSError:
                continue
            if len(tifs) >= 2:
                valid.append(s)

        if not valid:
            ttk.Label(self.sample_frame,
                      text="No sample folders with two tifs found here.\n"
                           "Did you pick the parent folder rather than a sample folder?",
                      foreground="#c00000", justify="left").pack(anchor="w")
            return

        grid = ttk.Frame(self.sample_frame)
        grid.pack(anchor="w")
        for i, s in enumerate(valid):
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(grid, text=s, variable=var).grid(
                row=i // 6, column=i % 6, sticky="w", padx=6, pady=2)
            self.sample_vars[s] = var

    def _set_all(self, state):
        for v in self.sample_vars.values():
            v.set(state)

    def _collect_config(self):
        cfg = {}
        for key, var in self.tune_vars.items():
            cfg[key] = var.get().strip()
        cfg["use_gpu"] = self.gpu_var.get()
        # translate the RAM dropdown into the Fiji heap size
        for label, mb in RAM_PRESETS:
            if label == self.ram_var.get():
                cfg["fiji_memory_mb"] = mb
                break
        fiji = self.fiji_var.get().strip()
        if fiji:
            cfg["fiji_dir"] = fiji
        # map GUI keys onto the pipeline's constant names
        rename = {
            "min_cell_mean": "MIN_CELL_MEAN",
            "very_high_signal_cutoff": "VERY_HIGH_SIGNAL_CUTOFF_8BIT",
        }
        out = {}
        for k, v in cfg.items():
            out[rename.get(k, k.upper())] = v
        return out

    def _validate(self):
        parent = self.parent_var.get().strip()
        if not os.path.isdir(parent):
            messagebox.showerror(APP_TITLE, "Choose a valid parent folder first.")
            return None
        chosen = [s for s, v in self.sample_vars.items() if v.get()]
        if not chosen:
            messagebox.showerror(APP_TITLE, "Tick at least one sample to run.")
            return None
        for key, var in self.tune_vars.items():
            try:
                float(var.get())
            except ValueError:
                messagebox.showerror(
                    APP_TITLE,
                    f"'{key}' must be a number (got '{var.get()}').\n"
                    "Check the Settings tab.")
                return None
        script = os.path.join(self.script_dir, PIPELINE_SCRIPT)
        if not os.path.isfile(script):
            messagebox.showerror(
                APP_TITLE,
                f"{PIPELINE_SCRIPT} is not next to this GUI.\n"
                f"Expected it in:\n{self.script_dir}")
            return None
        if not os.path.isfile(os.path.join(self.script_dir, MACRO_FILE)):
            messagebox.showerror(
                APP_TITLE,
                f"{MACRO_FILE} is not next to this GUI.\n"
                f"Expected it in:\n{self.script_dir}")
            return None
        return parent, chosen, script

    def _run(self):
        checked = self._validate()
        if not checked:
            return
        parent, chosen, script = checked

        cfg_path = os.path.join(self.script_dir, "_gui_config.json")
        try:
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(self._collect_config(), fh, indent=2)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Could not write config: {exc}")
            return

        cmd = [sys.executable, "-u", script, parent,
               "--samples", ",".join(chosen),
               "--config", cfg_path]
        out = self.output_var.get().strip()
        if out:
            cmd += ["--output", out]
        lm = self._label_map(verbose=True)
        if lm:
            lp = os.path.join(self.script_dir, "_gui_labels.txt")
            try:
                with open(lp, "w", encoding="utf-8") as fh:
                    fh.write(self.lbl_text.get("1.0", "end"))
                cmd += ["--labels", lp]
                self._pending_label_note = (
                    f"Sample names: using the pasted layout ({len(lm)} wells).")
            except OSError as exc:
                self._pending_label_note = (
                    f"Sample names: could NOT write the layout file ({exc}); "
                    "results will be named by well.")
        else:
            self._pending_label_note = (
                "Sample names: not used - "
                + (getattr(self, "_label_why", None) or "no layout available")
                + ".\n              Results will be named A1, A2, ... and the "
                  "Label column will be blank.")
        if self.force_var.get():
            cmd.append("--force")

        self.log.delete("1.0", "end")
        self._append(f"$ {' '.join(cmd)}\n\n")
        note = getattr(self, "_pending_label_note", "")
        if note:
            self._append(note + "\n\n",
                         "ok" if "using the pasted layout" in note else "warn")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="Running...")
        self.last_parent = parent

        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        try:
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NO_WINDOW
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                cwd=self.script_dir, creationflags=flags)
            for line in self.proc.stdout:
                self.log_queue.put(line)
            self.proc.wait()
            self.log_queue.put(("__DONE__", self.proc.returncode))
        except Exception:
            self.log_queue.put(traceback.format_exc())
            self.log_queue.put(("__DONE__", -1))

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._append("\n[stopped by user]\n", "warn")

    def _poll_log(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    code = item[1]
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if code == 0:
                        self.status.config(text="Finished")
                        self._append("\nFinished.\n", "ok")
                    else:
                        self.status.config(text=f"Finished with errors (exit {code})")
                        self._append(f"\nExited with code {code}.\n", "err")
                else:
                    tag = None
                    low = str(item).lower()
                    if "error" in low or "failed" in low:
                        tag = "err"
                    elif "warning" in low and "[warning]" not in low:
                        tag = "warn"
                    self._append(str(item), tag)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log)

    def _append(self, text, tag=None):
        self.log.insert("end", text, tag or ())
        self.log.see("end")

    def _open_results(self):
        parent = getattr(self, "last_parent", None) or self.parent_var.get().strip()
        if not os.path.isdir(parent):
            messagebox.showinfo(APP_TITLE, "No results folder yet.")
            return
        try:
            if os.name == "nt":
                os.startfile(parent)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", parent])
            else:
                subprocess.Popen(["xdg-open", parent])
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open folder: {exc}")

    # -------------------------------------------------------- setup check ---
    def _setup_check(self):
        self.setup_log.delete("1.0", "end")
        threading.Thread(target=self._setup_worker, daemon=True).start()

    def _sput(self, text, tag=None):
        self.setup_log.insert("end", text + "\n", tag or ())
        self.setup_log.see("end")

    def _setup_worker(self):
        ok, bad = "ok", "err"
        self._sput("Checking environment...\n")

        self._sput(f"Python  : {sys.version.split()[0]}  ({sys.executable})", ok)

        # required python packages
        for mod, why in [("numpy", "array maths"),
                         ("tifffile", "reading/writing tif"),
                         ("PIL", "image resizing"),
                         ("cellpose", "segmentation")]:
            try:
                m = __import__(mod)
                ver = getattr(m, "__version__", "?")
                self._sput(f"  {mod:<10} {ver}", ok)
            except ImportError:
                self._sput(f"  {mod:<10} MISSING  ({why})", bad)

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                self._sput(f"  torch CUDA available: {torch.cuda.get_device_name(0)}", ok)
            else:
                self._sput("  torch present but no CUDA GPU - will run on CPU (slower)", "warn")
        except ImportError:
            self._sput("  torch      MISSING (installed with cellpose)", bad)

        # Fiji
        self._sput("")
        fiji_dir = self.fiji_var.get().strip()
        if not fiji_dir:
            self._sput("Fiji folder not set (Settings tab).", bad)
        elif not os.path.isdir(fiji_dir):
            self._sput(f"Fiji folder does not exist: {fiji_dir}", bad)
        else:
            names = ["fiji-windows-x64.exe", "ImageJ-win64.exe",
                     "fiji-windows-x86.exe", "ImageJ-win32.exe",
                     "ImageJ-linux64", "Contents/MacOS/ImageJ-macosx"]
            found = None
            for n in names:
                p = os.path.join(fiji_dir, n)
                if os.path.isfile(p):
                    found = p
                    break
            if found:
                self._sput(f"Fiji    : {found}", ok)
            else:
                self._sput(f"No Fiji launcher in {fiji_dir}", bad)
                try:
                    self._sput("  contains: " + ", ".join(sorted(os.listdir(fiji_dir))[:20]))
                except OSError:
                    pass

        # companion files
        self._sput("")
        for fn in (PIPELINE_SCRIPT, MACRO_FILE):
            p = os.path.join(self.script_dir, fn)
            self._sput(f"{fn}: {'found' if os.path.isfile(p) else 'MISSING'}",
                       ok if os.path.isfile(p) else bad)

        # optional modules - report WHERE each was loaded from, because the
        # commonest cause of "the tab is not there" is an older copy of a file
        # sitting in a different folder from the one being edited.
        self._sput("")
        self._sput("Modules:")
        import importlib
        for name, why in (("cargo_labels", "Sample names tab"),
                          ("cargo_plot", "Figures tab"),
                          ("cargo_measure", "measurement"),
                          ("matplotlib", "plotting")):
            try:
                mod = importlib.import_module(name)
                where = getattr(mod, "__file__", "?")
                self._sput(f"  {name:<16} {where}", ok)
            except ImportError:
                self._sput(f"  {name:<16} NOT FOUND  ({why} unavailable)", bad)

        self._sput("")
        self._sput(f"This interface is running from:\n  "
                   f"{os.path.abspath(__file__)}")
        self._sput("If a tab you expect is missing, check that THIS file is the "
                   "one you copied from the package.", "warn")

        self._sput("\nDone. Anything red must be fixed before running.", "warn")

    # ---------------------------------------------------------- settings ---
    def _save_settings(self):
        data = {
            "fiji_dir": self.fiji_var.get().strip(),
            "parent": self.parent_var.get().strip(),
            "output": self.output_var.get().strip(),
            "use_gpu": self.gpu_var.get(),
            "ram": self.ram_var.get(),
            "tunables": {k: v.get() for k, v in self.tune_vars.items()},
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            messagebox.showinfo(APP_TITLE, "Settings saved.")
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Could not save: {exc}")

    def _load_settings(self):
        if not os.path.isfile(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        self.fiji_var.set(data.get("fiji_dir", ""))
        self.parent_var.set(data.get("parent", ""))
        self.output_var.set(data.get("output", ""))
        self.gpu_var.set(bool(data.get("use_gpu", True)))
        if data.get("ram"):
            self.ram_var.set(data["ram"])
        for k, v in (data.get("tunables") or {}).items():
            if k in self.tune_vars:
                self.tune_vars[k].set(v)
        if os.path.isdir(self.parent_var.get()):
            self._scan_samples()

    def _reset_settings(self):
        for key, _label, default, _tip in TUNABLES:
            self.tune_vars[key].set(default)
        self.gpu_var.set(True)
        messagebox.showinfo(APP_TITLE, "Analysis parameters reset to defaults.")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    CargoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()