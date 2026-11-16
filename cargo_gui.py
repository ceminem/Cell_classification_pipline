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
import re
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
except Exception as exc:          # ImportError, or RuntimeError (unsupported CPU)
    # Say WHY: "not installed" and "installed but its DLLs do not load" need
    # different fixes, and the second is the usual one on a fresh Windows PC.
    msg = str(exc)
    hint = ("numpy is installed but does not load. On Windows this is almost\n"
            "always the missing Microsoft Visual C++ runtime. Install it from\n"
            "    https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            "or run \"Setup on this PC.bat\" again, which offers to install it."
            if ("DLL" in msg or "dll" in msg) else
            "Run \"Setup on this PC.bat\", or in your environment:\n"
            "    pip install -r requirements.txt")
    sys.exit(f"ERROR: numpy could not be loaded:\n    {msg}\n\n{hint}")

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
    # NOTE: cellprob_threshold appears ONCE, above. A second entry with the
    # same key used to sit here; it replaced the first in tune_vars, so typing
    # into the first "Cellpose cellprob" box silently did nothing.
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
        self.tab_count = ttk.Frame(nb)
        self.tab_settings = ttk.Frame(nb)
        self.tab_labels = ttk.Frame(nb)
        self.tab_plot = ttk.Frame(nb)
        self.tab_cnn = ttk.Frame(nb)
        self.tab_setup = ttk.Frame(nb)
        nb.add(self.tab_run, text="  Run  ")
        nb.add(self.tab_count, text="  Cell count  ")
        nb.add(self.tab_labels, text="  Sample names  ")
        nb.add(self.tab_settings, text="  Settings  ")
        nb.add(self.tab_plot, text="  Figures  ")
        nb.add(self.tab_cnn, text="  CNN  ")
        nb.add(self.tab_setup, text="  Setup check  ")

        self._build_run_tab()
        self._build_count_tab()
        self._build_settings_tab()
        self._build_labels_tab()
        self._build_plot_tab()
        self._build_cnn_tab()
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
        ttk.Label(row, text="Fiji / ImageJ:", width=14).pack(side="left")
        self.fiji_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.fiji_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Browse...", command=self._pick_fiji).pack(side="left")

        ttk.Label(pbox,
                  text="Optional - the analysis runs in Python and does not need it. Folder with "
                       "fiji-windows-x64.exe / ImageJ-win64.exe (Fiji) or ImageJ.exe (ImageJ 1.x).",
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

    # ================================================================ CNN ===
    # The CNN tab runs cargo_cnn.train / cargo_cnn.predict as subprocesses with
    # sys.executable - the SAME Python this window runs in. "Cargo Analysis.bat"
    # starts the window in the best GPU environment, so training and
    # classification use the GPU without activating any environment by hand.

    CNN_LENGTHS = [("quick test  (15 epochs, minutes)", 15),
                   ("normal  (60 epochs, recommended)", 60),
                   ("thorough  (120 epochs)", 120)]

    def _build_cnn_tab(self):
        f = self.tab_cnn
        self.cnn_queue = queue.Queue()
        self.cnn_proc = None

        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(top, justify="left", wraplength=820, foreground="#444",
                  text="A neural network that learns 'released' vs 'not released' "
                       "from cells YOU labelled blind, by looking at each cell's "
                       "picture instead of two numbers. Optional: the normal "
                       "analysis is unchanged. Work through 1 -> 2 -> 3.")\
            .pack(anchor="w")
        self.cnn_dev_lbl = ttk.Label(top, text="Compute device: checking...",
                                     foreground="#666")
        self.cnn_dev_lbl.pack(anchor="w", pady=(4, 0))
        threading.Thread(target=self._cnn_detect_device, daemon=True).start()

        # ---- 1. label ----
        b1 = ttk.LabelFrame(f, text="1. Label cells  (curation tool)")
        b1.pack(fill="x", padx=8, pady=4)
        r = ttk.Frame(b1)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Button(r, text="Open curation tool...",
                   command=self._cnn_open_curator).pack(side="left")
        ttk.Label(r, justify="left", foreground="#666", font=("Segoe UI", 8),
                  text="Pick a SAMPLE folder (e.g. ...\\A1) that has been analysed. "
                       "Press B for blind mode, label with R / N under the mouse,\n"
                       "then 'Save labels'. Aim for 100+ cells per field and at least "
                       "3 fields from different images.").pack(side="left", padx=8)

        # ---- 2. train ----
        b2 = ttk.LabelFrame(f, text="2. Train a model")
        b2.pack(fill="x", padx=8, pady=4)
        r = ttk.Frame(b2)
        r.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Label(r, text="Folders with labelled samples (searched with all subfolders):")\
            .pack(side="left")
        r = ttk.Frame(b2)
        r.pack(fill="x", padx=6, pady=2)
        self.cnn_folders = tk.Listbox(r, height=3, font=("Consolas", 8))
        self.cnn_folders.pack(side="left", fill="x", expand=True)
        bcol = ttk.Frame(r)
        bcol.pack(side="left", padx=4)
        ttk.Button(bcol, text="Add folder...", command=self._cnn_add_folder).pack(fill="x")
        ttk.Button(bcol, text="Remove", command=lambda: [
            self.cnn_folders.delete(i) for i in reversed(self.cnn_folders.curselection())])\
            .pack(fill="x", pady=2)

        r = ttk.Frame(b2)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="Results folder (optional):").pack(side="left")
        self.cnn_results_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.cnn_results_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(r, text="Browse...", command=lambda: self._cnn_pick_dir(
            self.cnn_results_var, "Folder the pipeline saved results to")).pack(side="left")
        ToolTip(r, "Only needed if you ran the analysis with 'Save results to'.\n"
                   "The results files let training compare the CNN with the\n"
                   "rule and check the labels still fit the current masks.")

        r = ttk.Frame(b2)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Training length:").pack(side="left")
        self.cnn_len_var = tk.StringVar(value=self.CNN_LENGTHS[1][0])
        ttk.Combobox(r, textvariable=self.cnn_len_var, state="readonly", width=34,
                     values=[x[0] for x in self.CNN_LENGTHS]).pack(side="left", padx=6)
        self.cnn_train_btn = ttk.Button(r, text="Train model", command=self._cnn_train)
        self.cnn_train_btn.pack(side="right")

        # ---- 3. classify ----
        b3 = ttk.LabelFrame(f, text="3. Classify samples with a trained model")
        b3.pack(fill="x", padx=8, pady=4)
        r = ttk.Frame(b3)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Model:").pack(side="left")
        self.cnn_model_var = tk.StringVar()
        self.cnn_model_cb = ttk.Combobox(r, textvariable=self.cnn_model_var,
                                         state="readonly", width=70)
        self.cnn_model_cb.pack(side="left", padx=6)
        ttk.Button(r, text="Refresh", command=self._cnn_refresh_models).pack(side="left")
        r = ttk.Frame(b3)
        r.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(r, justify="left", foreground="#666", font=("Segoe UI", 8),
                  text="Uses the parent folder, the ticked samples and 'Save results "
                       "to' from the Run tab.\nRun the normal analysis on them first - "
                       "the CNN reuses its cell masks.").pack(side="left")
        self.cnn_pred_btn = ttk.Button(r, text="Classify ticked samples",
                                       command=self._cnn_predict)
        self.cnn_pred_btn.pack(side="right")

        r = ttk.Frame(f)
        r.pack(fill="x", padx=8, pady=2)
        self.cnn_stop_btn = ttk.Button(r, text="Stop", command=self._cnn_stop,
                                       state="disabled")
        self.cnn_stop_btn.pack(side="left")
        ttk.Button(r, text="Open training report", command=self._cnn_open_report)\
            .pack(side="left", padx=6)
        ttk.Button(r, text="Open models folder", command=lambda: self._open_path(
            os.path.join(self.script_dir, "models"))).pack(side="left")
        self.cnn_status = ttk.Label(r, text="", foreground="#666")
        self.cnn_status.pack(side="right")

        self.cnn_log = scrolledtext.ScrolledText(f, height=10, wrap="word",
                                                 font=("Consolas", 9))
        self.cnn_log.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self.cnn_log.tag_config("err", foreground="#c00000")
        self.cnn_log.tag_config("ok", foreground="#008000")
        self.cnn_log.tag_config("warn", foreground="#b06000")

        self._cnn_refresh_models()
        self.root.after(150, self._cnn_poll)

    # ---- helpers ----
    def _cnn_detect_device(self):
        """Ask torch which device training will use. Runs in a thread because
        importing torch takes a few seconds; the label is updated on the Tk
        thread via after()."""
        try:
            import torch
            if torch.cuda.is_available():
                kind = "AMD ROCm" if getattr(torch.version, "hip", None) else "NVIDIA CUDA"
                txt = (f"Compute device: GPU - {torch.cuda.get_device_name(0)} "
                       f"({kind}).  Training takes seconds to a few minutes.")
                col = "#008000"
            else:
                txt = ("Compute device: CPU only - works, but training takes "
                       "several minutes. Start the program with 'Cargo Analysis.bat' "
                       "so it picks your GPU environment.")
                col = "#b06000"
        except ImportError:
            txt, col = ("Compute device: torch is not installed in this Python "
                        "environment - the CNN tab cannot run.", "#c00000")
        except Exception as exc:
            txt, col = f"Compute device: could not be checked ({exc})", "#b06000"
        # never touch widgets from this thread - hand the result to _cnn_poll
        self.cnn_queue.put(("__DEVICE__", txt, col))

    def _cnn_pick_dir(self, var, title):
        d = filedialog.askdirectory(title=title)
        if d:
            var.set(os.path.normpath(d))

    def _cnn_add_folder(self):
        start = self.parent_var.get().strip() or None
        d = filedialog.askdirectory(title="Folder containing labelled samples",
                                    initialdir=start)
        if d and os.path.normpath(d) not in self.cnn_folders.get(0, "end"):
            self.cnn_folders.insert("end", os.path.normpath(d))

    def _cnn_open_curator(self):
        start = self.parent_var.get().strip() or None
        d = filedialog.askdirectory(title="Choose a SAMPLE folder to label (e.g. ...\\A1)",
                                    initialdir=start)
        if not d:
            return
        script = os.path.join(self.script_dir, "cargo_curate.py")
        if not os.path.isfile(script):
            messagebox.showerror(APP_TITLE, "cargo_curate.py is not next to this program.")
            return
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.Popen([sys.executable, script, os.path.normpath(d)],
                         cwd=self.script_dir, creationflags=flags)
        self.cnn_status.config(text="curation tool opened in its own window")

    def _cnn_models(self):
        """Trained models, newest first, with a one-line description each."""
        import glob
        out = []
        for p in sorted(glob.glob(os.path.join(self.script_dir, "models", "cnn_*",
                                               "model.pt")), reverse=True):
            desc = os.path.basename(os.path.dirname(p))
            try:
                with open(os.path.join(os.path.dirname(p), "model_info.json"),
                          encoding="utf-8") as fh:
                    meta = json.load(fh)
                n = sum(t["n"] for t in meta.get("trained_on", []))
                desc = f"{meta.get('created', '')[:16].replace('T', ' ')}  -  {n} cells"
                cv = meta.get("cv") or []

                def wavg(rows, key):
                    ok = [c for c in rows if c.get(key) is not None and c[key] == c[key]]
                    n = sum(c["n"] for c in ok)
                    return sum(c[key] * c["n"] for c in ok) / n if n else None

                if cv and "design" in cv[0]:
                    # newer models: two test designs + brightness yardstick
                    parts = []
                    for d, label in (("halves", "unseen cells"),
                                     ("fields", "unseen condition")):
                        rows = [c for c in cv if c.get("design") == d]
                        c_ = wavg(rows, "cnn_bal")
                        b_ = wavg(rows, "bright_bal")
                        if c_ is not None:
                            parts.append(f"{label} {c_ * 100:.0f}%"
                                         + (f" (brightness {b_ * 100:.0f}%)" if b_ is not None else ""))
                    if parts:
                        desc += "  -  " + " · ".join(parts)
                elif cv:
                    k = "cnn_bal" if all("cnn_bal" in c for c in cv) else "cnn_acc"
                    c_ = wavg(cv, k)
                    if c_ is not None:
                        desc += f"  -  held-out {c_ * 100:.1f}%"
            except (OSError, ValueError, KeyError, TypeError):
                pass
            out.append((desc, p))
        return out

    def _cnn_refresh_models(self):
        self._cnn_model_list = self._cnn_models()
        names = [d for d, _ in self._cnn_model_list]
        self.cnn_model_cb.config(values=names)
        if names:
            if self.cnn_model_var.get() not in names:
                self.cnn_model_var.set(names[0])
        else:
            self.cnn_model_var.set("(no trained model yet - do step 2)")

    def _cnn_selected_model(self):
        for d, p in getattr(self, "_cnn_model_list", []):
            if d == self.cnn_model_var.get():
                return p
        return None

    def _cnn_open_report(self):
        p = self._cnn_selected_model()
        rep = os.path.join(os.path.dirname(p), "training_report.txt") if p else None
        if not rep or not os.path.isfile(rep):
            messagebox.showinfo(APP_TITLE, "No trained model selected yet.")
            return
        self._open_path(rep)

    def _open_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open: {exc}")

    # ---- actions ----
    def _cnn_train(self):
        folders = list(self.cnn_folders.get(0, "end"))
        if not folders:
            messagebox.showinfo(APP_TITLE, "Add at least one folder that contains "
                                           "labelled samples (step 2, 'Add folder...').")
            return
        epochs = dict(self.CNN_LENGTHS).get(self.cnn_len_var.get(), 60)
        cmd = [sys.executable, "-u", "-m", "cargo_cnn.train",
               "--epochs", str(epochs), "--data"] + folders
        extra = [x for x in (self.cnn_results_var.get().strip(),
                             self.output_var.get().strip()) if x]
        if extra:
            cmd += ["--results"] + extra
        self._cnn_start(cmd, "training")

    def _cnn_predict(self):
        model = self._cnn_selected_model()
        if not model:
            messagebox.showinfo(APP_TITLE, "Train a model first (step 2).")
            return
        parent = self.parent_var.get().strip()
        if not os.path.isdir(parent):
            messagebox.showinfo(APP_TITLE, "Choose the parent folder on the Run tab first.")
            return
        chosen = [s for s, v in self.sample_vars.items() if v.get()]
        if not chosen:
            messagebox.showinfo(APP_TITLE, "Tick the samples to classify on the Run tab.")
            return
        cmd = [sys.executable, "-u", "-m", "cargo_cnn.predict", "--model", model,
               "--parent", parent, "--samples", ",".join(chosen)]
        out = self.output_var.get().strip()
        if out:
            cmd += ["--output", out]
        self.last_parent = parent
        self._cnn_start(cmd, "classifying")

    def _cnn_start(self, cmd, what):
        if self.cnn_proc and self.cnn_proc.poll() is None:
            messagebox.showinfo(APP_TITLE, "Something is already running in this tab.")
            return
        self.cnn_log.delete("1.0", "end")
        self.cnn_log.insert("end", f"$ {' '.join(cmd)}\n\n")
        for b in (self.cnn_train_btn, self.cnn_pred_btn):
            b.config(state="disabled")
        self.cnn_stop_btn.config(state="normal")
        self.cnn_status.config(text=f"{what}...")

        def work():
            try:
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.cnn_proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    bufsize=1, encoding="utf-8", errors="replace",
                    cwd=self.script_dir, creationflags=flags, env=env)
                for line in self.cnn_proc.stdout:
                    self.cnn_queue.put(line)
                self.cnn_proc.wait()
                self.cnn_queue.put(("__DONE__", self.cnn_proc.returncode, what))
            except Exception:
                self.cnn_queue.put(traceback.format_exc())
                self.cnn_queue.put(("__DONE__", -1, what))

        threading.Thread(target=work, daemon=True).start()

    def _cnn_stop(self):
        if self.cnn_proc and self.cnn_proc.poll() is None:
            self.cnn_proc.terminate()
            self.cnn_log.insert("end", "\n[stopped by user]\n", "warn")

    def _cnn_poll(self):
        try:
            while True:
                item = self.cnn_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DEVICE__":
                    self.cnn_dev_lbl.config(text=item[1], foreground=item[2])
                elif isinstance(item, tuple) and item and item[0] == "__DONE__":
                    _, code, what = item
                    for b in (self.cnn_train_btn, self.cnn_pred_btn):
                        b.config(state="normal")
                    self.cnn_stop_btn.config(state="disabled")
                    if code == 0:
                        self.cnn_status.config(text=f"{what} finished")
                        self.cnn_log.insert("end", "\nFinished.\n", "ok")
                        if what == "training":
                            self._cnn_refresh_models()
                            self.cnn_log.insert(
                                "end", "The new model is selected in step 3. Read the "
                                       "VERDICT above before using it.\n", "ok")
                    else:
                        self.cnn_status.config(text=f"{what} stopped (exit {code})")
                        self.cnn_log.insert("end", f"\nExited with code {code}.\n", "err")
                    self.cnn_log.see("end")
                else:
                    s = str(item)
                    low = s.lower()
                    tag = ("err" if ("error" in low or "traceback" in low) else
                           "ok" if "verdict: the cnn beats" in low else
                           "warn" if ("verdict" in low or "warning" in low
                                      or low.lstrip().startswith("!")) else None)
                    self.cnn_log.insert("end", s, tag or ())
                    self.cnn_log.see("end")
        except queue.Empty:
            pass
        self.root.after(150, self._cnn_poll)

    # ========================================================= cell count ===
    COUNT_EDGE = [
        ("Count every cell", "all"),
        ("Only cells completely inside the image", "exclude"),
        ("Counting frame: top/right edge count, bottom/left not (cells per area)", "unbiased"),
    ]
    COUNT_SPEED = [
        ("Settings tab (Cellpose downscale)", 0.0),
        ("Accurate: 1/2 size", 0.5),
        ("Faster: 1/3 size (~2x faster than 1/2)", 1 / 3.0),
        ("Fastest: 1/4 size (~4x faster than 1/2) - good on a laptop CPU", 0.25),
    ]
    COUNT_SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "cellcount_settings.json")

    def _build_count_tab(self):
        f = self.tab_count
        self.count_queue = queue.Queue()
        self.count_proc = None
        self.count_figs = []
        self.count_idx = 0
        self.count_photo = None

        ttk.Label(f, justify="left", wraplength=820, foreground="#444",
                  text="Brightfield images only: Cellpose finds the cells, they are "
                       "counted, and figures with clean cell outlines are saved as PNG "
                       "(full resolution) and PDF (vector outlines, for papers). Uses "
                       "the Cellpose settings from the Settings tab.")\
            .pack(anchor="w", padx=8, pady=(8, 2))

        b1 = ttk.LabelFrame(f, text="1. Images")
        b1.pack(fill="x", padx=8, pady=4)
        r = ttk.Frame(b1)
        r.pack(fill="x", padx=6, pady=4)
        self.count_list = tk.Listbox(r, height=4, font=("Consolas", 8),
                                     selectmode="extended")
        self.count_list.pack(side="left", fill="x", expand=True)
        col = ttk.Frame(r)
        col.pack(side="left", padx=4)
        ttk.Button(col, text="Add folder...", command=self._count_add_folder).pack(fill="x")
        ttk.Button(col, text="Add images...", command=self._count_add_files)\
            .pack(fill="x", pady=2)
        ttk.Button(col, text="Remove", command=lambda: [
            self.count_list.delete(i) for i in reversed(self.count_list.curselection())])\
            .pack(fill="x")
        ttk.Label(b1, foreground="#666", font=("Segoe UI", 8), justify="left",
                  text="Folders are searched with their subfolders. Where channels are "
                       "named (A1_brightfield / A1_fluorescence) only the brightfield is "
                       "used.").pack(anchor="w", padx=8, pady=(0, 4))
        r = ttk.Frame(b1)
        r.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(r, text="Save results to:").pack(side="left")
        self.count_out_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.count_out_var).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(r, text="Browse...", command=lambda: self._cnn_pick_dir(
            self.count_out_var, "Folder for the cell_count results")).pack(side="left")
        ttk.Label(r, text="(empty = next to the first folder)", foreground="#666")\
            .pack(side="left", padx=4)

        b2 = ttk.LabelFrame(f, text="2. Counting")
        b2.pack(fill="x", padx=8, pady=4)
        r = ttk.Frame(b2)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Pixels per µm:").pack(side="left")
        self.count_px_var = tk.StringVar(value="2.6428")
        e = ttk.Entry(r, textvariable=self.count_px_var, width=9)
        e.pack(side="left", padx=4)
        ToolTip(e, "Image calibration. 2.6428 px/µm = 20x objective on our "
                   "microscope;\nchange it for other objectives. Empty = no scale "
                   "bar, sizes in pixels.")
        ttk.Label(r, text="Ignore objects smaller than").pack(side="left", padx=(16, 4))
        self.count_min_var = tk.StringVar(value="0")
        e = ttk.Entry(r, textvariable=self.count_min_var, width=7)
        e.pack(side="left")
        ttk.Label(r, text="µm²  (debris; 0 = keep all)").pack(side="left", padx=4)
        r = ttk.Frame(b2)
        r.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(r, text="Speed:").pack(side="left")
        self.count_speed_var = tk.StringVar(value=self.COUNT_SPEED[0][0])
        cb = ttk.Combobox(r, textvariable=self.count_speed_var, state="readonly", width=66,
                          values=[x[0] for x in self.COUNT_SPEED])
        cb.pack(side="left", padx=6)
        ToolTip(cb, "Cellpose's time grows with the number of pixels it looks at.\n"
                    "Our cells are ~85 px across at full size, so 1/4 size (~21 px)\n"
                    "still finds them. The log warns if cells get too small.\n"
                    "With a GPU every setting takes seconds.")
        r = ttk.Frame(b2)
        r.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(r, text="Cells at the image edge:").pack(side="left")
        self.count_edge_var = tk.StringVar(value=self.COUNT_EDGE[0][0])
        ttk.Combobox(r, textvariable=self.count_edge_var, state="readonly", width=66,
                     values=[x[0] for x in self.COUNT_EDGE]).pack(side="left", padx=6)

        b3 = ttk.LabelFrame(f, text="3. Figure")
        b3.pack(fill="x", padx=8, pady=4)
        r = ttk.Frame(b3)
        r.pack(fill="x", padx=6, pady=4)
        ttk.Label(r, text="Outline colour:").pack(side="left")
        self.count_color_var = tk.StringVar(value="yellow")
        ttk.Combobox(r, textvariable=self.count_color_var, width=9,
                     values=["yellow", "cyan", "magenta", "green", "red", "orange",
                             "blue", "white", "black"]).pack(side="left", padx=4)
        ttk.Label(r, text="Line width (image px):").pack(side="left", padx=(12, 4))
        self.count_lw_var = tk.StringVar(value="3")
        ttk.Spinbox(r, from_=1, to=12, increment=0.5, textvariable=self.count_lw_var,
                    width=5).pack(side="left")
        self.count_opts = {}
        rows = (((("count_label", "n = … label", True),
                  ("scale_bar", "scale bar", True),
                  ("show_uncounted", "show uncounted cells dashed", True))),
                ((("numbers", "number every cell", False),
                  ("filled", "extra figure: filled mask", False),
                  ("svg", "also SVG", False))))
        for row in rows:
            r = ttk.Frame(b3)
            r.pack(fill="x", padx=6, pady=(0, 4))
            for key, text, default in row:
                v = tk.BooleanVar(value=default)
                self.count_opts[key] = v
                ttk.Checkbutton(r, text=text, variable=v).pack(side="left", padx=(0, 14))

        r = ttk.Frame(f)
        r.pack(fill="x", padx=8, pady=4)
        self.count_btn = ttk.Button(r, text="Count cells", command=self._count_run)
        self.count_btn.pack(side="left")
        self.count_force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="segment again (ignore saved masks)",
                        variable=self.count_force_var).pack(side="left", padx=8)
        self.count_stop_btn = ttk.Button(r, text="Stop", state="disabled",
                                         command=self._count_stop)
        self.count_stop_btn.pack(side="left")
        ttk.Button(r, text="Open results folder", command=lambda: self._open_path(
            self._count_outdir())).pack(side="left", padx=6)
        self.count_status = ttk.Label(r, text="", foreground="#666")
        self.count_status.pack(side="right")

        pw = ttk.PanedWindow(f, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self.count_log = scrolledtext.ScrolledText(pw, height=10, width=60, wrap="word",
                                                   font=("Consolas", 9))
        self.count_log.tag_config("err", foreground="#c00000")
        self.count_log.tag_config("ok", foreground="#008000")
        self.count_log.tag_config("warn", foreground="#b06000")
        pw.add(self.count_log, weight=1)
        prev = ttk.Frame(pw)
        pw.add(prev, weight=1)
        nav = ttk.Frame(prev)
        nav.pack(fill="x")
        ttk.Button(nav, text="<", width=3, command=lambda: self._count_show(-1))\
            .pack(side="left")
        ttk.Button(nav, text=">", width=3, command=lambda: self._count_show(+1))\
            .pack(side="left", padx=2)
        ttk.Button(nav, text="Open full size", command=lambda: self.count_figs and
                   self._open_path(self.count_figs[self.count_idx])).pack(side="left", padx=4)
        self.count_prev_lbl = ttk.Label(nav, text="Preview appears after counting.",
                                        foreground="#666")
        self.count_prev_lbl.pack(side="left", padx=6)
        self.count_canvas = tk.Label(prev, background="#222")
        self.count_canvas.pack(fill="both", expand=True, pady=(4, 0))
        self.count_canvas.bind("<Configure>", lambda e: self._count_show(0))

        self._count_load_settings()
        self.root.after(150, self._count_poll)

    # ---- helpers ----
    def _count_add_folder(self):
        d = filedialog.askdirectory(title="Folder with brightfield images")
        if d:
            self.count_list.insert("end", os.path.normpath(d))

    def _count_add_files(self):
        fs = filedialog.askopenfilenames(
            title="Brightfield images",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")])
        for p in fs:
            self.count_list.insert("end", os.path.normpath(p))

    def _count_outdir(self):
        out = self.count_out_var.get().strip()
        items = list(self.count_list.get(0, "end"))
        if not out and items:
            out = items[0] if os.path.isdir(items[0]) else os.path.dirname(items[0])
        return os.path.join(out, "cell_count") if out else self.script_dir

    def _count_settings(self):
        return {"inputs": list(self.count_list.get(0, "end")),
                "output": self.count_out_var.get().strip(),
                "px_per_um": self.count_px_var.get().strip(),
                "min_area": self.count_min_var.get().strip(),
                "edge": self.count_edge_var.get(),
                "speed": self.count_speed_var.get(),
                "color": self.count_color_var.get().strip(),
                "line_width": self.count_lw_var.get().strip(),
                "opts": {k: v.get() for k, v in self.count_opts.items()}}

    def _count_load_settings(self):
        try:
            with open(self.COUNT_SETTINGS, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return
        for p in d.get("inputs", []):
            if os.path.exists(p):
                self.count_list.insert("end", p)
        self.count_out_var.set(d.get("output", ""))
        self.count_px_var.set(d.get("px_per_um", self.count_px_var.get()))
        self.count_min_var.set(d.get("min_area", "0"))
        if d.get("speed") in [x[0] for x in self.COUNT_SPEED]:
            self.count_speed_var.set(d["speed"])
        if d.get("edge") in [x[0] for x in self.COUNT_EDGE]:
            self.count_edge_var.set(d["edge"])
        self.count_color_var.set(d.get("color", "yellow"))
        self.count_lw_var.set(d.get("line_width", "3"))
        for k, v in (d.get("opts") or {}).items():
            if k in self.count_opts:
                self.count_opts[k].set(bool(v))

    def _count_run(self):
        items = list(self.count_list.get(0, "end"))
        if not items:
            messagebox.showinfo(APP_TITLE, "Add a folder or images first (step 1).")
            return
        if self.count_proc and self.count_proc.poll() is None:
            messagebox.showinfo(APP_TITLE, "Counting is already running.")
            return

        def number(var, what):
            t = var.get().strip().replace(",", ".")
            if not t:
                return 0.0
            try:
                v = float(t)
                if v < 0:
                    raise ValueError
                return v
            except ValueError:
                messagebox.showerror(APP_TITLE, f"{what}: '{t}' is not a number.")
                raise
        try:
            px = number(self.count_px_var, "Pixels per µm")
            mn = number(self.count_min_var, "Smallest object")
            lw = number(self.count_lw_var, "Line width") or 3.0
        except ValueError:
            return
        try:
            with open(self.COUNT_SETTINGS, "w", encoding="utf-8") as fh:
                json.dump(self._count_settings(), fh, indent=2)
        except OSError:
            pass
        import tempfile
        cfg = os.path.join(tempfile.gettempdir(), "cellcount_gui_config.json")
        with open(cfg, "w", encoding="utf-8") as fh:
            json.dump(self._collect_config(), fh, indent=2)

        edge = dict(self.COUNT_EDGE).get(self.count_edge_var.get(), "all")
        cmd = [sys.executable, "-u", os.path.join(self.script_dir, "cell_count.py")] + items
        cmd += ["--edge", edge, "--color", self.count_color_var.get().strip() or "yellow",
                "--line-width", f"{lw:g}", "--config", cfg]
        sc = dict(self.COUNT_SPEED).get(self.count_speed_var.get(), 0.0)
        if sc:
            cmd += ["--scale", f"{sc:.4g}"]
        if px:
            cmd += ["--px-per-um", f"{px:g}"]
        if mn:
            cmd += ["--min-area-um2", f"{mn:g}"]
        if self.count_out_var.get().strip():
            cmd += ["--output", self.count_out_var.get().strip()]
        o = {k: v.get() for k, v in self.count_opts.items()}
        if not o["count_label"]:
            cmd.append("--no-count-label")
        if not o["scale_bar"]:
            cmd.append("--no-scale-bar")
        if not o["show_uncounted"]:
            cmd.append("--hide-uncounted")
        for k in ("numbers", "filled", "svg"):
            if o[k]:
                cmd.append("--" + k)
        if self.count_force_var.get():
            cmd.append("--force")

        self._count_last_q = -1
        self.count_log.delete("1.0", "end")
        self.count_log.insert("end", "Counting ... the first image also loads Cellpose "
                                     "(and downloads its model once).\n\n")
        self.count_btn.config(state="disabled")
        self.count_stop_btn.config(state="normal")
        self.count_status.config(text="counting...")

        def work():
            try:
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.count_proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    bufsize=1, encoding="utf-8", errors="replace",
                    cwd=self.script_dir, creationflags=flags, env=env)
                for line in self.count_proc.stdout:
                    self.count_queue.put(line)
                self.count_proc.wait()
                self.count_queue.put(("__DONE__", self.count_proc.returncode))
            except Exception:
                self.count_queue.put(traceback.format_exc())
                self.count_queue.put(("__DONE__", -1))

        threading.Thread(target=work, daemon=True).start()

    def _count_stop(self):
        if self.count_proc and self.count_proc.poll() is None:
            self.count_proc.terminate()
            self.count_log.insert("end", "\n[stopped by user]\n", "warn")

    def _count_poll(self):
        try:
            while True:
                item = self.count_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    code = item[1]
                    self.count_btn.config(state="normal")
                    self.count_stop_btn.config(state="disabled")
                    self.count_status.config(
                        text="finished" if code == 0 else f"finished with problems (exit {code})")
                    self.count_log.insert("end", "\nFinished.\n" if code == 0 else
                                          f"\nExited with code {code}.\n",
                                          "ok" if code == 0 else "err")
                    self.count_log.see("end")
                    self._count_find_figs()
                else:
                    s = str(item)
                    # download / progress bars (tqdm) arrive as hundreds of
                    # lines: show only every 25 %
                    m = re.search(r"(\d{1,3})%\|", s)
                    if m:
                        q = int(m.group(1)) // 25
                        if q == getattr(self, "_count_last_q", -1):
                            continue
                        self._count_last_q = q
                        s = "    " + s.strip() + "\n"
                    low = s.lower()
                    tag = ("err" if ("error" in low or "traceback" in low) else
                           "ok" if " counted " in low else
                           "warn" if "warning" in low else None)
                    if "still segmenting" in low:
                        # replace the previous heartbeat line instead of adding one
                        last = self.count_log.get("end-2l", "end-1c")
                        if "still segmenting" in last:
                            self.count_log.delete("end-2l", "end-1c")
                    self.count_log.insert("end", s, tag or ())
                    self.count_log.see("end")
        except queue.Empty:
            pass
        self.root.after(150, self._count_poll)

    def _count_find_figs(self):
        d = os.path.join(self._count_outdir(), "figures")
        try:
            # outline figures first, filled masks after them
            self.count_figs = sorted((os.path.join(d, n) for n in os.listdir(d)
                                      if n.lower().endswith(".png")),
                                     key=lambda p: (p.endswith("_mask.png"), p))
        except OSError:
            self.count_figs = []
        self.count_idx = 0
        self._count_show(0)

    def _count_show(self, step):
        if not self.count_figs:
            return
        self.count_idx = (self.count_idx + step) % len(self.count_figs)
        path = self.count_figs[self.count_idx]
        try:
            from PIL import Image, ImageTk
            w = max(200, self.count_canvas.winfo_width() - 4)
            h = max(200, self.count_canvas.winfo_height() - 4)
            with Image.open(path) as im:
                im.thumbnail((w, h))
                self.count_photo = ImageTk.PhotoImage(im.convert("RGB"))
            self.count_canvas.config(image=self.count_photo)
            self.count_prev_lbl.config(
                text=f"{self.count_idx + 1}/{len(self.count_figs)}  {os.path.basename(path)}")
        except Exception as exc:
            self.count_prev_lbl.config(text=f"(no preview: {exc})")

    # ============================================================== setup ===
    def _build_setup_tab(self):
        f = self.tab_setup
        self.setup_queue = queue.Queue()
        self.root.after(120, self._setup_poll)
        ttk.Label(f,
                  text="Checks that everything the pipeline needs is present and working.\n"
                       "Run this first on any new machine.",
                  justify="left").pack(anchor="w", padx=10, pady=8)
        r = ttk.Frame(f)
        r.pack(fill="x", padx=10)
        ttk.Button(r, text="Run setup check", command=self._setup_check).pack(side="left")
        ub = ttk.Button(r, text="Get latest version from GitHub",
                        command=self._update_from_git)
        ub.pack(side="left", padx=8)
        ToolTip(ub, "Downloads the newest version of these scripts (git pull).\n"
                    "Your data and settings are not touched. Restart the\n"
                    "program afterwards.")
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
        d = filedialog.askdirectory(title="Choose the Fiji or ImageJ folder")
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
        # The Fiji macro is only used by the legacy backend (USE_FIJI = True in
        # run_cargo_pipeline.py). Measurement runs in numpy by default, so a
        # missing macro must not block a run - it used to, which made a fresh
        # copy of the code unusable until an unrelated .ijm file was found.
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
        # read Tk variables HERE, on the Tk thread; the worker must not
        self._setup_fiji = self.fiji_var.get().strip()
        threading.Thread(target=self._setup_worker, daemon=True).start()

    def _sput(self, text, tag=None):
        """Queue a line for the setup log. Called from worker threads, so it
        must not touch the widget itself - _setup_poll writes it on the Tk
        thread (tkinter is not thread-safe)."""
        self.setup_queue.put((text, tag))

    def _setup_poll(self):
        try:
            while True:
                text, tag = self.setup_queue.get_nowait()
                self.setup_log.insert("end", text + "\n", tag or ())
                self.setup_log.see("end")
        except queue.Empty:
            pass
        self.root.after(120, self._setup_poll)

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
                ver = getattr(m, "__version__", None)
                if not ver:
                    # cellpose 4 has no __version__; ask the installed package
                    try:
                        from importlib.metadata import version as _v
                        ver = _v({"PIL": "pillow"}.get(mod, mod))
                    except Exception:
                        ver = "?"
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

        # Fiji / ImageJ - OPTIONAL. Measurement runs in Python (numpy); Fiji is
        # only used by the old macro backend (USE_FIJI = True in
        # run_cargo_pipeline.py), which nothing uses by default. So a missing
        # Fiji is never an error, and plain ImageJ 1.x (ImageJ.exe) counts too.
        self._sput("")
        fiji_dir = getattr(self, "_setup_fiji", "")
        note = "  (optional - the analysis does not need Fiji or ImageJ)"
        if not fiji_dir:
            self._sput("Fiji / ImageJ: not set" + note, ok)
        elif not os.path.isdir(fiji_dir):
            self._sput(f"Fiji / ImageJ folder does not exist: {fiji_dir}" + note, "warn")
        else:
            names = ["fiji-windows-x64.exe", "ImageJ-win64.exe",
                     "fiji-windows-x86.exe", "ImageJ-win32.exe",
                     "ImageJ-linux64", "Contents/MacOS/ImageJ-macosx",
                     "ImageJ.exe"]
            found = None
            for n in names:
                p = os.path.join(fiji_dir, n)
                if os.path.isfile(p):
                    found = p
                    break
            if found and os.path.basename(found) == "ImageJ.exe":
                self._sput(f"ImageJ  : {found}  (ImageJ 1.x)" + note, ok)
            elif found:
                self._sput(f"Fiji    : {found}" + note, ok)
            else:
                self._sput(f"No Fiji or ImageJ launcher in {fiji_dir}" + note, "warn")
                try:
                    self._sput("  contains: " + ", ".join(sorted(os.listdir(fiji_dir))[:20]))
                except OSError:
                    pass

        # companion files
        self._sput("")
        p = os.path.join(self.script_dir, PIPELINE_SCRIPT)
        self._sput(f"{PIPELINE_SCRIPT}: {'found' if os.path.isfile(p) else 'MISSING'}",
                   ok if os.path.isfile(p) else bad)
        p = os.path.join(self.script_dir, MACRO_FILE)
        self._sput(f"{MACRO_FILE}: "
                   + ("found" if os.path.isfile(p) else
                      "not present (only needed for the old Fiji backend - fine)"),
                   ok if os.path.isfile(p) else None)

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

        # CNN + version
        self._sput("")
        try:
            import importlib
            importlib.import_module("cargo_cnn.model")
            self._sput("  cargo_cnn        found (CNN tab available)", ok)
        except ImportError as exc:
            self._sput(f"  cargo_cnn        NOT usable ({exc})", "warn")
        try:
            import cargo_version
            g = cargo_version.git_state(self.script_dir)
            if g:
                self._sput(f"Code version: {g['describe']}"
                           + ("  - with UNCOMMITTED EDITS" if g["dirty"] else ""),
                           "warn" if g["dirty"] else ok)
            else:
                self._sput("Code version: not a git checkout (updates must be "
                           "copied by hand - see GETTING_STARTED.md)", "warn")
        except Exception:
            pass

        self._sput("\nDone. Anything red must be fixed before running.", "warn")

    def _update_from_git(self):
        """git pull in the program folder, with the result in plain words."""
        if not os.path.isdir(os.path.join(self.script_dir, ".git")):
            messagebox.showinfo(
                APP_TITLE,
                "This folder was not downloaded with git, so it cannot update "
                "itself.\n\nSee GETTING_STARTED.md: clone the repository once with "
                "GitHub Desktop, then this button works.")
            return
        self.setup_log.delete("1.0", "end")
        self._sput("Getting the latest version from GitHub...\n")

        def work():
            try:
                import cargo_version
                git = cargo_version.find_git()
                if not git:
                    raise FileNotFoundError("git")
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                p = subprocess.run([git, "pull", "--ff-only"], cwd=self.script_dir,
                                   capture_output=True, text=True, timeout=120,
                                   creationflags=flags)
                out = (p.stdout + p.stderr).strip()
                if p.returncode == 0 and "Already up to date" in out:
                    msg, tag = "You already have the latest version.", "ok"
                elif p.returncode == 0:
                    msg, tag = ("Updated. CLOSE AND RESTART the program to use the "
                                "new version."), "ok"
                elif "local changes" in out or "would be overwritten" in out:
                    msg, tag = ("Not updated: you edited some of these files yourself. "
                                "Commit or discard those edits in GitHub Desktop, "
                                "then try again."), "err"
                else:
                    msg, tag = "Update failed - details above.", "err"
            except FileNotFoundError:
                out, msg, tag = "", ("git is not installed. Use GitHub Desktop: "
                                     "'Fetch origin', then 'Pull origin'."), "err"
            except Exception as exc:
                out, msg, tag = "", f"Update failed: {exc}", "err"
            self._sput(out)
            self._sput("\n" + msg, tag)

        threading.Thread(target=work, daemon=True).start()

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