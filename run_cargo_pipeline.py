#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_cargo_pipeline.py
=============================================================================
End-to-end cargo-release pipeline for HeLa + iron-fumarate/calcein images.

For each selected sample folder (e.g. A1, A2, C2, D1 inside pre_glucose/):

  1. Find the two tifs. LOWER-sorted = brightfield, HIGHER-sorted = fluorescence.
  2. If <parent>_<sample>_cp_masks.png already exists in that folder -> SKIP
     Cellpose entirely and go straight to step 4. (This is the re-run case.)
  3. Otherwise: preprocess the brightfield exactly like
     Export_Brightfield_For_Cellpose.ijm did (8-bit -> 0.5x downscale ->
     JPEG q90), run Cellpose 4 / cpsam on it, and save the label mask as
     <parent>_<sample>_cp_masks.png in the sample folder.
  4. Run Cargo_Release_Classification_v3_headless.ijm in Fiji on the
     fluorescence tif + that mask, writing into <sample>/edited_results/.
  5. After all samples: collect every _summary.csv into batch_summary.csv
     at the PARENT level, one row per sample.

WHY THE MASK IS SAVED AT HALF RESOLUTION AND THAT IS FINE:
  The brightfield is downscaled 0.5x before Cellpose (segmentation time scales
  with pixel count). The ImageJ macro upscales the mask back to the full
  fluorescence dimensions with nearest-neighbour interpolation, which preserves
  the integer label IDs exactly. Nothing downstream needs to change.

USAGE (from an Anaconda Prompt with the cellpose env active):
    conda activate cellpose
    python run_cargo_pipeline.py

  It will then ask for the parent folder and which samples to run.
  Or give them on the command line:
    python run_cargo_pipeline.py --parent "D:\\data\\pre_glucose" --samples A1,A2,C2,D1
    python run_cargo_pipeline.py --parent "D:\\data\\pre_glucose" --samples all
    python run_cargo_pipeline.py --parent "D:\\data\\pre_glucose" --samples all --force
        (--force re-runs Cellpose even when a mask already exists)

REQUIREMENTS: cellpose 4.x, numpy, Pillow, tifffile (all already in your env),
and Fiji with the MorphoLibJ (IJPB-plugins) update site enabled.
=============================================================================
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

# =========================== CONFIGURATION ===========================
# Adjust these if you move things. Everything else is derived.

FIJI_DIR = r"C:\Users\cemin\Documents\Molbio\Fiji\Fiji"

# The headless macro. By default it is looked for next to this script.
MACRO_NAME = "Cargo_Release_Classification_v3_headless.ijm"

# --- brightfield preprocessing (mirrors Export_Brightfield_For_Cellpose.ijm) ---
# --- Cellpose 4 / CPSAM settings ---
CELLPOSE_MODEL      = "cpsam"

# FLOW_THRESHOLD (default 0.4) controls how far a predicted mask may deviate
# from the model's flow field. RAISING it accepts more irregular shapes, so
# more cells are kept - at the cost of some poorly-formed masks.
FLOW_THRESHOLD      = 0.4

# CELLPROB_THRESHOLD (default 0.0) is the main knob for MISSED CELLS.
# LOWER values accept fainter, lower-confidence cells:
#     0.0   default
#    -1.0   noticeably more cells, usually the first thing to try
#    -2.0   aggressive; picks up debris as well
# Raise above 0 only if the segmentation is finding things that are not cells.
CELLPROB_THRESHOLD  = 0.0

# Segmentation resolution. Measured on this data, cells are ~85 px across at
# full resolution, so at 0.5x Cellpose still sees ~43 px - well inside its
# working range, and the smallest cells are ~30 px. Full resolution therefore
# costs about 4x the time for little expected gain. Set to 1.0 if small or
# faint cells are still being missed after adjusting CELLPROB_THRESHOLD.
CELLPOSE_SCALE_FACTOR = 0.5

USE_GPU             = True    # falls back to CPU automatically if unavailable
# NOTE: CPSAM is largely scale-invariant; unlike cyto3 it does not need a
# diameter, so none is passed.

# ---- classification ------------------------------------------------------
# CLASSIFY_METHOD "brightness" is the default because it is what the data
# supports. Measured against 195 hand-labelled cells across four samples:
#     MeanWhole (brightness)  AUC 0.917   <- used
#     CoreFillRatio           AUC 0.451   <- chance
#     CoreCV                  AUC 0.453   <- chance
# Brightness alone reproduces the human labels on 87% of cells with one global
# threshold, 95% with a per-sample threshold. "texture" selects the old
# fill-ratio/CV rule and is kept only so the two can be compared.
CLASSIFY_METHOD = "brightness"

# Global fallback threshold, fitted to the hand labels.
BRIGHTNESS_THRESHOLD = 25.7

# Core CV cap. A cell must ALSO be smooth enough to count as released - low CV
# means an even fill, high CV means a few bright puncta among dark pixels.
# Fitted on 523 blind labels: brightness alone 80%, brightness AND CV 88%.
# Set to None to disable and use brightness only.
CORE_CV_CAP = 0.65

# Saturation veto. If this fraction of a cell sits at the top of the intensity
# range, the cell is bright because something saturated bled over it rather
# than because its cargo was released, and it is flagged rather than counted.
# 0.0 disables. Measured on hand labels, 6 of 50 bright cells were false
# positives of exactly this kind.
SATURATION_GUARD = 0.0

# AUTO_THRESHOLD picks a threshold per image (Otsu on the per-cell means).
# The best hand-fitted threshold ranged 14.2-53.3 across four samples because
# exposure differs between acquisitions, so per-image is usually the right
# choice. Whichever value is used is written into the summary CSV.
AUTO_THRESHOLD = True

# ---- legacy texture thresholds (only used when CLASSIFY_METHOD="texture") --
FILL_RATIO_THRESHOLD  = 0.65
CORE_CV_THRESHOLD     = 0.60
MIN_CELL_MEAN         = 5     # below this = no uptake, excluded from statistics
CORE_RADIUS_FRACTION  = 0.5   # inner 50% of the radius = "core"
ROLLING_BALL_RADIUS   = 50    # background subtraction, pixels

# veryHighSignalCutoff is on the 8-bit 0-255 scale. If the fluorescence tif
# turns out to be 16-bit, it is rescaled automatically (see scaled_cutoff()).
VERY_HIGH_SIGNAL_CUTOFF_8BIT = 200

# Measurement backend.
#   False (default) -> measure in numpy via cargo_measure.py. Same maths, about
#                      80x faster (6 s vs 500 s for 2400 cells), and with no
#                      ImageJ macro interpreter to crash.
#   True            -> use the old Fiji macro. Kept only as a fallback for
#                      comparing results; it is slower and has repeatedly died
#                      with no error message on large images.
USE_FIJI = False

# Java heap for Fiji, in MB. Only used when USE_FIJI is True.
FIJI_MEMORY_MB = 12288
# =====================================================================


def apply_settings(overrides):
    """
    Override the module-level settings above from a dict.

    The GUI (and the --config option) uses this so nobody has to edit this file
    by hand. Unknown keys are ignored so an old config never crashes a new
    version; type is taken from the existing default.
    """
    for key, value in (overrides or {}).items():
        name = key.upper()
        if name in globals() and not name.startswith("_"):
            current = globals()[name]
            try:
                if isinstance(current, bool):
                    value = str(value).strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(float(value))
                elif isinstance(current, float):
                    value = float(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                continue
            globals()[name] = value


# --------------------------------------------------------------------------
def find_fiji_executable(fiji_dir):
    """Fiji renamed its launcher across versions, so try the known names."""
    candidates = [
        "ImageJ-win64.exe",
        "fiji-windows-x64.exe",
        "ImageJ-win32.exe",
        "fiji-windows-x86.exe",
    ]
    for name in candidates:
        path = os.path.join(fiji_dir, name)
        if os.path.isfile(path):
            return path
    # not found - fail loudly and show what IS there, so the fix is obvious
    try:
        contents = "\n  ".join(sorted(os.listdir(fiji_dir)))
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot read Fiji folder {fiji_dir}\n  {exc}")
    raise SystemExit(
        f"ERROR: no Fiji launcher found in {fiji_dir}\n"
        f"Looked for: {', '.join(candidates)}\n"
        f"That folder contains:\n  {contents}\n"
        "Set FIJI_DIR at the top of this script to the folder holding the .exe."
    )


def find_channels(folder):
    """
    Identify the brightfield and fluorescence tifs in a sample folder.

    Two naming conventions are supported:

      1. EXPLICIT (preferred) - the filename says what it is:
             A1_brightfield.tif    -> segmentation input
             A1_fluorescence.tif   -> measured channel
             A1_overlay.tif        -> IGNORED, it is a derived image
         This is unambiguous, so it wins whenever it is present.

      2. POSITIONAL (legacy) - two timestamped files, lower-sorted is the
         brightfield. Fragile, but it is how the older data is named.

    Files this pipeline generates are always excluded, or the mask would be
    picked up as an input and the analysis would measure it against itself.
    Returns (brightfield_path, fluorescence_path) or (None, None).
    """
    generated = ("_cp_masks", "_classified_overlay", "_brightfield_for_cellpose",
                 "_bbox")
    tifs = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(f)[0].lower()
        if any(t in stem for t in generated):
            continue
        tifs.append(f)

    bf = next((f for f in tifs
               if "brightfield" in f.lower() or "_bf" in f.lower()), None)
    fl = next((f for f in tifs
               if "fluor" in f.lower() or "_fl" in f.lower()
               or "calcein" in f.lower()), None)

    if bf and fl:
        return os.path.join(folder, bf), os.path.join(folder, fl)

    # positional fallback: ignore anything that looks like a derived overlay
    plain = [f for f in tifs if "overlay" not in f.lower()]
    if len(plain) >= 2:
        return os.path.join(folder, plain[0]), os.path.join(folder, plain[-1])
    return None, None


def list_tifs(folder):
    """
    Sorted list of the ACQUISITION tifs only. Kept for compatibility; the
    channel identification now goes through find_channels().
    """
    generated = ("_cp_masks", "_classified_overlay", "_brightfield_for_cellpose",
                 "_bbox", "_overlay")
    tifs = []
    for f in os.listdir(folder):
        if not f.lower().endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(f)[0].lower()
        if any(t in stem for t in generated):
            continue
        tifs.append(f)
    tifs.sort()
    return tifs


def scaled_cutoff(fluor_path):
    """
    Return veryHighSignalCutoff on the scale of the actual image.

    You were unsure whether the acquisition is 8- or 16-bit, so this reads the
    real dtype instead of assuming. 200 on the 0-255 scale becomes the same
    *fraction* of full range on a 16-bit image (200/255 * 65535 = 51402).
    """
    try:
        import tifffile
        with tifffile.TiffFile(fluor_path) as tf:
            dtype = tf.series[0].dtype
        if dtype.itemsize >= 2:  # 16-bit or deeper
            return VERY_HIGH_SIGNAL_CUTOFF_8BIT / 255.0 * 65535.0
    except Exception as exc:
        print(f"    (could not read bit depth: {exc}; assuming 8-bit)")
    return float(VERY_HIGH_SIGNAL_CUTOFF_8BIT)


def load_brightfield(bf_path):
    """
    Load the brightfield tif and reduce it to a single 2D 8-bit plane for
    Cellpose. Returns the array; no intermediate JPEG is written.

    WHY NO JPEG ANY MORE: the original design mirrored
    Export_Brightfield_For_Cellpose.ijm, which wrote a JPG because you were
    dragging a file into the GUI by hand. Driving Cellpose from Python removes
    that need, and every extra step was a place to corrupt the image. Cellpose
    is now handed the pixels directly.
    """
    import numpy as np
    import tifffile

    arr = tifffile.imread(bf_path)
    raw_shape = arr.shape

    # --- reduce to a single 2D plane ---
    # Find the two LARGEST axes - those are always height and width - and
    # reduce every other axis (RGB channels, z, time) away. Assuming the
    # channel axis is axis 0 is wrong for (H, W, 3) images and shreds them.
    if arr.ndim > 2:
        big = sorted(sorted(range(arr.ndim), key=lambda i: arr.shape[i])[-2:])
        for ax in sorted([i for i in range(arr.ndim) if i not in big], reverse=True):
            if arr.shape[ax] <= 4:
                arr = arr.mean(axis=ax)   # RGB -> grayscale (your channels are identical)
            else:
                arr = arr.max(axis=ax)    # z-stack -> maximum projection

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"could not reduce {bf_path} to 2D (got shape {arr.shape})")

    # sanity check: a real microscopy field is never a few pixels wide
    if min(arr.shape) < 32:
        raise ValueError(
            f"brightfield reduced to an implausible shape {arr.shape} - "
            "the channel/stack axes were not identified correctly."
        )

    # --- 8-bit via PERCENTILE normalization ---
    # Clipping at the 1st/99th percentile ignores hot/dead pixels, which a plain
    # min-max stretch would let dictate the whole range.
    lo, hi = np.percentile(arr.astype(np.float32), [1.0, 99.0])
    if hi > lo:
        arr = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr = arr.astype(np.uint8)

    # --- optional downscale for speed ---
    if CELLPOSE_SCALE_FACTOR != 1.0:
        from PIL import Image
        img = Image.fromarray(arr)
        new_size = (max(1, int(round(img.width * CELLPOSE_SCALE_FACTOR))),
                    max(1, int(round(img.height * CELLPOSE_SCALE_FACTOR))))
        arr = np.array(img.resize(new_size, Image.BILINEAR))

    print(f"    brightfield {raw_shape} -> 2D -> {arr.shape} (std {arr.std():.1f})")
    if arr.std() < 3:
        print("    WARNING: very low contrast - Cellpose will likely find nothing.")
    return arr


def run_cellpose(img, mask_path, model_holder):
    """
    Segment the brightfield array and save the LABEL mask as PNG.

    The mask is written as a 16-bit PNG so label IDs above 255 survive - your
    B2 field alone has 1670 cells, and an 8-bit save would wrap 1415 of them
    around and silently merge distinct cells.
    """
    import numpy as np
    from cellpose import models
    from PIL import Image

    if model_holder.get("model") is None:
        print("    loading Cellpose model (first sample only)...")
        device, desc = pick_torch_device()
        print(f"    compute device -> {desc}")
        model_holder["device_desc"] = desc

        # Cellpose takes an explicit torch device. Passing the device found
        # above means AMD (ROCm or DirectML) and Apple Silicon are used when
        # present, instead of silently falling back to CPU.
        try:
            if device in (None, "cpu") or not USE_GPU:
                model_holder["model"] = models.CellposeModel(gpu=False)
            else:
                import torch
                dev = torch.device(device) if isinstance(device, str) else device
                model_holder["model"] = models.CellposeModel(gpu=True, device=dev)
        except Exception as exc:
            print(f"    GPU init failed ({exc}); falling back to CPU")
            model_holder["model"] = models.CellposeModel(gpu=False)

    model = model_holder["model"]

    result = model.eval(
        img,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        normalize=True,          # per-image percentile normalization, as the GUI does
    )
    masks = result[0]
    n_cells = int(masks.max())

    # If nothing was found, retry once with a permissive cellprob threshold.
    if n_cells == 0:
        print("    0 cells at default settings - retrying with cellprob_threshold=-2.0 ...")
        result = model.eval(
            img,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=-2.0,
            normalize=True,
        )
        masks = result[0]
        n_cells = int(masks.max())
        if n_cells > 0:
            print(f"    retry succeeded: {n_cells} cells. Consider setting "
                  "CELLPROB_THRESHOLD = -2.0 at the top of this script.")

    # Segmentation coverage: what fraction of the field ended up inside a cell.
    # A confluent HeLa field should be well covered; low coverage with cells
    # clearly visible in the brightfield means Cellpose is missing them, and
    # the fix is a LOWER cellprob_threshold, not anything downstream.
    covered = float((masks > 0).mean())
    print(f"    segmentation covers {covered*100:.0f}% of the field")
    if covered < 0.25:
        print("    NOTE: low coverage. If the brightfield shows clearly more "
              "cells than this, lower CELLPROB_THRESHOLD (try -1.0) and re-run "
              "with --force.")

    # Save the label mask as a 16-BIT TIFF, not PNG.
    # ImageJ's PNG reader handles 16-bit grayscale ("I;16") unreliably - it can
    # open it as RGB, after which the label IDs are destroyed and label
    # extraction finds nothing. A 16-bit TIFF is read unambiguously, and
    # tifffile writes the label IDs exactly.
    import tifffile
    tifffile.imwrite(mask_path, masks.astype(np.uint16))
    write_bbox_table(masks, mask_path)

    return n_cells


def write_bbox_table(masks, mask_path):
    """
    Write a table of each label's bounding box next to the mask.

    WHY: the ImageJ macro needs, for every label, the rectangle it occupies, so
    it can threshold inside that small box instead of scanning the whole
    5950x4468 image once per label. Computing it in the macro meant a loop over
    millions of pixels in interpreted macro language - slow, and it crashed the
    macro interpreter outright. Here the mask is already a numpy array, so the
    same information costs a fraction of a second and is completely reliable.

    Format: one line per present label, "label,x,y,w,h" in MASK pixel
    coordinates (the macro scales them up).
    """
    import numpy as np

    n = int(masks.max())
    if n < 1:
        return None

    # np.nonzero once, then group - far cheaper than slicing per label
    ys, xs = np.nonzero(masks)
    if len(ys) == 0:
        return None
    labels = masks[ys, xs]

    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    xs = xs[order]
    ys = ys[order]

    # boundaries between runs of equal label
    starts = np.searchsorted(labels, np.arange(1, n + 1), side="left")
    ends = np.searchsorted(labels, np.arange(1, n + 1), side="right")

    lines = []
    for lb in range(1, n + 1):
        s, e = starts[lb - 1], ends[lb - 1]
        if s >= e:
            continue                       # label id not present
        x0 = int(xs[s:e].min())
        x1 = int(xs[s:e].max())
        y0 = int(ys[s:e].min())
        y1 = int(ys[s:e].max())
        lines.append(f"{lb},{x0},{y0},{x1 - x0 + 1},{y1 - y0 + 1}")

    path = os.path.splitext(mask_path)[0] + "_bbox.csv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def pick_torch_device(verbose=True):
    """
    Find the fastest available compute device, in preference order:

      1. CUDA     - NVIDIA
      2. ROCm     - AMD. PyTorch-for-ROCm exposes AMD cards through the SAME
                    torch.cuda API, so this is detected together with CUDA and
                    told apart by torch.version.hip.
      3. DirectML - AMD/Intel on Windows via the torch-directml package. Works
                    on any DirectX 12 GPU, including Radeon cards that ROCm
                    does not cover.
      4. MPS      - Apple Silicon
      5. CPU

    Returns (device, description). Cellpose accepts a torch device object.
    """
    try:
        import torch
    except ImportError:
        return None, "torch not installed"

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            is_rocm = getattr(torch.version, "hip", None) is not None
            kind = "ROCm/HIP" if is_rocm else "CUDA"
            return "cuda", f"{kind}: {name}"
    except Exception:
        pass

    # DirectML - separate package, any DX12 GPU. The practical AMD path on Windows.
    try:
        import torch_directml
        if torch_directml.is_available():
            try:
                name = torch_directml.device_name(0)
            except Exception:
                name = "DirectML device"
            return torch_directml.device(), f"DirectML: {name}"
    except Exception:
        pass

    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", "Apple Metal (MPS)"
    except Exception:
        pass

    try:
        n = os.cpu_count() or 1
        torch.set_num_threads(n)
        return "cpu", f"CPU ({n} threads)"
    except Exception:
        return "cpu", "CPU"


def describe_gpu_options():
    """Report what acceleration is and is not available, and why."""
    lines = []
    try:
        import torch
        lines.append(f"torch {torch.__version__}")
        hip = getattr(torch.version, "hip", None)
        cuda = getattr(torch.version, "cuda", None)
        if hip:
            lines.append(f"  built for ROCm/HIP {hip}  (AMD)")
        elif cuda:
            lines.append(f"  built for CUDA {cuda}  (NVIDIA)")
        else:
            lines.append("  CPU-only build - no GPU support compiled in")
        lines.append(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    except ImportError:
        lines.append("torch not installed")

    try:
        import torch_directml
        lines.append(f"torch-directml present, available: {torch_directml.is_available()}")
    except ImportError:
        lines.append("torch-directml NOT installed  (the usual AMD path on Windows)")

    _, desc = pick_torch_device()
    lines.append(f"-> will use: {desc}")
    return "\n".join(lines)


def fiji_memory_flag():
    """
    Decide the Java heap for Fiji, capped by the RAM the machine actually has.

    A fixed --mem larger than available memory makes the JVM fail to start: it
    exits immediately, silently, with code 0 and no output at all. So the
    configured value is treated as a ceiling, not a demand - at most half of
    total RAM is requested, and the flag is dropped entirely if that would be
    too small to be worth setting.
    """
    want = int(FIJI_MEMORY_MB)
    total_mb = None
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            total_mb = (os.sysconf("SC_PAGE_SIZE") *
                        os.sysconf("SC_PHYS_PAGES")) // (1024 * 1024)
        elif os.name == "nt":
            import ctypes

            class MemStat(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemStat()
            st.dwLength = ctypes.sizeof(MemStat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                total_mb = int(st.ullTotalPhys // (1024 * 1024))
    except Exception:
        total_mb = None

    if total_mb:
        safe = max(1024, min(want, total_mb // 2))
        if safe != want:
            print(f"    (Fiji heap capped at {safe} MB; machine has {total_mb} MB RAM)")
        want = safe

    return want


def run_imagej(fiji_exe, macro_path, fluor_path, mask_path, out_dir, base,
               cutoff, timeout=1800):
    """Invoke the headless macro with pipe-separated arguments."""
    # ImageJ macro paths: forward slashes are safest, and the output dir must
    # end with a separator because the macro concatenates filenames onto it.
    def ij(p):
        return p.replace("\\", "/")

    out_dir_ij = ij(out_dir)
    if not out_dir_ij.endswith("/"):
        out_dir_ij += "/"

    macro_args = "|".join([
        ij(fluor_path),
        ij(mask_path),
        out_dir_ij,
        base,
        str(FILL_RATIO_THRESHOLD),
        str(CORE_CV_THRESHOLD),
        str(cutoff),
        str(MIN_CELL_MEAN),
        str(CORE_RADIUS_FRACTION),
        str(ROLLING_BALL_RADIUS),
    ])

    # NOTE ON --headless: Fiji's ROI Manager is an AWT window and CANNOT be
    # instantiated in true headless mode - it throws java.awt.HeadlessException
    # at RoiManager.<init>. Since the whole analysis is built on ROIs from the
    # Cellpose label map, headless is simply not an option here. Instead Fiji
    # is launched normally (a window flashes up briefly per sample) and closed
    # by the run("Quit") at the end of the macro.
    #
    # --mem raises the Java heap. With ~2500 ROIs held against a 5953x4469
    # image, the default heap fills and Java starts collecting garbage
    # continuously - one sample took 2.5 hours instead of ~9 minutes purely
    # from heap thrashing. Give it plenty of room.
    mem_mb = fiji_memory_flag()
    # --headless is NOT usable (the ROI Manager is an AWT window), but the
    # splash screen and the main window can still be suppressed:
    #   --no-splash          : no Fiji logo on startup
    #   -Djava.awt.headless=false : keep AWT usable for the ROI Manager
    #   -Dapple.awt.UIElement / -Dsun.java2d.noddraw : keep the window out of
    #                          the way on macOS/Windows respectively
    cmd = [fiji_exe,
           f"--mem={mem_mb}m",
           "--no-splash",
           "-batch", macro_path, macro_args]

    summary_file = os.path.join(out_dir, base + "_summary.csv")
    log_file = os.path.join(out_dir, base + "_log.txt")
    for stale in (summary_file, log_file):
        if os.path.isfile(stale):
            try:
                os.remove(stale)          # so a stale file can't fake success
            except OSError:
                pass

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except OSError as exc:
        print(f"    ERROR: could not launch Fiji: {exc}")
        return False

    # A JVM that cannot allocate the requested heap - or that rejects one of the
    # optional flags - exits immediately, silently, with no output. Retry with a
    # smaller heap and no optional flags. Falling back to Fiji's built-in
    # default heap is what makes a few thousand ROIs thrash for hours, so the
    # heap flag itself is kept.
    if not os.path.isfile(log_file) and not (proc.stdout or "").strip():
        fallback = max(1024, mem_mb // 2)
        print(f"    (Fiji produced nothing; retrying with --mem={fallback}m "
              "and no optional flags)")
        cmd = [fiji_exe, f"--mem={fallback}m", "-batch", macro_path, macro_args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except OSError as exc:
            print(f"    ERROR: could not launch Fiji: {exc}")
            return False

    out = (proc.stdout or "") + (proc.stderr or "")

    # Fiji's console output is unreliable on Windows ("GUI program launched from
    # unknown parent process"), so success is judged by whether the summary CSV
    # actually appeared on disk - not by parsing stdout.
    # Always surface the macro's own diagnostic lines (mask size, ROI count,
    # ERROR messages). Previously these were filtered away, so a mask that
    # loaded but yielded 0 ROIs looked like a clean run.
    for line in out.splitlines():
        s = line.strip()
        if s.startswith(("mask:", "ROIs from label map:", "ERROR ", "WARNING ")) \
                and not s.startswith("[WARNING]"):
            print("    " + s)

    if os.path.isfile(summary_file):
        for line in (proc.stdout or "").splitlines():
            s = line.strip()
            if s.startswith(base + ":"):
                print("    " + s)
        return True

    # No summary was written. The macro writes its own log file, which survives
    # even when Fiji's console output does not reach us.
    print("    analysis produced no summary file.")
    if os.path.isfile(log_file):
        print("    ----- macro log -----")
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.strip():
                        print("    " + line.rstrip())
        except OSError as exc:
            print(f"    (could not read log: {exc})")
    else:
        print("    (no log file was written - the macro never started)")

    print("    ----- full Fiji output -----")
    if out.strip():
        for line in out.strip().splitlines():
            print("    " + line)
    else:
        print("    (Fiji produced no output at all)")
    print(f"    ----- exit code: {proc.returncode} -----")
    return False


def collect_batch_summary(parent, samples, parent_name, output_root=None,
                          base_names=None):
    """
    Merge every per-sample _summary.csv into one table at parent level.

    base_names maps sample -> the base name actually used for that sample's
    files. It is required because a label mapping renames the outputs: the
    file is "A1_Fe-fum-plus-Histidin-5uL_pre_glucose_summary.csv", not
    "pre_glucose_A1_summary.csv". Rebuilding the old name here meant the file
    was never found and the batch table came out empty - with only a quiet
    "no summary" note to show for it.
    """
    rows = []
    header = None

    for sample in samples:
        base = (base_names or {}).get(sample) or f"{parent_name}_{sample}"
        if output_root:
            summary_path = os.path.join(output_root, sample, f"{base}_summary.csv")
        else:
            summary_path = os.path.join(parent, sample, "edited_results",
                                        f"{base}_summary.csv")
        if not os.path.isfile(summary_path):
            # fall back to whatever *_summary.csv is actually there, so a
            # naming mismatch degrades to a warning rather than silent loss
            folder = (os.path.join(output_root, sample) if output_root
                      else os.path.join(parent, sample, "edited_results"))
            found = None
            if os.path.isdir(folder):
                cands = [f for f in os.listdir(folder)
                         if f.endswith("_summary.csv")]
                if len(cands) == 1:
                    found = os.path.join(folder, cands[0])
            if found:
                print(f"  ! {sample}: expected {os.path.basename(summary_path)}, "
                      f"used {os.path.basename(found)} instead")
                summary_path = found
            else:
                print(f"  ! no summary for {sample} (analysis did not complete)")
                continue
        with open(summary_path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if len(lines) < 2:
            continue
        if header is None:
            header = lines[0]
        rows.append(lines[1])

    if not rows:
        print("  ! nothing to write into batch_summary.csv")
        return None

    # Name the batch file after the parent folder, so summaries from different
    # conditions stay distinguishable once they are copied somewhere together.
    dest = output_root if output_root else parent
    out_path = os.path.join(dest, f"{parent_name}_batch_summary.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(r + "\n")
    return out_path


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cellpose + cargo-release pipeline")
    # The parent folder can be given either as a bare path or with --parent.
    ap.add_argument("parent_pos", nargs="?", default=None,
                    help=r'Parent folder as a plain argument, e.g. "D:\data\pre_glucose"')
    ap.add_argument("--parent", help=r'Parent folder, e.g. "D:\data\pre_glucose"')
    ap.add_argument("--samples", help='Comma list e.g. "A1,A2,C2,D1", or "all"')
    ap.add_argument("--force", action="store_true",
                    help="Re-run Cellpose even if a mask already exists")
    ap.add_argument("--config", help="JSON file of setting overrides (used by the GUI)")
    ap.add_argument("--gpu-info", action="store_true",
                    help="Report what GPU acceleration is available, then exit")
    ap.add_argument("--output", help="Where to write results. A folder named "
                                     "<parent>_analysis is created inside it. "
                                     "Default: alongside each sample.")
    ap.add_argument("--labels", help="Text/TSV file holding a plate layout, so "
                                     "outputs are named by condition rather than "
                                     "by well alone.")
    args = ap.parse_args()

    if args.gpu_info:
        print(describe_gpu_options())
        return

    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as fh:
                apply_settings(json.load(fh))
        except Exception as exc:
            raise SystemExit(f"ERROR: could not read config {args.config}: {exc}")

    # ---- resolve parent folder ----
    parent = args.parent or args.parent_pos
    if not parent:
        parent = input("Parent folder (e.g. D:\\data\\pre_glucose): ")
    # Tolerate the usual copy/paste artefacts: surrounding quotes, stray
    # whitespace, and the trailing backslash Windows adds on tab-completion.
    parent = parent.strip().strip('"').strip("'").rstrip("\\/")
    parent = os.path.abspath(parent)
    if not os.path.isdir(parent):
        raise SystemExit(f"ERROR: not a folder: {parent}")

    parent_name = os.path.basename(parent.rstrip("\\/"))

    # ---- optional sample-name mapping ----
    # Without this a result is called "A1", which says nothing about what was
    # in the well. With it the condition travels with the file.
    label_map = {}
    if args.labels:
        try:
            import cargo_labels
            with open(args.labels, "r", encoding="utf-8") as fh:
                label_map, _notes = cargo_labels.parse_mask(fh.read())
            print(f"Labels   : {len(label_map)} wells mapped from "
                  f"{os.path.basename(args.labels)}")
        except Exception as exc:
            print(f"WARNING: could not read the label file ({exc}); "
                  "falling back to well ids.")
            label_map = {}

    # ---- where results go ----
    # With --output, everything is collected under one folder named after the
    # data's location, so results from different slides/conditions never get
    # confused once they are copied somewhere together. The name walks a few
    # levels up the source path, e.g.
    #   .../Fefum release studies 17072026/Slide I/pre_glucose
    #   -> "Fefum release studies 17072026_Slide I_pre_glucose_analysis"
    output_root = None
    if args.output:
        parts = []
        p = os.path.abspath(parent)
        for _ in range(3):
            p, tail = os.path.split(p)
            if not tail:
                break
            parts.insert(0, tail)
        stem = "_".join(parts) + "_analysis"
        # strip characters Windows will not accept in a folder name
        for ch in '<>:"/\\|?*':
            stem = stem.replace(ch, "-")
        output_root = os.path.join(os.path.abspath(args.output), stem)
        os.makedirs(output_root, exist_ok=True)
        print(f"Results  : {output_root}")

    # ---- which sample subfolders ----
    available = sorted([d for d in os.listdir(parent)
                        if os.path.isdir(os.path.join(parent, d))
                        and d != "batch_output"])
    if not available:
        raise SystemExit(f"ERROR: no subfolders in {parent}")

    sel = args.samples
    if not sel:
        print(f"\nSample folders in {parent_name}: {', '.join(available)}")
        sel = input('Which to run? Comma list (e.g. A1,A2,C2,D1) or "all": ').strip()

    if sel.lower() == "all":
        samples = available
    else:
        samples = [s.strip() for s in sel.split(",") if s.strip()]
        missing = [s for s in samples if s not in available]
        if missing:
            raise SystemExit(f"ERROR: not found in {parent_name}: {', '.join(missing)}")

    # ---- locate Fiji and the macro (only needed for the Fiji backend) ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    macro_path = os.path.join(script_dir, MACRO_NAME)
    fiji_exe = None
    if USE_FIJI:
        fiji_exe = find_fiji_executable(FIJI_DIR)
        if not os.path.isfile(macro_path):
            raise SystemExit(
                f"ERROR: macro not found: {macro_path}\n"
                f"Put {MACRO_NAME} in the same folder as this script."
            )
        print(f"\nFiji  : {fiji_exe}")
        print(f"Macro : {macro_path}")
    else:
        print("\nBackend: numpy (cargo_measure.py) - Fiji not required")
    print(f"Parent: {parent}")
    print(f"Samples ({len(samples)}): {', '.join(samples)}\n")

    # ---- run record ----
    # Written before any work, so a crashed or interrupted run still leaves a
    # record of what it was attempting. The classification rule has changed
    # several times; without this a summary CSV cannot be traced to the rule
    # that produced it.
    manifest = None
    try:
        import cargo_version
        current = {k: globals().get(k) for k in cargo_version.TRACKED_SETTINGS}
        manifest = cargo_version.build_manifest(
            current, script_dir, parent, samples, label_map)
        cargo_version.add_package_versions(manifest)
        print(f"Version  : analysis {manifest['analysis_version']}  "
              f"settings {manifest['settings_hash']}")
    except Exception as exc:
        print(f"(run record unavailable: {exc})")

    model_holder = {"model": None}   # lazy-loaded once, reused across samples
    ok, failed = [], []
    base_names = {}                  # sample -> the name its files were written under

    for idx, sample in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] {sample}")
        t0 = time.time()
        folder = os.path.join(parent, sample)
        if label_map.get(sample):
            import cargo_labels
            base = cargo_labels.build_name(sample, label_map[sample], parent_name)
        else:
            base = f"{parent_name}_{sample}"
        base_names[sample] = base

        # ---- find the channel pair ----
        bf_path, fluor_path = find_channels(folder)
        if not bf_path or not fluor_path:
            print(f"    SKIP: could not identify a brightfield/fluorescence pair")
            failed.append(sample)
            continue

        # Fingerprint the inputs only AFTER they have been identified. This
        # block used to sit above find_channels(), so the first sample raised a
        # NameError (silently swallowed - no fingerprint recorded) and every
        # later sample was recorded with the PREVIOUS sample's files.
        if manifest is not None:
            try:
                import cargo_version
                manifest["inputs"][sample] = {
                    "brightfield": cargo_version.file_fingerprint(bf_path),
                    "fluorescence": cargo_version.file_fingerprint(fluor_path),
                }
            except Exception:
                pass
        print(f"    brightfield  : {os.path.basename(bf_path)}")
        print(f"    fluorescence : {os.path.basename(fluor_path)}")
        if os.path.abspath(bf_path) == os.path.abspath(fluor_path):
            print("    SKIP: brightfield and fluorescence resolved to the same file")
            failed.append(sample)
            continue

        # ---- mask: reuse if present, otherwise segment ----
        # Masks are written as 16-bit TIFF (ImageJ reads these reliably).
        # An existing *_cp_masks.png from an earlier run or from the Cellpose
        # GUI is still accepted, so previous work is not thrown away.
        mask_path = os.path.join(folder, f"{base}_cp_masks.tif")
        existing_mask = None
        for cand in (mask_path,
                     os.path.join(folder, f"{base}_cp_masks.png")):
            if os.path.isfile(cand):
                existing_mask = cand
                break

        if existing_mask and not args.force:
            print(f"    mask exists -> skipping Cellpose ({os.path.basename(existing_mask)})")
            mask_path = existing_mask
            # A mask from an earlier version has no bounding-box table. Build it
            # now from the saved mask so the macro never has to scan pixels.
            bbox_path = os.path.splitext(mask_path)[0] + "_bbox.csv"
            if not os.path.isfile(bbox_path):
                try:
                    import tifffile
                    import numpy as np
                    from PIL import Image
                    if mask_path.lower().endswith((".tif", ".tiff")):
                        m = tifffile.imread(mask_path)
                    else:
                        m = np.array(Image.open(mask_path))
                    write_bbox_table(m, mask_path)
                    print(f"    built bounding-box table ({int(m.max())} labels)")
                except Exception as exc:
                    print(f"    WARNING: could not build bbox table: {exc}")
        else:
            try:
                bf_img = load_brightfield(bf_path)
            except Exception as exc:
                print(f"    ERROR reading brightfield: {exc}")
                failed.append(sample)
                continue
            print(f"    running Cellpose ({CELLPOSE_MODEL})...")
            try:
                n = run_cellpose(bf_img, mask_path, model_holder)
                print(f"    Cellpose found {n} cells -> {os.path.basename(mask_path)}")
                if n == 0:
                    print("    WARNING: 0 cells - analysis will produce empty results.")
            except Exception as exc:
                print(f"    ERROR in Cellpose: {exc}")
                failed.append(sample)
                continue

        # ---- analysis ----
        if output_root:
            out_dir = os.path.join(output_root, sample)
        else:
            out_dir = os.path.join(folder, "edited_results")
        os.makedirs(out_dir, exist_ok=True)

        cutoff = scaled_cutoff(fluor_path)
        if os.path.abspath(fluor_path) == os.path.abspath(mask_path):
            print("    SKIP: fluorescence path equals the mask path - "
                  "the analysis would measure the mask against itself")
            failed.append(sample)
            continue

        if USE_FIJI:
            print("    running ImageJ analysis...")
            try:
                success = run_imagej(fiji_exe, macro_path, fluor_path, mask_path,
                                     out_dir, base, cutoff)
            except subprocess.TimeoutExpired:
                print("    ERROR: Fiji timed out (30 min)")
                failed.append(sample)
                continue
        else:
            # Default path: measure in numpy. Same maths, no macro interpreter.
            print("    measuring (numpy)...")
            try:
                import cargo_measure
                res = cargo_measure.run_sample(
                    fluor_path, mask_path, out_dir, base,
                    well=sample,
                    label=label_map.get(sample, ""),
                    condition=parent_name,
                    version=(manifest or {}).get("analysis_version", ""),
                    settings_hash=(manifest or {}).get("settings_hash", ""),
                    cfg={
                        "method": CLASSIFY_METHOD,
                        "brightness_threshold": BRIGHTNESS_THRESHOLD,
                        "core_cv_cap": CORE_CV_CAP,
                        "saturation_guard": SATURATION_GUARD,
                        "auto_threshold": AUTO_THRESHOLD,
                        "min_cell_mean": MIN_CELL_MEAN,
                        "core_radius_fraction": CORE_RADIUS_FRACTION,
                        "fill_ratio_threshold": FILL_RATIO_THRESHOLD,
                        "core_cv_threshold": CORE_CV_THRESHOLD,
                        "very_high_signal_cutoff": cutoff,
                        "rolling_ball_radius": ROLLING_BALL_RADIUS,
                    })
                success = res is not None
            except Exception as exc:
                import traceback
                print(f"    ERROR in measurement: {exc}")
                traceback.print_exc()
                success = False

        if success:
            ok.append(sample)
            print(f"    done in {time.time() - t0:.0f}s -> {out_dir}")
        else:
            failed.append(sample)
            print("    ERROR: analysis failed (see Fiji output above)")
        print()

    # ---- batch summary ----
    print("=" * 60)
    print(f"Completed: {len(ok)}/{len(samples)}")
    if failed:
        print(f"Failed   : {', '.join(failed)}")

    if manifest is not None:
        try:
            import cargo_version
            manifest["data"]["samples_completed"] = ok
            manifest["data"]["samples_failed"] = failed
            dest = output_root if output_root else parent
            jp, tp = cargo_version.write_manifest(manifest, dest)
            print(f"Run record -> {os.path.basename(tp)}")
        except Exception as exc:
            print(f"(could not write the run record: {exc})")

    if ok:
        batch_path = collect_batch_summary(parent, ok, parent_name, output_root,
                                           base_names)
        if batch_path:
            print(f"Batch summary -> {batch_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()