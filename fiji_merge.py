#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fiji_merge.py - merge two images into a colour overlay via Fiji/ImageJ
=============================================================================
A drop-in helper: give it two image paths and it returns the path to a merged
RGB overlay, produced by Fiji so the blending matches what you'd get in the
GUI (Image > Color > Merge Channels).

WHY FIJI AND NOT PILLOW
  Pillow can add two arrays, but Fiji's Merge Channels handles bit depth,
  per-channel LUTs and display-range scaling the way ImageJ users expect, so
  an overlay made here looks the same as one made by hand in the program.
  The merge itself runs in a head-less Fiji called as a subprocess - there is
  no Python API for ImageJ, so we hand it a generated .ijm macro.

HOW IT WORKS
  merge_two(img_a, img_b, out_path, ...) writes a temporary macro that:
    1. opens both images
    2. assigns each to a channel colour (default: A -> green, B -> magenta,
       a colour-blind-safe pairing that still reads as a classic overlay)
    3. runs Merge Channels with "create" (composite) then flattens to RGB
    4. saves the result to out_path
  then invokes Fiji on that macro and returns out_path.

FINDING FIJI
  Pass fiji_path explicitly, or set the FIJI_PATH environment variable, or
  let it search the usual install locations. If it cannot be found a clear
  error is raised rather than a cryptic subprocess failure.
=============================================================================
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

# Channel colours Merge Channels understands, mapped to its c1..c7 slots.
# c1=red c2=green c3=blue c4=grey c5=cyan c6=magenta c7=yellow
_MERGE_SLOTS = {
    "red": "c1", "green": "c2", "blue": "c3", "grey": "c4", "gray": "c4",
    "cyan": "c5", "magenta": "c6", "yellow": "c7",
}

# Where Fiji tends to live, per OS. Checked in order after explicit path / env.
_CANDIDATE_FIJI = [
    # --- this machine (hard-coded) ---
    r"C:\Users\cemin\Documents\Molbio\Fiji\Fiji\fiji-windows-x64.exe",
    # Windows (older launcher name)
    r"C:\Fiji.app\ImageJ-win64.exe",
    r"C:\Program Files\Fiji.app\ImageJ-win64.exe",
    r"C:\Users\Public\Fiji.app\ImageJ-win64.exe",
    # macOS
    "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
    # Linux
    "/opt/Fiji.app/ImageJ-linux64",
    os.path.expanduser("~/Fiji.app/ImageJ-linux64"),
    os.path.expanduser("~/fiji/Fiji.app/ImageJ-linux64"),
]


def find_fiji(fiji_path=None):
    """
    Locate the Fiji executable.

    Order: explicit argument -> FIJI_PATH env var -> PATH -> known install
    locations. Returns the path, or raises FileNotFoundError with guidance.
    """
    candidates = []
    if fiji_path:
        candidates.append(fiji_path)
    env = os.environ.get("FIJI_PATH")
    if env:
        candidates.append(env)
    # anything named like Fiji/ImageJ already on PATH
    for exe in ("fiji-windows-x64.exe", "ImageJ-win64.exe", "ImageJ-linux64",
                "ImageJ-macosx", "fiji", "ImageJ"):
        found = shutil.which(exe)
        if found:
            candidates.append(found)
    candidates.extend(_CANDIDATE_FIJI)

    for c in candidates:
        if c and os.path.isfile(c):
            return c

    raise FileNotFoundError(
        "Could not find Fiji. Point to it with the fiji_path argument or the "
        "FIJI_PATH environment variable, e.g.\n"
        r"    set FIJI_PATH=C:\Fiji.app\ImageJ-win64.exe   (Windows)"
        "\n    export FIJI_PATH=/opt/Fiji.app/ImageJ-linux64   (Linux)")


def _ijm_escape(path):
    """Macro string literals use backslash escapes; make a Windows path safe."""
    return os.path.abspath(path).replace("\\", "\\\\")


def _build_macro(img_a, img_b, out_path, color_a, color_b, keep_composite):
    """
    Generate the .ijm macro text.

    Each image is opened and pushed into a Merge Channels slot by colour. The
    'create' flag makes a composite so the channels overlay rather than tile;
    'flatten' bakes it to RGB for a portable file that drops straight into the
    collage. If keep_composite is set the multi-channel composite is saved as
    TIFF instead, preserving the separate channels.
    """
    slot_a = _MERGE_SLOTS[color_a.lower()]
    slot_b = _MERGE_SLOTS[color_b.lower()]
    a = _ijm_escape(img_a)
    b = _ijm_escape(img_b)
    out = _ijm_escape(out_path)

    # title1/title2 let Merge Channels reference the images by window title.
    lines = [
        f'open("{a}");',
        'rename("A");',
        f'open("{b}");',
        'rename("B");',
        # both channels scaled to their display range, composite created
        f'run("Merge Channels...", "{slot_a}=A {slot_b}=B create");',
    ]
    if keep_composite:
        lines.append(f'saveAs("Tiff", "{out}");')
    else:
        lines += [
            'run("Flatten");',
            f'saveAs("PNG", "{out}");',
        ]
    lines.append('run("Quit");')
    return "\n".join(lines) + "\n"


def merge_two(img_a, img_b, out_path,
              color_a="green", color_b="magenta",
              fiji_path=None, keep_composite=False,
              timeout=180, headless=True):
    """
    Merge two images into a colour overlay using Fiji.

    img_a, img_b   : the two source image paths
    out_path       : where to write the result. Extension is normalised to
                     .png (overlay) or .tif (when keep_composite=True).
    color_a/color_b: channel colours - any of red, green, blue, grey, cyan,
                     magenta, yellow. Default green + magenta is colour-blind
                     safe and co-localisation shows as near-white.
    fiji_path      : explicit path to the Fiji executable (else auto-found).
    keep_composite : save a multi-channel TIFF composite instead of a flat RGB.
    timeout        : seconds before the Fiji call is abandoned.
    headless       : run Fiji without its GUI (recommended for batch use).

    Returns the path actually written.
    """
    for p, tag in ((img_a, "img_a"), (img_b, "img_b")):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{tag} does not exist: {p}")
    for c, tag in ((color_a, "color_a"), (color_b, "color_b")):
        if c.lower() not in _MERGE_SLOTS:
            raise ValueError(
                f"{tag}={c!r} is not a Merge Channels colour. "
                f"Use one of: {', '.join(sorted(set(_MERGE_SLOTS)))}.")
    if color_a.lower() == color_b.lower():
        raise ValueError("color_a and color_b must differ, or the two images "
                         "would land in the same channel.")

    exe = find_fiji(fiji_path)

    # normalise the output extension to match what the macro will actually save
    want_ext = ".tif" if keep_composite else ".png"
    if os.path.splitext(out_path)[1].lower() != want_ext:
        out_path = os.path.splitext(out_path)[0] + want_ext
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    macro = _build_macro(img_a, img_b, out_path, color_a, color_b, keep_composite)

    # Fiji runs a macro file passed after -macro; --headless suppresses the GUI.
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ijm", delete=False,
                                      encoding="utf-8")
    try:
        tmp.write(macro)
        tmp.close()

        cmd = [exe]
        if headless:
            cmd.append("--headless")
        # --console keeps output on stdout; -macro <file> runs then we quit.
        cmd += ["--console", "-macro", tmp.name]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Fiji did not finish within {timeout}s. Large images or a "
                "cold start can take a while - raise the timeout, or run once "
                "with headless=False to watch for a dialog waiting on input.")

        if not os.path.isfile(out_path):
            raise RuntimeError(
                "Fiji ran but produced no output file.\n"
                f"exit code: {proc.returncode}\n"
                f"stdout:\n{proc.stdout[-1500:]}\n"
                f"stderr:\n{proc.stderr[-1500:]}")
        return out_path
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Optional CLI:  python fiji_merge.py a.tif b.tif overlay.png [--composite]
# --------------------------------------------------------------------------
def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser(
        description="Merge two images into a colour overlay via Fiji.")
    ap.add_argument("image_a")
    ap.add_argument("image_b")
    ap.add_argument("out")
    ap.add_argument("--color-a", default="green")
    ap.add_argument("--color-b", default="magenta")
    ap.add_argument("--fiji", default=None, help="path to Fiji executable")
    ap.add_argument("--composite", action="store_true",
                    help="save a multi-channel TIFF instead of a flat overlay")
    ap.add_argument("--show", action="store_true",
                    help="run Fiji with its GUI (not headless)")
    a = ap.parse_args(argv)
    out = merge_two(a.image_a, a.image_b, a.out,
                    color_a=a.color_a, color_b=a.color_b,
                    fiji_path=a.fiji, keep_composite=a.composite,
                    headless=not a.show)
    print("wrote", out)


if __name__ == "__main__":
    _cli(sys.argv[1:])