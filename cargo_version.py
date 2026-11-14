#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_version.py - version stamping and run provenance
=============================================================================
Records what produced each result, so a number in a thesis can be traced back
to the exact settings and code that generated it.

WHY THIS MATTERS HERE MORE THAN USUAL
  The classification rule in this pipeline has changed three times, each time
  against new hand labels:

      v1  fill-ratio + core CV        ~25% agreement with blind labels
      v2  brightness alone            ~80%
      v3  brightness AND core CV      ~88%

  Results produced under any of those are not comparable with each other. A
  summary CSV saying "62% released" is meaningless without knowing which rule
  and thresholds produced it - and six months from now nobody will remember.

  So every run writes a manifest: code version, every setting that affects the
  numbers, the input files with their sizes and modification times, and a hash
  of the settings that makes two runs trivially comparable.

WHAT THIS IS NOT
  It is not git. It does not version the source code itself - use git for that
  if you want history and branching. This records what a RESULT came from,
  which is the part that goes stale silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Bump ANALYSIS_VERSION whenever a change alters the numbers a run produces.
# The history below is the record of why results differ between versions.
# ---------------------------------------------------------------------------
ANALYSIS_VERSION = "3.1"

VERSION_HISTORY = [
    ("1.0", "Fill-ratio + core CV thresholds (ImageJ macro). Agreed with blind "
            "labels on ~25% of cells - not usable."),
    ("2.0", "Measurement moved from ImageJ macro to numpy. Same maths, ~80x "
            "faster, no macro interpreter to crash."),
    ("2.1", "Classification switched to whole-cell brightness after 195 hand "
            "labels showed fill-ratio and CV were at chance (AUC 0.45). "
            "Brightness AUC 0.92, ~87% agreement."),
    ("3.0", "Per-image Otsu threshold; threshold recorded in every summary. "
            "Saturation veto added for cells bright only from blob bleed."),
    ("3.1", "Core CV restored as a co-criterion after 536 BLIND labels showed "
            "it is strongly predictive (AUC 0.074, i.e. 0.926 inverted). "
            "brightness>=25.7 AND CV<=0.65 gives 88% vs 80% for brightness "
            "alone. Nuclear/puncta/radial features added but not yet used for "
            "classification."),
]

# Settings that change the numbers. Anything not in this list is cosmetic.
TRACKED_SETTINGS = [
    "CLASSIFY_METHOD", "BRIGHTNESS_THRESHOLD", "CORE_CV_CAP", "AUTO_THRESHOLD",
    "MIN_CELL_MEAN", "CORE_RADIUS_FRACTION", "ROLLING_BALL_RADIUS",
    "SATURATION_GUARD", "FILL_RATIO_THRESHOLD", "CORE_CV_THRESHOLD",
    "VERY_HIGH_SIGNAL_CUTOFF_8BIT",
    "CELLPOSE_MODEL", "FLOW_THRESHOLD", "CELLPROB_THRESHOLD",
    "CELLPOSE_SCALE_FACTOR",
]


def settings_hash(settings):
    """
    Short stable hash of the settings that affect results.

    Two runs with the same hash used the same rules; different hashes mean the
    numbers are not directly comparable. Printed in the console and stored in
    every manifest so the check is a glance rather than a diff.
    """
    items = sorted((k, str(settings.get(k))) for k in TRACKED_SETTINGS
                   if k in settings)
    blob = ";".join(f"{k}={v}" for k, v in items)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def file_fingerprint(path):
    """Size and modification time - enough to notice an input changed."""
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path),
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    except OSError:
        return {"path": str(path), "size_bytes": None, "modified": None}


def code_fingerprint(script_dir):
    """
    Hash of every module that can affect the numbers.

    If a colleague reports a different result from the same settings, comparing
    these tells you immediately whether they are running the same code.
    """
    out = {}
    for fn in ("cargo_measure.py", "run_cargo_pipeline.py", "cargo_labels.py"):
        p = os.path.join(script_dir, fn)
        try:
            with open(p, "rb") as fh:
                out[fn] = hashlib.sha256(fh.read()).hexdigest()[:10]
        except OSError:
            out[fn] = "missing"
    return out


def build_manifest(settings, script_dir, parent, samples,
                   label_map=None, inputs=None, notes=None):
    """Assemble everything needed to reproduce or audit a run."""
    return {
        "analysis_version": ANALYSIS_VERSION,
        "settings_hash": settings_hash(settings),
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "settings": {k: settings.get(k) for k in TRACKED_SETTINGS
                     if k in settings},
        "code": code_fingerprint(script_dir),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.node(),
        },
        "data": {
            "parent": os.path.abspath(parent) if parent else None,
            "samples": list(samples or []),
            "labels_used": bool(label_map),
            "label_map": dict(label_map) if label_map else {},
        },
        "inputs": inputs or {},
        "notes": notes or "",
    }


def add_package_versions(manifest):
    """Record the versions of libraries that can change results between machines."""
    vers = {}
    for name in ("numpy", "scipy", "cellpose", "torch", "tifffile", "PIL"):
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "?")
        except ImportError:
            vers[name] = "not installed"
    try:
        import torch
        vers["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "CPU only")
    except Exception:
        vers["gpu"] = "unknown"
    manifest["environment"]["packages"] = vers
    return manifest


def write_manifest(manifest, out_dir, stem="run_manifest"):
    """
    Write the manifest as JSON plus a readable text summary.

    Both, deliberately: the JSON is for comparing runs programmatically, the
    text is what you would actually paste into a methods section or read six
    months later.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(out_dir, f"{stem}_{stamp}.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    tpath = os.path.join(out_dir, f"{stem}_{stamp}.txt")
    with open(tpath, "w", encoding="utf-8") as fh:
        fh.write(as_text(manifest))
    return jpath, tpath


def as_text(m):
    """Readable summary, suitable for a methods section."""
    L = []
    L.append("CARGO RELEASE ANALYSIS - RUN RECORD")
    L.append("=" * 60)
    L.append(f"analysis version : {m.get('analysis_version')}")
    L.append(f"settings hash    : {m.get('settings_hash')}")
    L.append(f"run started      : {m.get('run_started')}")
    d = m.get("data", {})
    L.append(f"data folder      : {d.get('parent')}")
    L.append(f"samples          : {', '.join(d.get('samples', [])) or '(none)'}")
    L.append("")
    L.append("SETTINGS THAT AFFECT THE NUMBERS")
    L.append("-" * 60)
    for k, v in sorted((m.get("settings") or {}).items()):
        L.append(f"  {k:<30} {v}")
    L.append("")
    if d.get("labels_used"):
        L.append("SAMPLE LABELS")
        L.append("-" * 60)
        for w, lab in sorted((d.get("label_map") or {}).items()):
            L.append(f"  {w:<6} {lab}")
        L.append("")
    env = m.get("environment", {})
    L.append("ENVIRONMENT")
    L.append("-" * 60)
    L.append(f"  python   {env.get('python')}")
    L.append(f"  platform {env.get('platform')}")
    L.append(f"  machine  {env.get('machine')}")
    for k, v in sorted((env.get("packages") or {}).items()):
        L.append(f"  {k:<9} {v}")
    L.append("")
    L.append("CODE FINGERPRINTS")
    L.append("-" * 60)
    for k, v in sorted((m.get("code") or {}).items()):
        L.append(f"  {k:<28} {v}")
    if m.get("inputs"):
        L.append("")
        L.append("INPUT FILES")
        L.append("-" * 60)
        for sample, files in sorted(m["inputs"].items()):
            L.append(f"  {sample}")
            for role, info in sorted(files.items()):
                L.append(f"    {role:<14} {os.path.basename(str(info.get('path')))}"
                         f"  {info.get('size_bytes')} bytes"
                         f"  {info.get('modified')}")
    L.append("")
    L.append("HOW TO READ THIS")
    L.append("-" * 60)
    L.append("  Two runs are directly comparable only if the settings hash and")
    L.append("  the analysis version match. The classification rule has changed")
    L.append("  several times as hand labels accumulated:")
    for v, why in VERSION_HISTORY:
        L.append(f"    {v}  {why}")
    return "\n".join(L)


def compare_manifests(a_path, b_path):
    """Report what differs between two runs - the usual 'why do these differ?'."""
    try:
        a = json.load(open(a_path, encoding="utf-8"))
        b = json.load(open(b_path, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"Could not read: {exc}"

    L = []
    if a.get("settings_hash") == b.get("settings_hash"):
        L.append("Settings are IDENTICAL - results are directly comparable.")
    else:
        L.append("Settings DIFFER - results are not directly comparable.")
    if a.get("analysis_version") != b.get("analysis_version"):
        L.append(f"  analysis version: {a.get('analysis_version')} vs "
                 f"{b.get('analysis_version')}")
    sa, sb = a.get("settings", {}), b.get("settings", {})
    for k in sorted(set(sa) | set(sb)):
        if sa.get(k) != sb.get(k):
            L.append(f"  {k}: {sa.get(k)} -> {sb.get(k)}")
    ca, cb = a.get("code", {}), b.get("code", {})
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k) != cb.get(k):
            L.append(f"  code {k}: {ca.get(k)} -> {cb.get(k)}")
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(compare_manifests(sys.argv[1], sys.argv[2]))
    else:
        print(f"cargo analysis version {ANALYSIS_VERSION}")
        print()
        for v, why in VERSION_HISTORY:
            print(f"  {v}  {why}")
        print()
        print("Usage: python cargo_version.py <manifest_a.json> <manifest_b.json>")
        print("       to compare two runs.")
