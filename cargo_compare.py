#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_compare.py - which classification rule best matches your labels?
=============================================================================
Run after re-analysing samples you have curated. It joins the automatic
measurements to your blind labels and reports how well each candidate rule
reproduces your judgement.

WHY THIS EXISTS
  Three new features were added (nuclear/cytoplasmic ratio, puncta counts,
  radial profile) on the reasoning that brightness and CV cannot separate
  ~9% of cells - 50 pairs inside the same image had nearly identical values
  but opposite labels. Whether the new features actually fix that is an
  empirical question, and this answers it on YOUR data instead of on the
  synthetic cells used to check the maths.

  Only BLIND labels are used. Labels made with the automatic colours visible
  tend to agree with the software, which would flatter whichever rule was
  active when they were made.

USAGE
  python cargo_compare.py <folder with *_results.csv and *_curated.csv>

  It searches recursively, so pointing it at a whole analysis folder works.
=============================================================================
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np


def load_pairs(root):
    """Join every *_curated.csv to the matching *_results.csv by CellID."""
    curated = {}
    results = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if fn.endswith("_curated.csv"):
                key = fn.replace("_curated.csv", "")
                curated[key] = p
            elif fn.endswith("_results.csv"):
                key = fn.replace("_results.csv", "")
                results[key] = p

    # curated files are named by well, results by the full label name, so match
    # on any shared well token rather than requiring identical stems
    pairs = []
    for ckey, cpath in curated.items():
        rpath = results.get(ckey)
        if rpath is None:
            tok = [t for t in ckey.replace("-", "_").split("_") if t]
            for rkey, rp in results.items():
                if any(t and t in rkey for t in tok if len(t) <= 3):
                    rpath = rp
                    break
        if rpath:
            pairs.append((ckey, cpath, rpath))
    return pairs


def read_joined(cpath, rpath):
    feats = {}
    with open(rpath, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            try:
                cid = int(r["CellID"])
            except (ValueError, KeyError):
                continue
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            feats[cid] = row

    out = []
    with open(cpath, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r.get("Source") != "manual_blind":
                continue
            if r.get("Label") not in ("released", "not_released"):
                continue
            try:
                cid = int(r["CellID"])
            except (ValueError, KeyError):
                continue
            f = feats.get(cid)
            if f is None:
                continue
            out.append((f, 1 if r["Label"] == "released" else 0))
    return out


def auc(vals, y):
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan")
    v = np.where(ok, v, np.nanmedian(v[ok]))
    o = np.argsort(v)
    rk = np.empty(len(v))
    rk[o] = np.arange(len(v))
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (rk[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def best_threshold(vals, y, direction="above"):
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return 0.0, 0.0
    v = v[ok]
    yy = y[ok]
    best = (0.0, 0.0)
    for t in np.unique(v):
        pred = (v >= t) if direction == "above" else (v <= t)
        a = (pred.astype(int) == yy).mean()
        if a > best[0]:
            best = (a, float(t))
    return best


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    pairs = load_pairs(root)
    if not pairs:
        print(f"No matching *_curated.csv / *_results.csv found under {root}")
        print("Re-run the analysis on samples you have curated first.")
        return

    all_rows = []
    print(f"{'sample':<28} {'blind labels':>13}")
    for key, cp, rp in sorted(pairs):
        rows = read_joined(cp, rp)
        print(f"{key[:28]:<28} {len(rows):>13}")
        all_rows.extend(rows)

    if len(all_rows) < 20:
        print("\nToo few blind labels to compare rules (need at least 20).")
        return

    y = np.array([r[1] for r in all_rows])
    print(f"\n{len(all_rows)} blind labels: {int(y.sum())} released, "
          f"{int(len(y) - y.sum())} not released\n")

    features = ["MeanWhole", "CoreCV", "CoreFillRatio", "NucCytoRatio",
                "PunctaCount", "PunctaArea", "PunctaMeanSize",
                "RadialSlope", "CentreEdgeRatio", "SatFraction"]

    print(f"{'feature':<20} {'AUC':>7} {'best single-threshold acc':>27}")
    print("-" * 58)
    have_new = False
    for f in features:
        vals = [r[0].get(f, np.nan) for r in all_rows]
        if not any(np.isfinite(np.asarray(vals, float))):
            continue
        a = auc(vals, y)
        if f in ("NucCytoRatio", "PunctaCount", "RadialSlope"):
            have_new = True
        d = "above" if a >= 0.5 else "below"
        acc, t = best_threshold(vals, y, d)
        arrow = ">=" if d == "above" else "<="
        print(f"{f:<20} {a:>7.3f}   {acc*100:>6.1f}%  ({arrow}{t:.3f})")

    if not have_new:
        print("\nThe new features are absent from these results files.")
        print("Re-run the analysis so they are measured, then run this again.")
        return

    # combined rules
    print("\ncombined rules")
    print("-" * 58)

    def acc_of(pred):
        return (pred.astype(int) == y).mean()

    mw = np.array([r[0].get("MeanWhole", np.nan) for r in all_rows], float)
    cv = np.array([r[0].get("CoreCV", np.nan) for r in all_rows], float)
    ncr = np.array([r[0].get("NucCytoRatio", np.nan) for r in all_rows], float)
    pc = np.array([r[0].get("PunctaCount", np.nan) for r in all_rows], float)

    def clean(a):
        ok = np.isfinite(a)
        return np.where(ok, a, np.nanmedian(a[ok]) if ok.any() else 0.0)

    mw, cv, ncr, pc = clean(mw), clean(cv), clean(ncr), clean(pc)

    b = max((acc_of(mw >= t), t) for t in np.unique(np.round(mw, 1)))
    print(f"brightness only                 {b[0]*100:>6.1f}%")

    b2 = max((acc_of((mw >= t) & (cv <= c)), t, c)
             for t in np.unique(np.round(mw, 1))[::2]
             for c in np.unique(np.round(cv, 2))[::2])
    print(f"brightness + CV  (current)      {b2[0]*100:>6.1f}%")

    b3 = max((acc_of(ncr >= t), t) for t in np.unique(np.round(ncr, 3)))
    print(f"nuclear ratio only              {b3[0]*100:>6.1f}%  (>={b3[1]:.3f})")

    b4 = max((acc_of((mw >= t) & (ncr >= n)), t, n)
             for t in np.unique(np.round(mw, 1))[::2]
             for n in np.unique(np.round(ncr, 2))[::2])
    print(f"brightness + nuclear ratio      {b4[0]*100:>6.1f}%")

    b5 = max((acc_of((mw >= t) & (ncr >= n) & (cv <= c)), t, n, c)
             for t in np.unique(np.round(mw, 1))[::3]
             for n in np.unique(np.round(ncr, 2))[::3]
             for c in np.unique(np.round(cv, 2))[::3])
    print(f"brightness + nuclear + CV       {b5[0]*100:>6.1f}%")

    print("\nRECOMMENDATION")
    scores = [("brightness + CV (current)", b2[0]),
              ("nuclear ratio only", b3[0]),
              ("brightness + nuclear", b4[0]),
              ("brightness + nuclear + CV", b5[0])]
    scores.sort(key=lambda s: -s[1])
    top = scores[0]
    print(f"  best here: {top[0]} at {top[1]*100:.1f}%")
    gain = (top[1] - b2[0]) * 100
    if gain < 1.5:
        print("  That is within noise of the current rule. The extra features do")
        print("  NOT solve the ambiguous cells, so the ceiling is label")
        print("  agreement rather than the measurement. Keep the simpler rule.")
    else:
        print(f"  That is {gain:.1f} points above the current rule - worth")
        print("  switching. Validate on a sample NOT used here before relying")
        print("  on it, or the gain may just be fitting to these images.")


if __name__ == "__main__":
    main()
