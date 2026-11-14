# -*- coding: utf-8 -*-
"""
cargo_cnn.predict - classify every cell of new samples with a trained model
=============================================================================
USAGE (repository folder, cellpose/rocm312 environment active)

    python -m cargo_cnn.predict --model models\\cnn_20261114_101500\\model.pt ^
                                --parent "D:\\...\\pre_glucose" --samples all

  Run the normal pipeline FIRST: this uses the Cellpose mask it saved
  (<sample>/<base>_cp_masks.tif) and, when present, its *_results.csv so the
  CNN's call and the rule's call sit side by side.

OUTPUT, per sample, next to the pipeline's own results:
    <base>_cnn.csv                one row per cell: P_released, CNN class,
                                  rule class, agreement
    <base>_cnn_summary.csv        same columns as *_summary.csv, Method=cnn
    <base>_cnn_overlay.png        green = released, red = not, grey = no signal
and at parent level
    <parent>_cnn_batch_summary.csv  - loads straight into the Figures tab

The SettingsHash column holds the MODEL's fingerprint, so the Figures tab
warns if CNN and rule results are mixed in one chart.

Cells are excluded exactly as in the pipeline: whole-cell mean below
min_cell_mean = no uptake, not part of the release percentage. The CNN is
only asked about cells that took something up.
=============================================================================
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cargo_cnn import data as D                           # noqa: E402


def load_model(path, device):
    import torch
    from cargo_cnn.model import CellNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    a = meta["arch"]
    model = CellNet(in_ch=a["in_ch"], width=a["width"], dropout=a["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    # a model trained with different crop settings must be fed the same way
    D.CROP = meta["input"]["crop"]
    D.CONTEXT = meta["input"]["context"]
    D.INTENSITY_SCALE = meta["input"]["intensity_scale"]
    D.MIN_BG_FRACTION = meta["input"]["min_bg_fraction"]
    with open(path, "rb") as fh:
        model_id = hashlib.sha256(fh.read()).hexdigest()[:10]
    return model, meta, model_id


def classify_sample(model, meta, device, fluor_path, mask_path, threshold=0.5,
                    tta=True, log=print):
    from cargo_cnn.train import predict
    raw, fl, labels, table = D.prepare_image(fluor_path, mask_path,
                                             meta["rolling_ball_radius"])
    rows, ex = [], []
    for lb in sorted(table):
        t = table[lb]
        ys, xs = t["slice"]
        r = {"CellID": lb, "Area": t["area"], "MeanWhole": t["mean"],
             "Bx": xs.start, "By": ys.start, "Bw": xs.stop - xs.start,
             "Bh": ys.stop - ys.start, "P_released": float("nan")}
        if t["mean"] < meta["min_cell_mean"]:
            r["Class"] = "excluded_negative"
        elif t["area"] < 20:
            r["Class"] = "excluded_too_small"
        else:
            img, own, other = D.extract_crop(fl, labels, lb, t["slice"])
            if not own.any():
                r["Class"] = "excluded_too_small"
            else:
                ex.append({"img": img, "own": own, "other": other})
                r["_i"] = len(ex) - 1
        rows.append(r)
    if ex:
        p = predict(model, ex, device, tta=tta)
        for r in rows:
            if "_i" in r:
                r["P_released"] = float(p[r.pop("_i")])
                r["Class"] = "released" if r["P_released"] >= threshold else "not_released"
    log(f"    {len(rows)} cells, {len(ex)} with signal classified")
    return raw, labels, rows


def _find_results(out_dir, base):
    p = os.path.join(out_dir, base + "_results.csv")
    if os.path.isfile(p):
        return p
    c = glob.glob(os.path.join(out_dir, "*_results.csv"))
    return c[0] if len(c) == 1 else None


def write_outputs(rows, raw, labels, out_dir, base, well, condition, model_id,
                  rule_rows):
    import cargo_measure
    os.makedirs(out_dir, exist_ok=True)

    agree_n = agree_k = 0
    with open(os.path.join(out_dir, base + "_cnn.csv"), "w", encoding="utf-8") as fh:
        fh.write("CellID;Area;MeanWhole;P_released;Class;RuleClass;Agree;Bx;By;Bw;Bh\n")
        for r in rows:
            rc = str(rule_rows.get(r["CellID"], {}).get("Class", ""))
            agree = ""
            if rc in ("released", "released_highsignal", "not_released") and \
                    r["Class"] in ("released", "not_released"):
                same = rc.startswith("released") == (r["Class"] == "released")
                agree = "1" if same else "0"
                agree_n += 1
                agree_k += int(same)
            p = r["P_released"]
            fh.write(f"{r['CellID']};{r['Area']};{r['MeanWhole']:.4f};"
                     f"{'NaN' if not np.isfinite(p) else f'{p:.4f}'};{r['Class']};"
                     f"{rc};{agree};{r['Bx']};{r['By']};{r['Bw']};{r['Bh']}\n")

    total = len(rows)
    rel = sum(r["Class"] == "released" for r in rows)
    notr = sum(r["Class"] == "not_released" for r in rows)
    neg = sum(r["Class"] == "excluded_negative" for r in rows)
    small = sum(r["Class"] == "excluded_too_small" for r in rows)
    sig = total - neg
    pct_rel = 100.0 * rel / (rel + notr) if rel + notr else 0.0
    pct_abs = 100.0 * rel / total if total else 0.0
    pct_up = 100.0 * sig / total if total else 0.0
    agree_pct = f"{100.0 * agree_k / agree_n:.2f}" if agree_n else ""
    hdr = ("Sample;Well;Label;Condition;TotalCellsInImage;CellsWithSignal;"
           "PercentUptake;Released;NotReleased;Excluded_NoSignal;Excluded_TooSmall;"
           "HighSignalSubset;PercentReleased_relative;PercentReleased_absolute;"
           "Method;Threshold;AnalysisVersion;SettingsHash;AgreementWithRule")
    line = (f"{base}_cnn;{well};;{condition};{total};{sig};{pct_up:.4f};{rel};{notr};"
            f"{neg};{small};0;{pct_rel:.4f};{pct_abs:.4f};cnn;0.5;"
            f"cnn-{model_id};{model_id};{agree_pct}")
    spath = os.path.join(out_dir, base + "_cnn_summary.csv")
    with open(spath, "w", encoding="utf-8") as fh:
        fh.write(hdr + "\n" + line + "\n")

    try:
        cargo_measure.write_overlay(raw, labels, rows,
                                    os.path.join(out_dir, base + "_cnn_overlay.png"))
    except Exception as exc:
        print(f"    (overlay failed: {exc})")
    return spath, rel, notr, pct_rel, agree_pct


def main(argv=None):
    ap = argparse.ArgumentParser(description="Classify cells with a trained CNN")
    ap.add_argument("--model", required=True, help="path to model.pt")
    ap.add_argument("--parent", required=True, help="folder containing the sample folders")
    ap.add_argument("--samples", default="all", help='"all" or e.g. "A1,A2,B1"')
    ap.add_argument("--output", help="the same --output folder used for the pipeline run, if any")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(released) at or above this = released")
    ap.add_argument("--no-tta", action="store_true",
                    help="skip averaging over the 8 rotations/mirrors (8x faster)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    import run_cargo_pipeline as rcp
    from cargo_cnn.model import pick_device, ensure_working_device

    device = ensure_working_device(pick_device(args.device), print, train=False)
    model, meta, model_id = load_model(args.model, device)
    print(f"model {model_id}  (trained {meta['created']} on "
          f"{sum(t['n'] for t in meta['trained_on'])} cells)  device {device}")

    parent = os.path.abspath(args.parent.strip().strip('"'))
    pname = os.path.basename(parent.rstrip("\\/"))
    subs = sorted(d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)))
    samples = subs if args.samples.lower() == "all" else \
        [s.strip() for s in args.samples.split(",") if s.strip()]

    out_root = None
    if args.output:
        # mirror run_cargo_pipeline's folder naming so results land together
        parts, p = [], parent
        for _ in range(3):
            p, tail = os.path.split(p)
            if not tail:
                break
            parts.insert(0, tail)
        stem = "_".join(parts) + "_analysis"
        for ch in '<>:"/\\|?*':
            stem = stem.replace(ch, "-")
        out_root = os.path.join(os.path.abspath(args.output), stem)

    summaries = []
    for s in samples:
        folder = os.path.join(parent, s)
        print(f"[{s}]")
        _bf, fl = rcp.find_channels(folder) if os.path.isdir(folder) else (None, None)
        masks = [m for m in glob.glob(os.path.join(folder, "*_cp_masks.*"))
                 if m.lower().endswith((".tif", ".tiff", ".png"))]
        masks.sort(key=lambda m: not m.lower().endswith((".tif", ".tiff")))
        if not fl or not masks:
            print("    SKIP: no fluorescence image or no Cellpose mask - run the pipeline first")
            continue
        mask = masks[0]
        base = os.path.basename(mask).rsplit("_cp_masks", 1)[0]
        out_dir = os.path.join(out_root, s) if out_root else os.path.join(folder, "edited_results")
        rule_rows = D.read_results(_find_results(out_dir, base))
        raw, labels, rows = classify_sample(model, meta, device, fl, mask,
                                            args.threshold, not args.no_tta)
        sp, rel, notr, pct, agree = write_outputs(
            rows, raw, labels, out_dir, base, s, pname, model_id, rule_rows)
        summaries.append(sp)
        print(f"    released {rel}, not released {notr}  ->  {pct:.1f}%"
              + (f"   (agrees with rule on {agree}% of cells)" if agree else ""))

    if summaries:
        dest = out_root or parent
        bpath = os.path.join(dest, f"{pname}_cnn_batch_summary.csv")
        lines = []
        for sp in summaries:
            with open(sp, encoding="utf-8") as fh:
                ls = [l.rstrip("\n") for l in fh if l.strip()]
            if not lines:
                lines.append(ls[0])
            lines.append(ls[1])
        with open(bpath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nBatch summary -> {bpath}")


if __name__ == "__main__":
    main()
