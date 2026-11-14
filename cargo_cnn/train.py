# -*- coding: utf-8 -*-
"""
cargo_cnn.train - learn released / not released from your blind labels
=============================================================================
USAGE (from the repository folder, cellpose/rocm312 environment active)

    python -m cargo_cnn.train --data "D:\\...\\Slide I" "D:\\...\\Slide II"
    python -m cargo_cnn.train --pairs my_labelled_fields.csv

  --data searches the folders for *_curated.csv files written by the curation
  tool and pairs each with the fluorescence image and mask of the same sample.
  --pairs lists them explicitly (see data.load_pairs_file).

WHAT IT DOES, IN ORDER
  1. Cuts every blind-labelled cell out of its image (data.py).
  2. CROSS-VALIDATION, leave one image out: for each labelled field, train on
     all the OTHER fields and predict this one. The network never sees the
     field it is tested on - not a single cell of it.
     Why whole fields and not random cells: cells in one image share exposure,
     background and focus. If cells from the test image were also in training,
     the network could learn "this image is bright" instead of "this cell
     released", and the accuracy would look better than it will be on a new
     experiment.
  3. Compares, ON THE SAME HELD-OUT CELLS, the network against the rule the
     pipeline actually used (the Class column of *_results.csv). That
     comparison is the only number that says whether the CNN is worth using.
     Note the rule's thresholds were fitted to these same labels, so the rule
     is being given a slight home advantage.
  4. Trains a final model on ALL labelled cells and saves it, with a report.

OUTPUT   models/cnn_<date>_<time>/
    model.pt              weights + every setting needed to use them
    training_report.txt   the cross-validation result, readable
    cv_predictions.csv    every held-out cell: label, CNN probability, rule
                          call - sort by disagreement to see where they differ
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cargo_cnn import CNN_VERSION                         # noqa: E402
from cargo_cnn import data as D                           # noqa: E402


# ──────────────────────────────────────────────────────────── metrics ──

def auc(p, y):
    """Area under the ROC curve via the rank-sum (Mann-Whitney) identity:
    the probability a random released cell scores above a random
    not-released one. 0.5 = chance, 1.0 = perfect."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):                       # average ranks for ties
        m = p == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores(pred, y):
    pred = np.asarray(pred, int)
    y = np.asarray(y, int)
    if len(y) == 0:
        return {"n": 0, "acc": float("nan"), "bal_acc": float("nan")}
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"n": len(y), "acc": (tp + tn) / len(y),
            "bal_acc": np.nanmean([sens, spec]),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────────────────── training ──

def train_model(train_ex, args, device, log, seed=0, val_ex=None):
    import torch
    from torch.utils.data import DataLoader
    from cargo_cnn.model import CellNet

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CellNet(width=args.width, dropout=args.dropout).to(device)

    y = np.array([e["y"] for e in train_ex])
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    # pos_weight balances the loss when one class is rarer, so the network
    # cannot score well by always answering the majority class
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32,
                              device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ds = D.CellDataset(train_ex, train=True, seed=seed,
                       exposure=(1 / args.exposure_aug, args.exposure_aug))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=len(ds) > args.batch)

    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in dl:
            if len(yb) < 2:            # batch-norm needs at least 2 samples
                continue
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        if (ep + 1) % max(1, args.epochs // 6) == 0 or ep == args.epochs - 1:
            msg = f"      epoch {ep + 1:>3}/{args.epochs}  train loss {tot / max(nb, 1):.3f}"
            if val_ex:
                p = predict(model, val_ex, device, tta=False)
                yy = [e["y"] for e in val_ex]
                msg += f"   held-out acc {scores(p >= 0.5, yy)['acc'] * 100:.1f}%"
            log(msg)
    return model


def predict(model, examples, device, tta=True, batch=256):
    """P(released) for each example. With tta, the average over the 8
    rotations/mirrors of the crop - a cell does not change class when turned,
    so averaging removes orientation noise from the prediction."""
    import torch
    model.eval()
    xs = np.stack([D.make_channels(e["img"], e["own"], e["other"])
                   for e in examples]).astype(np.float32)
    variants = D.DIHEDRAL if tta else [(0, False)]
    out = np.zeros(len(xs))
    with torch.no_grad():
        for k, f in variants:
            v = D.dihedral(xs, k, f)
            for i in range(0, len(v), batch):
                xb = torch.from_numpy(v[i:i + batch]).to(device)
                out[i:i + batch] += torch.sigmoid(model(xb)).cpu().numpy()
    return out / len(variants)


# ───────────────────────────────────────────────────────────── main ──

def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the cell classifier")
    ap.add_argument("--data", nargs="*", default=[],
                    help="folder(s) to search for *_curated.csv")
    ap.add_argument("--pairs", help="semicolon CSV: curated;fluorescence;mask;results")
    ap.add_argument("--out", default=os.path.join(_ROOT, "models"),
                    help="where the model folder is created")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16, help="network width (16 = ~0.3 M params)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--exposure-aug", type=float, default=1.25,
                    help="random exposure factor range 1/x..x during training")
    ap.add_argument("--rolling-ball", type=int, default=50)
    ap.add_argument("--min-cell-mean", type=float, default=5.0)
    ap.add_argument("--include-nonblind", action="store_true",
                    help="also use labels made with the colours visible (not advised)")
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from cargo_cnn.model import CellNet, n_parameters, pick_device

    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    device = pick_device(args.device)
    log(f"cargo_cnn {CNN_VERSION}   torch {torch.__version__}   device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    sets = []
    if args.pairs:
        sets += D.load_pairs_file(args.pairs)
    if args.data:
        sets += D.discover(args.data, log)
    if not sets:
        raise SystemExit("No labelled fields found. Give --data <folder> or --pairs <file>.")

    log(f"\nLabelled fields: {len(sets)}")
    t0 = time.time()
    ex = D.build_examples(sets, args.rolling_ball, args.include_nonblind, log)
    if len(ex) < 30:
        raise SystemExit(f"Only {len(ex)} labelled cells - too few to train on.")
    y_all = np.array([e["y"] for e in ex])
    groups = sorted({e["group"] for e in ex})
    log(f"\n{len(ex)} cells ({int(y_all.sum())} released, "
        f"{int(len(y_all) - y_all.sum())} not released) from {len(groups)} fields "
        f"- cut out in {time.time() - t0:.0f}s")
    log(f"network: {n_parameters(CellNet(width=args.width)):,} parameters")

    # ---- cross-validation ----
    cv_rows = []
    fold_table = []
    if not args.no_cv:
        if len(groups) >= 3:
            folds = [[g] for g in groups]
            log(f"\nCROSS-VALIDATION: leave one field out ({len(folds)} folds)")
        else:
            log("\nCROSS-VALIDATION: only 1-2 labelled fields, so whole fields "
                "cannot be held out.\n  Falling back to 5 random folds of CELLS. "
                "This OVERESTIMATES accuracy on a new\n  experiment - label a "
                "third field before trusting the number.")
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(len(ex))
            folds = [list(idx[i::5]) for i in range(5)]
        for fi, fold in enumerate(folds):
            if len(groups) >= 3:
                test = [e for e in ex if e["group"] in fold]
                train = [e for e in ex if e["group"] not in fold]
                name = fold[0]
            else:
                fs = set(fold)
                test = [e for i, e in enumerate(ex) if i in fs]
                train = [e for i, e in enumerate(ex) if i not in fs]
                name = f"random fold {fi + 1}"
            if len(train) < 20 or not test:
                continue
            log(f"  [{fi + 1}/{len(folds)}] test = {name} ({len(test)} cells), "
                f"train = {len(train)} cells")
            m = train_model(train, args, device, log, seed=args.seed + fi, val_ex=test)
            p = predict(m, test, device, tta=True)
            yt = [e["y"] for e in test]
            s_cnn = scores(p >= 0.5, yt)
            ruled = [(e["rule"], e["y"]) for e in test if e["rule"] is not None]
            s_rule = scores([r for r, _ in ruled], [t for _, t in ruled]) if ruled else None
            fold_table.append((name, s_cnn, s_rule, auc(p, yt)))
            for e, pi in zip(test, p):
                cv_rows.append({"field": e["group"], "CellID": e["cell_id"],
                                "label": e["y"], "p_cnn": round(float(pi), 4),
                                "cnn": int(pi >= 0.5),
                                "rule": "" if e["rule"] is None else e["rule"],
                                "MeanWhole": round(e["mean_whole"], 3)})

        # ---- summary ----
        log("\nRESULT ON HELD-OUT CELLS")
        log(f"  {'field':<34} {'n':>5} {'CNN acc':>8} {'rule acc':>9} {'CNN AUC':>8}")
        for name, sc, sr, a in fold_table:
            log(f"  {name[:34]:<34} {sc['n']:>5} {sc['acc'] * 100:>7.1f}% "
                f"{(sr['acc'] * 100 if sr else float('nan')):>8.1f}% {a:>8.3f}")
        yy = np.array([r["label"] for r in cv_rows])
        pp = np.array([r["p_cnn"] for r in cv_rows])
        s_cnn = scores(pp >= 0.5, yy)
        both = [r for r in cv_rows if r["rule"] != ""]
        s_rule = scores([r["rule"] for r in both], [r["label"] for r in both]) if both else None
        s_cnn_same = scores([r["cnn"] for r in both], [r["label"] for r in both]) if both else None
        log(f"\n  pooled CNN : accuracy {s_cnn['acc'] * 100:.1f}%   balanced "
            f"{s_cnn['bal_acc'] * 100:.1f}%   AUC {auc(pp, yy):.3f}   (n={len(yy)})")
        if s_rule:
            log(f"  same cells : CNN {s_cnn_same['acc'] * 100:.1f}%  vs  rule "
                f"{s_rule['acc'] * 100:.1f}%   (n={len(both)} cells that have a rule call)")
            gain = (s_cnn_same["acc"] - s_rule["acc"]) * 100
            log("")
            if gain >= 2.0:
                log(f"  VERDICT: the CNN beats the rule by {gain:.1f} points on images "
                    "it never saw.\n  Worth using - confirm on a newly labelled "
                    "field before switching a thesis figure.")
            elif gain > -2.0:
                log(f"  VERDICT: within {abs(gain):.1f} points of the rule - no real "
                    "difference.\n  Keep the rule (simpler, explainable). This "
                    "matches the earlier finding that ~9% of\n  cells are "
                    "ambiguous even to the labeller: that is a ceiling no model "
                    "can pass.")
            else:
                log(f"  VERDICT: the CNN is {-gain:.1f} points WORSE than the rule. "
                    "Do not use it.\n  More labelled fields (different days) "
                    "usually help more than a bigger network.")
        else:
            log("  (no *_results.csv found next to the labels, so the rule could "
                "not be compared)")

    # ---- final model ----
    log(f"\nFINAL MODEL: training on all {len(ex)} cells")
    model = train_model(ex, args, device, log, seed=args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"cnn_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import cargo_version
        git = cargo_version.git_state(_ROOT)
    except Exception:
        git = {}
    meta = {
        "cnn_version": CNN_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "arch": {"in_ch": D.N_CHANNELS, "width": args.width, "dropout": args.dropout},
        "input": {"crop": D.CROP, "context": D.CONTEXT,
                  "intensity_scale": D.INTENSITY_SCALE,
                  "min_bg_fraction": D.MIN_BG_FRACTION},
        "rolling_ball_radius": args.rolling_ball,
        "min_cell_mean": args.min_cell_mean,
        "training": {k: v for k, v in vars(args).items()
                     if k not in ("data", "pairs", "out")},
        "trained_on": [{"field": g,
                        "n": sum(1 for e in ex if e["group"] == g),
                        "released": sum(e["y"] for e in ex if e["group"] == g)}
                       for g in groups],
        "cv": [{"field": n, "n": sc["n"], "cnn_acc": sc["acc"],
                "rule_acc": (sr["acc"] if sr else None), "auc": a}
               for n, sc, sr, a in fold_table],
        "git": git,
        "torch": torch.__version__,
    }
    torch.save({"state_dict": model.cpu().state_dict(), "meta": meta},
               os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "model_info.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    if cv_rows:
        with open(os.path.join(out_dir, "cv_predictions.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cv_rows[0]), delimiter=";")
            w.writeheader()
            w.writerows(cv_rows)
    log(f"\nSaved -> {out_dir}")
    with open(os.path.join(out_dir, "training_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_dir


if __name__ == "__main__":
    main()
