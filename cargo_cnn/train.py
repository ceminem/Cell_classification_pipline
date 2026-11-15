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

MIN_EPOCH_SAMPLES = 1024     # cells drawn per epoch, at least (16 batches of 64)

def balanced_bce(p, y):
    """Cross-entropy averaged per class, then over the two classes, so the
    rare class counts as much as the common one. Smoother than balanced
    accuracy, which on a small stopping set jumps by whole cells."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    parts = [loss[y == c].mean() for c in (0, 1) if (y == c).any()]
    return float(np.mean(parts)) if parts else float("nan")


def split_for_stopping(train_ex, seed=0, fraction=0.15):
    """
    Set aside part of the TRAINING cells to decide when to stop - never test
    cells, which would let the test choose the model and inflate its score.

    About 15% of the cells of EVERY training field, separately for released
    and not released, so every condition and both classes stay in training
    and are represented in the stopping set.

    (A first version set aside a whole training field. On real data, where
    each field is a different condition, that removed a whole condition from
    training - one round trained on only 242 of 441 cells - and the stopping
    signal was erratic: chosen epochs 9, 51, 17, 17.)

    Returns (train_part, stop_part, description).
    """
    rng = np.random.default_rng(seed)
    stop_idx = set()
    for g in sorted({e["group"] for e in train_ex}):
        for c in (0, 1):
            idx = [i for i, e in enumerate(train_ex) if e["group"] == g and e["y"] == c]
            if len(idx) < 4:
                continue                    # too few to spare any
            rng.shuffle(idx)
            stop_idx.update(idx[:max(1, int(round(fraction * len(idx))))])
    return ([e for i, e in enumerate(train_ex) if i not in stop_idx],
            [e for i, e in enumerate(train_ex) if i in stop_idx],
            f"{int(fraction * 100)}% of the cells of every training field")


def fit_brightness_threshold(train_ex):
    """
    The simplest possible classifier, as a yardstick: one brightness
    threshold (MeanWhole >= t -> released), chosen on the TRAINING cells to
    maximise balanced accuracy, then applied unchanged to the test cells.
    It is fitted exactly like the CNN - on the same cells - so the
    comparison is fair, and it needs no results files.
    """
    v = np.array([e["mean_whole"] for e in train_ex], float)
    y = np.array([e["y"] for e in train_ex], int)
    best = (-1.0, float(np.median(v)) if len(v) else 25.7)
    for t in np.unique(v):
        b = scores(v >= t, y)["bal_acc"]
        if np.isfinite(b) and b > best[0]:
            best = (b, float(t))
    return best[1]


def make_folds(ex, design, seed=0):
    """
    Two ways to ask "how good is it on cells it never saw":

    fields  leave one FIELD out. Each round, one whole image is the test.
            When every field is a different condition (as here), this asks
            "does it work on a CONDITION it has never seen?" - the harder
            question, relevant if it will be used on new conditions.
    halves  split EVERY field into a left and a right half at the median
            cell position; train on all left halves, test on all right halves,
            then the reverse. Every condition is seen in training, the test
            cells are not. This is the normal use: new images/wells of the
            conditions that were labelled.

    Returns a list of (name, test_indices).
    """
    groups = sorted({e["group"] for e in ex})
    if design == "halves":
        med = {g: float(np.median([e["cx"] for e in ex if e["group"] == g])) for g in groups}
        left = [i for i, e in enumerate(ex) if e["cx"] < med[e["group"]]]
        right = [i for i, e in enumerate(ex) if e["cx"] >= med[e["group"]]]
        return [("left halves of all fields", left), ("right halves of all fields", right)]
    if len(groups) >= 3:
        return [(g, [i for i, e in enumerate(ex) if e["group"] == g]) for g in groups]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ex))
    return [(f"random fold {k + 1}", list(idx[k::5])) for k in range(5)]


def train_model(train_ex, args, device, log, seed=0, show_ex=None, stop_ex=None,
                epochs=None):
    """
    Train one network.

    stop_ex  EARLY STOPPING set (from the training fields). Checked after
             every epoch; the weights with the lowest balanced loss on it are
             kept, and training ends when it has not improved for
             args.patience epochs. Returns the epoch at which it was best.
    show_ex  the TEST field. Only printed, never used for any decision.
    epochs   overrides args.epochs (the final model uses the typical
             stopping epoch found in cross-validation).

    Returns (model, best_epoch).
    """
    import copy
    import torch
    from torch.utils.data import DataLoader
    from cargo_cnn.model import CellNet

    n_epochs = int(epochs or args.epochs)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CellNet(width=args.width, dropout=args.dropout).to(device)

    y = np.array([e["y"] for e in train_ex])
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    # BALANCED BATCHES. With e.g. 84% "released" labels, a network shown the
    # cells in their natural proportion learns that "always released" already
    # scores 84% and stops there - tested: it then caught 0 of 34
    # not-released cells. So cells are drawn with probability 1/(size of their
    # class): every batch is about half and half, and the rare class is seen
    # many times, each time differently augmented (rotated, re-exposed, new
    # background), which is what makes repeating it useful rather than
    # memorising.
    from torch.utils.data import WeightedRandomSampler
    w = np.where(y == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    # An "epoch" is at least MIN_EPOCH_SAMPLES draws. With a few hundred
    # labels a plain epoch is only 3-4 batches, so 60 epochs were ~200 weight
    # updates - too few: tested, the network was still answering "released"
    # for every cell. Fixing the amount of training per epoch makes the
    # epoch setting mean the same thing however many cells are labelled.
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=max(len(y), MIN_EPOCH_SAMPLES),
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    lossf = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    ds = D.CellDataset(train_ex, train=True, seed=seed,
                       exposure=(1 / args.exposure_aug, args.exposure_aug))
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=0,
                    drop_last=True)

    # Which epoch is "best" on the stopping set: highest BALANCED ACCURACY
    # (the number the report judges by); within half a point, the lower
    # balanced loss wins. Loss alone was tried first and kept picking epoch 2:
    # an early, uncertain network has a low loss because it is never
    # confidently wrong, while it is not yet good at the task. The first
    # WARMUP epochs are never chosen for the same reason.
    WARMUP = 5
    best_bal, best_loss, best_ep, best_state, bad = -1.0, float("inf"), n_epochs, None, 0
    y_stop = [e["y"] for e in stop_ex] if stop_ex else None
    every = max(1, n_epochs // 6)

    for ep in range(n_epochs):
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

        improved = False
        stop_msg = ""
        if stop_ex:
            ps = predict(model, stop_ex, device, tta=False)
            vl = balanced_bce(ps, y_stop)
            vb = scores(ps >= 0.5, y_stop)["bal_acc"]
            if ep + 1 >= WARMUP:
                better = (vb > best_bal + 0.005 or
                          (abs(vb - best_bal) <= 0.005 and vl < best_loss - 1e-4))
                if better:
                    best_bal, best_loss, best_ep, bad, improved = vb, vl, ep + 1, 0, True
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
            stop_msg = (f"   stop-set bal. {vb * 100:.1f}%"
                        f" loss {vl:.3f}{' *' if improved else ''}")

        last = ep == n_epochs - 1
        stopping = bool(stop_ex) and bad >= args.patience and ep + 1 >= args.min_epochs
        if (ep + 1) % every == 0 or last or stopping:
            msg = f"      epoch {ep + 1:>3}/{n_epochs}  train loss {tot / max(nb, 1):.3f}{stop_msg}"
            if show_ex:
                p = predict(model, show_ex, device, tta=False)
                yy = [e["y"] for e in show_ex]
                msg += (f"   [test field bal. {scores(p >= 0.5, yy)['bal_acc'] * 100:.1f}%"
                        " - shown only]")
            log(msg)
        if stopping:
            log(f"      early stop: no improvement on the stopping set for "
                f"{args.patience} epochs; using the weights from epoch {best_ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_ep if stop_ex else n_epochs)


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


def groups_hint(ex):
    return {e["group"] for e in ex}


# ───────────────────────────────────────────────────────────── main ──

def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the cell classifier")
    ap.add_argument("--data", nargs="*", default=[],
                    help="folder(s) to search for *_curated.csv")
    ap.add_argument("--pairs", help="semicolon CSV: curated;fluorescence;mask;results")
    ap.add_argument("--results", nargs="*", default=[],
                    help="extra folder(s) holding *_results.csv, if the pipeline "
                         "wrote them to a separate 'Save results to' folder")
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
    ap.add_argument("--cv", choices=["both", "halves", "fields"], default="both",
                    help="test design: halves = same conditions, unseen cells; "
                         "fields = leave one field (condition) out; both (default)")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="train every fold for the full number of epochs")
    ap.add_argument("--patience", type=int, default=15,
                    help="early stopping: epochs without improvement before stopping")
    ap.add_argument("--min-epochs", type=int, default=10,
                    help="early stopping: never stop before this epoch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from cargo_cnn.model import (CellNet, n_parameters, pick_device,
                                 ensure_working_device)

    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    device = ensure_working_device(pick_device(args.device), log, train=True)
    log(f"cargo_cnn {CNN_VERSION}   torch {torch.__version__}   device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    sets = []
    if args.pairs:
        sets += D.load_pairs_file(args.pairs)
    if args.data:
        sets += D.discover(args.data, log, results_roots=args.results)
    if not sets:
        raise SystemExit("No labelled fields found. Give --data <folder> or --pairs <file>.")

    log(f"\nLabelled fields: {len(sets)}")
    t0 = time.time()
    ex = D.build_examples(sets, args.rolling_ball, args.include_nonblind, log,
                          min_cell_mean=args.min_cell_mean)
    if len(ex) < 30:
        raise SystemExit(
            f"Only {len(ex)} labelled cells were found in {len(groups_hint(ex))} "
            "field(s) - too few to train on.\n"
            "If you added a single SAMPLE folder (e.g. ...\\B2), add the folder "
            "that CONTAINS all\nlabelled samples instead (e.g. ...\\20261114); "
            "all its subfolders are searched.")
    y_all = np.array([e["y"] for e in ex])
    groups = sorted({e["group"] for e in ex})
    log(f"\n{len(ex)} cells ({int(y_all.sum())} released, "
        f"{int(len(y_all) - y_all.sum())} not released) from {len(groups)} fields "
        f"- cut out in {time.time() - t0:.0f}s")
    n_pos = int(y_all.sum())
    n_neg = int(len(y_all) - n_pos)
    minority = min(n_pos, n_neg) / len(y_all)
    if minority < 0.25:
        rare = "not released" if n_neg < n_pos else "released"
        log(f"\nNOTE: only {min(n_pos, n_neg)} of {len(y_all)} labelled cells "
            f"({minority * 100:.0f}%) are '{rare}'.\n"
            f"  A model that called EVERY cell '{'released' if rare == 'not released' else 'not released'}' "
            f"would already reach {(1 - minority) * 100:.0f}% accuracy, so\n"
            "  plain accuracy flatters everything. The verdict below uses BALANCED "
            "accuracy\n  (mean of the hit rates in both classes; guessing = 50%).\n"
            f"  The network also learns '{rare}' from few examples - labelling more "
            f"'{rare}'\n  cells (blind) is the most useful thing to do next.")
    log(f"network: {n_parameters(CellNet(width=args.width)):,} parameters")

    # ---- cross-validation ----
    def pct(sc, k):
        return (sc[k] * 100) if (sc and np.isfinite(sc.get(k, np.nan))) else float("nan")

    cv_rows = []
    best_epochs = []
    meta_cv = []
    designs = {"both": ["halves", "fields"], "halves": ["halves"],
               "fields": ["fields"]}[args.cv]
    if not args.no_cv:
        for design in designs:
            folds = make_folds(ex, design, args.seed)
            if design == "halves":
                log("\nTEST A - SAME CONDITIONS, UNSEEN CELLS: every field split into a "
                    "left and a right half;\n  train on one half of every field, test on "
                    "the other half, then swap.\n  This is the normal use: new images of "
                    "conditions that were labelled.")
            elif len(groups) >= 3:
                log(f"\nTEST B - UNSEEN CONDITION: leave one field out ({len(folds)} rounds)."
                    "\n  Each field is a different condition here, so this asks whether it "
                    "works on a\n  condition it has NEVER seen - the harder question.")
            else:
                log("\nTEST B: only 1-2 labelled fields, so whole fields cannot be held out;"
                    "\n  using 5 random folds of cells instead (optimistic).")
            rows_d = []
            for fi, (name, test_idx) in enumerate(folds):
                ts = set(test_idx)
                test = [ex[i] for i in test_idx]
                train = [e for i, e in enumerate(ex) if i not in ts]
                if len(train) < 20 or not test:
                    continue
                log(f"  [{fi + 1}/{len(folds)}] test = {name} ({len(test)} cells), "
                    f"train = {len(train)} cells")
                stop = None
                if not args.no_early_stop:
                    train, stop, how = split_for_stopping(train, seed=args.seed + fi)
                    log(f"      early-stopping set: {how} ({len(stop)} cells); "
                        f"training on {len(train)}")
                thr = fit_brightness_threshold(train + (stop or []))
                m, best_ep = train_model(train, args, device, log, seed=args.seed + fi,
                                         show_ex=test, stop_ex=stop)
                best_epochs.append(best_ep)
                p = predict(m, test, device, tta=True)
                log(f"      brightness yardstick for this round: threshold {thr:.1f} "
                    "(learned on the same training cells)")
                for e, pi in zip(test, p):
                    rows_d.append({"design": design, "field": e["group"],
                                   "CellID": e["cell_id"], "label": e["y"],
                                   "p_cnn": round(float(pi), 4), "cnn": int(pi >= 0.5),
                                   "bright": int(e["mean_whole"] >= thr),
                                   "rule": "" if e["rule"] is None else e["rule"],
                                   "MeanWhole": round(e["mean_whole"], 3)})
            if not rows_d:
                continue
            cv_rows += rows_d

            # ---- summary for this design ----
            log(f"\n  RESULT, {'TEST A (unseen cells)' if design == 'halves' else 'TEST B (unseen condition)'}"
                "  - balanced accuracy (mean of the hit rates for both classes)")
            log(f"  {'field':<28} {'n':>5} {'CNN':>7} {'bright.':>8} {'rule':>7}")
            for g in sorted({r["field"] for r in rows_d}):
                rr = [r for r in rows_d if r["field"] == g]
                yy = [r["label"] for r in rr]
                sc = scores([r["cnn"] for r in rr], yy)
                sb = scores([r["bright"] for r in rr], yy)
                ru = [r for r in rr if r["rule"] != ""]
                sr = scores([r["rule"] for r in ru], [r["label"] for r in ru]) if ru else None
                log(f"  {g[-28:]:<28} {len(rr):>5} {pct(sc, 'bal_acc'):>6.1f}% "
                    f"{pct(sb, 'bal_acc'):>7.1f}% {pct(sr, 'bal_acc'):>6.1f}%")
                meta_cv.append({"design": design, "field": g, "n": len(rr),
                                "cnn_bal": sc["bal_acc"], "cnn_acc": sc["acc"],
                                "bright_bal": sb["bal_acc"],
                                "rule_bal": sr["bal_acc"] if sr else None,
                                "rule_acc": sr["acc"] if sr else None})
            yy = np.array([r["label"] for r in rows_d])
            pp = np.array([r["p_cnn"] for r in rows_d])
            sc = scores(pp >= 0.5, yy)
            sb = scores([r["bright"] for r in rows_d], yy)
            log(f"\n  pooled  CNN        : {sc['bal_acc'] * 100:5.1f}%   AUC {auc(pp, yy):.3f}"
                f"   not-released caught {sc['tn']}/{sc['tn'] + sc['fp']}, released "
                f"{sc['tp']}/{sc['tp'] + sc['fn']}")
            log(f"          brightness : {sb['bal_acc'] * 100:5.1f}%"
                f"            not-released caught {sb['tn']}/{sb['tn'] + sb['fp']}, released "
                f"{sb['tp']}/{sb['tp'] + sb['fn']}")
            ru = [r for r in rows_d if r["rule"] != ""]
            if ru:
                sr = scores([r["rule"] for r in ru], [r["label"] for r in ru])
                scs = scores([r["cnn"] for r in ru], [r["label"] for r in ru])
                log(f"          your rule  : {sr['bal_acc'] * 100:5.1f}%   (CNN on the same "
                    f"{len(ru)} cells: {scs['bal_acc'] * 100:.1f}%)")
                ref, ref_name, cnn_ref = sr["bal_acc"], "your rule", scs["bal_acc"]
            else:
                ref, ref_name, cnn_ref = sb["bal_acc"], "a brightness threshold", sc["bal_acc"]
            gain = (cnn_ref - ref) * 100
            what = "on unseen cells" if design == "halves" else "on an unseen condition"
            if gain >= 2.0:
                log(f"  VERDICT {what}: CNN better than {ref_name} by {gain:.1f} points.")
            elif gain > -2.0:
                log(f"  VERDICT {what}: CNN within {abs(gain):.1f} points of {ref_name} - "
                    "no real difference.")
            else:
                log(f"  VERDICT {what}: CNN {-gain:.1f} points WORSE than {ref_name}.")
        if not any(r["rule"] != "" for r in cv_rows):
            log("\n  (no *_results.csv found, so your pipeline rule itself was not "
                "compared - set\n   'Results folder' for that; the brightness "
                "yardstick is always computed)")

    # ---- final model ----
    final_epochs = args.epochs
    if best_epochs and not args.no_early_stop:
        final_epochs = max(args.min_epochs, int(round(float(np.median(best_epochs)))))
        log(f"\nEarly stopping chose epochs {best_epochs} in the folds -> the final "
            f"model trains for {final_epochs} (the median).")
    log(f"\nFINAL MODEL: training on all {len(ex)} cells for {final_epochs} epochs")
    model, _ = train_model(ex, args, device, log, seed=args.seed, epochs=final_epochs)

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
        "early_stopping": {"fold_best_epochs": best_epochs,
                           "final_epochs": final_epochs,
                           "enabled": not args.no_early_stop},
        "trained_on": [{"field": g,
                        "n": sum(1 for e in ex if e["group"] == g),
                        "released": sum(e["y"] for e in ex if e["group"] == g)}
                       for g in groups],
        "cv": meta_cv,
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
