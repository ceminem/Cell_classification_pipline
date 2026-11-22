#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_plot.py - figure and statistics page
=============================================================================
Turns batch_summary.csv files into a publication bar chart with significance
testing, tunable the way a graphing tool is: colours, labels, error bars,
which comparisons to test, and how to display them.

WHAT IT PLOTS
  One bar per GROUP. A group is a set of samples you assign to a condition
  (e.g. "pre glucose" = A1,A2,B1,B2 from one summary; "after glucose" = the
  same wells from the other). The bar height is the mean of the per-sample
  release percentages, with individual sample points overlaid - so the reader
  sees the actual spread, not just a bar.

STATISTICS - PLEASE READ
  The test compares GROUPS OF SAMPLES, not groups of cells. This matters:
  each image contains ~2000 cells, so treating cells as independent replicates
  would give absurdly small p-values that reflect the cell count, not the
  experiment. The unit of replication here is the image (or better, the
  biological replicate).

  With 2-4 images per condition, n is small. A t-test on n=2 is not
  meaningful, and the tool says so rather than printing a p-value that looks
  authoritative. Paired tests are offered when the same wells are measured
  before and after treatment, which is the stronger design.

  Significance stars are a convention, not evidence. The tool always shows
  the actual p-value and n alongside them.
=============================================================================
"""

from __future__ import annotations

import csv
import itertools
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# 12 colours each; the first four are the original ones, so earlier figures
# keep their colours. Every bar gets its own colour (see bar_colors).
PALETTES = {
    "Grey / Blue":   ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
                      "#937860", "#da8bc3", "#ccb974", "#64b5cd", "#2f4b7c", "#b07aa1"],
    "Colourblind":   ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#d55e00", "#56b4e9",
                      "#ca9161", "#fbafe4", "#949494", "#ece133", "#004d80", "#8c510a"],
    "Warm":          ["#c44e52", "#dd8452", "#937860", "#8c8c8c", "#e3a33b", "#a0522d",
                      "#d9776b", "#b5651d", "#f2c14e", "#7f3b3b", "#e88d67", "#6e4b3a"],
    "Mono":          ["#333333", "#777777", "#aaaaaa", "#cccccc", "#555555", "#999999",
                      "#bbbbbb", "#444444", "#888888", "#dddddd", "#666666", "#222222"],
    "Nature-ish":    ["#3b6ea5", "#c1524e", "#e0a458", "#5c8a5c", "#8e6c8a", "#4aa3a2",
                      "#d17a22", "#7a7a7a", "#a9c1d9", "#b5838d", "#2e4057", "#9bb35a"],
}


def _mix(hexcol, other, t):
    a = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def bar_colors(palette, n):
    """n colours from a palette, all different: the palette's own colours
    first, then lighter, then darker versions of them for more bars."""
    base = PALETTES.get(palette, PALETTES["Grey / Blue"])
    shades = [("#ffffff", 0.45), ("#000000", 0.35), ("#ffffff", 0.7), ("#000000", 0.6)]
    out = list(base)
    k = 0
    while len(out) < n:
        target, t = shades[(k // len(base)) % len(shades)]
        col = _mix(base[k % len(base)], target, t)
        j = 1
        while col in out and j < 20:                     # e.g. greys in "Mono"
            col = _mix(base[k % len(base)], target, min(0.9, t + 0.04 * j))
            j += 1
        out.append(col)
        k += 1
    return out[:max(n, 0)]


VALUE_COLUMNS = [
    ("PercentReleased_relative", "% released (of cells with signal)"),
    ("PercentReleased_absolute", "% released (of all cells)"),
    ("PercentUptake", "% of cells with any uptake"),
    # cell count (cell_count_summary.csv, one row per image)
    ("n_counted", "Cells counted per image"),
    ("density_per_mm2", "Cells per mm² (counting frame)"),
    ("n_unbiased_frame", "Cells in the counting frame"),
    ("coverage_pct", "Area covered by cells (%)"),
    ("median_area_um2", "Median cell area (µm²)"),
    ("median_diameter_um", "Median cell diameter (µm)"),
]
# default y-axis label per value
VALUE_YLABEL = {
    "PercentReleased_relative": "% released", "PercentReleased_absolute": "% released",
    "PercentUptake": "% uptake", "n_counted": "Cells per image",
    "density_per_mm2": "Cells per mm²", "n_unbiased_frame": "Cells per field",
    "coverage_pct": "Area covered (%)", "median_area_um2": "Cell area (µm²)",
    "median_diameter_um": "Cell diameter (µm)",
}
COUNT_TEXT = ("sample", "image", "file", "edge_rule", "model")


def read_summaries(paths):
    """
    Read one or more *_batch_summary.csv (or single *_summary.csv) files.
    Returns a list of dicts, one per sample row, with numeric fields parsed.
    """
    rows = []
    for p in paths:
        try:
            with open(p, newline="", encoding="utf-8-sig") as fh:
                head = fh.readline()
                fh.seek(0)
                delim = ";" if head.count(";") >= head.count(",") else ","
                reader = csv.DictReader(fh, delimiter=delim)
                fields = [f.strip() for f in (reader.fieldnames or [])]
                if "n_counted" in fields and "image" in fields:
                    rows.extend(_count_rows(reader, p))
                    continue
                for r in reader:
                    if not r.get("Sample"):
                        continue
                    out = {"Sample": r["Sample"].strip(),
                           "SourceFile": os.path.basename(p),
                           # text columns are kept as text; the figure maker
                           # uses Label for axis and legend names so the reader
                           # sees the condition, not a well id
                           "Well": (r.get("Well") or "").strip(),
                           "Label": (r.get("Label") or "").strip(),
                           "Condition": (r.get("Condition") or "").strip(),
                           "AnalysisVersion": (r.get("AnalysisVersion") or "").strip(),
                           "SettingsHash": (r.get("SettingsHash") or "").strip()}
                    for k, v in r.items():
                        if k in ("Sample", "Method", "Well", "Label", "Condition",
                                 "AnalysisVersion", "SettingsHash"):
                            continue
                        try:
                            out[k] = float(str(v).replace(",", "."))
                        except (TypeError, ValueError):
                            out[k] = v
                    out["Method"] = r.get("Method", "")
                    rows.append(out)
        except OSError:
            continue
    return rows


def _count_rows(reader, path):
    """cell_count_summary.csv (Cell count tab): one row per image. The image
    is the sample, its folder ("sample" column) the label, so bars default to
    one per folder with the images as points."""
    out = []
    for r in reader:
        img = (r.get("image") or "").strip()
        if not img:
            continue
        folder = (r.get("sample") or "").strip()
        row = {"Sample": img, "SourceFile": os.path.basename(path), "Well": "",
               "Label": folder, "Condition": "", "AnalysisVersion": "",
               "SettingsHash": "", "Method": "cell count", "Kind": "count"}
        for k, v in r.items():
            if k is None or k in COUNT_TEXT:
                continue
            try:
                row[k] = float(str(v).replace(",", "."))
            except (TypeError, ValueError):
                row[k] = v
        out.append(row)
    return out


def available_values(rows):
    """VALUE_COLUMNS entries that at least one loaded row has a number for."""
    have = []
    for k, d in VALUE_COLUMNS:
        if any(isinstance(r.get(k), float) and np.isfinite(r.get(k)) for r in rows):
            have.append((k, d))
    return have


# ── statistics ─────────────────────────────────────────────────────────────

def welch_t(a, b):
    """Welch's t-test (unequal variances). Returns (t, df, p) or None."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return None
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, _t_p(abs(t), df)


def paired_t(a, b):
    """Paired t-test. Requires equal length, matched order."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) != len(b) or len(a) < 2:
        return None
    d = a - b
    sd = d.std(ddof=1)
    if sd <= 0:
        return None
    t = d.mean() / (sd / np.sqrt(len(d)))
    return t, len(d) - 1, _t_p(abs(t), len(d) - 1)


def mannwhitney(a, b):
    """
    Mann-Whitney U with a normal approximation.

    Reported for completeness, but with n of 2-4 per group a rank test has
    almost no power - the smallest achievable p-value for n=3 vs n=3 is 0.1.
    The tool says so rather than letting a non-significant result be read as
    evidence of no effect.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        return None
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    ra = ranks[:na].sum()
    u = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = np.sqrt(na * nb * (na + nb + 1) / 12)
    if sd <= 0:
        return None
    z = (u - mu) / sd
    return u, z, 2 * (1 - _norm_cdf(abs(z)))


def _norm_cdf(z):
    return 0.5 * (1 + _erf(z / np.sqrt(2)))


def _erf(x):
    # Abramowitz & Stegun 7.1.26
    s = np.sign(x)
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _t_p(t, df):
    """Two-sided p for Student's t, via a continued-fraction incomplete beta."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betainc(df / 2, 0.5, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1 / d
        c = 1 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1 - c * d) < 1e-10:
            break
    return front * (f - 1)


def _lgamma(x):
    g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
         -176.61502916214059, 12.507343278686905, -0.13857109526572012,
         9.9843695780195716e-6, 1.5056327351493116e-7]
    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _lgamma(1 - x)
    x -= 1
    a = 0.99999999999980993
    t = x + 7.5
    for i, gi in enumerate(g):
        a += gi / (x + i + 1)
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def run_test(a, b, kind="welch"):
    """Returns (label, p, note). note carries the honest caveat."""
    note = ""
    n = min(len(a), len(b))
    if n < 2:
        return "not testable", float("nan"), (
            f"n={len(a)} vs {len(b)}: a test needs at least 2 samples per group.")
    if n < 3:
        note = (f"n={len(a)} vs {len(b)}: with two samples per group a p-value "
                "is essentially uninformative. Treat this as descriptive.")
    if kind == "wilcoxon":
        r = wilcoxon(a, b)
        if r is None:
            return "Wilcoxon (failed)", float("nan"), (
                "Wilcoxon needs equal numbers of matched samples, n>=3.")
        s, p = r
        return f"Wilcoxon W={s:.1f}", p, note
    if kind == "paired":
        r = paired_t(a, b)
        if r is None:
            return "paired t (failed)", float("nan"), (
                "Paired test needs equal numbers of matched samples.")
        t, df, p = r
        return f"paired t={t:.2f}, df={df}", p, note
    if kind == "mannwhitney":
        r = mannwhitney(a, b)
        if r is None:
            return "Mann-Whitney (failed)", float("nan"), note
        u, z, p = r
        if n <= 3:
            note += (" With this n a rank test cannot reach p<0.05 even for a "
                     "perfect separation.")
        return f"Mann-Whitney U={u:.1f}", p, note
    r = welch_t(a, b)
    if r is None:
        return "Welch t (failed)", float("nan"), note
    t, df, p = r
    return f"Welch t={t:.2f}, df={df:.1f}", p, note


# ── plotting ───────────────────────────────────────────────────────────────

def aggregate_replicates(samples, mapping, mode="mean"):
    """
    Collapse TECHNICAL replicates before testing.

    WHY THIS MATTERS
      Technical replicates - two wells of the same condition in the same
      experiment, or two fields from one well - are not independent
      observations of the biology. They share the same cell passage, the same
      treatment, the same day. Averaging them and treating the average as ONE
      data point is the correct handling; treating each as a separate n
      inflates the apparent sample size and shrinks p-values without any extra
      evidence being present. This is pseudoreplication, and it is one of the
      easiest ways to publish a result that does not replicate.

      mapping maps a sample name to its BIOLOGICAL replicate id. Samples
      sharing an id are averaged into a single value.

      If every sample has its own id, nothing is averaged and this is a no-op.

    Returns (values, n_biological, n_technical).
    """
    groups = {}
    for name, value in samples:
        key = mapping.get(name, name)
        groups.setdefault(key, []).append(value)
    if mode == "median":
        vals = [float(np.median(v)) for v in groups.values()]
    else:
        vals = [float(np.mean(v)) for v in groups.values()]
    return vals, len(groups), len(samples)


CHART_TYPES = ["bar", "box", "dot", "bar + dots", "violin", "line"]


# ── assumption checks and test recommendation ──────────────────────────────

def check_assumptions(groups_values):
    """
    Run the checks that decide WHICH test is appropriate, and report them.

    Two things are tested:
      Shapiro-Wilk  - is each group roughly normally distributed? A t-test and
                      ANOVA both assume it. With n < 3 the test cannot run, and
                      with n under about 8 it has almost no power - so a "pass"
                      at small n means "no evidence against", not "normal".
      Levene        - do the groups have similar variance? If not, Welch's
                      correction is needed rather than the classic test.

    Returning the reasoning, not just a verdict, matters: an automatically
    chosen test that the reader cannot inspect is worse than one they picked
    themselves.
    """
    out = {"normal": True, "equal_var": True, "notes": [], "n_min": 0}
    try:
        from scipy import stats as ss
    except ImportError:
        out["notes"].append("scipy not installed - assumption checks skipped.")
        return out

    vals = [np.asarray(v, float) for v in groups_values if len(v)]
    if not vals:
        return out
    out["n_min"] = min(len(v) for v in vals)

    for i, v in enumerate(vals):
        if len(v) < 3:
            out["notes"].append(
                f"group {i+1}: n={len(v)} is too small to test normality.")
            continue
        try:
            w, p = ss.shapiro(v)
            if p < 0.05:
                out["normal"] = False
                out["notes"].append(
                    f"group {i+1}: Shapiro-Wilk p={p:.3f} - not normal.")
        except Exception:
            pass

    if len(vals) >= 2 and all(len(v) >= 2 for v in vals):
        try:
            stat, p = ss.levene(*vals)
            if p < 0.05:
                out["equal_var"] = False
                out["notes"].append(
                    f"Levene p={p:.3f} - variances differ between groups.")
        except Exception:
            pass

    if out["n_min"] < 8:
        out["notes"].append(
            f"smallest group has n={out['n_min']}: normality tests have very "
            "little power here, so a clean result is weak reassurance.")
    return out


def recommend_test(groups_values, paired=False):
    """Suggest a test from the data shape and the assumption checks."""
    a = check_assumptions(groups_values)
    k = len([v for v in groups_values if len(v)])
    if k < 2:
        return "none", "Need at least two groups.", a
    if k == 2:
        if paired:
            name = "paired" if a["normal"] else "wilcoxon"
            why = ("paired t-test" if a["normal"]
                   else "Wilcoxon signed-rank (data not normal)")
        else:
            name = "welch" if a["normal"] else "mannwhitney"
            why = ("Welch t-test (does not assume equal variance)"
                   if a["normal"] else "Mann-Whitney (data not normal)")
        return name, why, a
    name = "anova" if a["normal"] else "kruskal"
    why = ("one-way ANOVA" if a["normal"]
           else "Kruskal-Wallis (data not normal)")
    return name, why, a


def anova(groups_values):
    """One-way ANOVA across 3+ groups."""
    try:
        from scipy import stats as ss
        vals = [np.asarray(v, float) for v in groups_values if len(v) >= 2]
        if len(vals) < 3:
            return None
        f, p = ss.f_oneway(*vals)
        return f, p
    except Exception:
        return None


def kruskal(groups_values):
    try:
        from scipy import stats as ss
        vals = [np.asarray(v, float) for v in groups_values if len(v) >= 2]
        if len(vals) < 3:
            return None
        h, p = ss.kruskal(*vals)
        return h, p
    except Exception:
        return None


def wilcoxon(a, b):
    try:
        from scipy import stats as ss
        if len(a) != len(b) or len(a) < 3:
            return None
        s, p = ss.wilcoxon(a, b)
        return s, p
    except Exception:
        return None


def make_figure(groups, opts):
    """
    groups : list of (label, [values])   -- simple mode, one bar per entry
             OR list of (label, {series: [values]}) -- GROUPED mode, where each
             label is a cluster on the x-axis and each series is a bar within
             it. Grouped mode is what puts A1_pre, A1_after and A1_1h next to
             each other, then B1_pre, B1_after, B1_1h, and so on.
    opts   : display options
    Returns a matplotlib Figure.
    """
    fig = plt.figure(figsize=(opts.get("width", 6.0), opts.get("height", 4.5)),
                     dpi=opts.get("dpi", 110))
    ax = fig.add_subplot(111)

    grouped = bool(groups) and isinstance(groups[0][1], dict)
    pal = PALETTES.get(opts.get("palette", "Grey / Blue"), PALETTES["Grey / Blue"])
    err_kind = opts.get("error", "SD")

    def spread(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return 0.0
        return v.std(ddof=1) if err_kind == "SD" else v.std(ddof=1) / np.sqrt(len(v))

    rng = np.random.default_rng(0)

    if grouped:
        # ---- clustered bars ----
        series = []
        for _lab, d in groups:
            for s in d:
                if s not in series:
                    series.append(s)
        if opts.get("series_order"):
            wanted = [s for s in opts["series_order"] if s in series]
            series = wanted + [s for s in series if s not in wanted]

        n_clusters = len(groups)
        n_series = max(len(series), 1)
        cluster_w = opts.get("bar_width", 0.8)
        bw = cluster_w / n_series
        x = np.arange(n_clusters)

        # one bar per cluster (no series to tell apart): colour by cluster,
        # so every bar still gets its own colour
        per_cluster = n_series == 1
        ccols = [opts.get("colors", {}).get(g[0], c)
                 for g, c in zip(groups, bar_colors(opts.get("palette", "Grey / Blue"),
                                                    n_clusters))]
        scols = bar_colors(opts.get("palette", "Grey / Blue"), n_series)
        for si, s in enumerate(series):
            means, errs, offs = [], [], []
            for ci, (_lab, d) in enumerate(groups):
                v = np.asarray(d.get(s, []), float)
                means.append(v.mean() if len(v) else np.nan)
                errs.append(spread(v))
                offs.append(x[ci] - cluster_w / 2 + bw * (si + 0.5))
            ax.bar(offs, means,
                   yerr=errs if opts.get("show_error", True) else None,
                   width=bw * 0.92, label=s,
                   color=ccols if per_cluster else opts.get("colors", {}).get(s, scols[si]),
                   edgecolor=opts.get("edge", "#333333"),
                   linewidth=opts.get("linewidth", 1.0),
                   capsize=opts.get("capsize", 3), zorder=2)
            if opts.get("show_points", True):
                for ci, (_lab, d) in enumerate(groups):
                    v = np.asarray(d.get(s, []), float)
                    if not len(v):
                        continue
                    j = (rng.random(len(v)) - 0.5) * bw * 0.45
                    ax.scatter(np.full(len(v), offs[ci]) + j, v,
                               s=opts.get("point_size", 20),
                               color=opts.get("point_color", "#222222"),
                               zorder=3, edgecolor="white", linewidth=0.5)
        if opts.get("legend", True) and n_series > 1:
            ax.legend(frameon=False, fontsize=opts.get("legend_size", 9),
                      loc=opts.get("legend_loc", "best"))
        labels = [g[0] for g in groups]
    else:
        # ---- one entry per group; several chart types ----
        labels = [g[0] for g in groups]
        vals = [np.asarray(g[1], float) for g in groups]
        means = [v.mean() if len(v) else np.nan for v in vals]
        errs = [spread(v) for v in vals]
        x = np.arange(len(groups))
        auto = bar_colors(opts.get("palette", "Grey / Blue"), len(groups))
        colors = [opts.get("colors", {}).get(labels[i], auto[i])
                  for i in range(len(groups))]
        ctype = opts.get("chart_type", "bar")

        if ctype in ("bar", "bar + dots"):
            ax.bar(x, means, yerr=errs if opts.get("show_error", True) else None,
                   capsize=opts.get("capsize", 4), color=colors,
                   edgecolor=opts.get("edge", "#333333"),
                   linewidth=opts.get("linewidth", 1.0),
                   width=opts.get("bar_width", 0.62), zorder=2)
        elif ctype == "box":
            bp = ax.boxplot([v for v in vals], positions=x, widths=0.5,
                            patch_artist=True, zorder=2,
                            medianprops=dict(color="#222222", linewidth=1.4))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_edgecolor(opts.get("edge", "#333333"))
        elif ctype == "violin":
            good = [v for v in vals if len(v) >= 2]
            if good:
                vp = ax.violinplot([v for v in vals if len(v) >= 2],
                                   positions=[x[i] for i, v in enumerate(vals)
                                              if len(v) >= 2],
                                   showmeans=True, widths=0.6)
                for i, b in enumerate(vp["bodies"]):
                    b.set_facecolor(colors[i % len(colors)])
                    b.set_alpha(0.75)
        elif ctype == "line":
            ax.plot(x, means, marker="o", color=colors[0], linewidth=1.6,
                    zorder=2)
            if opts.get("show_error", True):
                ax.errorbar(x, means, yerr=errs, fmt="none",
                            ecolor="#333333", capsize=opts.get("capsize", 4),
                            zorder=2)
        # "dot" draws nothing here - the points below are the whole plot

        want_points = opts.get("show_points", True) or ctype in ("dot", "bar + dots")
        if want_points and ctype != "violin":
            for i, v in enumerate(vals):
                if not len(v):
                    continue
                j = (rng.random(len(v)) - 0.5) * opts.get("jitter", 0.16)
                ax.scatter(x[i] + j, v, s=opts.get("point_size", 26),
                           color=(colors[i] if ctype == "dot"
                                  else opts.get("point_color", "#222222")),
                           zorder=3, edgecolor="white", linewidth=0.6)
            if ctype == "dot":
                # a mean line makes a dot plot readable
                for i, v in enumerate(vals):
                    if len(v):
                        ax.plot([x[i] - 0.22, x[i] + 0.22], [v.mean()] * 2,
                                color="#222222", linewidth=1.8, zorder=4)

    # ---- axes ----
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=opts.get("xrot", 0),
                       ha="right" if opts.get("xrot", 0) else "center",
                       fontsize=opts.get("tick_size", 10))
    ax.set_ylabel(opts.get("ylabel", "% released"),
                  fontsize=opts.get("label_size", 11))
    if opts.get("xlabel"):
        ax.set_xlabel(opts["xlabel"], fontsize=opts.get("label_size", 11))
    if opts.get("title"):
        ax.set_title(opts["title"], fontsize=opts.get("title_size", 12))
    ax.tick_params(labelsize=opts.get("tick_size", 10))

    ymin = opts.get("ymin")
    ymax = opts.get("ymax")
    try:
        ymin = float(ymin) if ymin not in (None, "") else 0.0
    except (TypeError, ValueError):
        ymin = 0.0
    try:
        ymax = float(ymax) if ymax not in (None, "") else None
    except (TypeError, ValueError):
        ymax = None
    if ymax is None:
        allv = []
        if grouped:
            for _l, d in groups:
                for v in d.values():
                    allv.extend(list(v))
        else:
            for _l, v in groups:
                allv.extend(list(v))
        top = max(allv) if allv else 1.0
        ymax = top * 1.3
    ax.set_ylim(ymin, ymax)

    if opts.get("ystep"):
        try:
            step = float(opts["ystep"])
            if step > 0:
                ax.set_yticks(np.arange(ymin, ymax + step / 2, step))
        except (TypeError, ValueError):
            pass

    for side in ("top", "right"):
        ax.spines[side].set_visible(not opts.get("hide_spines", True))
    if opts.get("grid", True):
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    # ---- significance brackets (simple mode only) ----
    comps = opts.get("comparisons", []) if not grouped else []
    if comps and opts.get("test", "welch") != "none":
        vals = [np.asarray(g[1], float) for g in groups]
        means = [v.mean() if len(v) else 0 for v in vals]
        errs = [spread(v) for v in vals]
        span = ymax - ymin
        step = span * 0.075
        level = max([m + e for m, e in zip(means, errs)]
                    + [v.max() for v in vals if len(v)] + [ymin]) + step * 0.5
        for k, (i, j) in enumerate(comps):
            if i >= len(vals) or j >= len(vals):
                continue
            _label, p, _note = run_test(vals[i], vals[j], opts.get("test", "welch"))
            y = level + k * step
            ax.plot([i, i, j, j], [y, y + step * 0.18, y + step * 0.18, y],
                    color="#333333", linewidth=1.0)
            txt = stars(p) if opts.get("star_style", "stars") == "stars" else (
                f"p={p:.3g}" if np.isfinite(p) else "n/a")
            ax.text((i + j) / 2, y + step * 0.22, txt, ha="center", va="bottom",
                    fontsize=opts.get("star_size", 11))
        need = level + len(comps) * step + step
        if need > ymax:
            ax.set_ylim(ymin, need)

    fig.tight_layout()
    return fig


def stats_report(groups, opts):
    """Plain-text statistics, including the caveats."""
    lines = []
    grouped = bool(groups) and isinstance(groups[0][1], dict)

    lines.append("GROUPS")
    if grouped:
        for lab, d in groups:
            lines.append(f"  {lab}")
            for s, v in d.items():
                v = np.asarray(v, float)
                if len(v):
                    sd = v.std(ddof=1) if len(v) > 1 else 0.0
                    lines.append(f"      {s:<20} n={len(v)}  mean={v.mean():.2f}  "
                                 f"SD={sd:.2f}  "
                                 f"[{', '.join(f'{x:.1f}' for x in v)}]")
    else:
        for lab, v in groups:
            v = np.asarray(v, float)
            if len(v):
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                lines.append(f"  {lab:<22} n={len(v)}  mean={v.mean():.2f}  "
                             f"SD={sd:.2f}  "
                             f"[{', '.join(f'{x:.1f}' for x in v)}]")
            else:
                lines.append(f"  {lab:<22} n=0")

    hashes = opts.get("settings_hashes") or []
    uniq = sorted({h for h in hashes if h})
    if len(uniq) > 1:
        lines.append("")
        lines.append("*** MIXED ANALYSIS SETTINGS ***")
        lines.append(f"These samples were produced under {len(uniq)} different")
        lines.append("settings: " + ", ".join(uniq))
        lines.append("The classification rule has changed between versions, so")
        lines.append("these numbers are NOT directly comparable. Re-run them all")
        lines.append("with the same settings before drawing the figure.")

    lines.append("")
    lines.append("The unit of replication is the IMAGE, not the cell. Each image")
    lines.append("holds ~2000 cells; treating those as replicates would give")
    lines.append("meaninglessly small p-values.")

    rep = opts.get("replicate_type", "technical")
    if rep == "technical":
        lines.append("")
        lines.append("*** TECHNICAL REPLICATES ***")
        lines.append("Wells from a single experiment measure the precision of the")
        lines.append("assay, not the reproducibility of the biology. A p-value")
        lines.append("across them answers 'were these wells different?', not")
        lines.append("'does the treatment change cargo release?'. Report as")
        lines.append("descriptive (mean +/- SD, n wells) and repeat on separate")
        lines.append("days before claiming an effect.")
    elif rep == "biological":
        lines.append("")
        lines.append("Biological replicates: independent experiments. A test here")
        lines.append("does speak to whether the effect reproduces. Check that the")
        lines.append("replicates really are independent - different passages and")
        lines.append("different days, not two wells from one flask.")
    else:
        lines.append("")
        lines.append("Replicate type not specified, so no test is reported.")

    if grouped:
        lines.append("")
        lines.append("Grouped bars are shown descriptively. For a test, switch to")
        lines.append("simple grouping so it is unambiguous which two sets are")
        lines.append("being compared.")
        return "\n".join(lines)

    lines.append("")
    comps = opts.get("comparisons", [])
    test = opts.get("test", "welch")
    if test == "none" or not comps:
        lines.append("No test requested (descriptive only).")
        return "\n".join(lines)

    vals = [np.asarray(g[1], float) for g in groups]

    # assumption checks, so the reader can see WHY this test
    a = check_assumptions(vals)
    lines.append("ASSUMPTION CHECKS")
    lines.append(f"  normality  : {'ok' if a['normal'] else 'VIOLATED'}")
    lines.append(f"  equal var  : {'ok' if a['equal_var'] else 'VIOLATED'}")
    for n in a["notes"]:
        lines.append(f"  - {n}")
    lines.append("")

    if len(vals) >= 3 and test in ("anova", "kruskal"):
        r = anova(vals) if test == "anova" else kruskal(vals)
        if r:
            stat, p = r
            nm = "One-way ANOVA F" if test == "anova" else "Kruskal-Wallis H"
            lines.append(f"OVERALL  {nm}={stat:.3f}  p={p:.4g}  {stars(p)}")
            lines.append("  This says only that the groups are not all equal.")
            lines.append("  Pairwise tests below are NOT corrected for multiple")
            lines.append("  comparisons - with many pairs, some will look")
            lines.append("  significant by chance alone.")
            lines.append("")

    lines.append(f"PAIRWISE  ({test})")
    for i, j in comps:
        if i >= len(vals) or j >= len(vals):
            continue
        label, p, note = run_test(vals[i], vals[j], test)
        lines.append(f"  {groups[i][0]}  vs  {groups[j][0]}")
        lines.append(f"    {label}   p={p:.4g}  {stars(p)}")
        if note:
            lines.append(f"    NOTE: {note}")
    return "\n".join(lines)