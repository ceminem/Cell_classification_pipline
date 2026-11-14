#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_labels.py - map well positions to real sample names
=============================================================================
Paste a plate layout copied straight out of Excel and it becomes a mapping
from well id (A1, A2, B1 ...) to the actual condition, so output files are
named A1_Fe-fum-Histidin-5uL_pre_glucose rather than a bare A1.

THE LAYOUT IT EXPECTS
  Excel copies as tab-separated text. A layout like

      Position | Fe-fum              | Fe-fum - His        | Fe-fum + BSA
      1        | Fe-fum + His 5 uL   | Fe-fum - His 5 uL   | ...
      2        | Fe-fum + His 3 uL   | Fe-fum - His 3 uL   | ...

  is filled COLUMN-MAJOR: the first data column is well letter A, the second
  is B, and so on, while the Position rows give the number. So the first
  column's two cells are A1 (upper) and A2 (lower), the next column gives B1
  and B2, then C1/C2, D1/D2.

  This matches how the plate is laid out physically - one condition per
  column, replicate positions down the column.

  A row-major layout (conditions along a row) is also supported by switching
  the orientation, because the same table can be typed either way and getting
  it backwards would silently mislabel every sample.

WHY NAMES ARE SANITISED
  "Fe-fum@ BSA-FL + Histidin 5 uL" contains characters Windows forbids in a
  filename (@, +) or that break CSV parsing (;). The label kept for display
  is the original; the one used in filenames is cleaned. Both are recorded so
  a file can always be traced back to the condition it came from.
=============================================================================
"""

from __future__ import annotations

import re
import string

# Characters Windows will not accept in a file name, plus the CSV delimiter.
_BAD_FILENAME = '<>:"/\\|?*;'


def clean_for_filename(text, max_len=60):
    """
    Turn a condition label into something safe to put in a filename, while
    keeping it readable and - crucially - keeping it DISTINCT.

    "+" and "-" carry the meaning in these labels: "Fe-fum + Histidin" and
    "Fe-fum - Histidin" are opposite conditions. Simply stripping them made
    both collapse to "Fe-fum-Histidin", so the two would be indistinguishable
    in a filename and one could overwrite the other. They are therefore
    translated to "plus"/"minus" rather than removed.

    "Fe-fum@ BSA-FL + Histidin 5 uL" -> "Fe-fum-BSA-FL_plus_Histidin-5uL"
    """
    if text is None:
        return ""
    s = str(text).strip()

    # preserve the sign words before any other substitution
    s = re.sub(r"\s*\+\s*", " plus ", s)
    # a minus is only a sign when it stands between words, not inside "Fe-fum"
    s = re.sub(r"(?<=\s)-\s*", "minus ", s)
    s = s.replace("@", " ").replace("&", " and ")

    for ch in _BAD_FILENAME:
        s = s.replace(ch, "-")

    # keep unit suffixes compact: "5 uL" -> "5uL"
    s = re.sub(r"(\d)\s+([a-zA-Z]{1,3})\b", r"\1\2", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    return s[:max_len]


def parse_mask(text, orientation="column-major", letters=None):
    """
    Parse a pasted plate layout into {well_id: label}.

    text        : tab- or comma-separated table copied from Excel
    orientation : "column-major" - each data COLUMN is a well letter, each
                  data ROW is a position number (the layout in the
                  iron-fumarate mask: A1/A2 in the first column, B1/B2 in
                  the second, and so on)
                  "row-major"    - each data ROW is a letter instead
    letters     : override the well letters, e.g. ["A","B","C","D"]

    The first row is treated as a header when its first cell is a word like
    "Position" or is empty, and the first column is treated as position
    numbers when it is numeric. Both are detected rather than assumed, so a
    table pasted without a header still works.

    Returns (mapping, warnings).
    """
    warnings = []
    if not text or not text.strip():
        return {}, ["Nothing pasted."]

    rows = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        cells = line.split("\t") if "\t" in line else (
            line.split(";") if ";" in line else line.split(","))
        rows.append([c.strip() for c in cells])

    if not rows:
        return {}, ["Nothing to parse."]

    # a title line spanning the table (one non-empty cell) is dropped
    if len(rows) > 1 and sum(1 for c in rows[0] if c) <= 1:
        rows = rows[1:]

    # header row?
    header = None
    first = rows[0]
    if first and (not first[0] or not first[0][0].isdigit()):
        looks_like_header = any(c and not c[0].isdigit() for c in first[1:])
        if looks_like_header:
            header = first
            rows = rows[1:]

    if not rows:
        return {}, ["Only a header was found - no data rows."]

    # position column?
    has_pos_col = all(r and r[0].strip().rstrip(".").isdigit() for r in rows if r)
    positions = []
    data = []
    for r in rows:
        if has_pos_col:
            try:
                positions.append(int(r[0].strip().rstrip(".")))
            except ValueError:
                positions.append(len(positions) + 1)
            data.append(r[1:])
        else:
            positions.append(len(positions) + 1)
            data.append(r)

    width = max((len(r) for r in data), default=0)
    if width == 0:
        return {}, ["No data cells found."]
    data = [r + [""] * (width - len(r)) for r in data]

    if letters is None:
        letters = list(string.ascii_uppercase)

    mapping = {}
    if orientation == "column-major":
        # column index -> letter, row index -> position number
        for ci in range(width):
            letter = letters[ci] if ci < len(letters) else f"col{ci+1}"
            for ri, row in enumerate(data):
                label = row[ci].strip()
                if not label:
                    continue
                mapping[f"{letter}{positions[ri]}"] = label
        if header and len(header) > 1:
            heads = header[1:] if has_pos_col else header
            for ci, h in enumerate(heads[:width]):
                if h and ci < len(letters):
                    warnings.append(f"{letters[ci]} = {h}")
    else:
        # each data row is a letter
        for ri, row in enumerate(data):
            letter = letters[ri] if ri < len(letters) else f"row{ri+1}"
            for ci in range(width):
                label = row[ci].strip()
                if not label:
                    continue
                mapping[f"{letter}{ci+1}"] = label

    if not mapping:
        warnings.append("No labels were found in the table.")
    elif len(mapping) < 2:
        # A single cell is almost never a real plate layout - it is usually a
        # stray line or a table that lost its columns on paste. Rejecting it is
        # safer than mapping one well and silently leaving the rest unnamed.
        warnings.append("Only one well was found - that does not look like a "
                        "plate layout. Check the table pasted with its columns "
                        "intact.")
        mapping = {}
    return mapping, warnings


def build_name(well, label, suffix="", include_well=True):
    """
    Compose the output name for a sample.

    well    : "A1"
    label   : "Fe-fum@ BSA-FL + Histidin 5 uL"
    suffix  : "pre_glucose" (the condition folder), optional

    -> "A1_Fe-fum-BSA-FL-Histidin-5uL_pre_glucose"

    The well id is kept at the front by default: it is the only part
    guaranteed unique, and losing it would make two identically-labelled
    wells overwrite each other.
    """
    parts = []
    if include_well and well:
        parts.append(str(well))
    cleaned = clean_for_filename(label)
    if cleaned:
        parts.append(cleaned)
    if suffix:
        parts.append(clean_for_filename(suffix, max_len=40))
    return "_".join(p for p in parts if p)


def preview(mapping, suffix=""):
    """Human-readable table of what the names will become."""
    if not mapping:
        return "(no mapping)"
    lines = [f"{'well':<6} {'label':<44} {'output name'}"]
    lines.append("-" * 100)
    for well in sorted(mapping, key=_well_key):
        lines.append(f"{well:<6} {mapping[well][:44]:<44} "
                     f"{build_name(well, mapping[well], suffix)}")
    return "\n".join(lines)


def _well_key(w):
    m = re.match(r"([A-Za-z]+)(\d+)", w)
    if m:
        return (m.group(1), int(m.group(2)))
    return (w, 0)


EXAMPLE = (
    "Position\tFe-fum\tFe-fum - His\tFe-fum + BSA\tFe-fum + BSA - His\n"
    "1\tFe-fum + Histidin 5 uL\tFe-fum - Histidin 5 uL\t"
    "Fe-fum@ BSA-FL + Histidin 5 uL\tFe-fum@ BSA-FL - Histidin 5 uL\n"
    "2\tFe-fum + Histidin 3 uL\tFe-fum - Histidin 3 uL\t"
    "Fe-fum@ BSA-FL + Histidin 3 uL\tFe-fum@ BSA-FL - Histidin 3 uL\n"
)