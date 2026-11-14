#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cargo_interactive.py - direct manipulation of the chart
=============================================================================
Adds mouse interaction to the figure: click a bar to select it, drag it left
or right to reorder the groups, double-click to rename, right-click for a
colour picker.

WHY REORDERING IS ONLY COSMETIC
  Dragging changes the ORDER the groups are drawn in, nothing else. The values
  inside each group, and every statistic computed from them, are untouched.
  That distinction matters: a figure tool that let you drag a bar to a new
  HEIGHT would be a tool for drawing conclusions you did not measure. Height
  here is always the mean of the data and cannot be edited by hand.

WHAT INTERACTION IS FOR
  Ordering is a real presentational decision - control first, or doses
  ascending, or grouped by treatment - and doing it by dragging is far quicker
  than retyping group names. Colour and label edits are the same kind of
  choice. None of them changes a number.
=============================================================================
"""

from __future__ import annotations

import numpy as np


class InteractiveChart:
    """
    Wraps a matplotlib axes and a group list, and handles mouse events.

    on_change(groups) is called whenever the order, a label, or a colour
    changes, so the owner can redraw and refresh its statistics.
    """

    def __init__(self, canvas, ax, groups, opts, on_change=None,
                 on_select=None):
        self.canvas = canvas
        self.ax = ax
        self.groups = groups          # list of (label, values) or (label, dict)
        self.opts = opts
        self.on_change = on_change
        self.on_select = on_select

        self.selected = None          # index of the selected group
        self._drag_from = None
        self._drag_x = None
        self._press_xy = None

        self._cids = [
            canvas.mpl_connect("button_press_event", self._press),
            canvas.mpl_connect("motion_notify_event", self._motion),
            canvas.mpl_connect("button_release_event", self._release),
        ]

    def disconnect(self):
        for cid in self._cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids = []

    # ---------------------------------------------------------------- hit ---
    def _index_at(self, event):
        """Which group is under the pointer, by x position in data space."""
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        i = int(round(event.xdata))
        if 0 <= i < len(self.groups):
            # only count it as a hit near the centre, so gaps do not select
            if abs(event.xdata - i) <= 0.45:
                return i
        return None

    # -------------------------------------------------------------- events ---
    def _press(self, event):
        # Heartbeat: proves an event actually reached this handler. Without it
        # a dead canvas and a logic bug look identical from the outside.
        if self.on_select:
            try:
                self.on_select(-1, "heartbeat")
            except Exception:
                pass

        idx = self._index_at(event)

        if event.button == 3:                      # right-click: colour
            if idx is not None and self.on_select:
                self.on_select(idx, "color")
            return

        if event.dblclick:                         # double-click: rename
            if idx is not None and self.on_select:
                self.on_select(idx, "rename")
            return

        if idx is None:
            self.selected = None
            if self.on_select:
                self.on_select(None, "deselect")
            return

        self.selected = idx
        self._drag_from = idx
        self._press_xy = (event.x, event.y)
        if self.on_select:
            self.on_select(idx, "select")

    def _motion(self, event):
        if self._drag_from is None or event.inaxes is not self.ax:
            return
        if event.xdata is None:
            return
        # show where the bar would land
        target = int(round(event.xdata))
        target = max(0, min(len(self.groups) - 1, target))
        if target != self._drag_x:
            self._drag_x = target
            self._hint(target)

    def _release(self, event):
        if self._drag_from is None:
            return
        src = self._drag_from
        self._drag_from = None
        dst = self._drag_x
        self._drag_x = None
        self._clear_hint()

        if dst is None or dst == src:
            return
        # move the group; values travel with it untouched
        item = self.groups.pop(src)
        self.groups.insert(dst, item)
        self.selected = dst
        if self.on_change:
            self.on_change(self.groups)

    # --------------------------------------------------------------- hints ---
    def _hint(self, target):
        self._clear_hint()
        try:
            self._hint_line = self.ax.axvline(
                target, color="#0173b2", linewidth=2, linestyle="--",
                alpha=0.8, zorder=10)
            self.canvas.draw_idle()
        except Exception:
            self._hint_line = None

    def _clear_hint(self):
        line = getattr(self, "_hint_line", None)
        if line is not None:
            try:
                line.remove()
            except Exception:
                pass
            self._hint_line = None
            self.canvas.draw_idle()


def highlight_selected(ax, index, n_groups):
    """Draw a soft band behind the selected group."""
    if index is None:
        return None
    try:
        return ax.axvspan(index - 0.5, index + 0.5, color="#0173b2",
                          alpha=0.10, zorder=0)
    except Exception:
        return None