#!/usr/bin/env python
"""House style, panel layout, the short-label scheme, and the figure QA gate.

DROP-IN TEMPLATE. Copy into the project's figures/ directory, then adjust:
  * TEXT_W_IN  to the venue's column width (measure it, do not guess)
  * C / SERIES to the paper's palette, keeping the roles semantic
  * font.serif to fonts that actually exist here (verify, do not assume)
  * the short-label block to this project's own data levels
Everything below that is machinery and should be kept as is.

Five things live here so every figure script in the paper agrees on them.

1. STYLE. Sizes matched to the paper's body text, and a save routine that lays
   the canvas out at the TRUE printed width, so a point in the script is a point
   on paper. This is the single most repeated failure in figure work.

2. PALETTE. Okabe-Ito, whose colour-vision-deficiency safety is CHECKED by
   cvd_report() rather than asserted. Run it after any palette edit.

3. PANEL LAYOUT. panels() / label_panels() / shared_legend(), because a figure
   with six sub-panels is judged on its arrangement before anyone reads a single
   axis. Compact, aligned, non-uniform, one legend.

4. SHORT LABELS. Raw schema levels ("long_middle", "upper_middle") are long,
   uneven, and uninformative. Every figure prints the level's own bin instead,
   which is shorter AND says more. Declared once here, mirrored in the paper's
   data table.

5. QA GATE. check_figure() measures the rendered bounding box of every Text a
   figure will actually draw and fails on overlap, clipping, undersized type,
   prose overload, and near-miss panel misalignment. That is what lets the
   redesign loop score itself instead of asking a human to squint at a PDF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette
#
# Okabe & Ito's Color Universal Design set: eight tones chosen to stay separable
# under the common forms of colour-vision deficiency, rather than chosen to look
# good on a screen. Keep the ROLES locked -- if blue means "retained" in one
# figure it means "retained" in every figure, and it never becomes decoration.
# Colour that carries no information is grey, and the caption says what grey
# means.
#
# After changing any of these, run:  python figstyle.py
# which prints cvd_report() and tells you whether the set is still separable.
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "black":   "#000000",
}

C = dict(proc=OKABE_ITO["blue"],   llm=OKABE_ITO["purple"],
         save=OKABE_ITO["green"],  cost=OKABE_ITO["orange"],
         acc=OKABE_ITO["vermil"],  data="#6B7280")     # grey = no information

# Categorical series order. Yellow is last on purpose: it is the weakest of the
# eight against white, so it should not be the second series anyone sees.
SERIES = [OKABE_ITO[k] for k in
          ("blue", "orange", "green", "purple", "vermil", "sky", "yellow")]

TEXT_W_IN = 5.5          # measured: CoRL 2026 \textwidth = 397.485pt (paper-template, 2026-09-24)
PNG_DPI = 400            # shipped raster. 300 is the print floor; 400 is headroom.

MIN_PT = 6.5             # smallest printed label we accept
# "Prose" is what a figure SAYS beyond naming its own axes and series. Titles,
# axis labels, tick labels and legend entries are structural and budgeted per
# item; free-floating annotations are sentences and are capped hard.
MAX_ANNOT = 120          # total chars of free-floating annotation text
MAX_TITLE = 42           # chars in any one panel title
MAX_AXLBL = 32           # chars in any one axis label

# ---- the short-label scheme (MIRROR THIS IN THE PAPER'S DATA TABLE) --------
# Replace with your own levels. Keep the mapping in one place: if a figure and
# the data table disagree, the reader has to learn two vocabularies.
DIST = ["short", "short_middle", "long_middle", "long", "very_long"]
DIST_LBL = ["<10", "10-20", "20-30", "30-40", "40-50"]        # miles
INCOME = ["low", "lower_middle", "middle", "upper_middle", "high"]
INCOME_LBL = ["Q1", "Q2", "Q3", "Q4", "Q5"]                   # Census quintiles

SHORT = {}
SHORT.update(dict(zip(DIST, DIST_LBL)))
SHORT.update(dict(zip(INCOME, INCOME_LBL)))


def short(level: str) -> str:
    """Raw level -> printed label, falling back to the raw string."""
    return SHORT.get(level, level)


def schema_order(levels, canonical):
    """Canonical order, never sorted(). Alphabetical order silently destroys
    every ordered factor's profile, and it is a defect no bounding-box gate can
    see: '10-20, 20-30, 30-40, <10, 40-50' passes every check and is nonsense."""
    return [lv for lv in canonical if lv in set(levels)]


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "axes.grid": True,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        # Box plots otherwise arrive with matplotlib's default orange median,
        # which is a colour this palette has already assigned a meaning to.
        # Anything drawn in a role colour must be carrying that role.
        "boxplot.medianprops.color": "#111111",
        "boxplot.medianprops.linewidth": 1.4,
        "boxplot.boxprops.linewidth": 1.0,
        "boxplot.whiskerprops.linewidth": 1.0,
        "boxplot.capprops.linewidth": 1.0,
        "boxplot.flierprops.markersize": 3,
        "figure.dpi": 110,
        "savefig.bbox": None,
        "pdf.fonttype": 42,     # TrueType, so Illustrator can edit the text
        "svg.fonttype": "none",  # keep text as <text>, editable in Inkscape
    })


# ===========================================================================
# Colour-vision check. The point is that the palette's safety is MEASURED.
#
# A comment claiming "colour-blind safe" is worth nothing: this file used to
# carry exactly that claim about a palette nobody had ever tested. Two checks
# run here, and both are computed from published constants rather than
# transcribed magic numbers:
#
#   (a) CIELAB dE76 between every pair. Below ~20 two series start to look
#       alike at small sizes even in full colour.
#   (b) L* (lightness) separation. Lightness survives EVERY form of CVD and
#       every greyscale photocopy, so it is the property that actually protects
#       a reader. A palette separated only in hue fails in the real world.
#
# If `colorspacious` is installed, a full deuteranopia / protanopia simulation
# runs as well. It is not vendored, because a simulation matrix typed from
# memory is exactly the kind of unverified number this function exists to
# replace.
# ===========================================================================
def _srgb_to_lab(hex_color: str):
    """sRGB hex -> CIELAB under D65. Standard IEC 61966-2-1 + CIE definitions."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    # sRGB D65 -> XYZ
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883          # D65 white point

    def f(t):
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def cvd_report(colors=None, min_de: float = 20.0, min_dl: float = 8.0,
               greyscale_n: int = 3):
    """Print pairwise separation for a categorical palette. Returns violations.

    min_de:      CIELAB dE76 floor for any pair, in full colour.
    min_dl:      L* floor, applied only within the FIRST `greyscale_n` series.
    greyscale_n: how many series this palette promises to keep separable in
                 greyscale.

    Why the lightness rule is scoped rather than global. Okabe-Ito's six
    chromatic tones all sit between L*=46 and L*=71, so NO ordering of six of
    them separates in greyscale -- the best achievable minimum gap is 0.8. The
    palette is engineered for colour-vision deficiency, not for a photocopier,
    and pretending otherwise would leave this gate permanently red, which is how
    a gate stops being read.

    So the honest contract is: the first three series are greyscale-separable
    (minimum gap about 11.7 L* as shipped), and a figure that needs a fourth
    must carry the distinction in something other than hue -- marker shape, line
    dash, hatch -- or split into panels. A legend with six entries usually wants
    to be six panels anyway.
    """
    colors = list(colors or SERIES)
    labs = [_srgb_to_lab(c) for c in colors]
    viol, worst_de = [], (1e9, None)

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            de = float(np.linalg.norm(np.subtract(labs[i], labs[j])))
            pair = f"{colors[i]} x {colors[j]}"
            if de < worst_de[0]:
                worst_de = (de, pair)
            if de < min_de:
                viol.append(f"CVD: {pair} differ by only dE={de:.1f} (min {min_de})")

    head = colors[:greyscale_n]
    worst_dl = (1e9, None)
    for i in range(len(head)):
        for j in range(i + 1, len(head)):
            dl = abs(labs[i][0] - labs[j][0])
            if dl < worst_dl[0]:
                worst_dl = (dl, f"{head[i]} x {head[j]}")
            if dl < min_dl:
                viol.append(f"CVD: {head[i]} x {head[j]} differ by only "
                            f"L*={dl:.1f} (min {min_dl}); the first "
                            f"{greyscale_n} series must survive greyscale")

    print(f"palette of {len(colors)}: closest pair dE={worst_de[0]:.1f} "
          f"({worst_de[1]}); first {greyscale_n} series closest lightness "
          f"dL*={worst_dl[0]:.1f} ({worst_dl[1]})")
    print(f"  greyscale-safe up to {greyscale_n} series. Beyond that, encode "
          f"redundantly (marker/dash/hatch) or use panels.")

    try:                                    # optional, and only if really present
        from colorspacious import cspace_convert
        for kind in ("deuteranomaly", "protanomaly"):
            space = {"name": "sRGB1+CVD", "cvd_type": kind, "severity": 100}
            sim = [cspace_convert(
                       [int(c[k:k + 2], 16) / 255 for k in (1, 3, 5)], space, "CIELab")
                   for c in colors]
            m = min(float(np.linalg.norm(np.subtract(sim[i], sim[j])))
                    for i in range(len(colors)) for j in range(i + 1, len(colors)))
            print(f"  simulated {kind:<14} closest pair dE={m:.1f}")
            if m < min_de:
                viol.append(f"CVD: under {kind} the closest pair falls to dE={m:.1f}")
    except ImportError:
        print("  (pip install colorspacious for a full CVD simulation)")

    for v in viol:
        print("  " + v)
    return viol


# ===========================================================================
# Panel layout
#
# A multi-panel figure is judged on its arrangement before anyone reads an axis.
# Three failures recur, and all three are avoidable here rather than in every
# figure script:
#
#   * The uniform grid. Six identical cells of the same chart type reads as a
#     contact sheet, not an argument. Give the panel that carries the claim more
#     room with width_ratios / height_ratios, and vary the chart TYPE across
#     panels so each one contributes different evidence.
#   * The duplicated legend. N panels sharing a series set need ONE legend for
#     the figure, not N copies eating N times the space.
#   * The near-miss. Panels whose edges are almost but not exactly aligned are
#     what "messy" actually means to a reader's eye. check_figure() fails on it.
# ===========================================================================
def panels(layout, width_ratios=None, height_ratios=None, sharex=False,
           sharey=False, figsize=(6.5, 3.0), **kw):
    """Named panels from an ASCII layout. Returns (fig, dict-of-axes).

        fig, ax = panels('''
            AAB
            CDB
        ''', width_ratios=[1, 1, 1.4])

    The layout string IS the design, visible in the diff when it changes.
    A '.' leaves a cell empty, which is often better than stretching something
    to fill it.

    On sharing: share an axis only when the panels genuinely share units.
    Sharing the y axis while leaving x free is what produces a figure that is
    wide at the top and narrow at the bottom, which reads as top-heavy and
    unstable. If you share one axis and not the other, look at the silhouette
    before shipping.
    """
    gs_kw = {k: v for k, v in
             dict(width_ratios=width_ratios, height_ratios=height_ratios).items()
             if v is not None}
    fig, axd = plt.subplot_mosaic(
        [list(r.strip()) for r in layout.strip().splitlines() if r.strip()],
        figsize=figsize, gridspec_kw=gs_kw, sharex=sharex, sharey=sharey, **kw)
    return fig, axd


def _open_row_gaps(fig):
    """Widen the gap between mosaic rows so each row has a lane for its label.

    In a stacked layout the natural row gap is exactly big enough for the upper
    panel's tick labels and axis label. A "(b)" placed above the lower panel
    then lands in that same strip and collides with the row above -- which is
    what the QA gate reported the first time this ran. The gap has to grow by
    about one line before the labels are placed.
    """
    tops = {round(ax.get_position().y1, 3) for ax in fig.get_axes()}
    if len(tops) < 2:
        return
    lab_frac = plt.rcParams["axes.titlesize"] * 1.9 / 72.0 / fig.get_size_inches()[1]
    heights = [ax.get_position().height for ax in fig.get_axes()]
    avg = max(sum(heights) / len(heights), 1e-6)
    fig.subplots_adjust(hspace=fig.subplotpars.hspace + lab_frac / avg)


def label_panels(axes, dy=0.008, weight="bold"):
    """Stamp (a), (b), (c) ... above the top-LEFT corner of each panel, in order.

    `axes` is a list of Axes in reading order (not a dict: the order is the
    argument, and it must be explicit).

    Two things here are not cosmetic:

    * The anchor is the axes' own left edge in FIGURE coordinates, so labels sit
      at a constant height across panels of different sizes. Placing them per
      axes in axes-fraction units is the usual mistake: each one lands at a
      slightly different height and the whole figure reads as sloppy.
    * va="bottom" puts the label ABOVE the anchor. With va="top" the text hangs
      down into the data area, which is how the first version of this function
      printed "(a)" on top of a line chart.

    Call this AFTER the layout has settled -- get_position() returns stale
    coordinates if tight_layout runs afterwards, and every label ends up in the
    wrong place. save() does the ordering for you; prefer passing
    panel_labels= to save() over calling this by hand.
    """
    for i, ax in enumerate(axes):
        bb = ax.get_position()
        ax.get_figure().text(bb.x0, bb.y1 + dy, f"({chr(97 + i)})",
                             ha="left", va="bottom", fontweight=weight,
                             fontsize=plt.rcParams["axes.titlesize"])


def shared_legend(fig, ax, ncol=None, pad=0.012, label_room=0.0):
    """One legend for the whole figure, above the panels, with room reserved.

    N panels sharing a series set need ONE legend, not N copies eating N times
    the space and inviting them to disagree.

    The legend is measured and then the panel block is pushed DOWN by exactly
    its height, rather than being dropped on top of the panels and hoped for.
    Reserving space only ever shrinks the data area, so nothing can be pushed
    off the canvas. Pass label_room to also reserve a line for panel labels.

    Like label_panels, this must run after the layout has settled.
    """
    h, l = ax.get_legend_handles_labels()
    if not h:
        return None
    fig.canvas.draw()
    leg = fig.legend(h, l, ncol=ncol or min(len(h), 4), loc="upper center",
                     bbox_to_anchor=(0.5, 1.0), frameon=False,
                     borderaxespad=0.0, handlelength=1.6, columnspacing=1.4)
    fig.canvas.draw()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    lh = leg.get_window_extent(fig.canvas.get_renderer()).height / fig_h_px
    top = 1.0 - lh - pad - label_room
    if top < fig.subplotpars.top:
        fig.subplots_adjust(top=top)
    return leg


def rank_barh(ax, labels, values, colors=None, highlight=None):
    """Ranked horizontal bars, largest at the BOTTOM, growing downward.

    The resulting silhouette is narrow at the top and wide at the base, which
    reads as stable. The same data with the largest bar on top reads as
    top-heavy and about to fall over, and gets rejected on looks alone no matter
    how correct the numbers are.

    Returns the y positions so a caller can annotate.
    """
    order = np.argsort(values)               # ascending -> largest ends at top...
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    y = np.arange(len(values))
    cols = ([colors[i] for i in order] if colors is not None
            else [C["proc"]] * len(values))
    if highlight is not None:
        cols = [c if labels[i] in highlight else C["data"]
                for i, c in enumerate(cols)]
    ax.barh(y, values, color=cols, height=0.72)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()                        # ...so flip: largest now at BOTTOM
    ax.grid(axis="y", visible=False)
    return y


def reveal_ylim(ax, values, pad=0.15, floor_frac=0.02):
    """Set a y range that SHOWS the differences instead of flattening them.

    Two failures this prevents, both of which shipped:
      * Twelve bars within 1% of each other, forced to start at zero, render as
        one flat line and the figure says nothing.
      * The smallest category sits at ~0 and is drawn as literally nothing, so a
        reader concludes that side of the chart is empty rather than small.

    Starting a bar axis away from zero exaggerates ratios, so say so in the
    caption whenever you do it. For a LINE or dot chart there is no such
    obligation, which is often the better answer for near-equal values.
    """
    v = np.asarray(list(values), dtype=float)
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        return
    if span / max(abs(hi), 1e-9) < 0.25:      # values are close: zoom in
        ax.set_ylim(lo - pad * span, hi + pad * span)
    else:                                     # keep zero, but keep the tail visible
        ax.set_ylim(0, hi * (1 + pad))
        floor = hi * floor_frac
        for p in getattr(ax, "patches", []):
            if 0 < p.get_height() < floor:
                p.set_height(floor)


# ===========================================================================
# QA gate
# ===========================================================================
def _boxes(fig):
    """Rendered bounding boxes of the Text artists that actually land on paper.

    findobj(Text) is not usable here. Matplotlib keeps a Text for every tick it
    *might* draw, including log-scale ticks outside the data range, and parks the
    unused ones at the axes origin, where they register as mutual overlaps and
    inflate the character count without ever being printed. Your gate then
    reports a pile of phantom failures and you start "fixing" a figure that was
    fine. So enumerate explicitly: titles, axis labels, IN-RANGE tick labels,
    legend entries, and hand-placed annotations.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts, in_legend = list(fig.texts), set()

    for leg in getattr(fig, "legends", []):
        for t in leg.get_texts():
            texts.append(t)
            in_legend.add(t)
        if leg.get_title() is not None:
            texts.append(leg.get_title())
            in_legend.add(leg.get_title())

    for ax in fig.get_axes():
        texts.append(ax.title)
        texts += [ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, "zaxis"):
            texts.append(ax.zaxis.label)
        for axis in ([ax.xaxis, ax.yaxis] +
                     ([ax.zaxis] if hasattr(ax, "zaxis") else [])):
            try:
                lo, hi = sorted(axis.get_view_interval())
            except Exception:
                lo, hi = -np.inf, np.inf
            for tick in axis.get_major_ticks():
                if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    texts.append(tick.label1)
        texts += list(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                texts.append(t)
                in_legend.add(t)
            if leg.get_title() is not None:
                texts.append(leg.get_title())
                in_legend.add(leg.get_title())

    out = []
    for t in texts:
        if t is None or not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_alpha() == 0:
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb, t in in_legend))
    return out


def _alignment_violations(fig, name, near=0.020, exact=0.002):
    """Panels almost-but-not-quite aligned are what 'messy' means to the eye.

    Perfectly aligned edges read as deliberate. Wildly different edges read as
    deliberate too. An edge off by half a percent of the figure width reads as
    an accident, and it is the most common reason a technically-correct
    multi-panel figure looks amateurish. Flag gaps in the near-miss band only.
    """
    axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    if len(axes) < 2:
        return []
    viol = []
    for side, idx in (("left", 0), ("right", 2), ("bottom", 1), ("top", 3)):
        vals = sorted({(ax.get_position().x0, ax.get_position().y0,
                        ax.get_position().x1, ax.get_position().y1)[idx]
                       for ax in axes})
        for a, b in zip(vals, vals[1:]):
            if exact < b - a < near:
                viol.append(f"{name}: MISALIGNED panel {side} edges "
                            f"{a:.3f} vs {b:.3f} (off by {b - a:.3f}); "
                            f"align them or separate them clearly")
    return viol


def check_figure(fig, name, scale):
    """Return a list of violation strings. `scale` = printed_in / canvas_in."""
    viol = []
    boxes = _boxes(fig)

    # (a) printed font size
    for t, _, _ in boxes:
        pt = t.get_fontsize() * scale
        if pt < MIN_PT:
            viol.append(f"{name}: '{t.get_text()[:28]}' prints at {pt:.1f}pt "
                        f"(min {MIN_PT})")

    # (b) pairwise overlap. Entries inside one legend are packed deliberately by
    # matplotlib and their boxes touch by design, so skip legend-legend pairs.
    n_over = 0
    for i in range(len(boxes)):
        ti, bi, leg_i = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj, leg_j = boxes[j]
            if leg_i and leg_j:
                continue
            ov_w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            ov_h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ov_w > 1.0 and ov_h > 1.0:
                area = ov_w * ov_h
                small = min(bi.width * bi.height, bj.width * bj.height)
                if area / max(small, 1.0) > 0.12:      # >12% of the smaller box
                    n_over += 1
                    if n_over <= 6:
                        viol.append(f"{name}: OVERLAP '{ti.get_text()[:20]}' x "
                                    f"'{tj.get_text()[:20]}'")
    if n_over > 6:
        viol.append(f"{name}: ... and {n_over - 6} more overlapping label pairs")

    # (c) clipping: a label whose box leaves the canvas is cut off in the PDF.
    fw, fh = fig.get_size_inches() * fig.dpi
    for t, bb, _ in boxes:
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fw + 0.5 or bb.y1 > fh + 0.5:
            viol.append(f"{name}: CLIPPED '{t.get_text()[:24]}' runs off the canvas")

    # (d) near-miss panel alignment
    viol += _alignment_violations(fig, name)

    # (e) prose load: a figure must not carry sentences. Numeric data labels
    # (bar values, heatmap cells) are DATA: tag them gid("data") and they are
    # exempt from the character budget, though still checked above for size,
    # overlap and clipping.
    def _prose(t):
        return t.get_text().strip() and t.get_gid() != "data"

    annot = sum(len(t.get_text()) for t in fig.texts if _prose(t))
    for ax in fig.get_axes():
        annot += sum(len(t.get_text()) for t in ax.texts if _prose(t))
        ttl = ax.title.get_text()
        if len(ttl) > MAX_TITLE:
            viol.append(f"{name}: title '{ttl[:30]}...' is {len(ttl)} chars "
                        f"(max {MAX_TITLE})")
        for lab in (ax.xaxis.label, ax.yaxis.label):
            txt = lab.get_text()
            if len(txt) > MAX_AXLBL:
                viol.append(f"{name}: axis label '{txt[:30]}' is {len(txt)} chars "
                            f"(max {MAX_AXLBL})")
    if annot > MAX_ANNOT:
        viol.append(f"{name}: {annot} chars of annotation (max {MAX_ANNOT}); "
                    f"move it to the caption")
    return viol


REPORT = []


def _fit(fig, width_in, rounds=4, w_pad=None, h_pad=None):
    """Lay the figure out at exactly width_in, growing HEIGHT until nothing clips.

    Width is the one dimension we may not touch. The figure is included at some
    fraction of \\linewidth, so the canvas width maps onto that column no matter
    what, and a wider canvas silently shrinks every font. Height is free:
    growing it moves labels back inside the canvas without rescaling anything.

    The insight worth internalising: a clipping problem cannot be solved by
    shrinking, only by growing the free dimension. Shrinking to fit is exactly
    what produces unreadable figures.
    """
    w, h = fig.get_size_inches()
    fig.set_size_inches(width_in, width_in * h / w)

    kw = {k: v for k, v in dict(w_pad=w_pad, h_pad=h_pad).items() if v is not None}

    def relayout():
        try:
            fig.tight_layout(**kw)
        except Exception:
            pass
        fig.canvas.draw()

    for _ in range(rounds):
        relayout()
        try:
            bb = fig.get_tightbbox(fig.canvas.get_renderer())   # inches
        except Exception:
            return
        cur_h = fig.get_size_inches()[1]
        over = max(0.0, -bb.y0) + max(0.0, bb.y1 - cur_h)
        if over < 0.01:
            return
        fig.set_size_inches(width_in, cur_h + over + 0.05)
    relayout()          # never leave the canvas resized without re-laying it out


def save(fig, out: Path, name: str, width_in: float = TEXT_W_IN, tight=True,
         w_pad=None, h_pad=None, dpi: int = PNG_DPI, vector: bool = False,
         legend_from=None, legend_ncol=None, panel_labels=None):
    """Lay out at the true printed width, QA, then write the PNG.

    Pass legend_from=<an Axes> for one shared legend, and panel_labels=<list of
    Axes in reading order> for (a)/(b)/(c). Both are applied HERE, after the
    layout has settled, because both read final axes positions and both were
    silently wrong when a figure script called them before tight_layout ran.

    PNG is the deliverable: it is what LaTeX includes, what you Read to inspect
    the figure, and what a co-author can open anywhere. At PNG_DPI a figure
    included at 6.5in carries ~2600 px across, comfortably above the 300 dpi
    print floor, and \\includegraphics[width=...] scales it to the column
    regardless of any dpi metadata.

    Pass vector=True to ALSO emit PDF and SVG. Do that when the venue demands
    vector art, or when a label will need hand-fixing in Illustrator or Inkscape
    later -- pdf.fonttype 42 and svg.fonttype "none" keep the text as text.
    Do not emit them by default: three files per figure per run is churn in the
    diff and a standing question about which one the paper actually includes.

    IMPORTANT: keep width_in in sync with \\includegraphics[width=...]. When you
    change one, grep for the other. Whenever fonts look slightly small, suspect
    this before anything else.
    """
    if tight:
        _fit(fig, width_in, w_pad=w_pad, h_pad=h_pad)
    else:
        w, h = fig.get_size_inches()
        fig.set_size_inches(width_in, width_in * h / w)

    # Order matters: layout first, then reserve legend room, then read the
    # settled axes positions to place the panel labels.
    if legend_from is not None:
        shared_legend(fig, legend_from, ncol=legend_ncol,
                      label_room=0.035 if panel_labels else 0.0)
    if panel_labels:
        _open_row_gaps(fig)
        label_panels(panel_labels)

    viol = check_figure(fig, name, 1.0)   # already laid out at printed size
    REPORT.append((name, viol))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi)
    if vector:
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.svg")
    plt.close(fig)
    return f"{name}.png"


def qa_summary(out: Path, tag: str = ""):
    """tag keeps two figure scripts from overwriting each other's report."""
    total = sum(len(v) for _, v in REPORT)
    lines = []
    for name, viol in REPORT:
        lines.append(f"{'PASS' if not viol else 'FAIL'}  {name}"
                     f"{'' if not viol else f'  ({len(viol)} issues)'}")
        lines += [f"        {v}" for v in viol]
    lines.append(f"\n{len(REPORT)} figures, {total} violations")
    txt = "\n".join(lines)
    (out / f"QA_REPORT{('_' + tag) if tag else ''}.txt").write_text(txt + "\n")
    print(txt)
    return total


if __name__ == "__main__":
    # Self-test: check the palette, then render a multi-panel figure through the
    # whole path and report violations. Run this after editing anything here.
    cvd_report()
    set_style()
    rng = np.random.default_rng(0)
    fig, ax = panels("""
        AAB
        CCB
    """, width_ratios=[1, 1, 1.3], figsize=(6.5, 3.4))
    ax["A"].plot(np.arange(20), rng.normal(size=20).cumsum(), color=SERIES[0],
                 label="method")
    ax["A"].plot(np.arange(20), rng.normal(size=20).cumsum(), color=SERIES[1],
                 label="baseline")
    ax["A"].set_ylabel("regret")
    ax["C"].boxplot([rng.normal(size=40), rng.normal(size=40) + 1], tick_labels=["a", "b"])
    ax["C"].set_ylabel("reward")
    rank_barh(ax["B"], list("vwxyz"), [3.0, 9.0, 5.0, 1.0, 7.0],
              highlight={"w", "z"})
    ax["B"].set_xlabel("share (%)")
    save(fig, Path("figstyle_selftest"), "selftest", width_in=TEXT_W_IN,
         legend_from=ax["A"], panel_labels=[ax["A"], ax["C"], ax["B"]])
    qa_summary(Path("figstyle_selftest"), tag="selftest")
