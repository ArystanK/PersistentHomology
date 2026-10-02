"""Matplotlib views of point clouds and persistence output.

Colour is assigned by *homological dimension* - an identity encoding, so the
categorical slots are used in fixed order and never cycled.  Every dimension
also gets its own marker shape, so identity survives colour-blind vision, print
and greyscale; the legend is always present.
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .persistence import PersistenceResult

# Categorical slots 1, 2, 3, 7 of the reference palette, in fixed order.
DIM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
DIM_MARKERS = ["o", "^", "s", "D"]
OTHER_COLOR = "#6b6a66"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8c8b85"
GRID = "#e6e5e1"


def dim_style(dim: int):
    if dim < len(DIM_COLORS):
        return DIM_COLORS[dim], DIM_MARKERS[dim]
    return OTHER_COLOR, "x"


def _as_diagrams(result) -> Dict[int, np.ndarray]:
    if isinstance(result, PersistenceResult):
        return result.diagrams
    if isinstance(result, dict):
        return result
    return {0: np.asarray(result, dtype=float).reshape(-1, 2)}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


#: Public alias: apply this module's axis styling to an axis you drew yourself,
#: so a hand-rolled panel sits beside the built-in ones without looking foreign.
style_axes = _style_axes


def _finite_max(diagrams: Dict[int, np.ndarray]) -> float:
    values = [
        d[np.isfinite(d)].max()
        for d in diagrams.values()
        if len(d) and np.isfinite(d).any()
    ]
    return float(max(values)) if values else 1.0


def persistence_diagram(result, ax=None, title: str = "Persistence diagram",
                        dims: Sequence[int] | None = None, alpha: float = 0.85,
                        top: float | None = None):
    """Scatter of (birth, death).  Distance above the diagonal is lifetime, so
    the meaningful features are the points far from it and the noise piles up
    along it.  Essential classes sit on the dashed infinity line at the top.

    ``top`` is the largest finite value the axes must show.  Pass the same value
    to several diagrams to put them on one scale: otherwise each one stretches
    to fit its own points, and a diagram of pure noise looks as dramatic as one
    with a real feature.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    diagrams = _as_diagrams(result)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.8))
    if dims is None:
        dims = sorted(diagrams)

    if top is None:
        top = _finite_max(diagrams)
    pad = 0.06 * top if top > 0 else 0.1
    infinity = top + 3 * pad

    ax.plot([-pad, infinity + pad], [-pad, infinity + pad],
            color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.fill_between([-pad, infinity + pad], [-pad, infinity + pad],
                    -pad, color=GRID, alpha=0.45, linewidth=0, zorder=0)
    ax.axhline(infinity, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    handles = []
    for d in dims:
        dgm = np.asarray(diagrams.get(d, np.empty((0, 2))), dtype=float).reshape(-1, 2)
        if len(dgm) == 0:
            continue
        color, marker = dim_style(d)
        death = np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], infinity)
        ax.scatter(dgm[:, 0], death, s=44, marker=marker, facecolor=color,
                   edgecolor=SURFACE, linewidth=1.0, alpha=alpha, zorder=3)
        # coincident points would otherwise read as one: say how many there are
        spots, counts = np.unique(np.round(np.column_stack([dgm[:, 0], death]), 9),
                                  axis=0, return_counts=True)
        for (x, y), n in zip(spots, counts):
            if n > 1:
                ax.annotate(f"×{n}", xy=(x, y), xytext=(7, 0), textcoords="offset points",
                            color=color, fontsize=9, va="center", zorder=4)
        handles.append(Line2D([], [], linestyle="none", marker=marker, color=color,
                              markeredgecolor=SURFACE, markersize=7,
                              label=f"$H_{d}$  ({len(dgm)})"))

    # infinity is a tick of its own on the death axis, so its label never sits
    # on top of the essential classes plotted at birth 0
    finite_ticks = [t for t in MaxNLocator(nbins=5).tick_values(0, top)
                    if 0 <= t <= infinity - 1.5 * pad]
    ax.yaxis.set_major_locator(FixedLocator(finite_ticks + [infinity]))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: r"$\infty$" if np.isclose(value, infinity) else f"{value:g}"))

    ax.set_xlim(-pad, infinity + pad)
    ax.set_ylim(-pad, infinity + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("birth (scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("death (scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    _style_axes(ax)
    if handles:
        legend = ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
    return ax


def barcode(result, ax=None, title: str = "Persistence barcode",
            dims: Sequence[int] | None = None, max_bars_per_dim: int = 60):
    """One horizontal bar per class, from birth to death.

    Bars are sorted longest-first within each dimension, and only the longest
    ``max_bars_per_dim`` are drawn: the short ones are sampling noise and would
    otherwise be a solid block at the left edge.
    """
    diagrams = _as_diagrams(result)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    if dims is None:
        dims = sorted(diagrams)

    top = _finite_max(diagrams)
    end = top * 1.08 if top > 0 else 1.0
    # one bar per unit of height: past ~30 bars a 2pt line is thicker than the
    # gap between neighbours and adjacent bars fuse into one
    n_shown = sum(min(len(np.asarray(diagrams.get(d, [])).reshape(-1, 2)), max_bars_per_dim)
                  for d in dims)
    bar_width = 2.0 if n_shown <= 30 else max(0.6, 2.0 * 30 / n_shown)

    y = 0
    handles, ticks, tick_labels = [], [], []
    for d in dims:
        dgm = np.asarray(diagrams.get(d, np.empty((0, 2))), dtype=float).reshape(-1, 2)
        if len(dgm) == 0:
            continue
        color, _ = dim_style(d)
        lifetimes = np.where(np.isfinite(dgm[:, 1]), dgm[:, 1] - dgm[:, 0], np.inf)
        order = np.argsort(-lifetimes)[:max_bars_per_dim]
        shown = dgm[order]
        start_y = y
        for birth, death in shown:
            finite = np.isfinite(death)
            ax.plot([birth, death if finite else end], [y, y],
                    color=color, linewidth=bar_width, solid_capstyle="butt", zorder=3)
            if not finite:
                ax.plot([end], [y], marker=">", color=color, markersize=5, zorder=3)
            y += 1
        ticks.append((start_y + y - 1) / 2)
        tick_labels.append(f"$H_{d}$")
        hidden = len(dgm) - len(shown)
        label = f"$H_{d}$  ({len(dgm)})" + (f", {hidden} short bars hidden" if hidden else "")
        handles.append(Line2D([], [], color=color, linewidth=2.5, label=label))
        y += 3  # a clear gap between dimension blocks

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=10, color=INK_SECONDARY)
    # keep the bars in the bottom 70% of the axes, so the legend above them
    # never lands on a bar whatever the figure size
    span = max(y - 2, 1) + 2
    ax.set_ylim(-2, -2 + (span / 0.70 if handles else span))
    ax.set_xlim(0, end * 1.02)
    ax.set_xlabel("scale", color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    if handles:
        legend = ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
    return ax


def betti_curves(result, ax=None, title: str = "Betti curves", n_steps: int = 400,
                 dims: Sequence[int] | None = None):
    """Betti number as a function of scale: how many classes are alive at each
    radius.  A wide plateau at b1 = 1 is the signature of a single robust loop.
    """
    diagrams = _as_diagrams(result)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))
    if dims is None:
        dims = sorted(diagrams)

    top = _finite_max(diagrams)
    grid = np.linspace(0, top * 1.02 if top > 0 else 1.0, n_steps)

    labels_at = {}  # end value -> labels already placed there
    for d in dims:
        dgm = np.asarray(diagrams.get(d, np.empty((0, 2))), dtype=float).reshape(-1, 2)
        if len(dgm) == 0:
            continue
        color, _ = dim_style(d)
        alive = ((dgm[:, 0][None, :] <= grid[:, None])
                 & (dgm[:, 1][None, :] > grid[:, None])).sum(axis=1)
        ax.plot(grid, alive, color=color, linewidth=2.0, label=f"$b_{d}$", zorder=3)
        # direct label at the right edge, so identity never rests on colour alone;
        # curves that end at the same value get their labels stacked, not overprinted
        stacked = labels_at.get(int(alive[-1]), 0)
        labels_at[int(alive[-1])] = stacked + 1
        ax.annotate(f"$b_{d}$", xy=(grid[-1], alive[-1]), xytext=(5, 11 * stacked),
                    textcoords="offset points", color=color, fontsize=10,
                    va="center", fontweight="bold")

    ax.set_xlabel("scale", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("classes alive", color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
    return ax


def plot_points(points: np.ndarray, ax=None, title: str = "Point cloud",
                color: str = "#52514e", size: float = 14):
    """Scatter of a 2D cloud; 3D clouds are drawn on a 3D axis."""
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 3:
        if ax is None:
            ax = plt.figure(figsize=(4.8, 4.8)).add_subplot(projection="3d")
        ax.scatter(*points.T, s=size, color=color, edgecolor=SURFACE,
                   linewidth=0.4, alpha=0.9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_facecolor(SURFACE)
        return ax
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(points[:, 0], points[:, 1], s=size, color=color,
               edgecolor=SURFACE, linewidth=0.6, alpha=0.9, zorder=3)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    _style_axes(ax)
    return ax


def _label_vertices(ax, points: np.ndarray, labels: Sequence[str]) -> None:
    """Put each vertex label just outside the drawing, pushed away from the
    centroid, so it never sits on an edge running into that vertex."""
    centre = points.mean(axis=0)
    for (x, y), text in zip(points, labels):
        away = np.array([x, y]) - centre
        norm = np.linalg.norm(away)
        away = away / norm if norm > 0 else np.array([0.0, 1.0])
        ax.annotate(text, (x, y), textcoords="offset points", xytext=tuple(11 * away),
                    ha="center", va="center", fontsize=11, color=INK, zorder=4)
    ax.margins(0.12)


def _picture_axes(ax, title: str) -> None:
    """A drawing, not a chart: no ticks, no grid, no axis lines."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    _style_axes(ax)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _fraction(value: float) -> str:
    """1, 2/3, 1/3 rather than 1.000, 0.667, 0.333 -- harmonic weights on small
    complexes are usually simple fractions, and the slide states them that way."""
    from fractions import Fraction
    exact = Fraction(value).limit_denominator(12)
    return str(exact) if abs(float(exact) - value) < 1e-6 else f"{value:.2f}"


def plot_complex(points: np.ndarray, complex_, ax=None, title: str = "",
                 vertex_size: float = 30, show_counts: bool = True,
                 vertex_labels: Sequence[str] | None = None):
    """Draw a 2D simplicial complex: vertices, edges, and shaded triangles.

    ``points`` supplies the coordinates of vertex ``i`` in row ``i``; the complex
    itself is combinatorial and carries none.  Simplices above dimension 2 are
    not drawn -- there is nothing sensible to draw in the plane -- but they are
    counted in the caption.
    """
    points = np.asarray(points, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0))

    for triangle in complex_.simplices(2):
        ax.fill(points[list(triangle), 0], points[list(triangle), 1],
                color=DIM_COLORS[0], alpha=0.16, linewidth=0, zorder=1)
    for u, v in complex_.simplices(1):
        ax.plot(points[[u, v], 0], points[[u, v], 1],
                color=DIM_COLORS[0], linewidth=1.6, alpha=0.85, zorder=2)
    ax.scatter(points[:, 0], points[:, 1], s=vertex_size, color=INK,
               edgecolor=SURFACE, linewidth=0.8, zorder=3)

    if show_counts:
        counts = complex_.counts()
        caption = "  ".join(f"$n_{k}$={n}" for k, n in enumerate(counts))
        betti = ", ".join(str(b) for b in complex_.betti_numbers())
        ax.set_xlabel(f"{caption}\n$\\beta$ = ({betti})",
                      color=INK_SECONDARY, fontsize=9)

    if vertex_labels is not None:
        _label_vertices(ax, points, vertex_labels)
    _picture_axes(ax, title)
    return ax


def plot_chain(points: np.ndarray, complex_, chain: np.ndarray, ax=None,
               title: str = "", scale: float = 1.0,
               vertex_labels: Sequence[str] | None = None, show_weights: bool = False):
    """Draw a 1-chain as a flow on the edges: an arrow per edge, pointing along
    the orientation when the coefficient is positive and against it when it is
    negative, with width proportional to magnitude.

    This is how a harmonic representative is meant to be read -- a circulation
    around the hole, not a set of edges.  ``show_weights`` prints each edge's
    magnitude relative to the largest, so a split such as 2/3 against 1/3 can be
    read off rather than guessed from arrow widths.
    """
    points = np.asarray(points, dtype=float)
    chain = np.asarray(chain, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0))

    peak = np.abs(chain).max() if len(chain) else 0.0
    peak = peak if peak > 0 else 1.0

    # the 2-simplices are drawn too: which triangles are filled is exactly what
    # explains why a harmonic chain's weights come out uneven
    for triangle in complex_.simplices(2):
        ax.fill(points[list(triangle), 0], points[list(triangle), 1],
                color=DIM_COLORS[0], alpha=0.14, linewidth=0, zorder=0)

    for (u, v), weight in zip(complex_.simplices(1), chain):
        start, end = (points[u], points[v]) if weight >= 0 else (points[v], points[u])
        magnitude = abs(weight) / peak
        if magnitude < 1e-9:
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=GRID, linewidth=1.2, zorder=1)
            continue
        ax.annotate("", xy=end, xytext=start, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=DIM_COLORS[1],
                                    linewidth=0.6 + 3.0 * magnitude * scale,
                                    alpha=0.35 + 0.65 * magnitude,
                                    shrinkA=4, shrinkB=6))
        if show_weights:
            # beside the edge's midpoint, on the side away from the centroid
            mid = (start + end) / 2
            normal = np.array([-(end - start)[1], (end - start)[0]])
            normal = normal / (np.linalg.norm(normal) or 1.0)
            if normal @ (mid - points.mean(axis=0)) < 0:
                normal = -normal
            ax.annotate(_fraction(magnitude), mid, textcoords="offset points",
                        xytext=tuple(10 * normal), ha="center", va="center",
                        fontsize=10, color=DIM_COLORS[1], fontweight="bold", zorder=4)
    ax.scatter(points[:, 0], points[:, 1], s=26, color=INK,
               edgecolor=SURFACE, linewidth=0.8, zorder=3)

    if vertex_labels is not None:
        _label_vertices(ax, points, vertex_labels)
    _picture_axes(ax, title)
    return ax


def plot_summary(points: np.ndarray, result: PersistenceResult,
                 title: str = "", figsize=(14.5, 4.4)):
    """Cloud, barcode and diagram side by side - the usual three-panel view."""
    fig = plt.figure(figsize=figsize, facecolor=SURFACE)
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 3:
        ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    else:
        ax0 = fig.add_subplot(1, 3, 1)
    plot_points(points, ax0, title=f"{len(points)} points")
    barcode(result, fig.add_subplot(1, 3, 2))
    persistence_diagram(result, fig.add_subplot(1, 3, 3))
    if title:
        fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1))
    return fig


def save(fig, path: str, dpi: int = 160):
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
