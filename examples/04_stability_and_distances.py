"""Example 4 - why persistence is usable on real data: stability.

The stability theorem says the bottleneck distance between the Rips diagrams of
two clouds is at most twice the Hausdorff distance between the clouds
(d_B <= 2 d_GH <= 2 d_H).  In other words a small perturbation of the input can
only move the diagram a little: the long bars cannot vanish, and the noise near
the diagonal cannot suddenly become significant.  This script measures both
sides of that inequality.

It then uses the bottleneck distance as a *metric on shapes*: a small distance
matrix over several clouds, where clouds with the same topology land close
together regardless of how they are drawn.
"""

from _common import header, out, timed

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import bottleneck_distance, diagram_distance_matrix
from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence

THRESHOLD = 2.2
NOISE_LEVELS = [0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20]


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def stability_experiment():
    base = ds.circle(70, seed=21)
    base_res = rips_persistence(base, max_dim=1, threshold=THRESHOLD)
    rng = np.random.default_rng(0)

    rows = []
    for noise in NOISE_LEVELS:
        pert = base + noise * rng.standard_normal(base.shape)
        res = rips_persistence(pert, max_dim=1, threshold=THRESHOLD)
        rows.append((
            noise,
            hausdorff(base, pert),
            bottleneck_distance(base_res[0], res[0]),
            bottleneck_distance(base_res[1], res[1]),
        ))
    return base_res, rows


def main() -> None:
    header("Example 4: stability, and diagrams as a metric on shapes")

    with timed("stability sweep"):
        _, rows = stability_experiment()

    print("\n  the bottleneck distance stays under 2 x the Hausdorff distance, always")
    print(f"  {'noise':>6}  {'Hausdorff':>10}  {'bottleneck H0':>14}  {'bottleneck H1':>14}  bound holds")
    for noise, haus, b0, b1 in rows:
        holds = "yes" if max(b0, b1) <= 2 * haus + 1e-9 else "VIOLATED"
        print(f"  {noise:6.2f}  {haus:10.4f}  {b0:14.4f}  {b1:14.4f}  {holds:>11}")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=viz.SURFACE)
    haus = [r[1] for r in rows]
    ax.plot(haus, [2 * h for h in haus], color=viz.INK_MUTED, linewidth=1.0,
            linestyle=(0, (4, 3)), label="stability bound (2x Hausdorff)", zorder=2)
    for dim, key in ((0, 2), (1, 3)):
        color, marker = viz.dim_style(dim)
        values = [r[key] for r in rows]
        ax.plot(haus, values, color=color, linewidth=2.0, marker=marker,
                markersize=6, markeredgecolor=viz.SURFACE, label=f"$H_{dim}$", zorder=3)
        ax.annotate(f"$H_{dim}$", xy=(haus[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=10,
                    va="center", fontweight="bold")
    ax.set_xlabel("Hausdorff distance between the clouds", color=viz.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("bottleneck distance between diagrams", color=viz.INK_SECONDARY, fontsize=10)
    ax.set_title("Diagrams move no faster than the data (both curves stay under the bound)",
                 color=viz.INK,
                 fontsize=11, loc="left", pad=10)
    viz._style_axes(ax)
    legend = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for text in legend.get_texts():
        text.set_color(viz.INK_SECONDARY)
    fig.tight_layout()
    print("\nwrote", viz.save(fig, out("04_stability.png")))

    # --- diagrams as a shape descriptor -------------------------------------
    header("Bottleneck distance between shape classes (H1)")
    clouds = {
        "circle A": ds.circle(70, noise=0.05, seed=1),
        "circle B": ds.circle(70, noise=0.05, seed=2),
        "ellipse": ds.circle(70, noise=0.05, seed=3) * np.array([1.35, 0.75]),
        "noise A": 2 * ds.uniform_box(70, seed=4) - 1,
        "noise B": 2 * ds.uniform_box(70, seed=5) - 1,
    }
    with timed("5 diagrams"):
        dgms = [rips_persistence(p, max_dim=1, threshold=THRESHOLD)[1] for p in clouds.values()]

    matrix = diagram_distance_matrix(dgms, metric="bottleneck")
    names = list(clouds)
    print("\n  " + "".join(f"{n:>10}" for n in names))
    for i, name in enumerate(names):
        print(f"  {name:>9} " + "".join(f"{matrix[i, j]:10.3f}" for j in range(len(names))))
    print("\n  circle-to-circle and noise-to-noise are small; circle-to-noise is large.")
    print("  the ellipse sits with the circles - the metric sees topology, not geometry.")

    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=viz.SURFACE)
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right",
                  color=viz.INK_SECONDARY, fontsize=9)
    ax.set_yticks(range(len(names)), names, color=viz.INK_SECONDARY, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            # value labels keep the cells readable without decoding the ramp
            light = matrix[i, j] > 0.6 * matrix.max()
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=viz.SURFACE if light else viz.INK_SECONDARY)
    ax.set_title("Bottleneck distance, $H_1$", color=viz.INK, fontsize=11, loc="left", pad=10)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=viz.INK_SECONDARY, labelsize=9, length=0)
    fig.tight_layout()
    print("\nwrote", viz.save(fig, out("04_diagram_distances.png")))


if __name__ == "__main__":
    main()
