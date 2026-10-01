"""Example 1 - the whole idea in one picture.

Two point clouds of the same size, in the same box.  One is sampled from a
circle, the other is uniform noise.  Neither is distinguishable by counting
points, by the mean, or by the covariance.  Persistent homology separates them
immediately: the circle has one H1 bar that survives across a wide range of
scales, and the noise has only short-lived ones.
"""

from _common import header, out, timed

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence

CLOUDS = {
    "circle (with noise)": ds.circle(80, radius=1.0, noise=0.06, seed=11),
    "uniform noise": 2 * ds.uniform_box(80, seed=11) - 1,
}


def main() -> None:
    header("Example 1: a loop vs. no loop")

    results = {}
    for name, points in CLOUDS.items():
        with timed(name):
            results[name] = rips_persistence(points, max_dim=1, threshold=2.0)

    for name, result in results.items():
        h1 = result.most_persistent(1, k=3)
        lifetimes = (h1[:, 1] - h1[:, 0]) if len(h1) else np.zeros(0)
        print(f"\n{name}")
        print(result.summary())
        print("  three longest H1 lifetimes: "
              + ", ".join(f"{v:.3f}" for v in lifetimes) if len(lifetimes) else "  no H1")

    # The separating statistic: the single longest H1 bar.
    scores = {
        name: float((r[1][:, 1] - r[1][:, 0]).max()) if len(r[1]) else 0.0
        for name, r in results.items()
    }
    print("\nlongest H1 bar (the 'is there a hole?' score)")
    for name, score in scores.items():
        print(f"  {name:22s} {score:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), facecolor=viz.SURFACE)
    for row, (name, points) in enumerate(CLOUDS.items()):
        viz.plot_points(points, axes[row, 0], title=name)
        viz.barcode(results[name], axes[row, 1], title=f"barcode - {name}")
        viz.persistence_diagram(results[name], axes[row, 2], title=f"diagram - {name}")
    fig.tight_layout()
    print("\nwrote", viz.save(fig, out("01_circle_vs_noise.png")))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), facecolor=viz.SURFACE)
    for ax, (name, result) in zip(axes, results.items()):
        viz.betti_curves(result, ax, title=f"Betti curves - {name}")
    fig.tight_layout()
    print("wrote", viz.save(fig, out("01_betti_curves.png")))


if __name__ == "__main__":
    main()
