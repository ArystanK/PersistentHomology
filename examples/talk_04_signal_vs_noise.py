"""Talk part 4 - which bars are the shape, and what if none are?

A diagram always has a longest bar.  Whether it is the shape of the data or the
noise of the sample is a separate question, answered here two ways on the three
clouds from Part 1:

1. a null model -- shuffle each coordinate independently, which keeps every
   marginal and destroys the joint shape; a bar counts only if it is longer
   than the null's 95th percentile;
2. a bootstrap band (Fasy et al. 2014) -- resample the points with replacement,
   take c = the 95th percentile of the bottleneck distance between the
   resampled diagram and the original; a bar longer than 2c lies outside the
   band and cannot be explained by sampling alone.

The loop passes both.  The blob passes neither: that is the "stable nowhere"
case, and the honest conclusion is "no feature larger than 2c at this sample
size", not "no structure at all".  Two warnings come out of the same numbers:
an automatic "most stable scale" still names a scale on the blob, and the
coordinate-shuffle null cannot see the two clusters, because shuffling each
coordinate keeps the gap that separates them.
"""

from _common import check, header, out

import matplotlib.pyplot as plt
import numpy as np

from talk_01_motivation import build_clouds

from persistent_homology import bottleneck_distance, rips_persistence
from persistent_homology import plotting as viz

THRESHOLD = 3.0
REPLICATES = 20


def finite(diagram: np.ndarray) -> np.ndarray:
    return diagram[np.isfinite(diagram[:, 1])]


def longest(diagram: np.ndarray) -> float:
    bars = finite(diagram)
    return float((bars[:, 1] - bars[:, 0]).max()) if len(bars) else 0.0


def main() -> None:
    header("Talk part 4: which bars are the shape?")
    rng = np.random.default_rng(0)
    found = {}

    print(f"\n  {REPLICATES} null replicates and {REPLICATES} bootstrap resamples per cloud")
    print(f"  {'cloud':18s} {'dim':>4s} {'longest':>8s} {'null 95%':>9s} {'2c':>7s}  verdict")
    for name, points in build_clouds().items():
        result = rips_persistence(points, max_dim=1, threshold=THRESHOLD)
        observed = {k: longest(result[k]) for k in (0, 1)}

        null = {0: [], 1: []}
        for _ in range(REPLICATES):
            shuffled = np.column_stack([rng.permutation(points[:, j])
                                        for j in range(points.shape[1])])
            replicate = rips_persistence(shuffled, max_dim=1, threshold=THRESHOLD)
            for k in (0, 1):
                null[k].append(longest(replicate[k]))

        wobble = {0: [], 1: []}
        for _ in range(REPLICATES):
            resample = points[rng.integers(0, len(points), len(points))]
            replicate = rips_persistence(resample, max_dim=1, threshold=THRESHOLD)
            wobble[0].append(bottleneck_distance(finite(result[0]), finite(replicate[0])))
            wobble[1].append(bottleneck_distance(result[1], replicate[1]))

        for k in (0, 1):
            null95 = float(np.percentile(null[k], 95))
            band = 2 * float(np.percentile(wobble[k], 95))
            beats_null, beats_band = observed[k] > null95, observed[k] > band
            verdict = ("real" if beats_null and beats_band
                       else "noise" if not (beats_null or beats_band) else "tests disagree")
            print(f"  {name:18s} {'H' + str(k):>4s} {observed[k]:8.3f} {null95:9.3f} "
                  f"{band:7.3f}  {verdict}")
            found[name, k] = (beats_null, beats_band)
            if k == 1:
                found[name, "h1"] = (result[1], null95, band)

        scale = result.most_stable_scale()
        found[name, "scale"] = (scale, result.betti_numbers(scale))

    print("\n  'Most stable scale' (widest interval with a constant, non-trivial Betti vector):")
    for name in ("(a) a blob", "(b) a loop", "(c) two clusters"):
        scale, betti = found[name, "scale"]
        print(f"    {name:18s} eps = {scale:.3f}, betti = {betti}")
    print("  It names a scale on the blob too.  Always test what it finds.")

    # ---------------------------------------------------------------- figure
    # Both thresholds are lines parallel to the diagonal: a point (b, d) has
    # persistence d - b, so "longer than t" means "above the line d = b + t".
    order = ["(b) a loop", "(a) a blob", "(c) two clusters"]
    top = max(float(finite(found[n, "h1"][0])[:, 1].max()) for n in order)
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), facecolor=viz.SURFACE)
    for ax, name in zip(axes, order):
        h1, null95, band = found[name, "h1"]
        real = all(found[name, 1])
        viz.persistence_diagram({1: h1}, ax, dims=[1], top=top,
                                title=f"{name}: {'a real loop' if real else 'noise'}")
        lo, hi = ax.get_xlim()
        xs = np.array([lo, hi])
        ax.fill_between(xs, xs, xs + band, color=viz.DIM_COLORS[0], alpha=0.16,
                        linewidth=0, zorder=1, label=f"bootstrap band, 2c = {band:.2f}")
        ax.plot(xs, xs + null95, color=viz.INK_SECONDARY, linestyle="--", linewidth=1.2,
                zorder=2, label=f"null 95%, {null95:.2f}")
        # no H1 class lives forever here, so crop away the empty infinity line
        pad = 0.06 * top
        ax.set_xlim(-pad, top + pad)
        ax.set_ylim(-pad, top + pad)
        legend = ax.legend(frameon=False, loc="lower right", fontsize=9)
        for text in legend.get_texts():
            text.set_color(viz.INK_SECONDARY)
    figure.suptitle(r"$H_1$ of the three Part 1 clouds: only a point above both lines is a feature",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    print("\nwrote", viz.save(figure, out("talk_04_signal_vs_noise.png")))

    print("\nVerdict")
    ok = True
    ok &= check("the loop's H1 bar beats the null and lies outside the bootstrap band",
                found["(b) a loop", 1] == (True, True))
    ok &= check("the blob's longest H1 bar passes neither test: stable nowhere",
                found["(a) a blob", 1] == (False, False))
    ok &= check("the two clusters have no H1 feature either",
                found["(c) two clusters", 1] == (False, False))
    ok &= check("the coordinate-shuffle null misses the clusters' H0 gap; the bootstrap sees it",
                found["(c) two clusters", 0] == (False, True))
    ok &= check("'most stable scale' still names a non-trivial Betti vector on the blob",
                found["(a) a blob", "scale"][1] != [1, 0])
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


if __name__ == "__main__":
    main()
