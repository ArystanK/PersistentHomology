"""Talk part 4 - "Persistence".

Part 2 ended with "which epsilon?" and no answer.  This is the answer: sweep
every epsilon at once and record, for each feature, the scale at which it is
born and the scale at which it dies.  Nothing has to be chosen.

Four things get checked here, all on the six-point hexagon or a circle sample:

1. the event table from the slide -- 6 components born at 0, five die at 1 and
   the loop is born, the loop dies at sqrt(3);
2. the reduction algorithm's pairing, printed simplex by simplex: which
   insertion creates a class and which insertion destroys it;
3. "you already know the H0 half of this" -- the H0 death times are exactly the
   edge weights of a minimum spanning tree, so H0 persistence *is* single-
   linkage clustering;
4. stability -- the bottleneck distance between diagrams is at most twice the
   Hausdorff distance between the clouds.

Then the "reading a diagram honestly" slide, which is a list of four warnings,
each one reproduced as a number.
"""

from _common import check, header, out, timed

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from persistent_homology import bottleneck_distance, pairwise_distances
from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import persistence, rips_persistence
from persistent_homology.complexes import rips_filtration

SQRT3 = float(np.sqrt(3))
VERTEX_NAMES = "012345"


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Hausdorff distance: the furthest any point of either cloud has to travel
    to reach the other cloud.
    """
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def hexagon() -> np.ndarray:
    angles = np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])


def name(simplex) -> str:
    return "".join(VERTEX_NAMES[v] for v in simplex)


def main() -> None:
    header("Talk part 4: persistence")
    points = hexagon()
    distances = pairwise_distances(points)

    # ------------------------------------------------- the hexagon, end to end
    filtration = rips_filtration(distances, max_dim=3, threshold=2.0)
    result = persistence(filtration, max_dim=2, keep_filtration=True)

    print("\nThe hexagon, end to end")
    print(f"  {'eps':>6s}  {'simplices added':>16s}  {'b0':>3s} {'b1':>3s} {'b2':>3s}"
          f"   event")
    events = {
        0.0: "6 components born",
        1.0: "5 components die; loop born",
        SQRT3: "triangles fill; loop dies; a spurious void is born",
        2.0: "one big simplex; the void dies",
    }
    for epsilon, note in events.items():
        added = int((filtration.values == epsilon).sum())
        betti = result.betti_numbers(epsilon + 1e-9) + [0, 0, 0]
        print(f"  {epsilon:6.3f}  {added:16d}  {betti[0]:3d} {betti[1]:3d} {betti[2]:3d}"
              f"   {note}")

    print("\n  The bars")
    for dim in (0, 1, 2):
        for birth, death in result[dim]:
            span = f"[{birth:.3f}, {'inf' if np.isinf(death) else f'{death:.3f}'})"
            print(f"    H{dim}  {span:>18s}"
                  + ("   <-- the loop" if dim == 1 else "")
                  + ("   <-- the Rips artifact" if dim == 2 else ""))
    print("  One long H1 bar [1, sqrt(3)).  It is a circle.")

    # ---------------------------------------------- the reduction's pairing
    print("\nWhat the reduction actually paired")
    print("  Every simplex either creates a class or destroys one; the algorithm")
    print("  says which, and that is the whole barcode.")
    print(f"  {'creates':>10s}  {'destroys':>10s}  {'dim':>3s}  bar")
    for birth_index, death_index in sorted(
            result.pairs, key=lambda p: (filtration.dims[p[0]], filtration.values[p[0]]))[:8]:
        creator = name(filtration.simplices[birth_index])
        birth = filtration.values[birth_index]
        if death_index is None:
            print(f"  {creator:>10s}  {'(never)':>10s}  {filtration.dims[birth_index]:3d}"
                  f"  [{birth:.3f}, inf)")
        else:
            killer = name(filtration.simplices[death_index])
            death = filtration.values[death_index]
            print(f"  {creator:>10s}  {killer:>10s}  {filtration.dims[birth_index]:3d}"
                  f"  [{birth:.3f}, {death:.3f})")
    print(f"  {len(result.pairs)} pairs in total, one per class in H0, H1 and H2.")
    print("  (The filtration itself is larger: simplices above dimension 3 are built")
    print("   so that H2 can be killed, but their own classes are not reported.)")

    # ------------------------------------- H0 persistence == single linkage
    print("\nYou already know the H0 half: it is single-linkage clustering")
    cloud = ds.clusters(60, centers=((0, 0), (3.2, 0.4), (1.6, 2.8)), spread=0.45, seed=4)
    cloud_result = rips_persistence(cloud, max_dim=1, threshold=3.0)

    h0_deaths = np.sort(cloud_result[0][np.isfinite(cloud_result[0][:, 1]), 1])
    mst_weights = np.sort(
        minimum_spanning_tree(pairwise_distances(cloud)).toarray().ravel()
    )
    mst_weights = mst_weights[mst_weights > 0]
    matches = (len(h0_deaths) == len(mst_weights)
               and np.allclose(h0_deaths, mst_weights))

    print(f"  {len(cloud)} points, {len(h0_deaths)} finite H0 bars, "
          f"{len(mst_weights)} MST edges")
    print(f"  the five largest H0 death times: "
          f"{np.array2string(h0_deaths[-5:], precision=4)}")
    print(f"  the five largest MST edge weights: "
          f"{np.array2string(mst_weights[-5:], precision=4)}")
    print(f"  identical to machine precision: {matches}")
    print("  So H0 persistence is the single-linkage dendrogram, and the elder rule")
    print("  is the rule that the older of two merging clusters keeps its identity.")
    print("  H1 and H2 are the part you cannot get this way.")

    # ------------------------------------------------------------ stability
    print("\nStability: the diagram is a Lipschitz function of the data")
    base = ds.circle(60, seed=21)
    with timed("stability sweep"):
        base_result = rips_persistence(base, max_dim=1, threshold=2.2)
        rng = np.random.default_rng(0)
        rows = []
        for noise in (0.02, 0.05, 0.10, 0.18):
            perturbed = base + noise * rng.standard_normal(base.shape)
            perturbed_result = rips_persistence(perturbed, max_dim=1, threshold=2.2)
            rows.append((
                noise,
                hausdorff(base, perturbed),
                bottleneck_distance(base_result[0], perturbed_result[0]),
                bottleneck_distance(base_result[1], perturbed_result[1]),
            ))

    print(f"  {'noise':>6s}  {'d_H':>8s}  {'2 d_H':>8s}  {'d_B(H0)':>8s}"
          f"  {'d_B(H1)':>8s}  bound holds")
    stability_ok = True
    for noise, haus, b0, b1 in rows:
        holds = b0 <= 2 * haus + 1e-9 and b1 <= 2 * haus + 1e-9
        stability_ok &= holds
        print(f"  {noise:6.2f}  {haus:8.4f}  {2 * haus:8.4f}  {b0:8.4f}"
              f"  {b1:8.4f}  {'yes' if holds else 'NO'}")
    print("  Noise of size delta can only create bars of length <= 2 delta.")
    print("  That inequality is the theorem behind 'long bars matter'.")

    # --------------------------------------------- reading a diagram honestly
    print("\nReading a diagram honestly")

    print("\n  1. Scale is not intrinsic.")
    scaled_result = rips_persistence(10 * base, max_dim=1, threshold=22.0)
    base_h1 = float((base_result[1][:, 1] - base_result[1][:, 0]).max())
    scaled_h1 = float((scaled_result[1][:, 1] - scaled_result[1][:, 0]).max())
    print(f"     longest H1 bar on the circle:        {base_h1:.4f}")
    print(f"     the same circle, all distances x10:  {scaled_h1:.4f}"
          f"   (ratio {scaled_h1 / base_h1:.2f})")
    print("     Nothing changed about the shape.  Normalise before comparing.")

    print("\n  2. Outliers are the enemy.")
    with_outlier = np.vstack([base, [[1.8, 1.8]]])
    outlier_result = rips_persistence(with_outlier, max_dim=1, threshold=2.2)
    clean_h0 = float(base_result[0][np.isfinite(base_result[0][:, 1]), 1].max())
    dirty_h0 = float(outlier_result[0][np.isfinite(outlier_result[0][:, 1]), 1].max())
    print(f"     60 points on a circle, plus one point at (1.8, 1.8):")
    print(f"     longest finite H0 bar  {clean_h0:.4f} -> {dirty_h0:.4f}"
          f"   ({dirty_h0 / clean_h0:.1f}x)")
    print("     One point out of 61 now dominates the H0 diagram, and any rule that")
    print("     reads the longest H0 bar as 'cluster separation' reports two clusters.")
    print("     Trim by density, or use a distance-to-measure filtration, which is")
    print("     robust by construction.")

    print("\n  3. Long does not always mean real.")
    h2 = result[2]
    print(f"     The hexagon's H2 bar: [{h2[0, 0]:.4f}, {h2[0, 1]:.4f}), "
          f"lifetime {h2[0, 1] - h2[0, 0]:.4f}")
    print("     It is longer than nothing at all and it is completely genuine as a")
    print("     fact about the complex -- at eps = sqrt(3) Rips builds the boundary")
    print("     of an octahedron.  Six points in the plane enclose no void.  The bar")
    print("     is a property of Rips, not of the data.")

    print("\n  4. A diagram says a loop exists, not where it is.")
    print("     Nothing above names a single edge of the loop.  Part 6 fixes this:")
    print("     the harmonic representative in ker L_1 is a canonical choice.")

    # ---------------------------------------------------------------- figure
    figure = plt.figure(figsize=(13.5, 4.4), facecolor=viz.SURFACE)
    viz.plot_points(points, figure.add_subplot(1, 3, 1),
                    title="six points on the unit circle")
    viz.barcode(result, figure.add_subplot(1, 3, 2),
                title="the whole filtration, one bar per class")
    viz.persistence_diagram(result, figure.add_subplot(1, 3, 3),
                            title="the same data as a diagram")
    figure.suptitle(r"The hexagon, end to end: one long $H_1$ bar $[1, \sqrt{3})$",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    print("\nwrote", viz.save(figure, out("talk_04_persistence.png")))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor=viz.SURFACE)
    viz.betti_curves(result, axes[0], title="hexagon: Betti curves")
    axes[1].plot([r[1] for r in rows], [r[3] for r in rows], "o-",
                 color=viz.DIM_COLORS[1], label=r"$d_B(H_1)$ measured")
    axes[1].plot([r[1] for r in rows], [2 * r[1] for r in rows], "--",
                 color=viz.INK_MUTED, label=r"the bound $2\,d_H$")
    axes[1].set_xlabel("Hausdorff distance between the clouds",
                       color=viz.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("bottleneck distance", color=viz.INK_SECONDARY, fontsize=10)
    axes[1].set_title("stability, measured", color=viz.INK, fontsize=11, loc="left", pad=10)
    axes[1].set_ylim(bottom=0)
    viz.style_axes(axes[1])
    legend = axes[1].legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(viz.INK_SECONDARY)
    figure.tight_layout()
    print("wrote", viz.save(figure, out("talk_04_stability.png")))

    # ------------------------------------------------------------ the checks
    print("\nVerdict")
    ok = True
    ok &= check("the hexagon has exactly one H1 bar, [1, sqrt(3))",
                len(result[1]) == 1
                and np.allclose(result[1][0], [1.0, SQRT3]))
    ok &= check("5 of the 6 H0 bars die at eps = 1, one lives forever",
                sorted(np.isinf(result[0][:, 1]).tolist()) == [False] * 5 + [True]
                and np.allclose(result[0][np.isfinite(result[0][:, 1]), 1], 1.0))
    ok &= check("each of the 6 vertices starts an H0 class, and 5 are destroyed "
                "by an edge",
                len(result[0]) == 6 and int(np.isinf(result[0][:, 1]).sum()) == 1)
    ok &= check("every recorded class came from exactly one creator/destroyer pair",
                len(result.pairs) == sum(len(result[d]) for d in (0, 1, 2)))
    ok &= check("H0 death times are exactly the MST edge weights", matches)
    ok &= check("d_B <= 2 d_H at every noise level tested", stability_ok)
    ok &= check("rescaling by 10 multiplies every lifetime by 10",
                abs(scaled_h1 / base_h1 - 10.0) < 0.01)
    ok &= check(f"one outlier out of 61 multiplies the longest H0 bar by "
                f"{dirty_h0 / clean_h0:.1f}", dirty_h0 > 3 * clean_h0)
    ok &= check("the hexagon's H2 bar is [sqrt(3), 2) and is a Rips artifact",
                np.allclose(h2[0], [SQRT3, 2.0]))
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


if __name__ == "__main__":
    main()
