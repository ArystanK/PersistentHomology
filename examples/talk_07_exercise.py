"""Talk part 7 - the closing exercise, worked, plus the appendix figure-eight.

The exercise on the slide:

    K has vertices 1..5; edges 12, 23, 13, 34, 45, 35; and one triangle, 123.
    The right-hand triangle 345 is NOT filled.

    1. Write down d_1 (5 x 6) and d_2 (6 x 1).
    2. Check d_1 d_2 = 0.
    3. Compute rank d_1 and rank d_2.
    4. Give b0 and b1.
    5. Verify with the Euler characteristic.

    Answers: rank d_1 = 4, rank d_2 = 1; b0 = 5 - 4 = 1, b1 = (6 - 4) - 1 = 1.
             Euler: 5 - 6 + 1 = 0 = b0 - b1.

Every one of those is recomputed below, then the same answer is reached a
second time through the Hodge Laplacian, so the two halves of the talk are seen
to agree on the same small example.

The appendix asks for a figure-eight -- two loops sharing a single vertex,
b1 = 2 -- which is at the end, both as an abstract complex and as a point cloud
put through the full Rips pipeline.
"""

from _common import check, header, out, print_matrix

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import SimplicialComplex, format_spectrum
from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence

# Vertices are 1..5 on the slide and 0..4 in the code, so print them shifted.
LAYOUT = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [2.0, 0.0]])


def label(simplex) -> str:
    return "".join(str(v + 1) for v in simplex)


def labels(complex_, k: int):
    return [label(s) for s in complex_.simplices(k)]


def exercise_complex() -> SimplicialComplex:
    """Edges 12, 23, 13, 34, 45, 35 and the single triangle 123, zero-indexed."""
    return SimplicialComplex([(0, 1, 2), (2, 3), (3, 4), (2, 4)])


def main() -> None:
    header("Talk part 7: the exercise, worked")
    K = exercise_complex()

    print("\nThe complex")
    print(f"  vertices: {', '.join(labels(K, 0))}")
    print(f"  edges:    {', '.join(labels(K, 1))}")
    print(f"  triangle: {', '.join(labels(K, 2))}   (345 is left empty)")
    print(f"  counts:   n0 = {K.count(0)}, n1 = {K.count(1)}, n2 = {K.count(2)}")

    # ------------------------------------------------------------- step 1
    print("\n1. The boundary matrices")
    d1, d2 = K.boundary_matrix(1), K.boundary_matrix(2)
    print(f"\n  d_1, shape {d1.shape}:")
    print_matrix(d1, rows=labels(K, 0), cols=labels(K, 1))
    print(f"\n  d_2, shape {d2.shape}:")
    print_matrix(d2, rows=labels(K, 1), cols=labels(K, 2))
    print("\n  d_2(123) = [23] - [13] + [12]: the only triangle, so the only column.")

    # ------------------------------------------------------------- step 2
    print("\n2. Check d_1 d_2 = 0")
    product = d1 @ d2
    print_matrix(product, rows=labels(K, 0), cols=labels(K, 2))
    print(f"  max |d_1 d_2| = {np.abs(product).max():g}.  Every boundary is a cycle.")

    # ------------------------------------------------------------- step 3
    print("\n3. The ranks")
    print(f"  rank d_1 = {K.rank(1)}   (5 vertices, connected, so n0 - 1 = 4)")
    print(f"  rank d_2 = {K.rank(2)}   (one triangle, and its column is not zero)")

    # ------------------------------------------------------------- step 4
    print("\n4. The Betti numbers")
    print(f"  b0 = n0 - rank d_1          = {K.count(0)} - {K.rank(1)} "
          f"= {K.betti_number(0)}")
    print(f"  b1 = (n1 - rank d_1) - rank d_2 = ({K.count(1)} - {K.rank(1)}) "
          f"- {K.rank(2)} = {K.betti_number(1)}")
    print("  One component, one loop.  The filled triangle 123 contributes no hole;")
    print("  the empty triangle 345 does.")

    # ------------------------------------------------------------- step 5
    print("\n5. The Euler characteristic")
    counts = K.counts()
    betti = K.betti_numbers()
    print(f"  sum (-1)^k n_k = {counts[0]} - {counts[1]} + {counts[2]} "
          f"= {K.euler_characteristic()}")
    print(f"  sum (-1)^k b_k = {betti[0]} - {betti[1]} = {betti[0] - betti[1]}")

    # -------------------------------------------------- the same, spectrally
    print("\nThe same answer through Part 6's Laplacian")
    for k in (0, 1):
        print(f"  L_{k}: spectrum {format_spectrum(K.spectrum(k))}")
        print(f"       dim ker = {K.betti_from_laplacian(k)} = b{k}")

    harmonic = K.harmonic_basis(1)[:, 0]
    harmonic = harmonic / np.abs(harmonic).max()
    print("\n  And the loop itself, as the harmonic chain over "
          f"({', '.join(labels(K, 1))}):")
    print("   ", np.round(harmonic, 4))
    carriers = [labels(K, 1)[i] for i in np.flatnonzero(np.abs(harmonic) > 1e-9)]
    print(f"  Non-zero exactly on {', '.join(carriers)} -- the empty triangle, and")
    print("  nothing else.  The filled side of the complex carries no circulation.")

    # ---------------------------------------------- appendix: figure-eight
    print("\nAppendix: the figure-eight, b1 = 2")
    eight = SimplicialComplex([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)])
    print(f"  two hollow triangles sharing vertex 3: n = {eight.counts()}, "
          f"b = {eight.betti_numbers()}")
    print(f"  Euler: {eight.euler_characteristic()} "
          f"= {eight.betti_numbers()[0]} - {eight.betti_numbers()[1]}")
    print(f"  dim ker L_1 = {eight.betti_from_laplacian(1)}, so the two loops show up")
    print("  as a two-dimensional kernel; any basis of it is a pair of independent")
    print("  cycles, and the minimum-norm one puts each loop on its own triangle.")

    print("\n  The same shape as a point cloud, through the full Rips pipeline:")
    cloud = ds.figure_eight(110, seed=6)
    cloud_result = rips_persistence(cloud, max_dim=1, threshold=2.0)
    scale = cloud_result.most_stable_scale(2.0)
    measured = cloud_result.betti_numbers(scale)
    print(f"  {len(cloud)} points, most stable scale {scale:.3f}, "
          f"b = {tuple(measured)}")
    top = cloud_result.most_persistent(1, k=3)
    print(f"  all {len(cloud_result[1])} H1 bars: "
          + ", ".join(f"[{b:.3f}, {d:.3f})" for b, d in top))
    print("  Two long bars and nothing else: two loops, exactly as the abstract")
    print("  complex says, recovered from nothing but pairwise distances.")

    # ---------------------------------------------------------------- figure
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), facecolor=viz.SURFACE)
    viz.plot_complex(LAYOUT, K, axes[0], title="the exercise complex")
    for index, (x, y) in enumerate(LAYOUT):
        axes[0].annotate(str(index + 1), (x, y), textcoords="offset points",
                         xytext=(6, 6), fontsize=11, color=viz.INK)
    viz.plot_chain(LAYOUT, K, K.harmonic_basis(1)[:, 0], axes[1],
                   title="its harmonic loop: only the empty triangle")
    viz.barcode(cloud_result, axes[2], title="figure-eight cloud: two long H1 bars")
    figure.suptitle("The exercise, and the appendix figure-eight",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    print("\nwrote", viz.save(figure, out("talk_07_exercise.png")))

    # ------------------------------------------------------------ the checks
    print("\nVerdict")
    ok = True
    ok &= check("d_1 is 5 x 6 and d_2 is 6 x 1",
                d1.shape == (5, 6) and d2.shape == (6, 1))
    ok &= check("d_1 d_2 = 0", np.abs(product).max() == 0)
    ok &= check("rank d_1 = 4 and rank d_2 = 1",
                K.rank(1) == 4 and K.rank(2) == 1)
    ok &= check("b0 = 1 and b1 = 1", K.betti_numbers() == [1, 1, 0])
    ok &= check("Euler: 5 - 6 + 1 = 0 = b0 - b1",
                K.euler_characteristic() == 0
                and K.euler_characteristic() == betti[0] - betti[1])
    ok &= check("the Laplacian agrees: dim ker L_0 = 1, dim ker L_1 = 1",
                [K.betti_from_laplacian(k) for k in (0, 1)] == [1, 1])
    ok &= check("the harmonic loop lives entirely on the empty triangle 345",
                sorted(carriers) == ["34", "35", "45"])
    ok &= check("the figure-eight has b1 = 2, abstractly and from a point cloud",
                eight.betti_numbers() == [1, 2] and tuple(measured) == (1, 2))
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


if __name__ == "__main__":
    main()
