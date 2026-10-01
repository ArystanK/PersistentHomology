"""Talk part 2 - "From points to complexes".

Six points on the unit circle, the example that runs through the whole talk.
Their only three distances are 1 (adjacent), sqrt(3) (short diagonal) and 2
(antipodal), so the Vietoris-Rips complex changes at exactly three moments, and
the slide's three pictures are the three regimes:

    eps < 1          dust: six isolated points
    1 <= eps < sqrt3 the hexagon closes, and the loop appears
    sqrt3 <= eps < 2 the triangles fill in, and the loop is gone

The script also builds the Cech complex of the same points, because the slide
claims Cech is the one with the right topology and Rips is the cheap
approximation.  That claim is worth seeing: at eps = sqrt(3) the two complexes
disagree, Rips has lost the loop and Cech has not, and the union of balls Cech
is modelled on plainly still has a hole in the middle.

Finally the "which epsilon?" slide: the same three regimes read as dust, loop
and blob, which is the problem persistence exists to solve.
"""

from _common import check, header, out, print_matrix

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import SimplicialComplex, pairwise_distances
from persistent_homology import plotting as viz
from persistent_homology.complexes import rips_filtration

SQRT3 = float(np.sqrt(3))


def hexagon() -> np.ndarray:
    """Six points spaced evenly on the unit circle."""
    angles = np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])


# --------------------------------------------------------------------- Cech

def enclosing_ball_radius(points: np.ndarray) -> float:
    """Radius of the smallest ball containing up to three points of the plane.

    One, two or three points only -- which is all the complexes here need, and
    all that can be drawn.  For three points the answer is the circumcircle when
    the triangle is acute, and the longest-edge ball when it is obtuse, because
    an obtuse triangle's circumcentre lies outside it.

    Note how much more this needs than Rips does: coordinates, not just the
    distance matrix.  That is the whole cost argument on the slide.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 1:
        return 0.0
    if len(points) == 2:
        return float(np.linalg.norm(points[0] - points[1]) / 2)
    if len(points) != 3:
        raise NotImplementedError("only up to three points, i.e. up to triangles")

    a, b, c = points
    sides = np.array([
        np.linalg.norm(b - c), np.linalg.norm(a - c), np.linalg.norm(a - b)
    ])
    longest = sides.max()
    # obtuse (or right) exactly when the longest side squared dominates
    if longest ** 2 >= (sides ** 2).sum() - longest ** 2:
        return float(longest / 2)

    u, v = b - a, c - a
    area = abs(u[0] * v[1] - u[1] * v[0]) / 2
    return float(sides.prod() / (4 * area))          # circumradius


def cech_complex(points: np.ndarray, epsilon: float, max_dim: int = 2,
                 tol: float = 1e-9) -> SimplicialComplex:
    """The Cech complex at scale ``epsilon``, up to dimension ``max_dim``.

    A set of points spans a simplex when the balls of radius ``epsilon / 2``
    around them share a common point -- equivalently, when the smallest ball
    enclosing them has radius at most ``epsilon / 2``.  By the nerve theorem
    this complex has *exactly* the topology of that union of balls.

    Written out by brute force over all subsets, which is honest about the cost.
    """
    from itertools import combinations

    points = np.asarray(points, dtype=float)
    simplices = [(i,) for i in range(len(points))]
    for size in range(2, max_dim + 2):
        for subset in combinations(range(len(points)), size):
            if enclosing_ball_radius(points[list(subset)]) <= epsilon / 2 + tol:
                simplices.append(subset)
    return SimplicialComplex(simplices)


# ---------------------------------------------------------------------- main

def main() -> None:
    header("Talk part 2: from points to complexes")
    points = hexagon()
    distances = pairwise_distances(points)

    print("\nThe input is a finite metric space and nothing else")
    print(f"  {len(points)} points, {len(points) * (len(points) - 1) // 2} distances, "
          f"taking only these values:")
    print("   ", np.unique(np.round(distances, 6)))
    print("  1 = adjacent, sqrt(3) = short diagonal, 2 = antipodal.")

    print("\nThe distance matrix (this is the entire input to Rips)")
    print_matrix(distances.round(0), rows=[f"p{i}" for i in range(6)],
                 cols=[f"p{j}" for j in range(6)], width=4)
    print("  (rounded; the sqrt(3) entries print as 2)")

    # ---------------------------------------------------------- Rips regimes
    filtration = rips_filtration(distances, max_dim=3, threshold=2.0)
    regimes = [
        ("eps < 1", 0.5, "dust"),
        ("1 <= eps < sqrt3", 1.0, "the loop"),
        ("sqrt3 <= eps < 2", SQRT3, "blob (and a spurious void)"),
    ]

    print("\nThe Vietoris-Rips complex, one regime at a time")
    print(f"  {'regime':20s} {'eps':>6s}  {'simplex counts':>22s}  {'betti':>14s}  reading")
    complexes = {}
    for label, epsilon, reading in regimes:
        complex_ = SimplicialComplex.from_filtration(filtration, epsilon + 1e-9)
        complexes[label] = (epsilon, complex_)
        counts = "  ".join(f"n{k}={n}" for k, n in enumerate(complex_.counts()))
        betti = "(" + ", ".join(str(b) for b in complex_.betti_numbers()) + ")"
        print(f"  {label:20s} {epsilon:6.3f}  {counts:>22s}  {betti:>14s}  {reading}")

    print("\n  At eps = sqrt(3) the complex is 6 vertices, 12 edges, 8 triangles:")
    print("  combinatorially the boundary of an octahedron.  b1 has dropped to 0,")
    print("  and a b2 = 1 has appeared that no six points in the plane can justify.")
    print("  That is a Rips artifact, and the 'reading a diagram honestly' slide")
    print("  is about exactly this bar.")

    # ----------------------------------------------------------- Rips vs Cech
    print("\nRips vs Cech on the same six points")
    print("  Both complexes are cut off at triangles, so only b0 and b1 are")
    print("  comparable -- the top dimension of a truncated complex counts cycles")
    print("  that the missing simplices would have filled in.")
    print(f"  {'eps':>6s}  {'Rips counts':>14s} {'(b0, b1)':>10s}"
          f"  {'Cech counts':>14s} {'(b0, b1)':>10s}")

    def first_two(complex_) -> tuple:
        betti = complex_.betti_numbers() + [0, 0]
        return betti[0], betti[1]

    disagreement = None
    for epsilon in (0.5, 1.0, 1.5, SQRT3, 1.9):
        rips = SimplicialComplex.from_filtration(filtration, epsilon + 1e-9)
        cech = cech_complex(points, epsilon)
        if first_two(rips) != first_two(cech) and disagreement is None:
            disagreement = (epsilon, first_two(rips), first_two(cech))
        print(f"  {epsilon:6.3f}  {'/'.join(str(n) for n in rips.counts()):>14s}"
              f" {str(first_two(rips)):>10s}"
              f"  {'/'.join(str(n) for n in cech.counts()):>14s}"
              f" {str(first_two(cech)):>10s}")

    if disagreement:
        epsilon, rips_betti, cech_betti = disagreement
        print(f"\n  They first disagree at eps = {epsilon:.3f}: "
              f"Rips says (b0, b1) = {rips_betti}, Cech says {cech_betti}.")
        print("  Cech is right about the union of balls -- at radius "
              f"{epsilon / 2:.3f} the six balls")
        print("  still do not reach the centre, which is 1.0 away from every point.")
        print("  Rips filled the hexagon in anyway, because it only ever looks at")
        print("  pairwise distances and never asks whether three balls actually meet.")

    print("\nThe classic three-point case, in one line")
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2]])   # equilateral, side 1
    circumradius = enclosing_ball_radius(triangle)
    print(f"  Unit equilateral triangle: all three distances are 1, so Rips puts the")
    print(f"  2-simplex in at eps = 1.  Cech needs balls of radius 0.5 to meet, and")
    print(f"  the smallest ball covering the three vertices has radius "
          f"{circumradius:.4f} > 0.5.")
    print(f"  So Rips_1 is a filled triangle and Cech_1 is a hollow one: "
          f"b1 = {SimplicialComplex([(0, 1, 2)]).betti_numbers()[1]} vs "
          f"{cech_complex(triangle, 1.0).betti_numbers()[1]}.")

    # ------------------------------------------------------------- which eps
    print("\nWhich epsilon?  The reason Part 4 exists")
    print(f"  {'eps':>6s}  {'b0':>4s} {'b1':>4s}  what you would report")
    for epsilon in (0.5, 0.99, 1.0, 1.5, SQRT3, 1.9, 2.0):
        complex_ = SimplicialComplex.from_filtration(filtration, epsilon + 1e-9)
        betti = complex_.betti_numbers() + [0, 0]
        if betti[0] > 1:
            verdict = f"{betti[0]} clusters (dust)"
        elif betti[1] == 1:
            verdict = "a circle"
        else:
            verdict = "one contractible blob"
        print(f"  {epsilon:6.3f}  {betti[0]:4d} {betti[1]:4d}  {verdict}")
    print("  Three different answers from one data set, and nothing in the data")
    print("  says which epsilon to take.  Part 4's answer: take all of them.")

    # ------------------------------------------------------------- the checks
    print("\nVerdict")
    ok = True
    ok &= check("eps < 1 gives six isolated points",
                complexes["eps < 1"][1].betti_numbers() == [6])
    ok &= check("eps = 1 closes the hexagon: b = (1, 1)",
                complexes["1 <= eps < sqrt3"][1].betti_numbers() == [1, 1])
    ok &= check("eps = sqrt(3) is the octahedron boundary: 6/12/8, b = (1, 0, 1)",
                complexes["sqrt3 <= eps < 2"][1].counts() == [6, 12, 8]
                and complexes["sqrt3 <= eps < 2"][1].betti_numbers() == [1, 0, 1])
    ok &= check("Cech still sees the loop at eps = sqrt(3), where Rips does not",
                cech_complex(points, SQRT3).betti_numbers()[1] == 1)
    ok &= check("Rips fills the unit equilateral triangle at eps = 1 and Cech does not",
                circumradius > 0.5)

    # ---------------------------------------------------------------- figure
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.4), facecolor=viz.SURFACE)
    for column, (label, epsilon, _) in enumerate(regimes):
        viz.plot_complex(points, complexes[label][1], axes[0, column],
                         title=f"Rips, {label}")
        axes[0, column].set_xlim(-1.4, 1.4)
        axes[0, column].set_ylim(-1.4, 1.4)
        # the union of balls the complex is a model of
        ax = axes[1, column]
        for point in points:
            ax.add_patch(plt.Circle(point, epsilon / 2, color=viz.DIM_COLORS[0],
                                    alpha=0.22, linewidth=0))
        ax.scatter(points[:, 0], points[:, 1], s=30, color=viz.INK, zorder=3)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"balls of radius eps/2 = {epsilon / 2:.2f}",
                     color=viz.INK, fontsize=11, loc="left", pad=8)
        viz.style_axes(ax)
        ax.grid(False)
    figure.suptitle("Six points on a circle: the complex, and the union of balls "
                    "it stands in for",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    print("\nwrote", viz.save(figure, out("talk_02_complexes.png")))

    # The same sweep as an animation, one PNG per frame for the animate package
    # on the Vietoris-Rips slide.  eps runs to 2.2, just past the last event at 2.
    from rips_animation import RipsScene, save_frames
    save_frames(RipsScene(points, 2.2), out("rips_hexagon_frames"), frames=90)
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


if __name__ == "__main__":
    main()
