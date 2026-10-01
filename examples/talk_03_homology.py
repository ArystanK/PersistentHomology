"""Talk part 3 - "Homology as linear algebra".

Every claim in this part of the talk is a statement about two sparse matrices
of 0s and +-1s, so every claim here is checked by multiplying them out.

    chains are vectors        C_k = R^(n_k)
    the boundary is a matrix  d_k, of shape (n_(k-1), n_k)
    d o d = 0                 d_k d_(k+1) = 0, literally
    Betti numbers             beta_k = nullity d_k - rank d_(k+1)

The running example is the triangle that runs through the whole talk: hollow it
has one loop, filled it has none, and the difference is one column in d_2.
Then the tetrahedron boundary for a void in dimension 2, the three sanity
checks, and the table of reference values from the end of the part.

Orientation convention, as on the slide: vertices carry a fixed total order and
every simplex is written with its vertices increasing.  That fixes the signs and
there is nothing else to remember.  The edge order used below is the
lexicographic one, ab / ac / bc, which is what the code sorts to; the slide
writes the same three edges as e1 = ab, e2 = bc, e3 = ac, so its middle and last
columns are swapped relative to these.  Nothing but the labels changes.
"""

from _common import check, header, out, print_matrix

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import SimplicialComplex
from persistent_homology import plotting as viz

VERTEX_NAMES = "abcdefg"


def name(simplex) -> str:
    return "".join(VERTEX_NAMES[v] for v in simplex)


def labels(complex_, k: int):
    return [name(s) for s in complex_.simplices(k)]


def show_boundary(complex_, k: int, title: str) -> np.ndarray:
    matrix = complex_.boundary_matrix(k)
    print(f"\n  {title}   shape {matrix.shape}  "
          f"(rows: {k - 1}-simplices, columns: {k}-simplices)")
    print_matrix(matrix, rows=labels(complex_, k - 1), cols=labels(complex_, k))
    return matrix


def main() -> None:
    header("Talk part 3: homology as linear algebra")

    hollow = SimplicialComplex([(0, 1), (0, 2), (1, 2)])
    filled = SimplicialComplex([(0, 1, 2)])

    # ------------------------------------------------- chains are vectors
    print("\nChains are vectors")
    for k, space in enumerate(filled.counts()):
        print(f"  C{k} = R^{space}   basis: {', '.join(labels(filled, k))}")
    print("  A 1-chain is a weighted set of edges -- a flow on the network.")

    # ------------------------------------------------ the boundary matrices
    print("\nThe boundary operator is a matrix")
    d1 = show_boundary(filled, 1, "d_1")
    print("\n  Read a column: d_1(ab) = b - a, head minus tail.  For k = 1 this")
    print("  is exactly the signed incidence matrix B of the graph.")
    d2 = show_boundary(filled, 2, "d_2")
    print("\n  d_2(abc) = [bc] - [ac] + [ab]: delete each vertex in turn, alternate signs.")

    # -------------------------------------------------------- d o d = 0
    print("\nd o d = 0 is a matrix product")
    product = d1 @ d2
    print_matrix(product, rows=labels(filled, 0), cols=labels(filled, 2))
    print("  Every codimension-2 face is produced twice, with opposite signs,")
    print("  and cancels.  Check it on something bigger than a triangle:")

    for label, complex_ in [("tetrahedron boundary", SimplicialComplex.boundary_of_simplex(4)),
                            ("solid 4-simplex", SimplicialComplex([(0, 1, 2, 3, 4)]))]:
        worst = max(
            float(np.abs(complex_.boundary_matrix(k)
                         @ complex_.boundary_matrix(k + 1)).max(initial=0.0))
            for k in range(1, complex_.dim + 1)
        )
        print(f"    {label:22s} n = {complex_.counts()},  "
              f"max |d_k d_(k+1)| over all k = {worst:g}")

    # --------------------------------------------------- hollow vs filled
    print("\nHollow versus filled")
    print(f"  {'':10s} {'n0':>3s} {'n1':>3s} {'n2':>3s}  {'rk d1':>6s} {'rk d2':>6s}"
          f"  {'b0':>3s} {'b1':>3s}")
    for label, complex_ in (("hollow", hollow), ("filled", filled)):
        counts = complex_.counts() + [0, 0]
        betti = complex_.betti_numbers() + [0, 0]
        print(f"  {label:10s} {counts[0]:3d} {counts[1]:3d} {counts[2]:3d}"
              f"  {complex_.rank(1):6d} {complex_.rank(2):6d}"
              f"  {betti[0]:3d} {betti[1]:3d}")

    cycle = hollow.harmonic_basis(1)[:, 0]
    cycle = cycle / np.abs(cycle).max()
    print(f"\n  The one cycle, as a vector over ({', '.join(labels(hollow, 1))}):"
          f" {np.round(cycle, 6)}")
    print("  Up to sign that is ab - ac + bc: go a -> b -> c and come back along ac.")
    print("  Filling the triangle puts that exact vector into the image of d_2,")
    print(f"  since d_2(abc) = {np.round(filled.boundary_matrix(2).ravel(), 6)}"
          f" in the same basis, so the cycle now bounds and b1 drops to 0.")

    # ----------------------------------------------------- the tetrahedron
    print("\nBoundary of a tetrahedron: a 2-sphere")
    sphere = SimplicialComplex.boundary_of_simplex(4)
    print(f"  counts n = {sphere.counts()} (4 vertices, 6 edges, 4 triangles, no solid)")
    print(f"  rank d1 = {sphere.rank(1)} (connected: n0 - 1)")
    print(f"  rank d2 = {sphere.rank(2)} (the four triangle columns sum to zero)")
    print(f"  b0 = {sphere.count(0)} - {sphere.rank(1)} = {sphere.betti_number(0)}")
    print(f"  b1 = ({sphere.count(1)} - {sphere.rank(1)}) - {sphere.rank(2)} "
          f"= {sphere.betti_number(1)}   (no loops on a sphere)")
    print(f"  b2 = ({sphere.count(2)} - {sphere.rank(2)}) - 0 "
          f"= {sphere.betti_number(2)}   (one enclosed void)")
    counts, betti = sphere.counts(), sphere.betti_numbers()
    print(f"  Euler: {counts[0]} - {counts[1]} + {counts[2]} "
          f"= {sphere.euler_characteristic()} "
          f"= {betti[0]} - {betti[1]} + {betti[2]}")

    solid = SimplicialComplex([(0, 1, 2, 3)])
    print(f"  Add the solid tetrahedron and b2 drops to "
          f"{solid.betti_number(2)}: the k = 2 analogue of filling the triangle.")

    # ------------------------------------------------- three sanity checks
    print("\nSanity check 1: b0 counts connected components")
    graph = SimplicialComplex.from_graph(9, [(0, 1), (1, 2), (0, 2), (3, 4), (5, 6), (6, 7)])
    components = _components(9, graph.simplices(1))
    print(f"  a graph with {graph.count(0)} vertices and {graph.count(1)} edges: "
          f"rank d1 = {graph.rank(1)} = n0 - c = {graph.count(0)} - {components}")
    print(f"  b0 = {graph.betti_number(0)}, components found by a walk = {components}")

    print("\nSanity check 2: b1 of a graph is its cycle rank, E - V + C")
    print(f"  {'graph':>22s}  {'V':>3s} {'E':>3s} {'C':>3s}  {'b1':>3s}  {'E-V+C':>6s}")
    graphs = {
        "triangle": SimplicialComplex.cycle(3),
        "hexagon": SimplicialComplex.cycle(6),
        "two triangles + tail": graph,
        "complete graph K5": SimplicialComplex.from_graph(
            5, [(i, j) for i in range(5) for j in range(i + 1, 5)]),
    }
    for label, complex_ in graphs.items():
        v, e = complex_.count(0), complex_.count(1)
        c = _components(v, complex_.simplices(1))
        print(f"  {label:>22s}  {v:3d} {e:3d} {c:3d}  {complex_.betti_number(1):3d}"
              f"  {e - v + c:6d}")
    print("  E - V + C is the number of edges outside a spanning forest -- what")
    print("  Kruskal's algorithm leaves out.")

    print("\nSanity check 3: the Euler characteristic")
    print(f"  {'complex':>22s}  {'counts':>16s}  {'sum (-1)^k n_k':>14s}"
          f"  {'sum (-1)^k b_k':>14s}")
    zoo = {
        "hollow triangle": hollow,
        "filled triangle": filled,
        "tetrahedron boundary": sphere,
        "solid tetrahedron": solid,
        "7-vertex torus": _torus(),
        "hexagonal disk": _disk(),
    }
    euler_ok = True
    for label, complex_ in zoo.items():
        from_counts = complex_.euler_characteristic()
        from_betti = sum((-1) ** k * b for k, b in enumerate(complex_.betti_numbers()))
        euler_ok &= from_counts == from_betti
        print(f"  {label:>22s}  {str(complex_.counts()):>16s}  {from_counts:14d}"
              f"  {from_betti:14d}")

    # ------------------------------------------------- reference values
    print("\nReference values from the end of the part")
    print(f"  {'space':>22s}  {'triangulation':>18s}  {'measured':>12s}  {'slide':>10s}")
    reference = [
        ("circle", SimplicialComplex.cycle(8), (1, 1, 0)),
        ("sphere S^2", sphere, (1, 0, 1)),
        ("torus", _torus(), (1, 2, 1)),
        ("disk", _disk(), (1, 0, 0)),
    ]
    reference_ok = True
    for label, complex_, expected in reference:
        betti = tuple((complex_.betti_numbers() + [0, 0, 0])[:3])
        reference_ok &= betti == expected
        print(f"  {label:>22s}  {str(complex_.counts()):>18s}  {str(betti):>12s}"
              f"  {str(expected):>10s}")

    # ---------------------------------------------------------------- figure
    # Drawn in the slide's edge order, e1 = ab, e2 = bc, e3 = ac, so every
    # matrix in the picture is the one on the board, sign for sign.
    slide_order = [0, 2, 1]
    edge_names = ["$e_1$=ab", "$e_2$=bc", "$e_3$=ac"]
    d1_slide = d1[:, slide_order]
    d2_slide = d2[slide_order, :]
    z = hollow.harmonic_basis(1)[:, 0]
    z = np.sign(z @ d2.ravel()) * z / np.abs(z).max()
    z_slide = z[slide_order]
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]])

    figure, axes = plt.subplots(1, 4, figsize=(14.0, 4.3), facecolor=viz.SURFACE,
                                gridspec_kw={"width_ratios": [1.1, 1.0, 0.8, 1.1]})
    viz.plot_chain(corners, hollow, z, axes[0],
                   title=r"hollow: the cycle $z = e_1 + e_2 - e_3$")
    axes[0].set_xlabel(_counts_caption(hollow), color=viz.INK_SECONDARY, fontsize=9)
    _label_vertices(axes[0], corners)

    _draw_matrix(axes[1], d1_slide, ["a", "b", "c"], edge_names,
                 rf"$\partial_1$: rank {hollow.rank(1)}, nullity "
                 rf"{hollow.count(1) - hollow.rank(1)}")
    axes[1].set_xlabel(r"$\ker\partial_1$ is spanned by $z$: $\partial_1 z = $"
                       f"{_as_ints(d1_slide @ z_slide)}",
                       color=viz.INK_SECONDARY, fontsize=9)

    _draw_matrix(axes[2], np.column_stack([d2_slide.ravel(), z_slide]),
                 edge_names, [r"$\partial_2 t$", "$z$"],
                 r"filling in $t$: $\partial_2 t = z$")
    axes[2].set_xlabel(r"$\partial_1\partial_2 = $"
                       f"{_as_ints(product.ravel())}",
                       color=viz.INK_SECONDARY, fontsize=9)

    viz.plot_complex(corners, filled, axes[3], title="filled: the cycle bounds")
    _label_vertices(axes[3], corners)

    figure.suptitle(r"$\beta_1 = \mathrm{nullity}\,\partial_1 - \mathrm{rank}\,\partial_2$:"
                    r" $1 - 0 = 1$ hollow, $1 - 1 = 0$ filled",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    print("\nwrote", viz.save(figure, out("talk_03_homology.png")))

    # --------------------------------------------------------- the checks
    print("\nVerdict")
    ok = True
    ok &= check("d_1 d_2 = 0 on the filled triangle", np.abs(product).max() == 0)
    ok &= check("hollow triangle: rank d1 = 2, b0 = 1, b1 = 1",
                hollow.rank(1) == 2 and hollow.betti_numbers() == [1, 1])
    ok &= check("filled triangle: rank d2 = 1, b1 = 0",
                filled.rank(2) == 1 and filled.betti_number(1) == 0)
    ok &= check("tetrahedron boundary: b = (1, 0, 1), Euler = 2",
                sphere.betti_numbers() == [1, 0, 1]
                and sphere.euler_characteristic() == 2)
    ok &= check("filling the solid kills the void: b2 = 0", solid.betti_number(2) == 0)
    ok &= check("b1 = E - V + C on every graph in the table",
                all(c.betti_number(1) == c.count(1) - c.count(0)
                    + _components(c.count(0), c.simplices(1))
                    for c in graphs.values()))
    ok &= check("sum (-1)^k n_k = sum (-1)^k b_k on all six complexes", euler_ok)
    ok &= check("circle (1,1,0), sphere (1,0,1), torus (1,2,1), disk (1,0,0)",
                reference_ok)
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


def _as_ints(vector) -> str:
    return "(" + ", ".join(str(int(round(v)) + 0) for v in vector) + ")"


def _counts_caption(complex_) -> str:
    counts = "  ".join(f"$n_{k}$={n}" for k, n in enumerate(complex_.counts()))
    betti = ", ".join(str(b) for b in complex_.betti_numbers())
    return f"{counts}\n$\\beta$ = ({betti})"


def _label_vertices(ax, points) -> None:
    centre = points.mean(axis=0)
    for label, point in zip(VERTEX_NAMES, points):
        offset = 0.13 * (point - centre) / np.linalg.norm(point - centre)
        ax.text(*(point + offset), label, color=viz.INK, fontsize=11,
                ha="center", va="center")
    ax.margins(0.18)


def _draw_matrix(ax, matrix, rows, cols, title: str) -> None:
    """A small 0/+-1 matrix drawn as a table: +1 and -1 get the two dimension
    colours, zeros stay blank, and every row and column keeps its simplex label.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_rows, n_cols = matrix.shape
    fills = {1: viz.DIM_COLORS[1], -1: viz.DIM_COLORS[0]}
    for i in range(n_rows):
        for j in range(n_cols):
            value = int(round(matrix[i, j]))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fills.get(value, viz.SURFACE),
                                       alpha=0.22 if value else 1.0,
                                       edgecolor=viz.GRID, linewidth=1.0))
            ax.text(j + 0.5, i + 0.5, f"{value:+d}" if value else "0",
                    ha="center", va="center", fontsize=12,
                    color=viz.INK if value else viz.INK_MUTED)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors=viz.INK_SECONDARY, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=viz.INK, fontsize=11, loc="left", pad=26)


def _components(n_vertices: int, edges) -> int:
    """Connected components by union-find -- an answer that owes nothing to
    linear algebra, so it is a real check on ``b0``.
    """
    parent = list(range(n_vertices))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        parent[find(u)] = find(v)
    return len({find(v) for v in range(n_vertices)})


def _torus() -> SimplicialComplex:
    """The 7-vertex Moebius torus, the smallest triangulation of a torus:
    triangles ``{i, i+1, i+3}`` and ``{i, i+2, i+3}`` mod 7.
    """
    return SimplicialComplex(
        [(i % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)]
        + [(i % 7, (i + 2) % 7, (i + 3) % 7) for i in range(7)]
    )


def _disk() -> SimplicialComplex:
    """A hexagon triangulated as a fan from one interior vertex."""
    return SimplicialComplex([(0, i, i + 1) for i in range(1, 6)])


if __name__ == "__main__":
    main()
