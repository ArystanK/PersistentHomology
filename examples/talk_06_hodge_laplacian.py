"""Talk part 6 - "The Hodge Laplacian".

Part 3 made a promise: the fact you already use, that the kernel dimension of
the graph Laplacian counts connected components, holds at every level k for a
matrix L_k that specialises to the graph Laplacian at k = 0.  This script
redeems it.

    L_k = d_k^T d_k  +  d_(k+1) d_(k+1)^T        symmetric, PSD, sparse
    L_0 = d_1 d_1^T  =  D - A                    the graph Laplacian
    beta_k = dim ker L_k                         counting holes is an eigenvalue
                                                 multiplicity

Five things get checked:

1. L_0 really is D - A, on several complexes;
2. the triangle: spectra {0, 3, 3} hollow and {3, 3, 3} filled, so filling it
   pushes the zero eigenvalue up to 3;
3. beta_k = dim ker L_k agrees with the rank-nullity answer everywhere;
4. the harmonic representative -- ker L_1 is a subspace, not a quotient, so
   "where is the loop?" has a canonical answer, which the barcode does not give;
5. subdivision: beta_1 never moves, the spectrum always does, and the spectral
   gap goes 3 -> 1 -> 0.268, which is the slide's "the spectrum sees more than
   beta_k does".

Then HodgeRank, which is the same decomposition applied to pairwise comparisons.
"""

from _common import check, header, out, print_matrix

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import SimplicialComplex, format_spectrum
from persistent_homology import plotting as viz

VERTEX_NAMES = "abcdefghijkl"


def name(simplex) -> str:
    return "".join(VERTEX_NAMES[v] for v in simplex)


def labels(complex_, k: int):
    return [name(s) for s in complex_.simplices(k)]


def polygon(n: int) -> np.ndarray:
    angles = np.arange(n) * 2 * np.pi / n + np.pi / 2
    return np.column_stack([np.cos(angles), np.sin(angles)])


def main() -> None:
    header("Talk part 6: the Hodge Laplacian")

    hollow = SimplicialComplex([(0, 1), (0, 2), (1, 2)])
    filled = SimplicialComplex([(0, 1, 2)])

    # ------------------------------------------------------- L_0 is D - A
    print("\nL_0 is the graph Laplacian you already use")
    print(f"  {'complex':>24s}  {'n0':>3s}  L_0 == D - A")
    zoo = {
        "hollow triangle": hollow,
        "hexagon (cycle graph)": SimplicialComplex.cycle(6),
        "tetrahedron boundary": SimplicialComplex.boundary_of_simplex(4),
        "path on 3 vertices": SimplicialComplex.from_graph(3, [(0, 1), (1, 2)]),
        "two disjoint triangles": SimplicialComplex([(0, 1, 2), (3, 4, 5)]),
    }
    laplacian_ok = True
    for label, complex_ in zoo.items():
        agrees = np.allclose(complex_.hodge_laplacian(0), complex_.graph_laplacian())
        laplacian_ok &= agrees
        print(f"  {label:>24s}  {complex_.count(0):3d}  {agrees}")

    print("\n  L_0 of the hollow triangle, built as d_1 d_1^T:")
    print_matrix(hollow.hodge_laplacian(0), rows=labels(hollow, 0),
                 cols=labels(hollow, 0))
    print(f"  spectrum {format_spectrum(hollow.spectrum(0))}, "
          f"dim ker = {hollow.betti_from_laplacian(0)} = b0.  "
          f"This is 3I - J.")

    print("\n  Path graph on 3 vertices, the slide's quick check:")
    path = zoo["path on 3 vertices"]
    print(f"  spectrum of L_0 is {format_spectrum(path.spectrum(0))}, "
          f"dim ker = {path.betti_from_laplacian(0)}: one component.")

    # -------------------------------------------------- back to the triangle
    print("\nBack to the triangle")
    print("\n  Hollow (d_2 = 0), so L_1 = d_1^T d_1:")
    print_matrix(hollow.hodge_laplacian(1), rows=labels(hollow, 1),
                 cols=labels(hollow, 1))
    print(f"  spectrum {format_spectrum(hollow.spectrum(1))}, "
          f"dim ker L_1 = {hollow.betti_from_laplacian(1)} = b1.")

    print("\n  Filled: add d_2 d_2^T with d_2 = "
          f"{filled.boundary_matrix(2).ravel().astype(int).tolist()}:")
    print_matrix(filled.hodge_laplacian(1), rows=labels(filled, 1),
                 cols=labels(filled, 1))
    print(f"  spectrum {format_spectrum(filled.spectrum(1))}, "
          f"dim ker L_1 = {filled.betti_from_laplacian(1)} = b1.")
    print(f"  Filling the triangle pushed the zero eigenvalue up to "
          f"{filled.spectral_gap(1):.0f}.")

    # ------------------------------------------- beta_k = dim ker L_k, always
    print("\nbeta_k = dim ker L_k, checked against rank-nullity")
    print(f"  {'complex':>24s}  {'counts':>14s}  {'rank-nullity':>14s}"
          f"  {'dim ker L_k':>14s}")
    catalogue = {
        "hollow triangle": hollow,
        "filled triangle": filled,
        "hexagon": SimplicialComplex.cycle(6),
        "tetrahedron boundary": SimplicialComplex.boundary_of_simplex(4),
        "solid tetrahedron": SimplicialComplex([(0, 1, 2, 3)]),
        "7-vertex torus": _torus(),
        "figure eight": _figure_eight(),
    }
    kernel_ok = True
    for label, complex_ in catalogue.items():
        by_rank = complex_.betti_numbers()
        by_kernel = [complex_.betti_from_laplacian(k) for k in range(complex_.dim + 1)]
        kernel_ok &= by_rank == by_kernel
        print(f"  {label:>24s}  {str(complex_.counts()):>14s}  {str(by_rank):>14s}"
              f"  {str(by_kernel):>14s}")
    print("  Two rank computations, or one eigensolver.  Same answer.")

    # ------------------------------------------------ harmonic representatives
    print("\nWhere is the loop?  The harmonic representative")
    print("  ker L_1 is a genuine subspace, not a quotient, so each class has a")
    print("  unique minimum-norm harmonic representative.  The barcode does not.")

    hexagon = SimplicialComplex.cycle(6)
    harmonic = hexagon.harmonic_basis(1)[:, 0]
    harmonic = harmonic / np.abs(harmonic).max()
    print(f"\n  Hexagon, the harmonic 1-chain over "
          f"({', '.join(labels(hexagon, 1))}):")
    print("   ", np.round(harmonic, 4))
    signs = np.sign(harmonic)
    odd_one_out = name(hexagon.simplices(1)[int(np.argmax(signs != np.sign(signs.sum())))])
    print("  Every edge carries the same magnitude: the loop has no preferred")
    print("  place to sit, so the circulation spreads out evenly.  The single sign")
    print(f"  flip is on {odd_one_out}, the wrap-around edge -- it is stored with its")
    print("  vertices increasing, which runs against the way the loop travels.")

    # a complex whose single loop has two routes home, so the harmonic chain
    # is genuinely non-uniform
    bridged = SimplicialComplex([(0, 1, 2), (0, 3), (2, 3)])
    bridged_harmonic = bridged.harmonic_basis(1)[:, 0]
    bridged_harmonic = bridged_harmonic / np.abs(bridged_harmonic).max()
    print(f"\n  A filled triangle abc with a two-edge path ad, cd bridging a to c:")
    print(f"  n = {bridged.counts()}, b1 = {bridged.betti_number(1)} -- one loop,")
    print("  and the harmonic chain representing it is")
    for edge, weight in zip(labels(bridged, 1), bridged_harmonic):
        print(f"    {edge:>4s}  {weight:+.4f}")
    print("  The weights are unequal.  Going home from c to a, the loop can take")
    print("  the direct edge ac or the two-edge path through b, and because the")
    print("  triangle abc is filled those two routes are homologous.  The harmonic")
    print("  representative splits the flow 2/3 to 1/3 between them -- the unique")
    print("  choice orthogonal to every boundary.  No barcode contains this.")

    # -------------------------------------------- subdivision and the spectrum
    print("\nThe spectrum sees more than beta_k does")
    print(f"  {'complex':>26s}  {'n1':>3s}  {'b1':>3s}  {'spectrum of L_1':>34s}"
          f"  {'gap':>7s}")
    gaps = []
    for times, label in ((0, "hollow triangle"), (1, "subdivided once (hexagon)"),
                         (2, "subdivided twice (12-gon)")):
        complex_ = SimplicialComplex.cycle(3 * 2 ** times)
        gaps.append(complex_.spectral_gap(1))
        spectrum = format_spectrum(complex_.spectrum(1))
        if len(spectrum) > 34:
            spectrum = spectrum[:31] + "...}"
        print(f"  {label:>26s}  {complex_.count(1):3d}  {complex_.betti_number(1):3d}"
              f"  {spectrum:>34s}  {gaps[-1]:7.3f}")
    print(f"  b1 stays 1 throughout -- it is the same one loop -- while the")
    print(f"  spectral gap runs {gaps[0]:.0f} -> {gaps[1]:.0f} -> {gaps[2]:.3f}.")
    print("  Two complexes can share a persistence diagram and have very different")
    print("  spectra.  The diagram is a coarsening of the Laplacian, not the reverse.")

    # ------------------------------------------------------------- HodgeRank
    print("\nHodgeRank: the same decomposition, applied to pairwise comparisons")
    print("  C_1 = im d_1^T (+) ker L_1 (+) im d_2, and the three pieces are a")
    print("  global ranking, an irreconcilably cyclic part, and local inconsistency.")

    comparisons = {
        "a beats b, b beats c, c beats a (pure cycle)": np.array([1.0, -1.0, 1.0]),
        "a > b > c, consistent (pure ranking)": np.array([1.0, 2.0, 1.0]),
    }
    print(f"\n  {'comparison edge-flow on the hollow triangle':>46s}"
          f"  {'||grad||':>9s} {'||harm||':>9s} {'||curl||':>9s}")
    hodge_ok = True
    for label, flow in comparisons.items():
        parts = hollow.hodge_decomposition(flow, 1)
        norms = {key: float(np.linalg.norm(value)) for key, value in parts.items()}
        print(f"  {label:>46s}  {norms['coboundary']:9.4f}"
              f"  {norms['harmonic']:9.4f} {norms['boundary']:9.4f}")
        hodge_ok &= np.allclose(
            parts["boundary"] + parts["harmonic"] + parts["coboundary"], flow)
    print("  (edges in the order ab, ac, bc, so 'b beats c' is +1 on bc and")
    print("   'c beats a' is -1 on ac.)")
    print("  The cyclic preference has no gradient part at all: no ranking of")
    print("  a, b, c explains it, and HodgeRank says so instead of inventing one.")

    # ---------------------------------------------------------------- figure
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), facecolor=viz.SURFACE)
    viz.plot_chain(polygon(6), hexagon, hexagon.harmonic_basis(1)[:, 0], axes[0],
                   title="hexagon: the harmonic loop")
    bridged_points = np.array([
        [0.0, 0.0], [0.8, -0.9], [1.6, 0.0], [0.8, 1.2],  # a, b, c, then d on top
    ])
    viz.plot_chain(bridged_points, bridged, bridged.harmonic_basis(1)[:, 0], axes[1],
                   title="two routes home: unequal weights",
                   vertex_labels=list("abcd"), show_weights=True)

    ax = axes[2]
    sizes = [3, 6, 12, 24, 48]
    ax.plot(sizes, [SimplicialComplex.cycle(n).spectral_gap(1) for n in sizes],
            "o-", color=viz.DIM_COLORS[1], label=r"spectral gap $\lambda_1(L_1)$")
    ax.plot(sizes, [SimplicialComplex.cycle(n).betti_number(1) for n in sizes],
            "s--", color=viz.DIM_COLORS[0], label=r"$\beta_1$")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("edges in the cycle (subdividing)", color=viz.INK_SECONDARY, fontsize=10)
    ax.set_title("the same loop, a moving spectrum",
                 color=viz.INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    viz.style_axes(ax)

    figure.suptitle(r"$\beta_k = \dim\ker L_k$, and the harmonic chain that "
                    "the barcode cannot give you",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    print("\nwrote", viz.save(figure, out("talk_06_hodge_laplacian.png")))

    # ------------------------------------------------------------ the checks
    print("\nVerdict")
    ok = True
    ok &= check("L_0 = d_1 d_1^T = D - A on every complex tested", laplacian_ok)
    ok &= check("hollow triangle: L_0 and L_1 both have spectrum {0, 3, 3}",
                np.allclose(hollow.spectrum(0), [0, 3, 3])
                and np.allclose(hollow.spectrum(1), [0, 3, 3]))
    ok &= check("filled triangle: L_1 = 3I, spectrum {3, 3, 3}, dim ker = 0",
                np.allclose(filled.hodge_laplacian(1), 3 * np.eye(3))
                and filled.betti_from_laplacian(1) == 0)
    ok &= check("beta_k = dim ker L_k on all seven complexes", kernel_ok)
    ok &= check("the hexagon's harmonic chain has equal weight on every edge",
                np.allclose(np.abs(harmonic), 1.0))
    ok &= check("the bridged complex's harmonic chain does not, because its "
                "loop has two routes home",
                not np.allclose(np.abs(bridged_harmonic),
                                np.abs(bridged_harmonic[0])))
    ok &= check("subdividing leaves b1 = 1 and moves the gap 3 -> 1 -> 0.268",
                all(SimplicialComplex.cycle(3 * 2 ** t).betti_number(1) == 1
                    for t in (0, 1, 2))
                and np.allclose(gaps, [3.0, 1.0, 0.268], atol=5e-4))
    ok &= check("the Hodge decomposition sums back to the original chain", hodge_ok)
    ok &= check("a pure preference cycle has no gradient component",
                np.linalg.norm(
                    hollow.hodge_decomposition(np.array([1.0, -1.0, 1.0]), 1)["coboundary"]
                ) < 1e-9)
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


def _torus() -> SimplicialComplex:
    return SimplicialComplex(
        [(i % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)]
        + [(i % 7, (i + 2) % 7, (i + 3) % 7) for i in range(7)]
    )


def _figure_eight() -> SimplicialComplex:
    """Two triangles sharing a single vertex: b0 = 1, b1 = 2."""
    return SimplicialComplex([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)])


if __name__ == "__main__":
    main()
