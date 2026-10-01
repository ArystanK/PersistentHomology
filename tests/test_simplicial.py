import numpy as np
import pytest

from persistent_homology import SimplicialComplex, format_spectrum
from persistent_homology.simplicial import subdivide_cycle

HOLLOW = [(0, 1), (0, 2), (1, 2)]


def hollow():
    return SimplicialComplex(HOLLOW)


def filled():
    return SimplicialComplex([(0, 1, 2)])


def torus():
    """The 7-vertex Moebius torus, the smallest triangulation of a torus."""
    return SimplicialComplex(
        [(i % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)]
        + [(i % 7, (i + 2) % 7, (i + 3) % 7) for i in range(7)]
    )


# ------------------------------------------------------------------ closure

def test_construction_closes_under_faces():
    K = SimplicialComplex([(0, 1, 2, 3)])
    assert K.counts() == [4, 6, 4, 1]
    assert (0,) in K and (1, 3) in K and (0, 2, 3) in K


def test_closure_reaches_all_the_way_down():
    # a single 4-simplex must drag in its vertices, not just its facets
    K = SimplicialComplex([(0, 1, 2, 3, 4)])
    assert K.counts() == [5, 10, 10, 5, 1]


def test_duplicate_and_unsorted_input_is_normalised():
    K = SimplicialComplex([(2, 1, 0), (0, 1, 2), (1, 0)])
    assert K.counts() == [3, 3, 1]


def test_empty_simplex_rejected():
    with pytest.raises(ValueError):
        SimplicialComplex([()])


# ---------------------------------------------------------- boundary matrices

def test_boundary_matrix_shapes_and_entries():
    K = filled()
    assert K.boundary_matrix(0).shape == (0, 3)
    assert K.boundary_matrix(1).shape == (3, 3)
    assert K.boundary_matrix(2).shape == (3, 1)
    assert K.boundary_matrix(3).shape == (1, 0)
    assert set(np.unique(K.boundary_matrix(1))) <= {-1.0, 0.0, 1.0}


def test_each_column_has_k_plus_one_nonzeros():
    K = SimplicialComplex([(0, 1, 2, 3)])
    for k in range(1, K.dim + 1):
        nonzeros = (K.boundary_matrix(k) != 0).sum(axis=0)
        assert (nonzeros == k + 1).all()


def test_boundary_of_edge_is_head_minus_tail():
    K = hollow()
    column = K.boundary_matrix(1)[:, K.index((0, 1))]
    assert column[0] == -1 and column[1] == 1 and column[2] == 0


@pytest.mark.parametrize("K", [hollow(), filled(), torus(),
                               SimplicialComplex([(0, 1, 2, 3, 4)]),
                               SimplicialComplex.boundary_of_simplex(5)])
def test_boundary_of_a_boundary_is_zero(K):
    for k in range(1, K.dim + 1):
        product = K.boundary_matrix(k) @ K.boundary_matrix(k + 1)
        assert np.abs(product).max(initial=0.0) == 0.0


# -------------------------------------------------------------- Betti numbers

def test_hollow_triangle_has_one_loop():
    assert hollow().betti_numbers() == [1, 1]


def test_filling_the_triangle_kills_the_loop():
    assert filled().betti_numbers() == [1, 0, 0]


def test_tetrahedron_boundary_is_a_sphere():
    assert SimplicialComplex.boundary_of_simplex(4).betti_numbers() == [1, 0, 1]


def test_solid_tetrahedron_has_no_void():
    assert SimplicialComplex([(0, 1, 2, 3)]).betti_numbers() == [1, 0, 0, 0]


def test_torus_has_two_loops_and_one_void():
    assert torus().betti_numbers() == [1, 2, 1]


def test_disjoint_pieces_are_counted_by_b0():
    K = SimplicialComplex([(0, 1, 2), (3, 4), (5,)])
    assert K.betti_number(0) == 3


def test_cycle_graph_has_one_loop_however_long():
    for n in (3, 6, 12, 25):
        assert SimplicialComplex.cycle(n).betti_numbers() == [1, 1]


def test_betti_number_outside_the_range_is_zero():
    K = hollow()
    assert K.betti_number(5) == 0 and K.betti_number(-1) == 0


@pytest.mark.parametrize("K", [hollow(), filled(), torus(),
                               SimplicialComplex.boundary_of_simplex(4),
                               SimplicialComplex.cycle(9)])
def test_euler_characteristic_matches_the_alternating_betti_sum(K):
    from_counts = K.euler_characteristic()
    from_betti = sum((-1) ** k * b for k, b in enumerate(K.betti_numbers()))
    assert from_counts == from_betti


# ----------------------------------------------------------- Hodge Laplacian

@pytest.mark.parametrize("K", [hollow(), filled(), torus(),
                               SimplicialComplex.cycle(7),
                               SimplicialComplex([(0, 1, 2), (3, 4, 5)])])
def test_L0_is_the_graph_laplacian(K):
    assert np.allclose(K.hodge_laplacian(0), K.graph_laplacian())


@pytest.mark.parametrize("K", [hollow(), filled(), torus(),
                               SimplicialComplex.boundary_of_simplex(4),
                               SimplicialComplex([(0, 1, 2, 3)]),
                               SimplicialComplex.cycle(11)])
def test_laplacian_is_symmetric_positive_semidefinite(K):
    for k in range(K.dim + 1):
        L = K.hodge_laplacian(k)
        assert np.allclose(L, L.T)
        assert K.spectrum(k).min() >= -1e-9


@pytest.mark.parametrize("K", [hollow(), filled(), torus(),
                               SimplicialComplex.boundary_of_simplex(4),
                               SimplicialComplex([(0, 1, 2, 3)]),
                               SimplicialComplex([(0, 1, 2), (3, 4), (5,)]),
                               SimplicialComplex.cycle(13)])
def test_kernel_dimension_equals_the_betti_number(K):
    for k in range(K.dim + 1):
        assert K.betti_from_laplacian(k) == K.betti_number(k)


def test_triangle_spectra_match_the_hand_computation():
    assert np.allclose(hollow().spectrum(0), [0, 3, 3])
    assert np.allclose(hollow().spectrum(1), [0, 3, 3])
    assert np.allclose(filled().hodge_laplacian(1), 3 * np.eye(3))
    assert np.allclose(filled().spectrum(1), [3, 3, 3])


def test_cycle_graph_spectrum_is_the_discrete_fourier_one():
    # eigenvalues of L_1 on an n-cycle are 2 - 2 cos(2 pi j / n)
    n = 6
    expected = np.sort(2 - 2 * np.cos(2 * np.pi * np.arange(n) / n))
    assert np.allclose(SimplicialComplex.cycle(n).spectrum(1), expected)


def test_subdividing_keeps_b1_and_shrinks_the_spectral_gap():
    gaps = [subdivide_cycle(3, times).spectral_gap(1) for times in (0, 1, 2)]
    assert all(subdivide_cycle(3, t).betti_number(1) == 1 for t in (0, 1, 2))
    assert np.allclose(gaps, [3.0, 1.0, 0.26794919], atol=1e-6)
    assert gaps[0] > gaps[1] > gaps[2]


def test_spectral_gap_of_a_complex_with_no_nonzero_eigenvalue():
    lone = SimplicialComplex([(0,)])
    assert lone.spectral_gap(0) == float("inf")


# ------------------------------------------------------------- Hodge theory

def test_harmonic_basis_is_orthonormal_and_in_the_kernel():
    K = torus()
    basis = K.harmonic_basis(1)
    assert basis.shape == (K.count(1), K.betti_number(1))
    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]))
    assert np.allclose(K.hodge_laplacian(1) @ basis, 0.0, atol=1e-9)


def test_harmonic_chains_are_cycles_orthogonal_to_boundaries():
    K = SimplicialComplex([(0, 1, 2), (0, 3), (2, 3)])
    chain = K.harmonic_basis(1)[:, 0]
    assert np.allclose(K.boundary_matrix(1) @ chain, 0.0, atol=1e-9)
    assert np.allclose(K.boundary_matrix(2).T @ chain, 0.0, atol=1e-9)


def test_harmonic_chain_on_a_cycle_graph_is_uniform():
    K = SimplicialComplex.cycle(8)
    chain = K.harmonic_basis(1)[:, 0]
    assert np.allclose(np.abs(chain), np.abs(chain[0]))


def test_hodge_decomposition_is_a_partition_of_the_chain():
    K = SimplicialComplex([(0, 1, 2), (0, 3), (2, 3)])
    rng = np.random.default_rng(0)
    chain = rng.normal(size=K.count(1))
    parts = K.hodge_decomposition(chain, 1)
    assert np.allclose(sum(parts.values()), chain)


def test_hodge_decomposition_pieces_are_mutually_orthogonal():
    K = SimplicialComplex([(0, 1, 2), (0, 3), (2, 3)])
    parts = K.hodge_decomposition(np.random.default_rng(1).normal(size=K.count(1)), 1)
    keys = list(parts)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert abs(float(parts[a] @ parts[b])) < 1e-9


def test_a_pure_preference_cycle_is_entirely_harmonic():
    # a beats b, b beats c, c beats a, on edges (ab, ac, bc)
    parts = hollow().hodge_decomposition(np.array([1.0, -1.0, 1.0]), 1)
    assert np.linalg.norm(parts["coboundary"]) < 1e-9
    assert np.linalg.norm(parts["boundary"]) < 1e-9
    assert np.linalg.norm(parts["harmonic"]) > 1.0


def test_a_consistent_ranking_has_no_harmonic_part():
    parts = hollow().hodge_decomposition(np.array([1.0, 2.0, 1.0]), 1)
    assert np.linalg.norm(parts["harmonic"]) < 1e-9


def test_decomposition_rejects_a_chain_of_the_wrong_length():
    with pytest.raises(ValueError):
        hollow().hodge_decomposition(np.zeros(2), 1)


# ------------------------------------------------------------ interop, misc

def test_from_filtration_matches_a_directly_built_complex():
    from persistent_homology import pairwise_distances, rips_filtration

    angles = np.arange(6) * np.pi / 3
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    filtration = rips_filtration(pairwise_distances(points), max_dim=2, threshold=2.0)

    at_one = SimplicialComplex.from_filtration(filtration, 1.0 + 1e-9)
    assert at_one.counts() == [6, 6]
    assert at_one.betti_numbers() == [1, 1]

    # at sqrt(3) the Rips complex is the boundary of an octahedron
    at_sqrt3 = SimplicialComplex.from_filtration(filtration, np.sqrt(3) + 1e-9)
    assert at_sqrt3.counts() == [6, 12, 8]
    assert at_sqrt3.betti_numbers() == [1, 0, 1]


def test_format_spectrum_reads_like_the_slides():
    assert format_spectrum([0.0, 1.0, 1.0, 3.0, 3.0, 4.0]) == "{0, 1, 1, 3, 3, 4}"
    assert format_spectrum([0.0, 0.2679491]) == "{0, 0.268}"


def test_repr_lists_the_counts():
    assert "n0=3" in repr(filled()) and "n2=1" in repr(filled())
