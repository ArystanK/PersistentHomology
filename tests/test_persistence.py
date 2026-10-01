import numpy as np
import pytest

from persistent_homology import (
    filtration_from_simplices,
    persistence,
    rips_filtration_from_points,
    rips_persistence,
)
from persistent_homology import datasets as ds
from persistent_homology.persistence import _reduce, _symmetric_difference


# --- the reduction itself ----------------------------------------------------

def test_symmetric_difference_is_gf2_addition():
    assert _symmetric_difference([1, 3, 5], [3, 4]) == [1, 4, 5]
    assert _symmetric_difference([1, 2], [1, 2]) == []
    assert _symmetric_difference([], [7]) == [7]


def test_reduced_columns_have_distinct_lowest_entries():
    filt = rips_filtration_from_points(ds.circle(20, seed=1), max_dim=2)
    reduced, pivots = _reduce(filt.boundary_columns(), filt.dims)
    lows = [col[-1] for col in reduced if col]
    assert len(lows) == len(set(lows))
    assert set(lows) == set(pivots)


def test_every_simplex_is_either_a_birth_or_a_death():
    """The pairing partitions the simplices: each one either opens a class or
    closes exactly one, and nothing is counted twice."""
    filt = rips_filtration_from_points(ds.two_circles(24, seed=2), max_dim=2)
    result = persistence(filt, min_persistence=-1.0)   # keep zero-length pairs too
    seen = [i for pair in result.pairs for i in pair if i is not None]
    assert len(seen) == len(set(seen)) == len(filt)


# --- hand-checkable complexes ------------------------------------------------

def test_hollow_triangle_has_an_essential_loop():
    filt = filtration_from_simplices(
        [((0,), 0), ((1,), 0), ((2,), 0), ((0, 1), 1), ((0, 2), 1), ((1, 2), 1)]
    )
    result = persistence(filt)
    assert result[1].tolist() == [[1.0, np.inf]]
    assert result.betti_numbers(1.0) == [1, 1]
    assert result.betti_numbers(0.5) == [3, 0]


def test_filling_the_triangle_kills_the_loop():
    filt = filtration_from_simplices(
        [((0,), 0), ((1,), 0), ((2,), 0), ((0, 1), 1), ((0, 2), 1), ((1, 2), 1),
         ((0, 1, 2), 2)]
    )
    result = persistence(filt)
    assert result[1].tolist() == [[1.0, 2.0]]
    # the complex contains a 2-simplex, so the Betti vector runs to H2
    assert result.betti_numbers(2.0) == [1, 0, 0]


def test_hollow_tetrahedron_has_a_void():
    vertices = [((i,), 0) for i in range(4)]
    edges = [((i, j), 1) for i in range(4) for j in range(i + 1, 4)]
    faces = [(t, 2) for t in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]
    result = persistence(filtration_from_simplices(vertices + edges + faces))
    assert result[2].tolist() == [[2.0, np.inf]]
    assert result.betti_numbers(2.0) == [1, 0, 1]


def test_two_disjoint_points_stay_two_components():
    filt = filtration_from_simplices([((0,), 0), ((1,), 0)])
    result = persistence(filt)
    assert result[0].tolist() == [[0.0, np.inf], [0.0, np.inf]]


# --- point clouds with known topology ----------------------------------------

def test_circle_loop_dies_at_the_inscribed_equilateral_triangle():
    """The Rips complex of a circle of radius r keeps its loop until the radius
    reaches r*sqrt(3), the side of the inscribed equilateral triangle.  On a
    regular n-gon only the chord lengths 2r*sin(k*pi/n) occur, so the loop dies
    at the first such chord that reaches sqrt(3).
    """
    n = 40
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    chords = 2 * np.sin(np.arange(1, n // 2 + 1) * np.pi / n)
    expected = float(chords[chords >= np.sqrt(3)][0])

    result = rips_persistence(circle, max_dim=1)
    (birth, death), = result.most_persistent(1)
    assert death == pytest.approx(expected)
    assert np.sqrt(3) <= death < np.sqrt(3) + 0.06     # and it is just above sqrt(3)
    assert birth < 0.3
    assert result.betti_numbers(1.0) == [1, 1]


def test_two_circles_give_two_components_and_two_loops():
    result = rips_persistence(ds.two_circles(80, separation=3.6, seed=5),
                              max_dim=1, threshold=2.2)
    assert result.betti_numbers(1.2) == [2, 2]


def test_figure_eight_is_one_component_with_two_loops():
    result = rips_persistence(ds.figure_eight(110, seed=6), max_dim=1, threshold=2.0)
    assert result.betti_numbers(1.07) == [1, 2]


def test_clusters_are_counted_by_h0():
    for k in (2, 3, 4):
        centers = [(4.0 * i, 0.0) for i in range(k)]
        points = ds.clusters(30 * k, centers=centers, spread=0.3, seed=k)
        result = rips_persistence(points, max_dim=0, threshold=2.0)
        assert result.betti_numbers(1.5) == [k]


def test_sphere_encloses_exactly_one_void():
    result = rips_persistence(ds.sphere(55, seed=3), max_dim=2, threshold=1.9)
    assert len(result[2]) == 1
    (birth, death), = result[2]
    assert result.betti_numbers((birth + death) / 2) == [1, 0, 1]


def test_torus_has_two_loops():
    result = rips_persistence(ds.flat_torus(200, seed=5), max_dim=1, threshold=1.8)
    assert result.betti_numbers(1.3) == [1, 2]


def test_noise_has_no_long_bars():
    result = rips_persistence(ds.uniform_box(70, seed=9), max_dim=1, threshold=0.8)
    circle = rips_persistence(ds.circle(70, seed=9), max_dim=1, threshold=2.2)
    noisiest = (result[1][:, 1] - result[1][:, 0]).max()
    loop = (circle[1][:, 1] - circle[1][:, 0]).max()
    assert noisiest < 0.25 * loop


# --- structural invariants ---------------------------------------------------

def test_h0_has_one_essential_class_per_connected_component():
    result = rips_persistence(ds.two_circles(40, separation=6.0, seed=7),
                              max_dim=0, threshold=2.0)
    essential = np.isinf(result[0][:, 1]).sum()
    assert essential == 2


def test_h0_class_count_equals_point_count():
    points = ds.uniform_box(35, seed=8)
    result = rips_persistence(points, max_dim=0, min_persistence=-1.0)
    assert len(result[0]) == len(points)


def test_births_never_exceed_deaths():
    result = rips_persistence(ds.annulus(60, seed=10), max_dim=1, threshold=2.0)
    for dgm in result.diagrams.values():
        assert (dgm[:, 0] <= dgm[:, 1]).all()


def test_persistence_matches_independent_rank_computation():
    """Betti numbers read off the diagrams must agree with the rank-nullity
    computation done directly on the reduced boundary matrix."""
    filt = rips_filtration_from_points(ds.two_circles(45, separation=3.6, seed=11), max_dim=2)
    result = persistence(filt, max_dim=1)
    for scale in (0.3, 0.7, 1.1, 1.5):
        # the filtration's own top dimension is unreliable, so compare H0 and H1
        assert result.betti_numbers(scale) == filt.betti_numbers(scale)[:2]


def test_min_persistence_drops_short_bars():
    points = ds.uniform_box(60, seed=12)          # noise: many short-lived loops
    everything = rips_persistence(points, max_dim=1, threshold=0.5)
    assert len(everything[1]) > 5
    pruned = rips_persistence(points, max_dim=1, threshold=0.5, min_persistence=0.05)
    assert len(pruned[1]) < len(everything[1])
    assert (pruned[1][:, 1] - pruned[1][:, 0] > 0.05).all()


def test_metrics_change_the_scale_but_not_the_topology():
    points = ds.circle(40, seed=13)
    for metric in ("euclidean", "manhattan", "chebyshev"):
        result = rips_persistence(points, max_dim=1, metric=metric)
        assert len(result.most_persistent(1)) == 1
        (birth, death), = result.most_persistent(1)
        assert result.betti_numbers((birth + death) / 2) == [1, 1]


def test_result_is_invariant_under_relabelling_the_points():
    points = ds.annulus(50, seed=14)
    order = np.random.default_rng(0).permutation(len(points))
    a = rips_persistence(points, max_dim=1, threshold=2.0)
    b = rips_persistence(points[order], max_dim=1, threshold=2.0)
    for dim in (0, 1):
        assert np.allclose(np.sort(a[dim], axis=0), np.sort(b[dim], axis=0))


def test_result_is_invariant_under_rotation():
    points = ds.annulus(50, seed=15)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    a = rips_persistence(points, max_dim=1, threshold=2.0)
    b = rips_persistence(points @ rotation.T, max_dim=1, threshold=2.0)
    for dim in (0, 1):
        assert np.allclose(np.sort(a[dim], axis=0), np.sort(b[dim], axis=0), atol=1e-9)


def test_scaling_the_cloud_scales_the_diagram():
    points = ds.circle(40, noise=0.04, seed=16)
    a = rips_persistence(points, max_dim=1, threshold=2.2)
    b = rips_persistence(3.0 * points, max_dim=1, threshold=6.6)
    assert np.allclose(np.sort(3.0 * a[1], axis=0), np.sort(b[1], axis=0), atol=1e-9)


def test_single_point_and_empty_cloud():
    single = rips_persistence(np.zeros((1, 2)), max_dim=1)
    assert single[0].tolist() == [[0.0, np.inf]]
    assert len(single[1]) == 0


def test_most_stable_scale_lands_on_the_widest_constant_interval():
    result = rips_persistence(ds.circle(60, noise=0.05, seed=9), max_dim=1, threshold=2.2)
    scale = result.most_stable_scale(2.2)
    assert result.betti_numbers(scale) == [1, 1]


def test_most_stable_scale_skips_the_trivial_blob_interval():
    # past the last death everything is one contractible blob, b = (1, 0), and
    # that interval is the widest one; it must not be the answer
    result = rips_persistence(ds.two_circles(80, separation=3.6, seed=5),
                              max_dim=1, threshold=2.4)
    assert result.betti_numbers(result.most_stable_scale(2.4)) != [1, 0]


def test_most_stable_scale_defaults_to_the_largest_finite_value():
    result = rips_persistence(ds.circle(40, noise=0.05, seed=2), max_dim=1, threshold=2.2)
    assert result.most_stable_scale() == result.most_stable_scale(
        max(d[np.isfinite(d[:, 1]), 1].max() for d in result.diagrams.values() if len(d))
    )
