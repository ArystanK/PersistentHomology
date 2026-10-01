import numpy as np
import pytest

from persistent_homology import (
    enclosing_radius,
    filtration_from_simplices,
    pairwise_distances,
    rips_filtration,
    rips_filtration_from_points,
)
from persistent_homology import datasets as ds


def test_pairwise_distances_matches_numpy():
    points = np.random.default_rng(0).normal(size=(12, 3))
    expected = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    assert np.allclose(pairwise_distances(points), expected)
    assert np.allclose(np.diag(pairwise_distances(points)), 0.0)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_distance_metrics_are_symmetric_and_nonnegative(metric):
    points = np.random.default_rng(1).normal(size=(8, 2))
    dist = pairwise_distances(points, metric)
    assert np.allclose(dist, dist.T)
    assert (dist >= 0).all()


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        pairwise_distances(np.zeros((3, 2)), "cosine")


def test_enclosing_radius_of_unit_square():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # every point's farthest neighbour is the opposite corner
    assert enclosing_radius(pairwise_distances(square)) == pytest.approx(np.sqrt(2))


def test_rips_on_four_points_has_the_right_simplex_counts():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    filt = rips_filtration_from_points(square, max_dim=3, threshold=10.0)
    counts = np.bincount(filt.dims)
    assert counts.tolist() == [4, 6, 4, 1]          # the full 3-simplex


def test_threshold_excludes_long_edges():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    filt = rips_filtration_from_points(square, max_dim=2, threshold=1.0)
    # the two diagonals are longer than 1, so only 4 edges and no triangles
    counts = np.bincount(filt.dims, minlength=3)
    assert counts.tolist() == [4, 4, 0]


def test_filtration_order_places_faces_first():
    filt = rips_filtration_from_points(ds.circle(15, seed=2), max_dim=2)
    for j, simplex in enumerate(filt.simplices):
        for k in range(len(simplex)):
            face = simplex[:k] + simplex[k + 1:]
            if face:
                assert filt.index(face) < j
    assert np.all(np.diff(filt.values) >= -1e-12)


def test_boundary_of_a_boundary_is_zero_mod_two():
    filt = rips_filtration_from_points(ds.circle(12, seed=3), max_dim=3)
    columns = filt.boundary_columns()
    for column in columns:
        twice: dict[int, int] = {}
        for face in column:
            for grandface in columns[face]:
                twice[grandface] = twice.get(grandface, 0) + 1
        assert all(count % 2 == 0 for count in twice.values())


def test_filtration_value_is_the_longest_edge():
    points = ds.uniform_box(10, seed=4)
    dist = pairwise_distances(points)
    filt = rips_filtration(dist, max_dim=3, threshold=10.0)
    for simplex, value in zip(filt.simplices, filt.values):
        longest = max(
            (dist[u, v] for i, u in enumerate(simplex) for v in simplex[i + 1:]),
            default=0.0,
        )
        assert value == pytest.approx(longest)


def test_explicit_filtration_rejects_a_missing_face():
    with pytest.raises(ValueError, match="missing its face"):
        filtration_from_simplices([((0,), 0.0), ((1,), 0.0), ((0, 1, 2), 1.0)])


def test_explicit_filtration_rejects_a_late_face():
    with pytest.raises(ValueError, match="after its coface"):
        filtration_from_simplices([((0,), 0.0), ((1,), 5.0), ((0, 1), 1.0)])


def test_max_simplices_guard_trips():
    with pytest.raises(MemoryError):
        rips_filtration_from_points(
            ds.uniform_box(40, seed=5), max_dim=4, threshold=10.0, max_simplices=1000
        )
