import numpy as np
import pytest

from persistent_homology import (
    bottleneck_distance,
    diagram_distance_matrix,
    rips_persistence,
    wasserstein_distance,
)
from persistent_homology import datasets as ds

EMPTY = np.empty((0, 2))


def test_distance_to_self_is_zero():
    dgm = np.array([[0.0, 1.0], [0.2, 3.0], [0.5, 0.9]])
    assert bottleneck_distance(dgm, dgm) == 0.0
    assert wasserstein_distance(dgm, dgm) == 0.0


def test_empty_diagrams():
    assert bottleneck_distance(EMPTY, EMPTY) == 0.0
    assert wasserstein_distance(EMPTY, EMPTY) == 0.0


def test_a_lone_point_is_matched_to_the_diagonal():
    # the cheapest option for [0, 1] is its own projection, at height 0.5
    assert bottleneck_distance(np.array([[0.0, 1.0]]), EMPTY) == pytest.approx(0.5)
    assert wasserstein_distance(np.array([[0.0, 1.0]]), EMPTY, order=1) == pytest.approx(0.5)


def test_a_pure_shift_costs_the_shift():
    a = np.array([[0.0, 1.0], [0.0, 2.0]])
    b = a + np.array([0.1, 0.1])
    assert bottleneck_distance(a, b) == pytest.approx(0.1)


def test_bottleneck_takes_the_worst_pair_and_wasserstein_the_sum():
    a = np.array([[0.0, 4.0], [0.0, 6.0]])
    b = np.array([[0.1, 4.0], [0.3, 6.0]])
    assert bottleneck_distance(a, b) == pytest.approx(0.3)
    assert wasserstein_distance(a, b, order=1) == pytest.approx(0.4)


def test_distances_are_symmetric():
    a = np.array([[0.0, 1.0], [0.4, 2.2]])
    b = np.array([[0.1, 1.4], [0.0, 3.0], [0.9, 1.0]])
    assert bottleneck_distance(a, b) == pytest.approx(bottleneck_distance(b, a))
    assert wasserstein_distance(a, b) == pytest.approx(wasserstein_distance(b, a))


def test_bottleneck_never_exceeds_wasserstein():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = np.sort(rng.uniform(0, 3, (4, 2)), axis=1)
        b = np.sort(rng.uniform(0, 3, (5, 2)), axis=1)
        assert bottleneck_distance(a, b) <= wasserstein_distance(a, b, order=1) + 1e-9


def test_triangle_inequality():
    rng = np.random.default_rng(1)
    for _ in range(10):
        a, b, c = (np.sort(rng.uniform(0, 3, (3, 2)), axis=1) for _ in range(3))
        assert (bottleneck_distance(a, c)
                <= bottleneck_distance(a, b) + bottleneck_distance(b, c) + 1e-9)


def test_mismatched_essential_counts_are_infinitely_far_apart():
    finite = np.array([[0.0, 1.0]])
    essential = np.array([[0.0, np.inf]])
    assert bottleneck_distance(finite, essential) == float("inf")
    assert wasserstein_distance(finite, essential) == float("inf")


def test_essential_classes_are_matched_on_their_births():
    a = np.array([[0.0, np.inf], [1.0, np.inf]])
    b = np.array([[0.2, np.inf], [1.1, np.inf]])
    assert bottleneck_distance(a, b) == pytest.approx(0.2)


def test_bottleneck_agrees_with_brute_force_on_small_diagrams():
    """Check the binary-search-plus-matching result against every possible
    matching, enumerated exhaustively."""
    from itertools import permutations

    rng = np.random.default_rng(2)
    for _ in range(20):
        a = np.sort(rng.uniform(0, 2, (3, 2)), axis=1)
        b = np.sort(rng.uniform(0, 2, (3, 2)), axis=1)
        n = len(a)
        # cost of matching a_i to b_j, or either to the diagonal
        cross = np.abs(a[:, None, :] - b[None, :, :]).max(axis=-1)
        diag_a = (a[:, 1] - a[:, 0]) / 2
        diag_b = (b[:, 1] - b[:, 0]) / 2
        best = float("inf")
        for mask in range(1 << n):                       # which a's go to the diagonal
            kept_a = [i for i in range(n) if not (mask >> i) & 1]
            for perm in permutations(range(n), len(kept_a)):
                used = set(perm)
                cost = max(
                    [cross[i, j] for i, j in zip(kept_a, perm)]
                    + [diag_a[i] for i in range(n) if (mask >> i) & 1]
                    + [diag_b[j] for j in range(n) if j not in used]
                    + [0.0]
                )
                best = min(best, cost)
        assert bottleneck_distance(a, b) == pytest.approx(best)


def test_stability_bound_holds_on_perturbed_clouds():
    base = ds.circle(45, seed=3)
    base_dgm = rips_persistence(base, max_dim=1, threshold=2.2)[1]
    rng = np.random.default_rng(4)
    for noise in (0.02, 0.05, 0.1):
        pert = base + noise * rng.standard_normal(base.shape)
        hausdorff = float(max(
            np.linalg.norm(base[:, None] - pert[None], axis=-1).min(axis=1).max(),
            np.linalg.norm(base[:, None] - pert[None], axis=-1).min(axis=0).max(),
        ))
        pert_dgm = rips_persistence(pert, max_dim=1, threshold=2.2)[1]
        assert bottleneck_distance(base_dgm, pert_dgm) <= 2 * hausdorff + 1e-9


def test_distance_matrix_is_symmetric_with_a_zero_diagonal():
    dgms = [
        rips_persistence(cloud, max_dim=1, threshold=2.2)[1]
        for cloud in (ds.circle(40, noise=0.05, seed=1),
                      ds.circle(40, noise=0.05, seed=2),
                      2 * ds.uniform_box(40, seed=3) - 1)
    ]
    matrix = diagram_distance_matrix(dgms)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    # the two circles are closer to each other than either is to the noise
    assert matrix[0, 1] < matrix[0, 2] and matrix[0, 1] < matrix[1, 2]
