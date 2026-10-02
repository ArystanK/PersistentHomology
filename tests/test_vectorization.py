import numpy as np
import pytest

from persistent_homology import bottleneck_distance, rips_persistence
from persistent_homology import datasets as ds
from persistent_homology import vectorization as vec

GRID = np.linspace(0.0, 2.0, 50)
DIAGRAM = np.array([[0.2, 1.4], [0.5, 0.7], [0.9, 1.0]])


# --------------------------------------------------------------- Betti curve

def test_betti_curve_counts_bars_alive():
    curve = vec.betti_curve(np.array([[0.0, 1.0], [0.5, 2.0]]), np.array([0.25, 0.75, 1.5]))
    assert list(curve) == [1.0, 2.0, 1.0]


def test_betti_curve_is_half_open_at_the_death():
    curve = vec.betti_curve(np.array([[0.0, 1.0]]), np.array([0.0, 1.0]))
    assert list(curve) == [1.0, 0.0]        # alive at birth, dead at death


def test_betti_curve_counts_essential_classes_everywhere():
    curve = vec.betti_curve(np.array([[0.0, np.inf]]), GRID)
    assert (curve == 1.0).all()


def test_empty_diagram_gives_a_zero_curve():
    assert (vec.betti_curve(np.empty((0, 2)), GRID) == 0.0).all()


# ---------------------------------------------------------------- landscape

def test_landscape_has_a_fixed_shape_whatever_the_diagram():
    for diagram in (np.empty((0, 2)), DIAGRAM, DIAGRAM[:1]):
        assert vec.persistence_landscape(diagram, GRID, n_layers=4).shape == (4, len(GRID))


def test_landscape_layers_are_ordered_and_nonnegative():
    landscape = vec.persistence_landscape(DIAGRAM, GRID, n_layers=3)
    assert (landscape >= 0).all()
    assert (landscape[0] >= landscape[1]).all()
    assert (landscape[1] >= landscape[2]).all()


def test_a_single_bar_peaks_at_half_its_length():
    landscape = vec.persistence_landscape(np.array([[0.0, 1.0]]),
                                          np.linspace(0, 1, 101), n_layers=1)
    assert np.isclose(landscape[0].max(), 0.5)
    assert np.isclose(landscape[0][0], 0.0) and np.isclose(landscape[0][-1], 0.0)


def test_landscape_ignores_essential_classes():
    with_essential = np.vstack([DIAGRAM, [[0.1, np.inf]]])
    assert np.allclose(vec.persistence_landscape(with_essential, GRID),
                       vec.persistence_landscape(DIAGRAM, GRID))


def test_landscape_is_one_lipschitz_in_the_bottleneck_distance():
    circle = ds.circle(50, noise=0.05, seed=4)
    base = rips_persistence(circle, max_dim=1, threshold=2.2)[1]
    rng = np.random.default_rng(0)
    for noise in (0.02, 0.06, 0.12):
        other = rips_persistence(circle + noise * rng.standard_normal(circle.shape),
                                 max_dim=1, threshold=2.2)[1]
        gap = np.abs(vec.persistence_landscape(other, GRID)
                     - vec.persistence_landscape(base, GRID)).max()
        assert gap <= bottleneck_distance(base, other) + 1e-9


def test_landscape_is_linear_in_the_scale():
    scaled = vec.persistence_landscape(2 * DIAGRAM, 2 * GRID)
    assert np.allclose(scaled, 2 * vec.persistence_landscape(DIAGRAM, GRID))


# ------------------------------------------------------------------- image

def test_image_has_the_requested_resolution():
    for diagram in (np.empty((0, 2)), DIAGRAM):
        assert vec.persistence_image(diagram, resolution=(12, 9)).shape == (9, 12)


def test_image_is_nonnegative_and_nonzero_for_a_real_diagram():
    image = vec.persistence_image(DIAGRAM, resolution=(16, 16))
    assert (image >= 0).all() and image.sum() > 0


def test_default_window_puts_the_most_persistent_point_inside_the_image():
    # one point, birth 0.4 and lifetime 1.2: the brightest pixel must be that
    # point, not an edge or a corner of the grid
    image = vec.persistence_image(np.array([[0.4, 1.6]]), resolution=(20, 20))
    row, col = np.unravel_index(image.argmax(), image.shape)
    assert 0 < row < 19 and 0 < col < 19
    edge = max(image[0].max(), image[-1].max(), image[:, 0].max(), image[:, -1].max())
    assert image.max() > 2 * edge


def test_linear_weighting_suppresses_points_on_the_diagonal():
    on_diagonal = np.array([[0.5, 0.5], [0.9, 0.9]])
    assert vec.persistence_image(on_diagonal, resolution=(8, 8)).sum() == 0.0


def test_unweighted_image_keeps_short_bars():
    short = np.array([[0.5, 0.52]])
    weighted = vec.persistence_image(short, resolution=(8, 8), sigma=0.1)
    unweighted = vec.persistence_image(short, resolution=(8, 8), sigma=0.1,
                                       weight="none")
    assert unweighted.sum() > weighted.sum()


def test_unknown_weight_rejected():
    with pytest.raises(ValueError):
        vec.persistence_image(DIAGRAM, weight="quadratic")


def test_fixed_ranges_make_two_images_comparable():
    ranges = dict(birth_range=(0.0, 2.0), lifetime_range=(0.0, 2.0), sigma=0.15)
    a = vec.persistence_image(DIAGRAM, resolution=(10, 10), **ranges)
    b = vec.persistence_image(DIAGRAM + 0.01, resolution=(10, 10), **ranges)
    assert a.shape == b.shape
    assert np.abs(a - b).max() < a.max()      # a small move is a small change


# --------------------------------------------------------------- the bundle

def test_diagram_features_have_a_shape_set_only_by_the_parameters():
    circle = rips_persistence(ds.circle(40, noise=0.05, seed=1), max_dim=1, threshold=2.0)
    noise = rips_persistence(ds.uniform_box(40, seed=1), max_dim=1, threshold=2.0)
    a = vec.diagram_features(circle, max_scale=2.0, n_layers=3, n_steps=20)
    b = vec.diagram_features(noise, max_scale=2.0, n_layers=3, n_steps=20)
    assert a.shape == b.shape == (2 * 3 * 20,)
    assert not np.allclose(a, b)


def test_default_grid_spans_the_finite_part():
    grid = vec.default_grid(DIAGRAM, n_steps=10)
    assert len(grid) == 10 and grid[0] == 0.0 and grid[-1] >= DIAGRAM.max()
