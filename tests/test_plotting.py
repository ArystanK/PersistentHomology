import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence

RESULT = rips_persistence(ds.circle(30, noise=0.05, seed=1), max_dim=1, threshold=2.2)


def test_dimension_colours_are_distinct_and_never_cycled():
    used = [viz.dim_style(d)[0] for d in range(len(viz.DIM_COLORS))]
    assert len(set(used)) == len(used)
    markers = [viz.dim_style(d)[1] for d in range(len(viz.DIM_MARKERS))]
    assert len(set(markers)) == len(markers)
    # past the fixed slots we fall back to one neutral, we do not reuse a hue
    assert viz.dim_style(99)[0] == viz.OTHER_COLOR


def test_persistence_diagram_draws_every_class():
    from matplotlib.collections import PathCollection

    ax = viz.persistence_diagram(RESULT)
    drawn = sum(c.get_offsets().shape[0]
                for c in ax.collections if isinstance(c, PathCollection))
    assert drawn == sum(len(d) for d in RESULT.diagrams.values())
    assert ax.get_legend() is not None
    plt.close("all")


def test_barcode_draws_one_line_per_bar_up_to_the_cap():
    ax = viz.barcode(RESULT, max_bars_per_dim=5)
    expected = sum(min(5, len(d)) for d in RESULT.diagrams.values())
    # each bar is one Line2D; infinite bars add an arrow marker line
    assert len(ax.lines) >= expected
    plt.close("all")


def test_betti_curves_start_at_the_point_count():
    ax = viz.betti_curves(RESULT)
    line = ax.lines[0]
    assert line.get_ydata()[0] == 30
    plt.close("all")


def test_summary_figure_has_three_panels_and_saves(tmp_path):
    points = ds.circle(30, noise=0.05, seed=1)
    fig = viz.plot_summary(points, RESULT, title="test")
    assert len(fig.axes) == 3
    path = viz.save(fig, str(tmp_path / "summary.png"))
    assert (tmp_path / "summary.png").stat().st_size > 0
    assert path.endswith("summary.png")


def test_plots_handle_an_empty_diagram():
    empty = {0: np.empty((0, 2)), 1: np.empty((0, 2))}
    viz.persistence_diagram(empty)
    viz.barcode(empty)
    viz.betti_curves(empty)
    plt.close("all")


def test_three_dimensional_clouds_get_a_3d_axis():
    ax = viz.plot_points(ds.sphere(20, seed=2))
    assert ax.name == "3d"
    plt.close("all")


def test_plot_complex_draws_edges_and_triangles():
    from persistent_homology import SimplicialComplex

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]])
    complex_ = SimplicialComplex([(0, 1, 2), (1, 3), (2, 3)])
    ax = viz.plot_complex(points, complex_)
    assert len(ax.lines) == complex_.count(1)          # one line per edge
    assert len(ax.patches) == complex_.count(2)        # one filled polygon per triangle
    plt.close("all")


def test_plot_chain_draws_one_arrow_per_nonzero_edge():
    from persistent_homology import SimplicialComplex

    points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    complex_ = SimplicialComplex.cycle(4)
    chain = complex_.harmonic_basis(1)[:, 0]
    ax = viz.plot_chain(points, complex_, chain)
    arrows = [child for child in ax.texts if child.arrow_patch is not None]
    assert len(arrows) == int((np.abs(chain) > 1e-9).sum())
    plt.close("all")


def test_plot_chain_draws_a_plain_line_where_the_chain_vanishes():
    from persistent_homology import SimplicialComplex

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0]])
    complex_ = SimplicialComplex([(0, 1, 2), (0, 3), (1, 3)])
    chain = np.zeros(complex_.count(1))
    ax = viz.plot_chain(points, complex_, chain)
    assert len(ax.lines) == complex_.count(1)
    plt.close("all")
