"""Turning a persistence diagram into something a model can eat.

A diagram is a *multiset* of points of varying cardinality.  It has no addition,
no scalar multiplication and no fixed dimension, so it cannot be handed to a
linear layer, an SVM or a transformer.  The fix is to map it into a vector
space, and the maps worth using are the ones that are Lipschitz in the
bottleneck distance -- then the stability theorem survives into feature space
and a small perturbation of the data is still a small perturbation of the
features.

Three standard choices, cheapest first:

``betti_curve``           step function counting classes alive at each scale.
``persistence_landscape`` ranked tent functions; 1-Lipschitz, so means and
                          hypothesis tests are well defined (Bubenik 2015).
``persistence_image``     weighted Gaussians on the (birth, lifetime) plane,
                          discretised to a fixed grid -- a direct CNN input
                          (Adams et al. 2017).

All three take a diagram as an ``(n, 2)`` array of ``(birth, death)`` and return
a fixed-size array whatever ``n`` is, which is the entire point.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _finite(diagram: np.ndarray) -> np.ndarray:
    """Drop essential classes.  They carry no lifetime, so every vectorisation
    here would have to invent one; dropping them is the honest default, and the
    count of them is a separate feature you can concatenate yourself.
    """
    diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
    if len(diagram) == 0:
        return diagram
    return diagram[np.isfinite(diagram).all(axis=1)]


def default_grid(diagram: np.ndarray, n_steps: int = 100) -> np.ndarray:
    """A scale grid spanning the finite part of a diagram."""
    finite = _finite(diagram)
    top = float(finite.max()) if len(finite) else 1.0
    return np.linspace(0.0, top * 1.02, n_steps)


def betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``t -> #{i : b_i <= t < d_i}``, evaluated on ``grid``.

    The cheapest vectorisation, and not Lipschitz: one bar moving by ``eps`` can
    change the curve by a whole unit.  Still a perfectly good baseline feature,
    and the thing plotted as a "Betti curve".  Essential classes are kept here,
    since a bar alive forever is alive at every grid point.
    """
    diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
    grid = np.asarray(grid, dtype=float).ravel()
    if len(diagram) == 0:
        return np.zeros_like(grid)
    alive = (diagram[:, 0][None, :] <= grid[:, None]) & (diagram[:, 1][None, :] > grid[:, None])
    return alive.sum(axis=1).astype(float)


def persistence_landscape(diagram: np.ndarray, grid: np.ndarray,
                          n_layers: int = 5) -> np.ndarray:
    """Persistence landscape (Bubenik 2015), as an ``(n_layers, len(grid))`` array.

    Each bar ``[b, d)`` contributes a tent
    ``L(t) = max(0, min(t - b, d - t))`` -- zero outside the bar, peaking at
    ``(d - b) / 2`` in the middle.  Layer ``j`` is the ``j``-th largest tent
    value at each ``t``, so layer 0 tracks the dominant feature, layer 1 the
    runner-up, and so on.

    The result is 1-Lipschitz in the bottleneck distance, which is what makes it
    safe to average landscapes across a batch and do statistics on the mean.
    """
    diagram = _finite(diagram)
    grid = np.asarray(grid, dtype=float).ravel()
    if len(diagram) == 0:
        return np.zeros((n_layers, len(grid)))

    births, deaths = diagram[:, 0][None, :], diagram[:, 1][None, :]
    tents = np.maximum(0.0, np.minimum(grid[:, None] - births, deaths - grid[:, None]))

    # sort each row descending, then pad to n_layers if there are fewer bars
    ranked = -np.sort(-tents, axis=1)
    if ranked.shape[1] < n_layers:
        ranked = np.pad(ranked, ((0, 0), (0, n_layers - ranked.shape[1])))
    return ranked[:, :n_layers].T


def persistence_image(diagram: np.ndarray, resolution: Tuple[int, int] = (20, 20),
                      sigma: float | None = None,
                      birth_range: Tuple[float, float] | None = None,
                      lifetime_range: Tuple[float, float] | None = None,
                      weight: str = "linear") -> np.ndarray:
    """Persistence image (Adams et al. 2017), as a ``resolution`` array.

    Each point is first moved to ``(birth, lifetime)`` coordinates -- so the
    diagonal, where the noise lives, becomes the horizontal axis -- then
    replaced by a Gaussian of width ``sigma``, weighted so that points near the
    diagonal count for little.  The surface is sampled on a fixed grid, and that
    grid is the feature vector.

    ``weight`` is ``"linear"`` (weight = lifetime, the usual choice, and the one
    that makes the map stable) or ``"none"``.  Fixing ``birth_range`` and
    ``lifetime_range`` across a dataset is what keeps the images comparable;
    left unset, they are read off this diagram alone.
    """
    diagram = _finite(diagram)
    n_birth, n_lifetime = resolution

    points = np.column_stack([diagram[:, 0], diagram[:, 1] - diagram[:, 0]]) \
        if len(diagram) else np.zeros((0, 2))

    if birth_range is None:
        birth_range = (0.0, float(points[:, 0].max()) if len(points) else 1.0)
    if lifetime_range is None:
        lifetime_range = (0.0, float(points[:, 1].max()) if len(points) else 1.0)
    if sigma is None:
        span = max(birth_range[1] - birth_range[0], lifetime_range[1] - lifetime_range[0])
        sigma = max(span, 1e-12) / max(max(resolution), 1) * 2.0

    birth_axis = np.linspace(*birth_range, n_birth)
    lifetime_axis = np.linspace(*lifetime_range, n_lifetime)
    image = np.zeros((n_lifetime, n_birth))
    if len(points) == 0:
        return image

    weights = points[:, 1] if weight == "linear" else np.ones(len(points))
    if weight not in ("linear", "none"):
        raise ValueError(f"unknown weight {weight!r}")

    for (birth, lifetime), w in zip(points, weights):
        if w == 0.0:
            continue
        bump = (np.exp(-((lifetime_axis - lifetime) ** 2) / (2 * sigma ** 2))[:, None]
                * np.exp(-((birth_axis - birth) ** 2) / (2 * sigma ** 2))[None, :])
        image += w * bump

    return image / (2 * np.pi * sigma ** 2)


def diagram_features(result, max_scale: float, dims: Sequence[int] = (0, 1),
                     n_layers: int = 3, n_steps: int = 40) -> np.ndarray:
    """One flat feature vector per point cloud: landscapes of each dimension,
    concatenated.

    The shape depends only on ``dims``, ``n_layers`` and ``n_steps``, never on
    how many bars a particular cloud produced -- so a stack of these is a design
    matrix you can hand straight to any estimator.

    ``max_scale`` fixes the grid, and must be the *same* for every cloud in a
    dataset: a landscape sampled on a grid of its own choosing is not comparable
    with anything.  Use one value derived from the whole dataset, computed
    inside the cross-validation loop if it depends on the data.
    """
    grid = np.linspace(0.0, float(max_scale), n_steps)
    blocks = [
        persistence_landscape(result[dim], grid, n_layers).ravel()
        for dim in dims
    ]
    return np.concatenate(blocks) if blocks else np.zeros(0)
