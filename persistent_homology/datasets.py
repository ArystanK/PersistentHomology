"""Synthetic point clouds with known topology, for testing and demos."""

from __future__ import annotations

import numpy as np


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def circle(n: int = 100, radius: float = 1.0, noise: float = 0.0, seed=0) -> np.ndarray:
    """Points on a circle.  b0 = 1, b1 = 1."""
    rng = _rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = radius * np.column_stack([np.cos(theta), np.sin(theta)])
    return pts + noise * rng.standard_normal(pts.shape)


def annulus(n: int = 200, r_inner: float = 0.7, r_outer: float = 1.0, seed=0) -> np.ndarray:
    """Uniform sample of a 2D annulus.  b0 = 1, b1 = 1."""
    rng = _rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    # sqrt keeps the sample uniform in area rather than in radius
    r = np.sqrt(rng.uniform(r_inner ** 2, r_outer ** 2, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def two_circles(n: int = 120, radius: float = 1.0, separation: float = 3.0,
                noise: float = 0.03, seed=0) -> np.ndarray:
    """Two disjoint circles.  b0 = 2, b1 = 2."""
    rng = _rng(seed)
    a = circle(n // 2, radius, noise, rng)
    b = circle(n - n // 2, radius, noise, rng) + np.array([separation, 0.0])
    return np.vstack([a, b])


def figure_eight(n: int = 150, radius: float = 1.0, noise: float = 0.02, seed=0) -> np.ndarray:
    """Two circles meeting at a point.  b0 = 1, b1 = 2."""
    rng = _rng(seed)
    a = circle(n // 2, radius, noise, rng) - np.array([radius, 0.0])
    b = circle(n - n // 2, radius, noise, rng) + np.array([radius, 0.0])
    return np.vstack([a, b])


def sphere(n: int = 150, radius: float = 1.0, dim: int = 2, noise: float = 0.0, seed=0) -> np.ndarray:
    """Uniform sample of the ``dim``-sphere in ``dim + 1`` dimensions.

    For ``dim = 2``: b0 = 1, b1 = 0, b2 = 1.
    """
    rng = _rng(seed)
    pts = rng.standard_normal((n, dim + 1))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return radius * pts + noise * rng.standard_normal(pts.shape)


def torus(n: int = 300, r_major: float = 2.0, r_minor: float = 1.0,
          noise: float = 0.0, seed=0) -> np.ndarray:
    """Sample of a torus in R^3 via rejection sampling (uniform on the surface).

    b0 = 1, b1 = 2, b2 = 1.
    """
    rng = _rng(seed)
    thetas = []
    while len(thetas) < n:
        candidate = rng.uniform(0, 2 * np.pi, n)
        accept = rng.uniform(0, 1, n) < (r_major + r_minor * np.cos(candidate)) / (r_major + r_minor)
        thetas.extend(candidate[accept].tolist())
    theta = np.array(thetas[:n])
    phi = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([
        (r_major + r_minor * np.cos(theta)) * np.cos(phi),
        (r_major + r_minor * np.cos(theta)) * np.sin(phi),
        r_minor * np.sin(theta),
    ])
    return pts + noise * rng.standard_normal(pts.shape)


def flat_torus(n: int = 200, seed=0) -> np.ndarray:
    """The product of two unit circles, embedded in R^4 as
    ``(cos a, sin a, cos b, sin b)``.

    Topologically a torus (b0 = 1, b1 = 2, b2 = 1), and far friendlier to
    Vietoris-Rips than the doughnut in R^3: every point looks the same, so a
    few hundred samples already resolve both loops.
    """
    rng = _rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    b = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)])


def uniform_box(n: int = 100, dim: int = 2, seed=0) -> np.ndarray:
    """Pure noise: a uniform sample of the unit cube.  b0 = 1, everything else 0."""
    return _rng(seed).uniform(0, 1, (n, dim))


def clusters(n: int = 150, centers=((0, 0), (5, 0), (2.5, 4)), spread: float = 0.4,
             seed=0) -> np.ndarray:
    """Gaussian blobs; b0 equals the number of centers over a range of scales."""
    rng = _rng(seed)
    centers = np.asarray(centers, dtype=float)
    which = rng.integers(0, len(centers), n)
    return centers[which] + spread * rng.standard_normal((n, centers.shape[1]))


def takens_embedding(series: np.ndarray, dimension: int = 3, delay: int = 1) -> np.ndarray:
    """Delay embedding of a 1D signal: row ``i`` is ``[x_i, x_{i+tau}, ...]``.

    A periodic signal traces a loop in the embedded cloud, so H1 detects
    periodicity without any spectral method.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series) - (dimension - 1) * delay
    if n <= 0:
        raise ValueError("series is too short for this dimension/delay")
    return np.column_stack([series[i * delay: i * delay + n] for i in range(dimension)])


def lorenz(n: int = 2000, dt: float = 0.01, sigma: float = 10.0, rho: float = 28.0,
           beta: float = 8.0 / 3.0, burn_in: int = 1000) -> np.ndarray:
    """A trajectory on the Lorenz attractor (RK4), for a non-toy demo cloud."""
    def deriv(state):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    state = np.array([1.0, 1.0, 1.0])
    out = np.empty((n, 3))
    for i in range(n + burn_in):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= burn_in:
            out[i - burn_in] = state
    return out
