"""Distances between persistence diagrams.

Both are computed on the standard *augmented* bipartite problem: every point of
one diagram is matched either to a point of the other diagram or to its own
orthogonal projection onto the diagonal, and leftover diagonal slots match each
other for free.  Bottleneck minimises the largest matched cost, Wasserstein-q
the q-th root of the summed q-th powers.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Diagram = np.ndarray


def _split(dgm: Diagram) -> Tuple[np.ndarray, np.ndarray]:
    dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
    finite = np.isfinite(dgm[:, 1])
    return dgm[finite], dgm[~finite]


def _augmented_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(n+m) x (n+m)`` L-infinity cost matrix for the augmented problem."""
    n, m = len(a), len(b)
    size = n + m
    cost = np.zeros((size, size))

    if n and m:
        cost[:n, :m] = np.abs(a[:, None, :] - b[None, :, :]).max(axis=-1)
    # a_i -> diagonal, only on its own dedicated slot
    diag_a = (a[:, 1] - a[:, 0]) / 2.0 if n else np.zeros(0)
    diag_b = (b[:, 1] - b[:, 0]) / 2.0 if m else np.zeros(0)
    cost[:n, m:] = np.inf
    cost[n:, :m] = np.inf
    if n:
        cost[np.arange(n), m + np.arange(n)] = diag_a
    if m:
        cost[n + np.arange(m), np.arange(m)] = diag_b
    # diagonal-to-diagonal costs nothing
    return cost


class _HopcroftKarp:
    """Maximum bipartite matching, used by the bottleneck binary search."""

    def __init__(self, adjacency: List[List[int]], n_right: int):
        self.adj = adjacency
        self.n_left = len(adjacency)
        self.n_right = n_right

    def max_matching(self) -> int:
        INF = float("inf")
        # augmenting paths are explored recursively and can be as long as the
        # left vertex set
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, self.n_left * 2 + 1000))
        match_l = [-1] * self.n_left
        match_r = [-1] * self.n_right
        result = 0
        while True:
            dist = [INF] * self.n_left
            queue = []
            for u in range(self.n_left):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
            found = False
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in self.adj[u]:
                    w = match_r[v]
                    if w == -1:
                        found = True
                    elif dist[w] == INF:
                        dist[w] = dist[u] + 1
                        queue.append(w)
            if not found:
                sys.setrecursionlimit(limit)
                return result

            def try_augment(u: int) -> bool:
                for v in self.adj[u]:
                    w = match_r[v]
                    if w == -1 or (dist[w] == dist[u] + 1 and try_augment(w)):
                        match_l[u] = v
                        match_r[v] = u
                        return True
                dist[u] = INF
                return False

            for u in range(self.n_left):
                if match_l[u] == -1 and try_augment(u):
                    result += 1


def _min_max_matching(cost: np.ndarray) -> float:
    """Smallest ``eps`` admitting a perfect matching using only edges of cost
    ``<= eps``.  Exact: binary search over the sorted distinct costs.
    """
    size = cost.shape[0]
    if size == 0:
        return 0.0
    candidates = np.unique(cost[np.isfinite(cost)])
    if len(candidates) == 0:
        return float("inf")

    def feasible(eps: float) -> bool:
        adjacency = [np.flatnonzero(cost[u] <= eps).tolist() for u in range(size)]
        return _HopcroftKarp(adjacency, size).max_matching() == size

    if not feasible(candidates[-1]):
        return float("inf")

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def bottleneck_distance(dgm1: Diagram, dgm2: Diagram) -> float:
    """Exact bottleneck distance between two persistence diagrams.

    Essential (infinite) classes are matched among themselves on their birth
    values; if the two diagrams disagree on how many they have, the distance is
    infinite -- they are topologically different at every scale.
    """
    a, a_inf = _split(dgm1)
    b, b_inf = _split(dgm2)

    if len(a_inf) != len(b_inf):
        return float("inf")

    finite = _min_max_matching(_augmented_cost(a, b)) if len(a) + len(b) else 0.0

    essential = 0.0
    if len(a_inf):
        cost = np.abs(a_inf[:, 0][:, None] - b_inf[:, 0][None, :])
        essential = _min_max_matching(cost)

    return max(finite, essential)


def wasserstein_distance(dgm1: Diagram, dgm2: Diagram, order: float = 2.0) -> float:
    """Exact q-Wasserstein distance (an optimal assignment, via Hungarian)."""
    a, a_inf = _split(dgm1)
    b, b_inf = _split(dgm2)
    if len(a_inf) != len(b_inf):
        return float("inf")

    total = 0.0
    if len(a) + len(b):
        cost = _augmented_cost(a, b)
        # Hungarian cannot see infinities: replace them with a prohibitive cost.
        finite_max = cost[np.isfinite(cost)].max(initial=0.0)
        cost = np.where(np.isfinite(cost), cost, finite_max * (cost.shape[0] + 1) + 1.0)
        rows, cols = linear_sum_assignment(cost ** order)
        total += float((cost[rows, cols] ** order).sum())
    if len(a_inf):
        cost = np.abs(a_inf[:, 0][:, None] - b_inf[:, 0][None, :])
        rows, cols = linear_sum_assignment(cost ** order)
        total += float((cost[rows, cols] ** order).sum())
    return total ** (1.0 / order)


def diagram_distance_matrix(
    diagrams: Sequence[Diagram], metric: str = "bottleneck", **kwargs
) -> np.ndarray:
    """Symmetric matrix of pairwise diagram distances."""
    fn = {"bottleneck": bottleneck_distance, "wasserstein": wasserstein_distance}[metric]
    n = len(diagrams)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = fn(diagrams[i], diagrams[j], **kwargs)
    return out
