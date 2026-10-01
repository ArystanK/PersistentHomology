"""The standard persistence algorithm: column reduction of the boundary matrix
over GF(2), with the clearing ("twist") optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .complexes import Filtration, rips_filtration_from_points


def _symmetric_difference(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Add two GF(2) columns held as ascending index lists."""
    out: List[int] = []
    i = j = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        ai, bj = a[i], b[j]
        if ai < bj:
            out.append(ai)
            i += 1
        elif ai > bj:
            out.append(bj)
            j += 1
        else:                      # 1 + 1 = 0
            i += 1
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def _reduce(
    columns: List[List[int]],
    dims: np.ndarray,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Reduce the boundary matrix left to right so that no two non-empty columns
    share a lowest non-zero entry.

    Returns the reduced columns and a ``low -> column`` pivot map.  Every pivot
    ``(i, j)`` is a persistence pair: simplex ``i`` creates a class that simplex
    ``j`` destroys.

    Columns are processed in order of *decreasing* dimension so that clearing
    applies: once index ``i`` is known to be a pivot it is a birth simplex that
    got killed, its own column is provably reducible to zero, and we can skip
    reducing it entirely.  On Rips complexes this removes most of the work.
    """
    n = len(columns)
    reduced: List[List[int]] = [None] * n  # type: ignore[list-item]
    pivot_to_col: Dict[int, int] = {}
    cleared: set[int] = set()

    max_d = int(dims.max()) if n else -1
    by_dim: List[List[int]] = [[] for _ in range(max_d + 1)]
    for j in range(n):
        by_dim[int(dims[j])].append(j)

    for d in range(max_d, -1, -1):
        for j in by_dim[d]:
            if j in cleared:
                reduced[j] = []
                continue
            col = columns[j]
            while col and col[-1] in pivot_to_col:
                col = _symmetric_difference(col, reduced[pivot_to_col[col[-1]]])
            reduced[j] = col
            if col:
                low = col[-1]
                pivot_to_col[low] = j
                cleared.add(low)

    return reduced, pivot_to_col


@dataclass
class PersistenceResult:
    """Persistence pairs of a filtration, grouped by homological dimension."""

    diagrams: Dict[int, np.ndarray]
    filtration: Filtration | None = None
    pairs: List[Tuple[int, int | None]] = field(default_factory=list)

    def __getitem__(self, dim: int) -> np.ndarray:
        return self.diagrams.get(dim, np.empty((0, 2)))

    @property
    def max_dim(self) -> int:
        return max(self.diagrams) if self.diagrams else -1

    def betti_numbers(self, threshold: float) -> List[int]:
        """Betti numbers of the subcomplex at ``threshold``, read off the
        diagrams: count the bars alive at that value.
        """
        return [
            int(((self[d][:, 0] <= threshold) & (self[d][:, 1] > threshold)).sum())
            for d in range(self.max_dim + 1)
        ]

    def most_stable_scale(self, cap: float | None = None) -> float:
        """Midpoint of the widest interval on which the Betti vector is constant.

        Every birth and every death is a point where some Betti number changes,
        so between two consecutive events nothing happens.  The widest such gap
        is the scale at which the shape's own topology is most robust to how the
        cloud was sampled -- the principled answer to "which epsilon?" once the
        whole filtration is in hand.

        Intervals where the complex has collapsed to a single featureless blob,
        ``b = (1, 0, ...)``, are skipped: that is the trivial answer every cloud
        eventually gives, and being unbounded it would otherwise always win.
        """
        finite = [
            v for dgm in self.diagrams.values()
            for v in np.concatenate([dgm[:, 0], dgm[np.isfinite(dgm[:, 1]), 1]]).tolist()
        ]
        if cap is None:
            cap = max(finite) if finite else 1.0

        events = {0.0, float(cap)} | {v for v in finite if v <= cap}
        grid = np.array(sorted(events))
        if len(grid) < 2:
            return float(grid.mean())

        trivial = [1] + [0] * self.max_dim
        midpoints = (grid[:-1] + grid[1:]) / 2
        for k in np.argsort(-np.diff(grid)):
            if self.betti_numbers(float(midpoints[k])) != trivial:
                return float(midpoints[k])
        return float(midpoints[int(np.argmax(np.diff(grid)))])

    def most_persistent(self, dim: int, k: int = 1) -> np.ndarray:
        """The ``k`` longest bars in dimension ``dim`` (infinite bars first)."""
        dgm = self[dim]
        if len(dgm) == 0:
            return dgm
        lifetimes = dgm[:, 1] - dgm[:, 0]
        return dgm[np.argsort(-lifetimes)[:k]]

    def summary(self) -> str:
        lines = []
        for d in range(self.max_dim + 1):
            dgm = self[d]
            finite = dgm[np.isfinite(dgm[:, 1])]
            n_inf = len(dgm) - len(finite)
            longest = (finite[:, 1] - finite[:, 0]).max() if len(finite) else 0.0
            lines.append(
                f"  H{d}: {len(dgm):4d} classes "
                f"({n_inf} essential, longest finite bar {longest:.4f})"
            )
        return "\n".join(lines)


def persistence(
    filtration: Filtration,
    max_dim: int | None = None,
    min_persistence: float = 0.0,
    keep_filtration: bool = False,
) -> PersistenceResult:
    """Compute persistent homology of a filtration over GF(2)."""
    columns = filtration.boundary_columns()
    dims = filtration.dims
    values = filtration.values
    reduced, pivot_to_col = _reduce(columns, dims)

    if max_dim is None:
        max_dim = filtration.max_dim

    births_that_die = set(pivot_to_col)
    diagrams: Dict[int, List[Tuple[float, float]]] = {d: [] for d in range(max_dim + 1)}
    pairs: List[Tuple[int, int | None]] = []

    for low, j in pivot_to_col.items():
        d = int(dims[low])
        if d > max_dim:
            continue
        birth, death = float(values[low]), float(values[j])
        if death - birth > min_persistence:
            diagrams[d].append((birth, death))
            pairs.append((low, j))

    for i in range(len(filtration)):
        if reduced[i] or i in births_that_die:
            continue
        d = int(dims[i])
        if d > max_dim:
            continue
        diagrams[d].append((float(values[i]), float("inf")))
        pairs.append((i, None))

    arrays = {
        d: np.array(sorted(v), dtype=float).reshape(-1, 2)
        for d, v in diagrams.items()
    }
    return PersistenceResult(arrays, filtration if keep_filtration else None, pairs)


def rips_persistence(
    points: np.ndarray,
    max_dim: int = 1,
    threshold: float | None = None,
    metric: str = "euclidean",
    min_persistence: float = 0.0,
    **kwargs,
) -> PersistenceResult:
    """Persistent homology of the Vietoris-Rips filtration of a point cloud.

    ``max_dim`` is the highest *homology* dimension wanted; the complex itself is
    built one dimension higher, since a ``d``-cycle can only be filled in by
    ``(d+1)``-simplices.
    """
    filt = rips_filtration_from_points(points, max_dim + 1, threshold, metric, **kwargs)
    return persistence(filt, max_dim=max_dim, min_persistence=min_persistence)
