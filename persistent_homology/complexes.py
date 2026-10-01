"""Construction of simplicial complexes and filtrations from point clouds."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Simplex = Tuple[int, ...]


def pairwise_distances(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Dense pairwise distance matrix for a ``(n, d)`` array of points.

    ``metric`` is one of ``euclidean``, ``manhattan`` or ``chebyshev``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"expected a (n, d) array of points, got shape {points.shape}")

    diff = points[:, None, :] - points[None, :, :]
    if metric == "euclidean":
        dist = np.sqrt((diff ** 2).sum(axis=-1))
    elif metric == "manhattan":
        dist = np.abs(diff).sum(axis=-1)
    elif metric == "chebyshev":
        dist = np.abs(diff).max(axis=-1)
    else:
        raise ValueError(f"unknown metric {metric!r}")

    np.fill_diagonal(dist, 0.0)
    return dist


def enclosing_radius(dist: np.ndarray) -> float:
    """Smallest radius at which the Rips complex becomes a cone (a contractible
    blob).  Beyond it no new topology can appear, so it is the natural default
    threshold: ``min_i max_j d(i, j)``.
    """
    if dist.shape[0] == 0:
        return 0.0
    return float(dist.max(axis=1).min())


class Filtration:
    """A finite filtered simplicial complex.

    Simplices are stored in *filtration order*: sorted by filtration value, then
    by dimension.  Because every face of a simplex has a value no larger and a
    dimension strictly smaller, this order always places faces before their
    cofaces -- exactly what the persistence reduction algorithm requires.
    """

    __slots__ = ("simplices", "values", "dims", "_index")

    def __init__(self, entries: Iterable[Tuple[Simplex, float]]):
        ordered = sorted(entries, key=lambda item: (item[1], len(item[0]), item[0]))
        self.simplices: List[Simplex] = [s for s, _ in ordered]
        self.values: np.ndarray = np.array([v for _, v in ordered], dtype=float)
        self.dims: np.ndarray = np.array([len(s) - 1 for s in self.simplices], dtype=np.int32)
        self._index = {s: i for i, s in enumerate(self.simplices)}

    def __len__(self) -> int:
        return len(self.simplices)

    def index(self, simplex: Simplex) -> int:
        return self._index[simplex]

    @property
    def max_dim(self) -> int:
        return int(self.dims.max()) if len(self.dims) else -1

    def boundary_columns(self) -> List[List[int]]:
        """Boundary matrix over GF(2), one column per simplex.

        Column ``j`` lists the filtration indices of the codimension-1 faces of
        simplex ``j``, sorted ascending.  Signs are irrelevant mod 2.
        """
        columns: List[List[int]] = []
        for simplex in self.simplices:
            if len(simplex) == 1:
                columns.append([])
                continue
            faces = [
                self._index[simplex[:k] + simplex[k + 1:]]
                for k in range(len(simplex))
            ]
            faces.sort()
            columns.append(faces)
        return columns

    def betti_numbers(self, threshold: float) -> List[int]:
        """Betti numbers of the subcomplex at ``threshold``, computed directly by
        rank arguments (an independent check on the persistence output).

        Note the top dimension is meaningless in a truncated complex: with no
        ``(d+1)``-simplices present, nothing can kill a ``d``-cycle, so the last
        entry counts cycles that a fuller complex would fill in.
        """
        from .persistence import _reduce  # local import avoids a cycle

        keep = self.values <= threshold
        n_kept = int(keep.sum())
        columns = self.boundary_columns()[:n_kept]
        dims = self.dims[:n_kept]
        reduced, _ = _reduce(columns, dims)

        max_d = int(dims.max()) if n_kept else -1
        ranks = [0] * (max_d + 2)          # rank of boundary map out of dim d
        n_simplices = [0] * (max_d + 2)
        for j, col in enumerate(reduced):
            d = int(dims[j])
            n_simplices[d] += 1
            if col:
                ranks[d] += 1
        return [n_simplices[d] - ranks[d] - ranks[d + 1] for d in range(max_d + 1)]


def rips_filtration(
    dist: np.ndarray,
    max_dim: int,
    threshold: float | None = None,
    max_simplices: int | None = 2_000_000,
) -> Filtration:
    """Vietoris-Rips filtration up to and including dimension ``max_dim``.

    A set of vertices spans a simplex as soon as *all* of its pairwise distances
    are below the current radius, so the filtration value of a simplex is the
    largest edge length inside it.  Simplices are grown one dimension at a time:
    a ``d``-simplex is extended by any vertex adjacent to all of its vertices
    and larger than its last vertex, which enumerates every clique exactly once.
    """
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    if dist.shape != (n, n):
        raise ValueError("distance matrix must be square")
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")
    if threshold is None:
        threshold = enclosing_radius(dist)

    entries: List[Tuple[Simplex, float]] = [((i,), 0.0) for i in range(n)]

    neighbours = [set(np.flatnonzero(dist[i] <= threshold).tolist()) - {i} for i in range(n)]

    # (vertices, filtration value, candidate vertices that may extend it)
    current: List[Tuple[Simplex, float, set]] = []
    for i in range(n):
        for j in sorted(v for v in neighbours[i] if v > i):
            cofaces = {v for v in neighbours[i] & neighbours[j] if v > j}
            current.append(((i, j), float(dist[i, j]), cofaces))
            entries.append(((i, j), float(dist[i, j])))

    for _ in range(2, max_dim + 1):
        nxt: List[Tuple[Simplex, float, set]] = []
        for verts, value, candidates in current:
            for v in sorted(candidates):
                new_value = max(value, max(float(dist[u, v]) for u in verts))
                new_verts = verts + (v,)
                entries.append((new_verts, new_value))
                nxt.append((new_verts, new_value, {w for w in candidates & neighbours[v] if w > v}))
        if max_simplices is not None and len(entries) > max_simplices:
            raise MemoryError(
                f"Rips complex exceeded {max_simplices} simplices at dimension {len(nxt[0][0]) - 1 if nxt else '?'}. "
                "Lower `threshold`, lower `max_dim`, or subsample the points."
            )
        current = nxt
        if not current:
            break

    return Filtration(entries)


def rips_filtration_from_points(
    points: np.ndarray,
    max_dim: int,
    threshold: float | None = None,
    metric: str = "euclidean",
    **kwargs,
) -> Filtration:
    return rips_filtration(pairwise_distances(points, metric), max_dim, threshold, **kwargs)


def filtration_from_simplices(entries: Sequence[Tuple[Sequence[int], float]]) -> Filtration:
    """Build a filtration from an explicit list of ``(vertices, value)`` pairs.

    Faces are *not* inferred; every face must be present, and its value must not
    exceed the value of its cofaces.
    """
    normalised = [(tuple(sorted(vs)), float(v)) for vs, v in entries]
    filt = Filtration(normalised)
    for simplex in filt.simplices:
        if len(simplex) == 1:
            continue
        for k in range(len(simplex)):
            face = simplex[:k] + simplex[k + 1:]
            if face not in filt._index:
                raise ValueError(f"simplex {simplex} is missing its face {face}")
            if filt.values[filt.index(face)] > filt.values[filt.index(simplex)]:
                raise ValueError(f"face {face} appears after its coface {simplex}")
    return filt
