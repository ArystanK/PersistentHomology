"""Unfiltered simplicial complexes over the reals: signed boundary matrices,
Betti numbers by rank-nullity, and the Hodge (combinatorial) Laplacian.

The rest of the package works over GF(2), where signs vanish and persistence is
a column reduction.  This module is the other half of the story: a *fixed*
complex, real coefficients, and dense linear algebra.  It exists because two
things only make sense with signs and an inner product:

* the identity ``d o d = 0`` as an actual matrix product with cancelling signs;
* the Hodge Laplacian ``L_k = d_k^T d_k + d_{k+1} d_{k+1}^T``, whose kernel
  dimension is ``beta_k`` and whose kernel vectors are canonical representatives
  of the homology classes.

``L_0`` is the graph Laplacian ``D - A``, so everything here specialises at
``k = 0`` to the matrix behind spectral clustering.

Orientation convention: every simplex is stored as a tuple of vertices in
increasing order, and

    d[v_0, ..., v_k] = sum_i (-1)^i [v_0, ..., v_i-hat, ..., v_k].

Matrices are dense.  These complexes are meant to have tens or hundreds of
simplices -- blackboard examples, unit tests, and the small demonstrations in
the tutorial -- not the hundreds of thousands a Rips filtration produces.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Simplex = Tuple[int, ...]

# Eigenvalues below this are read as zero.  Chosen well above the rounding error
# of a symmetric eigensolver on a 0,+-1 matrix of this size, and well below the
# smallest nonzero eigenvalue any of these complexes produces.
ZERO_TOL = 1e-9


class SimplicialComplex:
    """A finite abstract simplicial complex with real coefficients.

    Construction closes the given simplices under taking faces, so
    ``SimplicialComplex([(0, 1, 2)])`` is the *filled* triangle: three vertices,
    three edges and the triangle itself.
    """

    __slots__ = ("by_dim", "_index")

    def __init__(self, simplices: Iterable[Sequence[int]]):
        closed: set[Simplex] = set()
        pending: List[Simplex] = []
        for raw in simplices:
            vertices = tuple(sorted(set(int(v) for v in raw)))
            if not vertices:
                raise ValueError("a simplex needs at least one vertex")
            pending.append(vertices)

        # close under taking faces: a simplex drags in its facets, which drag in
        # theirs, all the way down to the vertices
        while pending:
            simplex = pending.pop()
            if simplex in closed:
                continue
            closed.add(simplex)
            if len(simplex) > 1:
                pending.extend(
                    simplex[:k] + simplex[k + 1:] for k in range(len(simplex))
                )

        max_dim = max(len(s) for s in closed) - 1 if closed else -1
        self.by_dim: List[List[Simplex]] = [[] for _ in range(max_dim + 1)]
        for simplex in sorted(closed, key=lambda s: (len(s), s)):
            self.by_dim[len(simplex) - 1].append(simplex)

        self._index: Dict[Simplex, int] = {}
        for row in self.by_dim:
            for i, simplex in enumerate(row):
                self._index[simplex] = i

    # ---------------------------------------------------------------- basics

    @classmethod
    def from_graph(cls, n_vertices: int, edges: Iterable[Sequence[int]]) -> "SimplicialComplex":
        """A 1-dimensional complex: vertices ``0..n-1`` and the given edges, no
        triangles.  This is a graph, held as a complex.
        """
        return cls([(v,) for v in range(n_vertices)] + [tuple(e) for e in edges])

    @classmethod
    def cycle(cls, n: int) -> "SimplicialComplex":
        """The cycle graph on ``n`` vertices: a hollow ``n``-gon, ``beta = (1, 1)``."""
        return cls.from_graph(n, [(i, (i + 1) % n) for i in range(n)])

    @classmethod
    def boundary_of_simplex(cls, n_vertices: int) -> "SimplicialComplex":
        """All *proper* faces of the simplex on ``n`` vertices -- a hollow shell,
        topologically the sphere ``S^(n-2)``.  ``n = 4`` is the tetrahedron
        boundary, a 2-sphere.
        """
        full = tuple(range(n_vertices))
        return cls([full[:k] + full[k + 1:] for k in range(n_vertices)])

    @classmethod
    def from_filtration(cls, filtration, threshold: float) -> "SimplicialComplex":
        """The subcomplex of a :class:`~persistent_homology.complexes.Filtration`
        at a fixed scale: every simplex whose filtration value is ``<= threshold``.
        """
        keep = [s for s, v in zip(filtration.simplices, filtration.values) if v <= threshold]
        return cls(keep)

    def __len__(self) -> int:
        return sum(len(row) for row in self.by_dim)

    def __contains__(self, simplex: Sequence[int]) -> bool:
        return tuple(sorted(simplex)) in self._index

    def __repr__(self) -> str:
        counts = ", ".join(f"n{k}={len(row)}" for k, row in enumerate(self.by_dim))
        return f"SimplicialComplex({counts})"

    @property
    def dim(self) -> int:
        return len(self.by_dim) - 1

    def simplices(self, k: int) -> List[Simplex]:
        """The ``k``-simplices, in the fixed order used by every matrix here."""
        if 0 <= k < len(self.by_dim):
            return self.by_dim[k]
        return []

    def count(self, k: int) -> int:
        """``n_k``, the number of ``k``-simplices."""
        return len(self.simplices(k))

    def counts(self) -> List[int]:
        return [len(row) for row in self.by_dim]

    def index(self, simplex: Sequence[int]) -> int:
        """Position of a simplex among those of its own dimension."""
        return self._index[tuple(sorted(simplex))]

    # ------------------------------------------------------------- operators

    def boundary_matrix(self, k: int) -> np.ndarray:
        """``d_k``, of shape ``(n_{k-1}, n_k)``, with entries in ``{0, +-1}``.

        Column ``j`` holds the signed faces of the ``j``-th ``k``-simplex.
        ``d_0`` is the empty ``(0, n_0)`` matrix: a vertex has no boundary.
        """
        if k <= 0 or k > self.dim:
            rows = self.count(k - 1) if k > 0 else 0
            return np.zeros((rows, self.count(k)))

        rows, cols = self.count(k - 1), self.count(k)
        matrix = np.zeros((rows, cols))
        for j, simplex in enumerate(self.simplices(k)):
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1:]
                matrix[self._index[face], j] = (-1.0) ** i
        return matrix

    def hodge_laplacian(self, k: int) -> np.ndarray:
        """``L_k = d_k^T d_k + d_{k+1} d_{k+1}^T``, of shape ``(n_k, n_k)``.

        Symmetric positive semidefinite.  The first term is the *down* Laplacian
        (how ``k``-simplices meet through shared faces), the second the *up*
        Laplacian (how they meet through shared cofaces).  At ``k = 0`` the down
        term vanishes and what is left is the graph Laplacian ``D - A``.
        """
        down = self.boundary_matrix(k)
        up = self.boundary_matrix(k + 1)
        return down.T @ down + up @ up.T

    def graph_laplacian(self) -> np.ndarray:
        """``D - A`` of the 1-skeleton, built directly from degrees and
        adjacency -- an independent construction to check ``L_0`` against.
        """
        n = self.count(0)
        adjacency = np.zeros((n, n))
        for u, v in self.simplices(1):
            adjacency[u, v] = adjacency[v, u] = 1.0
        return np.diag(adjacency.sum(axis=1)) - adjacency

    # ------------------------------------------------------------- invariants

    def rank(self, k: int) -> int:
        """``rank d_k``."""
        matrix = self.boundary_matrix(k)
        if matrix.size == 0:
            return 0
        return int(np.linalg.matrix_rank(matrix))

    def betti_number(self, k: int) -> int:
        """``beta_k = nullity d_k - rank d_{k+1} = n_k - rank d_k - rank d_{k+1}``."""
        if k < 0 or k > self.dim:
            return 0
        return self.count(k) - self.rank(k) - self.rank(k + 1)

    def betti_numbers(self) -> List[int]:
        return [self.betti_number(k) for k in range(self.dim + 1)]

    def euler_characteristic(self) -> int:
        """``sum_k (-1)^k n_k``, which also equals ``sum_k (-1)^k beta_k``."""
        return sum((-1) ** k * n for k, n in enumerate(self.counts()))

    def spectrum(self, k: int, tol: float = ZERO_TOL) -> np.ndarray:
        """Eigenvalues of ``L_k``, ascending, with near-zero values snapped to 0.

        The snapping is cosmetic -- it keeps ``-3.1e-16`` from being printed as a
        negative eigenvalue of a positive semidefinite matrix.
        """
        laplacian = self.hodge_laplacian(k)
        if laplacian.size == 0:
            return np.zeros(0)
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues[np.abs(eigenvalues) < tol] = 0.0
        return eigenvalues

    def spectral_gap(self, k: int, tol: float = ZERO_TOL) -> float:
        """The smallest *nonzero* eigenvalue of ``L_k``, or ``inf`` if there is
        none.  Unlike ``beta_k`` this keeps moving as the complex is refined.
        """
        eigenvalues = self.spectrum(k, tol)
        nonzero = eigenvalues[eigenvalues > tol]
        return float(nonzero[0]) if len(nonzero) else float("inf")

    def betti_from_laplacian(self, k: int, tol: float = ZERO_TOL) -> int:
        """``beta_k = dim ker L_k``: the multiplicity of the eigenvalue 0.

        The same number as :meth:`betti_number`, reached through an eigensolver
        instead of two rank computations.
        """
        if k < 0 or k > self.dim:
            return 0
        return int((np.abs(self.spectrum(k, tol)) <= tol).sum())

    # ------------------------------------------------------------ Hodge theory

    def harmonic_basis(self, k: int, tol: float = ZERO_TOL) -> np.ndarray:
        """An orthonormal basis of ``ker L_k``, as columns of an ``(n_k, beta_k)``
        array.

        These are the *harmonic* chains: cycles that are also orthogonal to every
        boundary.  Each homology class contains exactly one of them (up to the
        choice of basis within the kernel), which is the canonical answer to
        "where is the hole?" that a barcode does not give.
        """
        laplacian = self.hodge_laplacian(k)
        if laplacian.size == 0:
            return np.zeros((self.count(k), 0))
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        return eigenvectors[:, np.abs(eigenvalues) <= tol]

    def hodge_decomposition(self, chain: np.ndarray, k: int) -> Dict[str, np.ndarray]:
        """Split a ``k``-chain into the three orthogonal pieces of

            C_k = im d_{k+1}  (+)  ker L_k  (+)  im d_k^T.

        Returns the ``boundary``, ``harmonic`` and ``coboundary`` components.
        For ``k = 1`` and a chain of pairwise comparisons this is HodgeRank: the
        coboundary part is a global ranking, the harmonic part is genuinely
        cyclic preference, and the boundary part is local inconsistency.
        """
        chain = np.asarray(chain, dtype=float).ravel()
        if chain.shape[0] != self.count(k):
            raise ValueError(
                f"chain has length {chain.shape[0]}, expected n_{k} = {self.count(k)}"
            )

        boundary = _project_onto_columns(self.boundary_matrix(k + 1), chain)
        coboundary = _project_onto_columns(self.boundary_matrix(k).T, chain)
        return {
            "boundary": boundary,
            "harmonic": chain - boundary - coboundary,
            "coboundary": coboundary,
        }


def _project_onto_columns(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Orthogonal projection of ``vector`` onto the column space of ``matrix``.

    Uses an SVD-derived orthonormal basis rather than the normal equations,
    which are ill-conditioned exactly when a boundary matrix is rank-deficient
    -- and boundary matrices always are.
    """
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros_like(vector)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    basis = u[:, singular_values > ZERO_TOL * max(1.0, float(singular_values[0]))]
    if basis.shape[1] == 0:
        return np.zeros_like(vector)
    return basis @ (basis.T @ vector)


def subdivide_cycle(n: int, times: int = 1) -> SimplicialComplex:
    """The hollow ``n``-gon with every edge subdivided ``times`` over.

    ``beta_1`` stays 1 no matter how often you subdivide -- it is still one loop
    -- but the spectrum of ``L_1`` keeps changing.  That contrast is the point.
    """
    return SimplicialComplex.cycle(n * 2 ** times)


def format_spectrum(eigenvalues: Sequence[float], decimals: int = 3) -> str:
    """``{0, 1, 1, 3, 3, 4}`` -- eigenvalues the way the slides write them."""
    def fmt(value: float) -> str:
        rounded = round(float(value), decimals)
        if abs(rounded - round(rounded)) < 10 ** -decimals:
            return str(int(round(rounded)))
        return f"{rounded:g}"

    return "{" + ", ".join(fmt(v) for v in eigenvalues) + "}"
