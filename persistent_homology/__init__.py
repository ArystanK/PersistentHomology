"""A small, dependency-light persistent homology library.

Everything is implemented from scratch on NumPy: the Vietoris-Rips filtration,
the GF(2) boundary-matrix reduction that produces persistence pairs, and exact
bottleneck / Wasserstein distances between the resulting diagrams.

    >>> from persistent_homology import datasets, rips_persistence
    >>> result = rips_persistence(datasets.circle(60, noise=0.05), max_dim=1)
    >>> result.betti_numbers(threshold=0.6)
    [1, 1]
"""

from . import datasets, plotting, vectorization
from .complexes import (
    Filtration,
    enclosing_radius,
    filtration_from_simplices,
    pairwise_distances,
    rips_filtration,
    rips_filtration_from_points,
)
from .distances import (
    bottleneck_distance,
    diagram_distance_matrix,
    wasserstein_distance,
)
from .persistence import PersistenceResult, persistence, rips_persistence
from .simplicial import SimplicialComplex, format_spectrum

__version__ = "0.1.0"

__all__ = [
    "Filtration",
    "PersistenceResult",
    "SimplicialComplex",
    "bottleneck_distance",
    "datasets",
    "diagram_distance_matrix",
    "enclosing_radius",
    "filtration_from_simplices",
    "format_spectrum",
    "pairwise_distances",
    "persistence",
    "plotting",
    "rips_filtration",
    "rips_filtration_from_points",
    "rips_persistence",
    "vectorization",
    "wasserstein_distance",
]
