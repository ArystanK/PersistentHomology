# Persistent Homology

Persistent homology from scratch, in NumPy — the Vietoris–Rips filtration, the
GF(2) boundary-matrix reduction that produces persistence pairs, exact
bottleneck and Wasserstein distances between diagrams, and five worked examples
ending in an application to time-series analysis.

No TDA library is used or wrapped. The point is that the algorithm is short
enough to read.

```
persistent_homology/
  complexes.py      distance matrices, Vietoris-Rips filtration, boundary matrix
  persistence.py    the reduction algorithm, persistence pairs, diagrams
  distances.py      bottleneck (binary search + Hopcroft-Karp), Wasserstein (Hungarian)
  simplicial.py     fixed complexes over R: signed boundaries, Hodge Laplacian
  vectorization.py  Betti curves, persistence landscapes, persistence images
  datasets.py       synthetic clouds with known topology, Takens embedding, Lorenz
  plotting.py       barcodes, diagrams, Betti curves, complexes, edge flows
examples/           thirteen runnable scripts, figures land in output/
tests/              140 tests: hand-checked complexes, known topologies, invariants
```

## Quick start

```bash
pip install -r requirements.txt
python run_examples.py        # the five library examples, ~2 minutes
python run_examples.py --talk # the seven tutorial examples, ~4 minutes
pytest                        # 140 tests, ~40 seconds
```

```python
from persistent_homology import datasets as ds, rips_persistence

points = ds.circle(80, noise=0.06, seed=1)      # a noisy circle in the plane
result = rips_persistence(points, max_dim=1, threshold=2.0)

result.betti_numbers(0.8)       # [1, 1]  - one component, one hole
result.most_persistent(1)       # [[0.50, 1.63]] - the loop, born early, dies late
result[1]                       # the full H1 diagram as an (n, 2) array
```

## What the algorithm actually does

**1. Build a shape out of the points.** At radius *r*, a set of points spans a
simplex whenever all its pairwise distances are ≤ *r*. Growing *r* from 0 to ∞
gives a nested sequence of simplicial complexes — a *filtration*. Each simplex
enters at one specific value, its longest internal edge, and never leaves.

**2. Track features across all radii at once.** Small *r* leaves isolated points;
large *r* fills everything into one contractible blob. Neither extreme says
anything. What is informative is the *lifetime* of each topological feature: a
connected component (H0), a loop (H1), an enclosed void (H2). Every feature is
born at some radius and dies at another, giving an interval — a *bar*. Long bars
are structure; short bars are sampling noise.

**3. Compute the bars by reducing a matrix.** Order the simplices by filtration
value (faces always land before their cofaces) and write the boundary matrix
over GF(2): column *j* lists the codimension-1 faces of simplex *j*. Reduce
columns left to right until no two share a lowest non-zero entry. Every pivot
(*i*, *j*) that survives is a persistence pair: simplex *i* creates a class,
simplex *j* destroys it, and the bar is [value(*i*), value(*j*)). Columns that
reduce to zero without ever being a pivot are the essential classes — bars that
never die.

That is all of it: `_reduce` in `persistence.py` is about forty lines, clearing
optimisation included.

**4. Compare the results.** The bottleneck distance between two diagrams matches
their points up one-to-one — each point either to a point of the other diagram
or to its own projection onto the diagonal — and reports the largest displacement
needed. The stability theorem then bounds it by twice the Hausdorff distance
between the input clouds, which is what makes any of this usable on measured
data: a small perturbation of the input cannot make a long bar disappear.

## Reading the output

A **barcode** draws one horizontal line per class. A **persistence diagram**
plots the same data as points (birth, death); distance above the diagonal is
lifetime, so real features sit high above it and noise piles up along it.
**Betti curves** count how many classes are alive at each radius — a wide plateau
is a robust answer.

## The examples

| | what it shows |
|---|---|
| `01_circle_vs_noise.py` | Two clouds identical in every summary statistic; one has a hole. The longest H1 bar is 1.16 for the circle and 0.19 for the noise. |
| `02_shapes_gallery.py` | Betti numbers recovered automatically for three clusters (3,0), an annulus (1,1), two circles (2,2), and a figure eight (1,2) — with the scale chosen as the widest interval on which the Betti vector never changes. |
| `03_higher_dimensions.py` | H2 on a 2-sphere finds exactly one void; H1 on a flat torus finds exactly two loops. |
| `04_stability_and_distances.py` | The stability bound measured directly across eight noise levels, then bottleneck distance used as a metric on shapes — circles cluster with circles, noise with noise, and a squashed ellipse lands with the circles. |
| `05_periodicity_detection.py` | An actual application. Takens delay embedding turns a signal into a cloud; a periodic signal traces a loop, so the longest H1 bar is a periodicity score. It ranks a triangle wave and a sine highest, places a chaotic Lorenz trajectory in between, and white noise last — with no assumed frequency and no assumption that the waveform is sinusoidal. |

Each writes figures into `output/` and prints its numbers to the terminal.

`rips_animation.py` is interactive rather than a check: a window with a slider for ε, showing the Vietoris–Rips complex and the balls it stands in for next to the barcode, with a line at the current ε. `--data` picks the hexagon (default), a noisy circle, two circles, a figure eight or clusters; `--save out.gif` writes a GIF instead of opening a window.

## The tutorial examples

`tda_ai_2h_1.tex` is a two-hour introduction to TDA for an audience of AI
researchers, built around homology as linear algebra. Seven more scripts, one
per part of the talk, recompute what that part asserts:

```bash
python run_examples.py --talk
```

| | what it checks |
|---|---|
| `talk_01_motivation.py` | The opening slide's three clouds — blob, loop, two clusters — built to share a mean and, to within sampling error, a covariance. PCA is blind to all three and k-means cuts the loop into three "clusters"; the longest H0 and H1 bars fire on exactly one cloud each, and on nothing else. |
| `talk_02_complexes.py` | Six points on the unit circle at the three Rips regimes: dust, the closed hexagon, and the octahedron boundary at ε = √3. Includes a real Čech complex (smallest enclosing ball), which keeps the loop at ε = √3 where Rips has already lost it — so the cost argument for Rips is shown with the price attached. Also writes the 90 frames of the ε sweep that the deck animates on the Rips slide. |
| `talk_03_homology.py` | ∂₁ and ∂₂ printed with labelled rows and columns, ∂∂ = 0 multiplied out, hollow vs filled, the tetrahedron boundary, and the three sanity checks: β₀ against a union-find component count, β₁ = E − V + C, and the Euler characteristic both ways. Ends with the slide's reference table — circle (1,1,0), sphere (1,0,1), torus (1,2,1), disk (1,0,0) — on explicit triangulations. The figure puts the hollow/filled board computation next to the matrices behind it, in the slide's edge order. |
| `talk_04_persistence.py` | The hexagon's event table and barcode; the reduction's creator/destroyer pairing printed simplex by simplex; H0 death times shown to be *exactly* the MST edge weights, so H0 persistence is single-linkage clustering; the stability bound measured; and each of the four "read a diagram honestly" warnings reproduced as a number, including the H2 bar [√3, 2) that is a Rips artifact and not a fact about the data. |
| `talk_05_pipeline.py` | Betti curve, landscape and image on one diagram, with the landscape's 1-Lipschitz property measured rather than asserted. Then a full pipeline on closed loops vs 85–93% arcs — a task where moment features genuinely struggle: baseline 0.67, topological 0.83, concatenated 0.88. Ends with the null-model comparison the talk closes on. |
| `talk_06_hodge_laplacian.py` | L₀ = ∂₁∂₁ᵀ = D − A checked on five complexes; the triangle's spectra {0,3,3} hollow and {3,3,3} filled; β_k = dim ker L_k against rank-nullity on seven complexes; the harmonic representative that answers "*where* is the loop?"; subdivision moving the spectral gap 3 → 1 → 0.268 while β₁ never budges; and HodgeRank splitting a cyclic preference from a consistent ranking. |
| `talk_07_exercise.py` | The closing exercise worked in full — all five steps, then the same answer again through L₀ and L₁ — plus the appendix figure-eight, both as an abstract complex and through the Rips pipeline on a point cloud. |

Every script ends with a list of the claims it checked and whether each one came
out, so a slide and the code that backs it can be compared line by line.

The figures land in `output/` and the deck includes them directly — nine frames,
at least one per part, each captioned with what to read off it and credited to the script
that produced it. Build it with `pdflatex tda_ai_2h_1.tex` (twice). The deck
still compiles if you have not run the examples: each figure frame falls back to
a note saying which file is missing. To drop all nine frames and keep the
original running time, set `\computedslidesfalse` in the preamble. The Rips
animation uses the `animate` package: it plays in Adobe Reader and Okular, and
other viewers (browsers, macOS Preview) show a still of the loop regime instead.

## Cost, and when to use something else

A Rips complex needs (d+1)-simplices to see d-dimensional homology, so the
simplex count grows like n^(d+2). Runtime tracks the simplex count, not the
point count — measured on one laptop core:

| homology | points | threshold | simplices | time |
|---|---|---|---|---|
| H0–H1 | 300 | 0.6 | 129k | 2.6 s |
| H0–H1 | 200 | 1.8 | 97k | 4.1 s |
| H0–H1 | 100 | 2.0 (everything) | 167k | 8.9 s |
| H0–H2 | 60 | 1.9 | 290k | 15.8 s |

Two levers keep it tractable: `threshold` (drop edges longer than a radius you
do not care about — the default is the enclosing radius, past which no new
topology can appear) and `max_dim`. A guard raises `MemoryError` before the
complex explodes rather than after.

For production-scale work use [Ripser](https://github.com/Ripser/ripser) or
[GUDHI](https://gudhi.inria.fr/): same output, but a cohomology formulation with
apparent-pairs and emergent-pairs shortcuts that skip most of the matrix. This
implementation is here to show what those libraries compute.

## API

```python
# filtrations
pairwise_distances(points, metric="euclidean"|"manhattan"|"chebyshev")
enclosing_radius(dist)
rips_filtration(dist, max_dim, threshold=None, max_simplices=2_000_000)
rips_filtration_from_points(points, max_dim, threshold=None, metric="euclidean")
filtration_from_simplices([(vertices, value), ...])   # build one by hand

Filtration.boundary_columns()      # the GF(2) boundary matrix
Filtration.betti_numbers(t)        # independent rank-based check

# persistence
persistence(filtration, max_dim=None, min_persistence=0.0)
rips_persistence(points, max_dim=1, threshold=None, metric="euclidean")

PersistenceResult[dim]                 # (n, 2) array of (birth, death)
PersistenceResult.betti_numbers(t)
PersistenceResult.most_persistent(dim, k=1)
PersistenceResult.summary()

# comparing diagrams
bottleneck_distance(dgm1, dgm2)                 # exact
wasserstein_distance(dgm1, dgm2, order=2.0)     # exact
diagram_distance_matrix(diagrams, metric="bottleneck")

# data
datasets.circle / annulus / two_circles / figure_eight / sphere / torus
datasets.flat_torus / uniform_box / clusters / takens_embedding / lorenz

# fixed complexes over R (signs, the Hodge Laplacian)
SimplicialComplex(simplices)                 # closed under faces on construction
SimplicialComplex.cycle(n) / .boundary_of_simplex(n) / .from_graph(n, edges)
SimplicialComplex.from_filtration(filt, t)   # a filtration frozen at one scale

K.boundary_matrix(k)        # d_k, signed, shape (n_(k-1), n_k)
K.hodge_laplacian(k)        # L_k = d_k^T d_k + d_(k+1) d_(k+1)^T
K.graph_laplacian()         # D - A of the 1-skeleton, to check L_0 against
K.betti_numbers()           # by rank-nullity
K.betti_from_laplacian(k)   # by eigenvalue multiplicity: dim ker L_k
K.spectrum(k) / K.spectral_gap(k) / K.euler_characteristic()
K.harmonic_basis(k)         # canonical representatives of the homology classes
K.hodge_decomposition(chain, k)   # boundary (+) harmonic (+) coboundary

# vectorising a diagram for a model
vectorization.betti_curve(dgm, grid)
vectorization.persistence_landscape(dgm, grid, n_layers)   # 1-Lipschitz in d_B
vectorization.persistence_image(dgm, resolution, sigma)
vectorization.diagram_features(result, max_scale)          # one flat vector

# figures
plotting.barcode / persistence_diagram / betti_curves / plot_points / plot_summary
plotting.plot_complex(points, K)          # vertices, edges, shaded triangles
plotting.plot_chain(points, K, chain)     # a 1-chain drawn as a flow on the edges
```

## Implementation notes

- **Clearing.** Columns are reduced in order of decreasing dimension. Once index
  *i* is known to be a pivot, its own column provably reduces to zero and is
  skipped outright. On Rips complexes this removes most of the work.
- **Columns as sorted index lists.** GF(2) addition is a merge (symmetric
  difference), and the lowest non-zero entry is the last element — no dense
  matrix is ever allocated.
- **Exact bottleneck.** Binary search over the sorted distinct costs, testing
  each candidate with Hopcroft–Karp on the augmented bipartite graph. Verified
  against exhaustive enumeration of all matchings on small diagrams.
- **Essential classes.** Kept as `inf` rather than truncated, and matched among
  themselves on their birth values when diagrams are compared; diagrams with
  different numbers of essential classes are infinitely far apart.
- **Colour in the figures** encodes homological dimension in a fixed order that
  is never cycled, with a distinct marker shape per dimension so identity does
  not rest on colour alone.
- **Two coefficient fields, on purpose.** Persistence runs over GF(2), where
  signs vanish and the reduction is a symmetric difference. `simplicial.py` runs
  over R with dense matrices, because signs are what make `d o d = 0` a
  cancellation you can watch, and an inner product is what makes the Hodge
  Laplacian exist at all. The two agree on every Betti number, which the tests
  check on seven complexes.
- **Near-zero eigenvalues are snapped to zero** at `1e-9` before a kernel is
  counted — above the rounding error of a symmetric eigensolver on a 0,±1
  matrix of these sizes, and far below the smallest nonzero eigenvalue these
  complexes produce.
