"""Talk part 1 - "Three data sets your tools cannot tell apart".

The claim on the slide is specific and checkable, so this script checks it.
Three clouds -- a blob, a loop, two clusters -- are each built to have an
isotropic covariance and then rescaled to unit variance, so all three share a
mean of 0 and a covariance of I to three decimals.  Then:

  * PCA reports identical explained variance for all three;
  * a k-means silhouette flags (c) and cannot separate (a) from (b);
  * H0 and H1 separate all three, because the loop is not a linear subspace
    and not a set of separated clusters.

The two clusters are made isotropic honestly, not by whitening them afterwards:
they are two vertical streaks side by side, elongated just enough that the
spread *within* a cluster along one axis matches the separation *between*
clusters along the other.  Whitening would have worked on the numbers but would
have squashed the gap that makes them two clusters in the first place.
"""

from _common import check, header, out, timed

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence

SEED = 7


def normalise(points: np.ndarray) -> np.ndarray:
    """Centre, then divide by a single scalar so the total variance is 2.

    Deliberately *not* a whitening: a scalar cannot reshape a cloud, so the
    covariances below agree because the clouds were built to agree, not because
    a transform forced them to.
    """
    centred = points - points.mean(axis=0)
    return centred / np.sqrt(centred.var(axis=0).mean())


def kmeans(points: np.ndarray, k: int, seed: int = 0, iterations: int = 60):
    """Lloyd's algorithm, k-means++ seeding.  Written out because the point of
    the slide is what a standard clustering algorithm reports, and importing one
    would leave a reader wondering what it does.
    """
    rng = np.random.default_rng(seed)
    centres = [points[rng.integers(len(points))]]
    for _ in range(k - 1):
        squared = np.min(
            ((points[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1
        )
        centres.append(points[rng.choice(len(points), p=squared / squared.sum())])
    centres = np.array(centres)

    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new_labels = distances.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for j in range(k):
            if (labels == j).any():
                centres[j] = points[labels == j].mean(axis=0)
    return labels


def silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient: how much better a point fits its own cluster
    than the nearest other one.  Above ~0.5 is usually read as real structure.
    """
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    scores = np.zeros(len(points))
    for i in range(len(points)):
        i_label = labels[i]
        own = labels == i_label
        same = distances[i][own]
        a = same.sum() / max(own.sum() - 1, 1)
        b = min(distances[i][labels == other].mean()
                for other in unique if other != i_label)
        scores[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(scores.mean())


DOMINANCE = 2.0     # "the top bar is at least twice the runner-up"


def top_lifetimes(diagram: np.ndarray, k: int = 2) -> np.ndarray:
    """The ``k`` longest *finite* bar lengths, descending, zero-padded."""
    diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
    finite = diagram[np.isfinite(diagram[:, 1])]
    lifetimes = np.sort(finite[:, 1] - finite[:, 0])[::-1]
    return np.pad(lifetimes, (0, max(0, k - len(lifetimes))))[:k]


def dominance(diagram: np.ndarray) -> float:
    """How far the longest bar stands above the runner-up in the same diagram.

    This is the whole "long bars matter" heuristic as one number, and it needs
    no scale to be chosen and nothing to compare against: a feature of the
    underlying shape towers over the sampling noise *of its own cloud*, while
    the longest bar of a featureless cloud is just the luckiest piece of noise
    and sits in among the rest.  Being a ratio, it also survives the rescaling
    that the "reading a diagram honestly" slide warns about.
    """
    top, runner_up = top_lifetimes(diagram, 2)
    if top == 0.0:
        return 0.0
    return float("inf") if runner_up == 0.0 else float(top / runner_up)


def build_clouds(n: int = 110) -> dict:
    """Three clouds whose covariance is isotropic by construction.

    The blob and the loop are isotropic for free.  For the two clusters, put the
    centres at ``(+-a, 0)`` and give each cluster a within-cluster standard
    deviation of ``s`` across and ``sqrt(a^2 + s^2)`` along: the between-cluster
    variance ``a^2`` in x is then matched exactly by the within-cluster variance
    in y, and the total covariance is a multiple of the identity.
    """
    rng = np.random.default_rng(SEED)

    # a uniform disk, not a Gaussian: a Gaussian's tail plants isolated
    # outliers, which are genuinely extra components and would be the diagram's
    # honest answer rather than the blob's shape
    angle, radius = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.uniform(0, 1, n))
    blob = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])

    loop = ds.circle(n, radius=1.0, noise=0.07, seed=SEED)

    a, s = 2.0, 0.3
    side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)   # exactly balanced
    pair = np.column_stack([
        side * a + s * rng.standard_normal(n),
        np.sqrt(a ** 2 + s ** 2) * rng.standard_normal(n),
    ])

    return {
        "(a) a blob": normalise(blob),
        "(b) a loop": normalise(loop),
        "(c) two clusters": normalise(pair),
    }


def main() -> None:
    header("Talk part 1: three data sets your tools cannot tell apart")
    clouds = build_clouds()

    print("\nFirst and second moments")
    print(f"  {'cloud':18s} {'n':>4s}  {'|mean|':>8s}  {'cov eigenvalues':>18s}"
          f"  {'PCA var ratio':>14s}")
    for name, points in clouds.items():
        eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
        ratio = eigenvalues[::-1] / eigenvalues.sum()
        print(f"  {name:18s} {len(points):4d}  {np.linalg.norm(points.mean(0)):8.2e}"
              f"  {np.array2string(eigenvalues, precision=3):>18s}"
              f"  {np.array2string(ratio, precision=2):>14s}")
    print("  -> the same to within sampling error: PCA has nothing to work with.")

    print("\nk-means + silhouette (the clustering answer)")
    ks = (2, 3, 4, 5)
    print(f"  {'cloud':18s} " + "".join(f"{f'k={k}':>9s}" for k in ks) + "     best k")
    best_k = {}
    for name, points in clouds.items():
        row = [silhouette(points, kmeans(points, k, seed=SEED)) for k in ks]
        best_k[name] = ks[int(np.argmax(row))]
        print(f"  {name:18s} " + "".join(f"{v:9.3f}" for v in row)
              + f"     {best_k[name]}")
    print("  -> the criterion answers 'k > 1' for the loop as readily as for the")
    print("     clusters: it partitions the circle into arcs and scores it well.")

    print("\nPersistent homology (the topological answer)")
    results = {}
    for name, points in clouds.items():
        with timed(name):
            results[name] = rips_persistence(points, max_dim=1, threshold=2.6)

    print(f"\n  {'cloud':18s} {'H0: top two':>16s} {'x':>6s}   "
          f"{'H1: top two':>16s} {'x':>6s}   reading")
    verdicts = {}
    for name in clouds:
        row = []
        for dim in (0, 1):
            top = top_lifetimes(results[name][dim])
            ratio = dominance(results[name][dim])
            row.append(f"  {top[0]:6.3f} {top[1]:7.3f} {ratio:6.1f} ")
        reading = [
            label for dim, label in ((0, "separated clusters"), (1, "a hole"))
            if dominance(results[name][dim]) >= DOMINANCE
        ]
        verdicts[name] = reading or ["neither"]
        print(f"  {name:18s}" + " ".join(row) + f"  {', '.join(verdicts[name])}")
    print(f"  ('x' is the top bar divided by the runner-up; it fires at {DOMINANCE}.)")

    print("\nVerdict")
    covariances = [np.cov(p, rowvar=False) for p in clouds.values()]
    ok = True
    ok &= check("all three have the same mean and near-identical covariance, "
                "so PCA is blind",
                np.allclose(covariances, np.eye(2), atol=0.15))
    ok &= check(f"k-means splits the loop into {best_k['(b) a loop']} clusters "
                "although the loop is one connected component",
                best_k["(b) a loop"] > 1 and results["(b) a loop"].betti_numbers(
                    results["(b) a loop"].most_stable_scale(2.6))[0] == 1)
    ok &= check("H1 fires on the loop and on nothing else",
                verdicts["(b) a loop"] == ["a hole"])
    ok &= check("H0 fires on the two clusters and on nothing else",
                verdicts["(c) two clusters"] == ["separated clusters"])
    ok &= check("nothing fires on the blob", verdicts["(a) a blob"] == ["neither"])

    # One scale for all three diagrams, as the clouds above share one: on its own
    # axes the blob's noise would sit as far from the diagonal as the loop's bar.
    shared_top = max(float(d[np.isfinite(d)].max())
                     for result in results.values()
                     for d in result.diagrams.values() if len(d))
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), facecolor=viz.SURFACE)
    for column, (name, points) in enumerate(clouds.items()):
        viz.plot_points(points, axes[0, column], title=name)
        axes[0, column].set_xlim(-3.2, 3.2)
        axes[0, column].set_ylim(-3.2, 3.2)
        viz.persistence_diagram(results[name], axes[1, column], title=f"diagram - {name}",
                                top=shared_top)
    figure.suptitle("Same mean, same covariance, different topology",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    print("\nwrote", viz.save(figure, out("talk_01_motivation.png")))
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


if __name__ == "__main__":
    main()
