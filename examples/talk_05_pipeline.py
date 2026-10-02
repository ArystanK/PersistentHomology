"""Talk part 5 - "Demo: TDA in an AI pipeline".

A persistence diagram is a multiset of points of varying cardinality.  It is not
a vector, it has no fixed dimension, and it cannot be handed to a linear layer.
So this script does the three things the slides say to do.

1. Vectorise.  Betti curve, persistence landscape and persistence image, each
   computed on the same diagram, each a fixed-size array whatever the diagram
   looked like.  The landscape is the one that is 1-Lipschitz in the bottleneck
   distance, so stability survives into feature space -- and that is checked
   here, not asserted.

2. Run the pipeline end to end.  Clouds -> Rips filtration -> diagrams ->
   landscapes -> a classifier, with the vectorisation fitted inside the
   cross-validation loop and never outside it.  Topological features are
   compared against, and then concatenated with, the sort of summary statistics
   you would otherwise have used.

3. Compare against a null model, which is the closing line of the part.  A bar
   only counts if it beats what shuffled data of the same size and dimension
   produces.

The classifier is nearest-centroid, written out in five lines.  Anything
stronger would obscure what the features are doing, which is the point.
"""

from _common import check, header, out, timed

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import bottleneck_distance
from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence
from persistent_homology import vectorization as vec

THRESHOLD = 2.4
GRID = np.linspace(0.0, THRESHOLD, 60)
# The persistence image's (birth, lifetime) window, fixed so the figure can
# label its axes in the same units the image was computed in.
IMAGE_WINDOW = (0.0, 1.6)
SEED = 3


# --------------------------------------------------------------- the data set

def make_dataset(n_per_class: int = 12, n_points: int = 60, seed: int = SEED):
    """Did the trajectory close the loop, or nearly close it?

    Class 1 is a full noisy circle.  Class 0 is an arc covering 85-93% of one,
    randomly rotated -- a cycle that almost came back to where it started.
    This is the talk's recurring-policy and periodicity question: a patrol route
    that closes is topologically a circle, and one that does not is an interval.

    The two classes are built to agree on everything a moment-based summary can
    see.  Both sit on the unit circle, so the radial mean and spread match; the
    random rotation leaves the covariance near-isotropic for both.  What differs
    is whether the loop ever closes, and how long it survives once it does.
    """
    rng = np.random.default_rng(seed)
    clouds, labels = [], []
    for _ in range(n_per_class):
        for closed in (True, False):
            span = 2 * np.pi if closed else rng.uniform(0.85, 0.93) * 2 * np.pi
            angle = rng.uniform(0, 2 * np.pi) + rng.uniform(0, span, n_points)
            arc = np.column_stack([np.cos(angle), np.sin(angle)])
            clouds.append(arc + 0.09 * rng.standard_normal(arc.shape))
            labels.append(1 if closed else 0)
    return clouds, np.array(labels)


def baseline_features(cloud: np.ndarray) -> np.ndarray:
    """The features you would reach for first: spread, shape, radial moments."""
    centred = cloud - cloud.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    radii = np.linalg.norm(centred, axis=1)
    return np.array([
        radii.mean(), radii.std(), radii.min(), radii.max(),
        eigenvalues[0], eigenvalues[1], eigenvalues[0] / eigenvalues[1],
    ])


# ------------------------------------------------------------- the classifier

def nearest_centroid_cv(features: np.ndarray, labels: np.ndarray, folds: int = 6):
    """Leave-one-fold-out nearest-centroid accuracy.

    Standardisation is fitted on the training fold only.  With a fixed
    vectorisation grid that is the whole of the leakage risk here, but the
    principle is the one from the slide: anything data-dependent goes inside
    the loop.
    """
    n = len(labels)
    order = np.random.default_rng(SEED).permutation(n)
    correct = 0
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)

        mean, std = features[train].mean(0), features[train].std(0)
        std = np.where(std > 0, std, 1.0)
        train_z, test_z = (features[train] - mean) / std, (features[test] - mean) / std

        centroids = np.array([train_z[labels[train] == c].mean(0) for c in (0, 1)])
        distances = ((test_z[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        correct += int((distances.argmin(axis=1) == labels[test]).sum())
    return correct / n


# ---------------------------------------------------------------------- main

def main() -> None:
    header("Talk part 5: TDA in an AI pipeline")

    # ------------------------------------------------------ 1. vectorisations
    print("\nA diagram is not a vector")
    circle = ds.circle(70, noise=0.08, seed=1)
    result = rips_persistence(circle, max_dim=1, threshold=THRESHOLD)
    h1 = result[1]
    print(f"  the H1 diagram of one noisy circle has {len(h1)} points;")
    print(f"  a second sample of the same circle gives "
          f"{len(rips_persistence(ds.circle(70, noise=0.08, seed=2), max_dim=1, threshold=THRESHOLD)[1])}.")
    print("  No fixed dimension, so no linear layer, no SVM, no transformer.")

    print("\nThree vectorisations of that same H1 diagram")
    curve = vec.betti_curve(h1, GRID)
    landscape = vec.persistence_landscape(h1, GRID, n_layers=4)
    image = vec.persistence_image(h1, resolution=(16, 16), birth_range=IMAGE_WINDOW,
                                  lifetime_range=IMAGE_WINDOW)
    for label, array in (("Betti curve", curve), ("landscape (4 layers)", landscape),
                         ("persistence image", image)):
        print(f"  {label:22s} shape {str(array.shape):>10s}  "
              f"-> {array.size:4d} numbers, whatever the diagram looked like")

    # the property that makes the landscape the right default
    print("\n  Is the landscape really Lipschitz?  Measure it.")
    print(f"  {'noise':>6s}  {'d_B(H1)':>9s}  {'||dL||_inf':>11s}  {'ratio':>7s}")
    rng = np.random.default_rng(0)
    base_landscape = vec.persistence_landscape(h1, GRID, n_layers=4)
    lipschitz_ok = True
    for noise in (0.01, 0.03, 0.06, 0.10):
        perturbed = circle + noise * rng.standard_normal(circle.shape)
        perturbed_h1 = rips_persistence(perturbed, max_dim=1, threshold=THRESHOLD)[1]
        bottleneck = bottleneck_distance(h1, perturbed_h1)
        gap = float(np.abs(
            vec.persistence_landscape(perturbed_h1, GRID, n_layers=4) - base_landscape
        ).max())
        lipschitz_ok &= gap <= bottleneck + 1e-9
        print(f"  {noise:6.2f}  {bottleneck:9.4f}  {gap:11.4f}"
              f"  {gap / bottleneck if bottleneck else 0:7.2f}")
    print("  ||L(D) - L(D')||_inf <= d_B(D, D') at every level: the stability")
    print("  theorem from Part 4 survives into feature space, which is exactly why")
    print("  you are allowed to average landscapes over a batch.")

    # ------------------------------------------------------- 2. the pipeline
    print("\nThe pipeline, end to end")
    clouds, labels = make_dataset()
    print(f"  {len(clouds)} clouds of 60 points: {int(labels.sum())} closed loops, "
          f"{int((1 - labels).sum())} arcs covering 85-93% of a loop")
    print("  Same radial mean, same radial spread, near-isotropic covariance for")
    print("  both classes.  The only difference is whether the loop closes.")

    with timed("diagrams for the whole data set"):
        results = [rips_persistence(c, max_dim=1, threshold=THRESHOLD) for c in clouds]

    topological = np.array([
        vec.diagram_features(r, max_scale=THRESHOLD, dims=(0, 1), n_layers=3, n_steps=40)
        for r in results
    ])
    baseline = np.array([baseline_features(c) for c in clouds])
    combined = np.hstack([baseline, topological])

    print(f"\n  {'features':>28s}  {'dimension':>9s}  {'CV accuracy':>11s}")
    accuracies = {}
    for label, matrix in (("baseline (moments, radii)", baseline),
                          ("topological (landscapes)", topological),
                          ("both concatenated", combined)):
        accuracies[label] = nearest_centroid_cv(matrix, labels)
        print(f"  {label:>28s}  {matrix.shape[1]:9d}  {accuracies[label]:11.3f}")
    print("  The topological features beat the moment baseline, and concatenating")
    print("  the two beats either alone -- which is the slide's point exactly.")
    print("  Topological features are a complement, not a replacement: the right")
    print("  move on real data is to concatenate them and measure the lift.")

    # --------------------------------------------------------- 3. null model
    print("\nAlways compare against a null model")
    print("  A bar only counts if it beats what structureless data of the same size")
    print("  and dimension produces.  Shuffling each coordinate independently keeps")
    print("  every marginal and destroys the shape -- the cleanest null there is.")

    observed = float((h1[:, 1] - h1[:, 0]).max())
    null_scores = []
    with timed("20 null replicates"):
        null_rng = np.random.default_rng(11)
        for _ in range(20):
            shuffled = np.column_stack([
                null_rng.permutation(circle[:, 0]), null_rng.permutation(circle[:, 1])
            ])
            null_h1 = rips_persistence(shuffled, max_dim=1, threshold=THRESHOLD)[1]
            null_scores.append(
                float((null_h1[:, 1] - null_h1[:, 0]).max()) if len(null_h1) else 0.0)
    null_scores = np.array(null_scores)
    percentile_95 = float(np.percentile(null_scores, 95))

    print(f"  longest H1 bar, observed:              {observed:.4f}")
    print(f"  longest H1 bar, null (mean +- sd):     "
          f"{null_scores.mean():.4f} +- {null_scores.std():.4f}")
    print(f"  null 95th percentile:                  {percentile_95:.4f}")
    print(f"  the observed loop clears the null by   "
          f"{observed / percentile_95:.1f}x")

    # ---------------------------------------------------------------- figure
    figure = plt.figure(figsize=(14.0, 7.6), facecolor=viz.SURFACE)

    viz.plot_points(circle, figure.add_subplot(2, 3, 1), title="one noisy circle")
    viz.persistence_diagram(result, figure.add_subplot(2, 3, 2),
                            title="its diagram (not a vector)")

    def finish(ax, title, xlabel, ylabel=None, legend=False):
        ax.set_title(title, color=viz.INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel(xlabel, color=viz.INK_SECONDARY, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, color=viz.INK_SECONDARY, fontsize=10)
        viz.style_axes(ax)
        if legend:
            for text in ax.legend(frameon=False, fontsize=8).get_texts():
                text.set_color(viz.INK_SECONDARY)

    ax = figure.add_subplot(2, 3, 3)
    ax.plot(GRID, curve, color=viz.DIM_COLORS[1], linewidth=2.0)
    ax.set_yticks(range(int(curve.max()) + 1))
    finish(ax, "Betti curve", "scale", r"$\beta_1$")

    ax = figure.add_subplot(2, 3, 4)
    for layer in range(landscape.shape[0]):
        ax.plot(GRID, landscape[layer], linewidth=2.0 - 0.4 * layer,
                color=viz.DIM_COLORS[1], alpha=1.0 - 0.22 * layer,
                label=f"layer {layer + 1}")
    finish(ax, "persistence landscape", "scale", legend=True)

    # Drawn in the units it was computed in, so the bright spot can be read off
    # as the loop: born near 0.4, living about 1.3.
    ax = figure.add_subplot(2, 3, 5)
    ax.imshow(image, origin="lower", cmap="magma",
              extent=(*IMAGE_WINDOW, *IMAGE_WINDOW), aspect="equal")
    finish(ax, "persistence image", "birth", "lifetime")
    ax.grid(False)

    ax = figure.add_subplot(2, 3, 6)
    ax.hist(null_scores, bins=8, color=viz.INK_MUTED, alpha=0.75,
            label="null replicates")
    ax.axvline(observed, color=viz.DIM_COLORS[1], linewidth=2.2,
               label="observed circle")
    finish(ax, "observed vs. null", r"longest $H_1$ bar", "replicates", legend=True)

    figure.suptitle("From a diagram to a feature vector, and a null model to "
                    "compare it against",
                    color=viz.INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    print("\nwrote", viz.save(figure, out("talk_05_pipeline.png")))

    # ------------------------------------------------------------ the checks
    print("\nVerdict")
    ok = True
    ok &= check("every vectorisation has a fixed size, independent of the diagram",
                curve.shape == (len(GRID),) and landscape.shape == (4, len(GRID))
                and image.shape == (16, 16))
    ok &= check("the landscape is 1-Lipschitz in the bottleneck distance",
                lipschitz_ok)
    ok &= check("topological features beat the moment baseline at closed-vs-open "
                f"({accuracies['topological (landscapes)']:.2f} vs "
                f"{accuracies['baseline (moments, radii)']:.2f})",
                accuracies["topological (landscapes)"]
                > accuracies["baseline (moments, radii)"])
    ok &= check("concatenating the two beats either one alone "
                f"({accuracies['both concatenated']:.2f})",
                accuracies["both concatenated"]
                >= max(accuracies["topological (landscapes)"],
                       accuracies["baseline (moments, radii)"]))
    ok &= check("the observed loop clears the null model's 95th percentile",
                observed > percentile_95)
    print("all claims reproduced" if ok else "SOME CLAIMS FAILED")


if __name__ == "__main__":
    main()
