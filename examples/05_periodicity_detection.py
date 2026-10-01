"""Example 5 - an actual application: detecting periodicity in a signal.

Takens' delay embedding turns a 1D time series into a point cloud: row i is
``[x_i, x_{i+tau}, x_{i+2tau}]``.  A periodic signal traces a closed curve in
that cloud, so it has a long H1 bar.  A non-periodic one does not.  The longest
H1 bar is therefore a *periodicity score* that needs no assumed frequency, no
window length, and no stationarity - unlike a periodogram, it does not care
whether the waveform is a sinusoid.

The last signal is a chaotic Lorenz trajectory: aperiodic but structured, and
the score correctly places it between clean periodicity and white noise.
"""

from _common import header, out, timed

import matplotlib.pyplot as plt
import numpy as np

from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence

N = 400
T = np.linspace(0, 8 * 2 * np.pi, N)
RNG = np.random.default_rng(17)


def normalise(x: np.ndarray) -> np.ndarray:
    """Unit variance, so the persistence scales are comparable across signals."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


SIGNALS = {
    "clean sine": normalise(np.sin(T)),
    "noisy sine": normalise(np.sin(T) + 0.35 * RNG.standard_normal(N)),
    # anharmonic: a periodogram sees two peaks here, H1 still sees one loop
    "triangle wave": normalise(np.abs(((T / (2 * np.pi)) % 1.0) - 0.5)),
    "amplitude drift": normalise(np.sin(T) * np.linspace(0.4, 1.6, N)),
    "white noise": normalise(RNG.standard_normal(N)),
    "Lorenz x(t)": normalise(ds.lorenz(N, dt=0.02)[:, 0]),
}

# tau = a quarter period spreads the embedded loop out as widely as possible
DELAY = max(1, N // (8 * 4))
SUBSAMPLE = 130


def score(signal: np.ndarray):
    cloud = ds.takens_embedding(signal, dimension=3, delay=DELAY)
    # persistent homology is expensive in the cloud size, and a loop survives
    # subsampling, so thin the trajectory before building the complex
    idx = np.linspace(0, len(cloud) - 1, min(SUBSAMPLE, len(cloud))).astype(int)
    cloud = cloud[idx]
    result = rips_persistence(cloud, max_dim=1, threshold=3.0)
    h1 = result[1]
    longest = float((h1[:, 1] - h1[:, 0]).max()) if len(h1) else 0.0
    return cloud, result, longest


def main() -> None:
    header("Example 5: periodicity as a topological question")
    print(f"  embedding: dimension 3, delay {DELAY}, {SUBSAMPLE} points per cloud\n")
    print(f"  {'signal':>18}  {'longest H1 bar':>15}  {'vs noise':>9}   verdict")

    computed = {}
    with timed("6 signals"):
        for name, signal in SIGNALS.items():
            computed[name] = score(signal)

    # white noise is the null model: score everything relative to it
    baseline = computed["white noise"][2]
    ranked = sorted(computed.items(), key=lambda kv: -kv[1][2])
    for name, (_, _, longest) in ranked:
        ratio = longest / baseline if baseline else float("inf")
        verdict = ("periodic" if ratio > 2.5 else "structured" if ratio > 1.5 else "no loop")
        print(f"  {name:>18}  {longest:15.3f}  {ratio:8.1f}x   {verdict}")

    fig, axes = plt.subplots(len(SIGNALS), 3, figsize=(14.5, 2.5 * len(SIGNALS)),
                             facecolor=viz.SURFACE)
    for row, (name, signal) in enumerate(SIGNALS.items()):
        cloud, result, longest = computed[name]

        ax = axes[row, 0]
        ax.plot(signal[:200], color=viz.INK_SECONDARY, linewidth=1.2)
        ax.set_title(f"{name} (first 200 samples)", color=viz.INK, fontsize=10, loc="left")
        ax.set_yticks([])
        viz._style_axes(ax)

        ax = axes[row, 1]
        ax.plot(cloud[:, 0], cloud[:, 1], color=viz.GRID, linewidth=0.8, zorder=1)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=10, color=viz.INK_SECONDARY,
                   edgecolor=viz.SURFACE, linewidth=0.4, zorder=3)
        ax.set_title("delay embedding (2 of 3 axes)", color=viz.INK, fontsize=10, loc="left")
        ax.set_aspect("equal")
        viz._style_axes(ax)

        viz.persistence_diagram(result, axes[row, 2],
                                title=f"diagram - longest $H_1$ = {longest:.2f}")
    fig.tight_layout()
    print("\nwrote", viz.save(fig, out("05_periodicity.png")))


if __name__ == "__main__":
    main()
