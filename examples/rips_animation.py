"""The Vietoris-Rips complex with a slider for epsilon.

Left: the point cloud, the balls of radius eps/2 around each point, and the
Rips complex at the current scale -- an edge wherever d(x, y) <= eps, a filled
triangle wherever all three of its edges are present.  Right: the barcode of the
whole filtration, with a vertical line at the current eps.  The bars it crosses
are exactly the classes alive at that scale, so the number of H_k bars under
the line is beta_k of the complex on the left.

    python examples/rips_animation.py                       # the talk's hexagon
    python examples/rips_animation.py --data circle --n 30 --noise 0.08
    python examples/rips_animation.py --data figure-eight --n 40
    python examples/rips_animation.py --save output/rips.gif  # write a GIF instead
    python examples/rips_animation.py --frames-dir output/rips_hexagon_frames --frames 90
                                                            # PNG frames for the deck

Interactive controls: drag the slider, press Play to sweep, space bar to
play/pause, left/right arrow keys to step, "b" to toggle the balls.

The complex is built once, up to eps_max, and every frame just shows the
simplices whose filtration value is <= eps -- the same subcomplex that
SimplicialComplex.from_filtration(filtration, eps) would give.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from persistent_homology import datasets, pairwise_distances, persistence  # noqa: E402
from persistent_homology.complexes import rips_filtration  # noqa: E402

SQRT3 = float(np.sqrt(3))


def hexagon() -> np.ndarray:
    """Six points spaced evenly on the unit circle, as on the slides."""
    angles = np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])


def load_points(name: str, n: int, noise: float, seed: int) -> np.ndarray:
    if name == "hexagon":
        return hexagon()
    if name == "circle":
        return datasets.circle(n, noise=noise, seed=seed)
    if name == "two-circles":
        return datasets.two_circles(n, seed=seed)
    if name == "figure-eight":
        return datasets.figure_eight(n, noise=noise, seed=seed)
    if name == "clusters":
        return datasets.clusters(n, centers=((0, 0), (3, 0), (1.5, 2.5)), seed=seed)
    raise ValueError(f"unknown data set {name!r}")


class RipsScene:
    """Everything that depends on the data but not on eps, computed once."""

    def __init__(self, points: np.ndarray, eps_max: float):
        self.points = np.asarray(points, dtype=float)
        self.eps_max = eps_max
        distances = pairwise_distances(self.points)
        # dimension 2 is enough to draw, and to fill in every 1-cycle that dies
        filtration = rips_filtration(distances, max_dim=2, threshold=eps_max)
        self.result = persistence(filtration, max_dim=1)

        simplices, values = filtration.simplices, filtration.values
        edges = [(s, v) for s, v in zip(simplices, values) if len(s) == 2]
        triangles = [(s, v) for s, v in zip(simplices, values) if len(s) == 3]
        self.edge_segments = np.array([self.points[list(s)] for s, _ in edges]).reshape(-1, 2, 2)
        self.edge_values = np.array([v for _, v in edges])
        self.triangle_polys = np.array([self.points[list(s)] for s, _ in triangles]).reshape(-1, 3, 2)
        self.triangle_values = np.array([v for _, v in triangles])

    def counts(self, eps: float):
        return (len(self.points),
                int((self.edge_values <= eps).sum()),
                int((self.triangle_values <= eps).sum()))

    def betti(self, eps: float):
        return self.result.betti_numbers(eps)


def build_figure(scene: RipsScene, show_balls: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
    from matplotlib.patches import Circle

    from persistent_homology import plotting as viz

    figure = plt.figure(figsize=(12.0, 6.2), facecolor=viz.SURFACE)
    ax_complex = figure.add_axes([0.03, 0.17, 0.46, 0.72])
    ax_bars = figure.add_axes([0.56, 0.17, 0.41, 0.72])

    # ------------------------------------------------------------ the complex
    points = scene.points
    span = np.ptp(points, axis=0).max()
    pad = 0.08 * span + 0.35 * scene.eps_max  # room for most of the largest balls
    lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
    ax_complex.set_xlim(lo[0], hi[0])
    ax_complex.set_ylim(lo[1], hi[1])
    ax_complex.set_aspect("equal")
    ax_complex.set_xticks([])
    ax_complex.set_yticks([])
    viz.style_axes(ax_complex)
    ax_complex.grid(False)

    balls = PatchCollection([Circle(p, 0.0) for p in points], facecolor=viz.DIM_COLORS[0],
                            edgecolor="none", alpha=0.13, zorder=0)
    balls.set_visible(show_balls)
    ax_complex.add_collection(balls)
    triangles = PolyCollection([], facecolor=viz.DIM_COLORS[0], edgecolor="none",
                               alpha=0.22, zorder=1)
    ax_complex.add_collection(triangles)
    edges = LineCollection([], colors=viz.DIM_COLORS[0], linewidths=1.4, alpha=0.9, zorder=2)
    ax_complex.add_collection(edges)
    ax_complex.scatter(points[:, 0], points[:, 1], s=26 if len(points) > 12 else 40,
                       color=viz.INK, edgecolor=viz.SURFACE, linewidth=0.8, zorder=3)
    title = ax_complex.set_title("", color=viz.INK, fontsize=13, loc="left", pad=10)
    caption = ax_complex.text(0.0, -0.06, "", transform=ax_complex.transAxes,
                              color=viz.INK_SECONDARY, fontsize=10, va="top")

    # ------------------------------------------------------------ the barcode
    bars = []  # (dim, birth, death) in the order they are drawn, bottom to top
    for dim in (0, 1):
        diagram = scene.result[dim]
        diagram = diagram[np.argsort(diagram[:, 0] - diagram[:, 1], kind="stable")]
        bars.extend((dim, float(b), float(d)) for b, d in diagram)
    bar_artists = []
    for row, (dim, birth, death) in enumerate(bars):
        end = min(death, scene.eps_max)
        (line,) = ax_bars.plot([birth, end], [row, row], color=viz.DIM_COLORS[dim],
                               linewidth=3.2 if len(bars) < 40 else 1.8,
                               solid_capstyle="butt")
        if not np.isfinite(death):
            ax_bars.annotate("", xy=(scene.eps_max * 1.02, row), xytext=(end, row),
                             arrowprops=dict(arrowstyle="->", color=viz.DIM_COLORS[dim], lw=1.2))
        bar_artists.append(line)
    n_h0 = sum(1 for dim, _, _ in bars if dim == 0)
    if len(bars) > n_h0:
        ax_bars.axhline(n_h0 - 0.5, color=viz.GRID, linewidth=1.0)
    ax_bars.set_yticks([(n_h0 - 1) / 2, n_h0 + (len(bars) - n_h0 - 1) / 2]
                       if len(bars) > n_h0 else [(n_h0 - 1) / 2])
    ax_bars.set_yticklabels(["$H_0$", "$H_1$"][: 2 if len(bars) > n_h0 else 1])
    ax_bars.set_xlim(0, scene.eps_max * 1.05)
    ax_bars.set_ylim(-1, len(bars))
    ax_bars.set_xlabel(r"$\varepsilon$", color=viz.INK_SECONDARY, fontsize=11)
    ax_bars.set_title("barcode: bars under the line are alive",
                      color=viz.INK, fontsize=11, loc="left", pad=10)
    viz.style_axes(ax_bars)
    ax_bars.grid(False)
    sweep = ax_bars.axvline(0.0, color=viz.INK, linewidth=1.2, alpha=0.8)

    def draw(eps: float) -> None:
        eps = float(eps)
        balls.set_paths([Circle(p, eps / 2) for p in points])
        edges.set_segments(scene.edge_segments[scene.edge_values <= eps])
        triangles.set_verts(scene.triangle_polys[scene.triangle_values <= eps])

        n0, n1, n2 = scene.counts(eps)
        betti = scene.betti(eps)
        title.set_text(rf"Vietoris–Rips complex at $\varepsilon = {eps:.3f}$")
        caption.set_text(f"{n0} vertices, {n1} edges, {n2} triangles      "
                         rf"$\beta_0 = {betti[0]}$,  $\beta_1 = {betti[1]}$")

        sweep.set_xdata([eps, eps])
        for line, (_, birth, death) in zip(bar_artists, bars):
            line.set_alpha(1.0 if birth <= eps < death else 0.25)

    return figure, draw, balls


def run_interactive(scene: RipsScene, eps0: float, show_balls: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    from persistent_homology import plotting as viz

    figure, draw, balls = build_figure(scene, show_balls)
    ax_slider = figure.add_axes([0.13, 0.05, 0.62, 0.035], facecolor=viz.GRID)
    ax_play = figure.add_axes([0.79, 0.04, 0.08, 0.055])
    ax_balls = figure.add_axes([0.885, 0.04, 0.08, 0.055])
    slider = Slider(ax_slider, r"$\varepsilon$", 0.0, scene.eps_max, valinit=eps0,
                    color=viz.DIM_COLORS[0])
    play = Button(ax_play, "Play")
    balls_button = Button(ax_balls, "Balls")

    step = scene.eps_max / 200
    state = {"playing": False}
    timer = figure.canvas.new_timer(interval=40)

    def advance():
        value = slider.val + step
        if value > scene.eps_max:
            value = 0.0
        slider.set_val(value)

    def toggle_play(_=None):
        state["playing"] = not state["playing"]
        play.label.set_text("Pause" if state["playing"] else "Play")
        (timer.start if state["playing"] else timer.stop)()
        figure.canvas.draw_idle()

    def toggle_balls(_=None):
        balls.set_visible(not balls.get_visible())
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            toggle_play()
        elif event.key == "right":
            slider.set_val(min(slider.val + step * 5, scene.eps_max))
        elif event.key == "left":
            slider.set_val(max(slider.val - step * 5, 0.0))
        elif event.key == "b":
            toggle_balls()

    timer.add_callback(advance)
    slider.on_changed(lambda value: (draw(value), figure.canvas.draw_idle()))
    play.on_clicked(toggle_play)
    balls_button.on_clicked(toggle_balls)
    figure.canvas.mpl_connect("key_press_event", on_key)

    draw(eps0)
    plt.show()


def save_animation(scene: RipsScene, path: str, frames: int, fps: int, show_balls: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.animation import FuncAnimation, PillowWriter

    figure, draw, _ = build_figure(scene, show_balls)
    # sweep up, hold at the top for a moment, so the loop in a GIF reads cleanly
    values = np.concatenate([np.linspace(0.0, scene.eps_max, frames),
                             np.full(fps, scene.eps_max)])
    animation = FuncAnimation(figure, lambda i: draw(values[i]), frames=len(values))
    if path.lower().endswith(".gif"):
        animation.save(path, writer=PillowWriter(fps=fps), dpi=90)
    else:
        animation.save(path, fps=fps, dpi=110)  # .mp4 needs ffmpeg on the PATH
    print("wrote", path)


def save_frames(scene: RipsScene, directory: str, frames: int = 90, dpi: int = 110) -> int:
    """Write the sweep as ``frame-000.png``, ``frame-001.png``, ... for the LaTeX
    ``animate`` package, which needs one image per frame rather than a GIF.
    Stale frames from an earlier, longer run are removed first, so the deck's
    frame range always matches what is on disk.
    """
    import glob

    import matplotlib

    matplotlib.use("Agg")
    from persistent_homology import plotting as viz

    os.makedirs(directory, exist_ok=True)
    for stale in glob.glob(os.path.join(directory, "frame-*.png")):
        os.remove(stale)
    figure, draw, _ = build_figure(scene)
    # a smaller canvas at a higher dpi: same pixels, but the labels come out
    # large enough to read once the frame is scaled down onto a slide
    figure.set_size_inches(9.0, 4.65)
    for i, eps in enumerate(np.linspace(0.0, scene.eps_max, frames)):
        draw(eps)
        figure.savefig(os.path.join(directory, f"frame-{i:03d}.png"), dpi=dpi,
                       facecolor=viz.SURFACE)
    import matplotlib.pyplot as plt
    plt.close(figure)
    print(f"wrote {frames} frames to {directory}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="hexagon",
                        choices=["hexagon", "circle", "two-circles", "figure-eight", "clusters"])
    parser.add_argument("--n", type=int, default=24, help="number of points (not for hexagon)")
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-max", type=float, default=None,
                        help="top of the slider (default: 2.2 for the hexagon, "
                             "else the enclosing radius)")
    parser.add_argument("--eps", type=float, default=None, help="starting epsilon")
    parser.add_argument("--no-balls", action="store_true", help="start with the balls hidden")
    parser.add_argument("--save", default=None, help="write a .gif (or .mp4) instead of opening a window")
    parser.add_argument("--frames-dir", default=None,
                        help="write numbered PNG frames here, for the LaTeX animate package")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    points = load_points(args.data, args.n, args.noise, args.seed)
    if args.eps_max is not None:
        eps_max = args.eps_max
    elif args.data == "hexagon":
        eps_max = 2.2
    else:
        from persistent_homology import enclosing_radius
        eps_max = float(enclosing_radius(pairwise_distances(points)))

    scene = RipsScene(points, eps_max)
    if args.frames_dir:
        save_frames(scene, args.frames_dir, args.frames)
    elif args.save:
        save_animation(scene, args.save, args.frames, args.fps, not args.no_balls)
    else:
        eps0 = args.eps if args.eps is not None else (1.2 if args.data == "hexagon" else eps_max / 3)
        run_interactive(scene, eps0, not args.no_balls)


if __name__ == "__main__":
    main()
