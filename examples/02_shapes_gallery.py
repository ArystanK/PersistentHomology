"""Example 2 - reading Betti numbers off a barcode.

Four planar clouds with known topology.  For each one the script finds the
*most stable scale* - the widest range of radii over which the Betti vector
does not change at all - reads the Betti numbers there, and checks them against
what the shape should give.  This is the "count the components and the holes"
step that persistent homology automates, with the scale chosen by the data
rather than by hand.
"""

from _common import header, out, timed

import numpy as np

from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence

SHAPES = [
    ("three clusters", ds.clusters(90, spread=0.35, seed=3), 1.6, (3, 0)),
    ("annulus", ds.annulus(110, r_inner=0.65, seed=4), 2.0, (1, 1)),
    ("two circles", ds.two_circles(90, separation=3.6, seed=5), 2.2, (2, 2)),
    ("figure eight", ds.figure_eight(110, seed=6), 2.0, (1, 2)),
]


def main() -> None:
    header("Example 2: Betti numbers from barcodes")
    print(f"  {'shape':>16}  {'scale':>6}  {'measured':>12}  {'expected':>10}   ok")

    all_ok = True
    for name, points, threshold, expected in SHAPES:
        with timed(name):
            result = rips_persistence(points, max_dim=1, threshold=threshold)
        scale = result.most_stable_scale(threshold)
        measured = tuple(result.betti_numbers(scale))
        ok = measured == expected
        all_ok &= ok
        print(f"  {name:>16}  {scale:6.3f}  {str(measured):>12}  {str(expected):>10}   "
              f"{'yes' if ok else 'NO'}")

        fig = viz.plot_summary(
            points, result, title=f"{name}  -  b = {measured} at scale {scale:.2f}")
        viz.save(fig, out(f"02_{name.replace(' ', '_')}.png"))

    print(f"\n  all four recovered correctly: {'yes' if all_ok else 'no'}")
    print(f"  wrote 4 figures to {out('')}")


if __name__ == "__main__":
    main()
