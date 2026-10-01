"""Example 3 - voids and multiple loops.

H0 counts components and H1 counts loops; H2 counts enclosed voids.  Here a
sphere (b = 1, 0, 1) and a torus (b = 1, 2, 1) are recovered from nothing but
pairwise distances, with no knowledge of the embedding.

A warning about cost: a Rips complex needs (d+1)-simplices to see d-dimensional
homology, so the simplex count grows like n^(d+2).  H2 on 60 points already
means ~300k simplices; that is why the sphere here is sampled sparsely and the
torus is only taken to H1.  Real work at this scale uses a cohomology-based
engine with an apparent-pairs optimisation (Ripser); the point of this script is
that the plain algorithm gives the same answer.
"""

from _common import header, out, timed

import numpy as np

from persistent_homology import datasets as ds
from persistent_homology import plotting as viz
from persistent_homology import rips_persistence


def main() -> None:
    header("Example 3: the 2-sphere and the torus")

    print("\n2-sphere in R^3, 60 points, homology to H2  (expect b = 1, 0, 1)")
    sphere = ds.sphere(60, dim=2, seed=3)
    with timed("sphere"):
        s_res = rips_persistence(sphere, max_dim=2, threshold=1.9)
    print(s_res.summary())
    h2 = s_res.most_persistent(2, k=1)[0]
    print(f"  the void: born {h2[0]:.3f}, dies {h2[1]:.3f}")
    print(f"  betti at scale {h2.mean():.2f}: {s_res.betti_numbers(h2.mean())}")
    viz.save(viz.plot_summary(sphere, s_res, title="2-sphere: one void, no loops"),
             out("03_sphere.png"))

    print("\nflat torus in R^4, 200 points, homology to H1  (expect b0 = 1, b1 = 2)")
    torus = ds.flat_torus(200, seed=5)
    with timed("torus"):
        t_res = rips_persistence(torus, max_dim=1, threshold=1.8)
    print(t_res.summary())
    top = t_res.most_persistent(1, k=4)
    print("  four longest H1 bars (the top two are the torus, the rest is noise):")
    for birth, death in top:
        print(f"    [{birth:.3f}, {death:.3f})   lifetime {death - birth:.3f}")
    scale = float(top[:2, 1].min() * 0.75)
    print(f"  betti at scale {scale:.2f}: {t_res.betti_numbers(scale)}")
    viz.save(viz.plot_summary(torus[:, :3], t_res,
                              title="Flat torus (first 3 of 4 coordinates shown): two loops"),
             out("03_torus.png"))

    print(f"\nwrote figures to {out('')}")


if __name__ == "__main__":
    main()
