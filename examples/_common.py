"""Shared plumbing for the example scripts."""

import os
import sys
import time
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # examples write files; no interactive window needed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT = os.path.join(ROOT, "output")
os.makedirs(OUTPUT, exist_ok=True)


def out(name: str) -> str:
    return os.path.join(OUTPUT, name)


def header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"  [{label}: {time.perf_counter() - start:.2f}s]")


def print_matrix(matrix, rows=None, cols=None, indent: str = "  ", width: int = 5) -> None:
    """Print a small integer-valued matrix with row and column labels.

    Boundary matrices are only readable when you can see which face is which
    row, so the labels are the point of this.
    """
    matrix = [[m for m in row] for row in matrix]
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows else 0
    rows = list(rows) if rows is not None else [str(i) for i in range(n_rows)]
    cols = list(cols) if cols is not None else [str(j) for j in range(n_cols)]
    label_width = max((len(r) for r in rows), default=0)

    print(f"{indent}{'':{label_width}}  " + "".join(f"{c:>{width}}" for c in cols))
    for label, row in zip(rows, matrix):
        cells = "".join(f"{int(round(v)):>{width}d}" for v in row)
        print(f"{indent}{label:>{label_width}}  {cells}")


def check(claim: str, ok: bool) -> bool:
    """Print a claim from the slides next to whether the code reproduces it."""
    print(f"  [{'ok' if ok else 'FAILED'}] {claim}")
    return ok
