"""Run the example scripts, writing figures to ``output/``.

    python run_examples.py                 # the five library examples
    python run_examples.py --talk          # the eight tutorial examples
    python run_examples.py --all           # everything
    python run_examples.py 03_higher_dimensions.py talk_06_hodge_laplacian.py
"""

import runpy
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")

SCRIPTS = [
    "01_circle_vs_noise.py",
    "02_shapes_gallery.py",
    "03_higher_dimensions.py",
    "04_stability_and_distances.py",
    "05_periodicity_detection.py",
]

# One per part of the tutorial in tda_ai_2h_1.tex; each one recomputes the
# numbers that part asserts and prints whether they came out.
TALK_SCRIPTS = [
    "talk_01_motivation.py",
    "talk_02_complexes.py",
    "talk_03_homology.py",
    "talk_04_persistence.py",
    "talk_04_signal_vs_noise.py",
    "talk_05_pipeline.py",
    "talk_06_hodge_laplacian.py",
    "talk_07_exercise.py",
]

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        chosen = SCRIPTS
    elif args == ["--talk"]:
        chosen = TALK_SCRIPTS
    elif args == ["--all"]:
        chosen = SCRIPTS + TALK_SCRIPTS
    else:
        chosen = args

    sys.path.insert(0, EXAMPLES)
    for name in chosen:
        print("\n" + "=" * 72)
        print(f"running {name}")
        print("=" * 72)
        runpy.run_path(os.path.join(EXAMPLES, name), run_name="__main__")
