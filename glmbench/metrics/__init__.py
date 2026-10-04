"""The metrics themselves.

    tables.py    reading and writing the per-sample metric tables
    unseen.py    P@k and CROSS P@k on the unseen orders          (Fig. 4-6)
    pairwise.py  one unseen order against the other              (Fig. 7)
    reference.py the reference-set benchmark that chooses layers (Fig. 1-2)
    runner.py    running all of it in parallel and writing the tables

Each metric is a mean over the 50 reference samples, and every model is scored on
the same samples, which is what makes the comparisons in statistics.py paired.
"""

from . import pairwise, reference, tables, unseen  # noqa: F401
