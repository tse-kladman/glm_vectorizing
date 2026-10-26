"""Significance marks drawn on the comparison figures.

Every comparison in a figure is paired over the reference samples (the same 50 draws,
or the 10 of the word-length sweep), so the test is always the two-sided Wilcoxon
signed-rank test of ``statistics.compare_pair`` and the mark is ``statistics.stars``
of a Benjamini-Hochberg adjusted p-value:

    ***  p < 0.001     **  p < 0.01     *  p < 0.05     ns  not significant

Which family the adjustment runs over depends on where the p-values come from:

    study table   Fig. 4-6 (unseen orders) read ``results/statistics/pairwise_model_tests.csv``
                  and use its ``p_bh_all``, so a mark on a bar is exactly the one the
                  box plots and the text use (BH over all distinct comparisons).
    per figure    the other figures compare things the study table does not hold
                  (layers, word lengths, scaling/distance variants); their p-values are
                  adjusted together, one family per figure.

What the marks do and do not say is the same as for the tests themselves (see
``statistics``): robustness to the draw of the reference set, not to the choice of
viruses.
"""

import numpy as np
import pandas as pd

from .. import config, statistics

STUDY_TABLE = "pairwise_model_tests.csv"


def paired_p(first: pd.Series, second: pd.Series) -> float:
    """Raw two-sided Wilcoxon p of two per-sample series (aligned on their index)."""
    return float(statistics.compare_pair(first, second)["p_wilcoxon"])


def adjust(raw: dict) -> dict:
    """BH-adjust a {key: raw p} family; returns {key: adjusted p}."""
    if not raw:
        return {}
    keys = list(raw)
    adjusted = statistics.benjamini_hochberg(np.array([raw[key] for key in keys], dtype=float))
    return dict(zip(keys, adjusted))


# ── the study-wide table (Fig. 4-6) ─────────────────────────────────────────

def study_tests(selection: str = "reference", statistics_dir=None) -> pd.DataFrame:
    """Rows of the study-wide test table under one layer-selection rule.

    The table is written by the ``stats`` stage (``figures/statistics.build``); the
    ``figures`` stage builds it before the main figures, so the marks are never read
    from a table older than the metric tables they are drawn on.  Empty when it does
    not exist yet (the figures are then drawn without marks).
    """
    path = (statistics_dir or config.STATISTICS) / STUDY_TABLE
    if not path.exists():
        return pd.DataFrame()
    tests = pd.read_csv(path)
    return tests[tests["selection"] == selection]


def study_p(tests: pd.DataFrame, panel: str, first: str, second: str) -> float:
    """Study-wide BH p of one pair in one panel (order of the pair does not matter)."""
    if tests.empty:
        return float("nan")
    rows = tests[(tests["panel"] == panel)
                 & (((tests["model_a"] == first) & (tests["model_b"] == second))
                    | ((tests["model_a"] == second) & (tests["model_b"] == first)))]
    return float(rows["p_bh_all"].iloc[0]) if len(rows) else float("nan")


def label(p: float) -> str:
    """Mark of one adjusted p; empty when there was nothing to test."""
    return statistics.stars(p)


# ── drawing ─────────────────────────────────────────────────────────────────

def put(axis, x: float, y: float, text: str, fontsize: float = 9) -> None:
    """Write a mark centred above a bar or point; 'ns' is smaller and grey."""
    if not text:
        return
    significant = text != "ns"
    axis.text(x, y, text, ha="center", va="bottom",
              fontsize=fontsize if significant else fontsize - 1.5,
              color="black" if significant else "0.45")


def against_best(per_sample: pd.DataFrame) -> tuple:
    """For a table rows = candidates (layers, k), columns = samples: the best row by
    mean (NaN samples ignored) and the RAW p of every other row against it.

    Raw, not adjusted: a figure holds several such tables (models x metrics, methods x
    metrics) and they form one BH family together, so the caller pools the raw
    p-values of all of them and adjusts once (``adjust``).  Rows are compared sample
    by sample: both rows come from the same table, so their columns are the same
    reference samples.
    """
    best = per_sample.mean(axis=1, skipna=True).idxmax()
    raw = {row: paired_p(per_sample.loc[row], per_sample.loc[best])
           for row in per_sample.index if row != best}
    return best, raw

