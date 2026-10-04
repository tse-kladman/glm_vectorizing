"""Paired tests between models, and the multiple-testing corrections.

Every model is scored on the same 50 reference samples, so model comparisons are
paired: the difference per sample removes the variation that is common to all
models and leaves only the difference between them.

    Wilcoxon signed-rank   primary, two-sided, on the 50 paired differences
    paired t-test          secondary
    rank-biserial r        effect size, +1 meaning A wins on every sample

The Wilcoxon test has a floor: with n samples the smallest two-sided p it can
return is 2/2^n, so values at that floor mean "A wins on every sample", not an
exact probability.

Corrections.  Within one panel, Holm over its ten pairs.  Across the study, Holm
(family-wise) and Benjamini-Hochberg (false discovery rate) over the distinct
comparisons -- distinct, because some panels read the same table and must not be
counted twice.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (controls the false discovery rate)."""
    n = len(p)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((n / np.arange(n, 0, -1)) * p[order][::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: +1 = A larger on every sample."""
    differences = differences[differences != 0]
    if not len(differences):
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum())


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p the exact Wilcoxon test can return for n pairs."""
    return 2 / 2 ** n


def compare_pair(a_values: pd.Series, b_values: pd.Series) -> dict:
    """One paired comparison: effect sizes and both raw p-values."""
    pair = pd.concat([a_values, b_values], axis=1, keys=["a", "b"]).dropna()
    differences = (pair["a"] - pair["b"]).to_numpy()
    return {
        "n": len(differences),
        "mean_a": pair["a"].mean(), "mean_b": pair["b"].mean(),
        "mean_diff": differences.mean(),
        "rank_biserial_r": rank_biserial(differences),
        "p_wilcoxon": (stats.wilcoxon(pair["a"], pair["b"]).pvalue
                       if differences.any() else 1.0),
        "p_ttest": (stats.ttest_rel(pair["a"], pair["b"]).pvalue
                    if differences.any() else 1.0),
    }


def panel_tests(values: dict, order: list, context: dict) -> pd.DataFrame:
    """Every pair of series in one panel, Holm-corrected within that panel."""
    rows = []
    for a, b in combinations([key for key in order if key in values], 2):
        row = {**context, "model_a": a, "model_b": b}
        row.update(compare_pair(values[a], values[b]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["p_holm_panel"] = holm(frame["p_wilcoxon"].to_numpy())
    return frame


def correct(tests: pd.DataFrame, key_columns=("source", "model_a", "model_b")) -> pd.DataFrame:
    """Study-wide Holm and BH over the *distinct* comparisons, then significance."""
    tests = tests.copy()
    tests["comparison"] = tests[list(key_columns)].astype(str).agg("|".join, axis=1)
    unique = tests.drop_duplicates("comparison").set_index("comparison")
    raw = unique["p_wilcoxon"].to_numpy()
    adjusted = pd.DataFrame({"p_holm_all": holm(raw), "p_bh_all": benjamini_hochberg(raw)},
                            index=unique.index)
    tests = tests.join(adjusted, on="comparison")
    tests["signif"] = tests["p_bh_all"].map(stars)
    return tests
