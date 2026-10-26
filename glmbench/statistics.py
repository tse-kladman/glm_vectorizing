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

What the paired test covers.  Wilcoxon over the 50 overlapping reference samples tests
robustness to the *draw of the reference set* only -- not to the choice of viruses
(which unseen orders, which of their genomes).  With n = 50 the exact p-value hits its
floor 2/2^50 = 1.8e-15 whenever one model wins on every sample, so such values are
"won every draw", not an estimate.

Corrections.  Within one panel, Holm over its pairs of models.  Across the study, Holm
(family-wise) and Benjamini-Hochberg (false discovery rate) over the distinct
comparisons -- distinct, because some panels read the same table and must not be
counted twice.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _finite_only(adjust):
    """Apply a correction to the finite p-values only; NaN stays NaN in place.

    A comparison that could not be made (no overlapping samples) has no p-value.  It
    must neither count towards the size of the family nor -- through ``np.minimum`` /
    ``np.maximum`` propagating NaN along the running extremum -- turn the adjusted
    p-values of real comparisons into NaN.
    """
    def wrapped(p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        out = np.full(p.shape, np.nan)
        finite = np.isfinite(p)
        if finite.any():
            out[finite] = adjust(p[finite])
        return out
    wrapped.__doc__ = adjust.__doc__
    wrapped.__name__ = adjust.__name__
    return wrapped


@_finite_only
def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate).

    NaN entries are left out of the family and returned as NaN.
    """
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


@_finite_only
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (controls the false discovery rate).

    NaN entries are left out of the family and returned as NaN.
    """
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
    """Significance mark of an (adjusted) p; '' when there is no p-value at all."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p the exact Wilcoxon test can return for n pairs."""
    return 2 / 2 ** n


def compare_pair(a_values: pd.Series, b_values: pd.Series) -> dict:
    """One paired comparison: effect sizes and both raw p-values.

    The two series are aligned on their index (the reference-sample number), so a
    sample missing from one side drops out of both.  No common sample at all gives
    NaN p-values -- there is nothing to test, which is not the same as "no
    difference" (identical values on every sample give p = 1).
    """
    pair = pd.concat([a_values, b_values], axis=1, keys=["a", "b"]).dropna()
    differences = (pair["a"] - pair["b"]).to_numpy()
    if not len(differences):
        return {"n": 0, "mean_a": np.nan, "mean_b": np.nan, "mean_diff": np.nan,
                "rank_biserial_r": np.nan, "p_wilcoxon": np.nan, "p_ttest": np.nan}
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
