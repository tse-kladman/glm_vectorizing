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
"won every draw", not an estimate.  For uncertainty over genomes see
``bootstrap_ci_by_genome`` below.

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


# ── cluster bootstrap over genomes ──────────────────────────────────────────

def genome_matrix(per_genome: pd.DataFrame, metric: str, layer=None) -> pd.DataFrame:
    """Per-genome table (``results/metrics/<model>/per_genome_*.csv.gz``) -> samples x genomes.

    ``layer`` selects one layer (as the string stored in the file); ``None`` is only
    valid for layer-less models, where the single layer is taken.
    """
    frame = per_genome[per_genome["metric"] == metric]
    layers = frame["layer"].astype(str).unique()
    if layer is None:
        if len(layers) != 1:
            raise ValueError(f"{len(layers)} layers present; say which one")
        layer = layers[0]
    frame = frame[frame["layer"].astype(str) == str(layer)]
    return frame.pivot(index="sample", columns="genome", values="value")


def bootstrap_ci_by_genome(values, n_boot: int = 10_000, alpha: float = 0.05,
                           seed: int | None = None) -> dict:
    """Cluster bootstrap over genomes: a CI that treats the genome as the unit.

    ``values`` is ``(n_samples, n_genomes)`` -- one row per reference draw, one column
    per genome of the unseen order (a 1-D array is read as one row).  Fragments of a
    genome are not independent observations, and neither are the 50 reference draws
    (they share a population of reference viruses), so what varies *between
    independent units* is the genome.  Each replicate resamples genomes with
    replacement, then averages over those genomes and over the reference draws.

    The interval answers "how uncertain is this score given which genomes of the
    order happened to be available"; the Wilcoxon test over reference samples does not
    (it only reflects the choice of reference draw).  With few genomes the percentile
    interval is optimistic and is reported with ``n_genomes`` so a reader can see that.

    ``seed`` is required: every random choice in the package takes an explicit seed.
    NaNs (a genome missing from a draw) are ignored in the averages.
    """
    if seed is None:
        raise ValueError("bootstrap_ci_by_genome needs an explicit seed")
    matrix = np.atleast_2d(np.asarray(values, dtype=float))
    n_genomes = matrix.shape[1]
    if n_genomes < 2:
        raise ValueError("a bootstrap over genomes needs at least two genomes")

    per_genome = np.nanmean(matrix, axis=0)              # average over reference draws
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_genomes, size=(n_boot, n_genomes))
    replicates = per_genome[picks].mean(axis=1)
    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(per_genome.mean()), "low": float(low), "high": float(high),
            "n_genomes": int(n_genomes), "n_boot": int(n_boot), "alpha": alpha,
            "seed": int(seed)}


def bootstrap_diff_by_genome(first, second, n_boot: int = 10_000, alpha: float = 0.05,
                             seed: int | None = None) -> dict:
    """Cluster bootstrap of the paired difference ``first - second`` over genomes.

    Both are ``(n_samples, n_genomes)`` over the *same* genomes (the model comparison
    on one unseen order).  The same resampled genomes are used for both models.
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("the two models must be scored on the same samples and genomes")
    return bootstrap_ci_by_genome(first - second, n_boot, alpha, seed)
