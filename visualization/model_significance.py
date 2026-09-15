"""
model_significance.py -- pairwise significance tests between models.

Every metric in data/metrics/ is computed on the same 50 reference samples for every
model (column i = data/random_samples/sample_i), so model comparisons are *paired*.
For each figure panel we take each model at the layer shown in the figure (the layer
with the highest mean over the 50 samples) and compare every pair of models with

  * the Wilcoxon signed-rank test on the 50 paired differences (primary, two-sided),
  * the paired t-test (secondary, two-sided),

and correct the Wilcoxon p-values with Holm's method within each panel (10 pairs of
5 models).  Effect sizes: mean paired difference (A - B) and the matched-pairs
rank-biserial correlation r (from -1 to 1, positive = A higher).

Reads only data/metrics/**.xlsx (no embeddings, no cluster access needed) and writes
results/statistics/pairwise_model_tests.csv.

    python visualization/model_significance.py
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import figure_io as fio
from figure_io import MODELS

OUT = fio.RESULTS / "statistics"


# --------------------------------------------------------------------------- #
#  Loading: one 50-long vector per model, taken at the figure's best layer.     #
# --------------------------------------------------------------------------- #
def _best_row(mat: pd.DataFrame) -> tuple[str, pd.Series]:
    """Row (layer) with the highest mean over samples, as in the figures."""
    means = mat.mean(axis=1, skipna=True)
    layer = means.idxmax()
    return str(layer), mat.loc[layer]


def load_reference(model: str, metric: str):
    """Fig. 1 tables: rows = layers (0..n), columns = sample index 0..49."""
    mat = pd.read_excel(fio.DATA_METRICS / model / f"{metric}.xlsx")
    mat.columns = [int(c) for c in mat.columns]
    layer, row = _best_row(mat)
    return ("-" if model == "kmer" else layer), row


def load_pk_matrix(model: str, filename: str):
    """Pipeline tables: 'layer' column + sample_<i> columns in arbitrary order."""
    mat = pd.read_excel(fio.DATA_METRICS / model / filename).set_index("layer")
    mat.columns = [int(c.split("_")[1]) for c in mat.columns]
    return _best_row(mat.sort_index(axis=1))


# --------------------------------------------------------------------------- #
#  Panels: (figure, panel label, loader) -- one family of 10 comparisons each.  #
# --------------------------------------------------------------------------- #
def panels():
    for metric in ["ARI", "P5", "silh"]:
        yield "fig1", metric, lambda m, metric=metric: load_reference(m, metric)

    for name, label in [("Egovirales_preproc", "P@5 Egovirales"),
                        ("Mriyavirales_preproc", "P@5 Mriyavirales"),
                        ("REVERSECOMPLEMENTEgovirales_preproc", "P@5 Egovirales (RC)"),
                        ("REVERSECOMPLEMENTMriyavirales_preproc", "P@5 Mriyavirales (RC)")]:
        yield "fig4", label, lambda m, f=f"P@5_matrix_{name}.xlsx": load_pk_matrix(m, f)

    for name, label in [("Egovirales_preproc", "CROSS P@5 Egovirales"),
                        ("Mriyavirales_preproc", "CROSS P@5 Mriyavirales")]:
        yield "fig5", label, lambda m, f=f"cross_P@5_matrix_{name}.xlsx": load_pk_matrix(m, f)

    for virus in ["Egovirales", "Mriyavirales"]:
        for k in range(1, 11):
            yield ("fig6", f"P@{k} {virus}",
                   lambda m, f=f"P@{k}_matrix_{virus}_preproc.xlsx": load_pk_matrix(m, f))


# --------------------------------------------------------------------------- #
#  Statistics.                                                                  #
# --------------------------------------------------------------------------- #
def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of paired differences d."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def compare_panel(figure: str, panel: str, loader) -> pd.DataFrame:
    data = {m: loader(m) for m in MODELS}
    rows = []
    for a, b in combinations(MODELS, 2):
        (la, xa), (lb, xb) = data[a], data[b]
        pair = pd.concat([xa, xb], axis=1, keys=["a", "b"]).dropna()
        d = (pair["a"] - pair["b"]).to_numpy()
        rows.append({
            "figure": figure, "panel": panel,
            "model_a": a, "layer_a": la, "model_b": b, "layer_b": lb,
            "n": len(d),
            "mean_a": pair["a"].mean(), "mean_b": pair["b"].mean(),
            "mean_diff": d.mean(),
            "rank_biserial_r": rank_biserial(d),
            "p_wilcoxon": stats.wilcoxon(pair["a"], pair["b"]).pvalue,
            "p_ttest": stats.ttest_rel(pair["a"], pair["b"]).pvalue,
        })
    df = pd.DataFrame(rows)
    df["p_wilcoxon_holm"] = holm(df["p_wilcoxon"].to_numpy())
    return df


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([compare_panel(*p) for p in panels()], ignore_index=True)
    df["signif"] = df["p_wilcoxon_holm"].map(stars)
    path = OUT / "pairwise_model_tests.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"  tests    ->  {path.relative_to(fio.ROOT).as_posix()}  ({len(df)} comparisons)")
    return df


if __name__ == "__main__":
    main()
