"""
model_significance.py -- pairwise significance tests between models, with figures.

Every metric in data/metrics/ is computed on the same 50 reference samples for every
model (column i = data/random_samples/sample_i), so model comparisons are *paired*.
For each panel of Fig. 1, 4, 5 and 6 we take each model at the layer shown in that
figure (the layer with the highest mean over the 50 samples) and compare every pair of
models with

  * the Wilcoxon signed-rank test on the 50 paired differences (primary, two-sided),
  * the paired t-test (secondary, two-sided).

Multiple testing.  The raw p-values are corrected three ways:

  * ``p_holm_panel`` -- Holm within one panel (10 pairs of 5 models); the relevant
    correction when a single panel is read on its own,
  * ``p_holm_all``   -- Holm over *all* comparisons of the whole study (FWER),
  * ``p_bh_all``     -- Benjamini-Hochberg over all comparisons (FDR); this is the
    column ``signif`` and every figure annotation are based on.

Effect sizes: mean paired difference (A - B) and the matched-pairs rank-biserial
correlation r (-1..1, positive = A higher).

The script is split like the figure notebooks:

  Stage 1  data/metrics/**.xlsx  ->  results/figure_data/statistics/*.csv
  Stage 2  results/figure_data/statistics/*.csv  ->  results/figures/statistics/*.{png,pdf}

    python visualization/model_significance.py
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import figure_io as fio
from figure_io import MODELS, DISPLAY, MODEL_COLOR, save_figure

FIG_DATA = fio.FIG_DATA / "statistics"

# Panels drawn as box plots / heat maps; Fig. 6 (P@k for k = 1..10) stays in the CSV
# only -- it is 20 more panels of the same two comparisons.
PLOTTED = {
    "fig1": ("Reference benchmark (Fig. 1)", 3),
    "fig4": ("Generalisation to unseen orders (Fig. 4)", 2),
    "fig5": ("Forward vs reverse complement (Fig. 5)", 2),
}


# --------------------------------------------------------------------------- #
#  Selecting the layer.                                                         #
#                                                                               #
#  The layer shown in a figure is the one with the highest mean, chosen on the   #
#  same 50 samples the result is then reported on -- which biases that result    #
#  upwards.  ``split`` fixes this: even-numbered samples choose the layer, odd-  #
#  numbered ones are used for every reported number and every test.  ``in_sample #
#  reproduces the figures as they stand; both are written out so the size of the #
#  selection effect stays visible.                                              #
# --------------------------------------------------------------------------- #
SELECTIONS = ("split", "in_sample", "oracle")   # primary first


def _best_row(mat: pd.DataFrame, selection: str) -> tuple[str, pd.Series]:
    """Pick the layer and return its values, under one of three selection rules.

    split     choose on even samples, report on odd ones  -- primary, unbiased
    in_sample choose and report on all 50                 -- reproduces the figures
    oracle    choose and report on the odd samples        -- same reporting samples
              as ``split``, so the difference between the two is the selection
              optimism itself, uncontaminated by which half is scored
    """
    even = [c for c in mat.columns if c % 2 == 0]
    odd = [c for c in mat.columns if c % 2 == 1]

    if selection == "in_sample":
        choose = report = list(mat.columns)
    elif selection == "split":
        choose, report = even, odd
    elif selection == "oracle":
        choose = report = odd
    else:
        raise ValueError(f"unknown selection rule: {selection}")

    layer = mat[choose].mean(axis=1, skipna=True).idxmax()
    return str(layer), mat.loc[layer, report]


def load_reference(model: str, metric: str, selection: str):
    """Fig. 1 tables: rows = layers (0..n), columns = sample index 0..49."""
    mat = pd.read_excel(fio.DATA_METRICS / model / f"{metric}.xlsx")
    mat.columns = [int(c) for c in mat.columns]
    layer, row = _best_row(mat, selection)
    return ("-" if model == "kmer" else layer), row


def load_pk_matrix(model: str, filename: str, selection: str):
    """Pipeline tables: 'layer' column + sample_<i> columns in arbitrary order."""
    mat = pd.read_excel(fio.DATA_METRICS / model / filename).set_index("layer")
    mat.columns = [int(c.split("_")[1]) for c in mat.columns]
    return _best_row(mat.sort_index(axis=1), selection)


# --------------------------------------------------------------------------- #
#  Panels: (figure, panel label, loader) -- one family of 10 comparisons each.  #
# --------------------------------------------------------------------------- #
def panels():
    """Yield (figure, panel label, source, loader).

    ``source`` identifies the underlying metric table.  Fig. 4 and the k = 5 panels of
    Fig. 6 are the same tables, so they share a source and are counted once by the
    study-wide multiple-testing correction.
    """
    for metric, label in [("ARI", "ARI"), ("P5", "Precision@5"), ("silh", "Silhouette")]:
        yield ("fig1", label, metric,
               lambda m, s, metric=metric: load_reference(m, metric, s))

    for name, label in [("Egovirales_preproc", "P@5 Egovirales"),
                        ("Mriyavirales_preproc", "P@5 Mriyavirales"),
                        ("REVERSECOMPLEMENTEgovirales_preproc", "P@5 Egovirales (RC)"),
                        ("REVERSECOMPLEMENTMriyavirales_preproc", "P@5 Mriyavirales (RC)")]:
        f = f"P@5_matrix_{name}.xlsx"
        yield "fig4", label, f, lambda m, s, f=f: load_pk_matrix(m, f, s)

    for name, label in [("Egovirales_preproc", "CROSS P@5 Egovirales"),
                        ("Mriyavirales_preproc", "CROSS P@5 Mriyavirales")]:
        f = f"cross_P@5_matrix_{name}.xlsx"
        yield "fig5", label, f, lambda m, s, f=f: load_pk_matrix(m, f, s)

    for virus in ["Egovirales", "Mriyavirales"]:
        for k in range(1, 11):
            f = f"P@{k}_matrix_{virus}_preproc.xlsx"
            yield "fig6", f"P@{k} {virus}", f, lambda m, s, f=f: load_pk_matrix(m, f, s)


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
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (controls the false discovery rate)."""
    n = len(p)
    order = np.argsort(p)
    adj = np.minimum.accumulate((n / np.arange(n, 0, -1)) * p[order][::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------- #
#  Stage 1 -- per-sample values and the tests built from them.                  #
# --------------------------------------------------------------------------- #
def _extract_one(selection: str):
    """All panels for one selection rule ("split" or "in_sample")."""
    samples, tests = [], []

    for figure, panel, source, loader in panels():
        data = {m: loader(m, selection) for m in MODELS}

        for model, (layer, values) in data.items():
            for sample, value in values.items():
                samples.append({"selection": selection, "figure": figure, "panel": panel,
                                "model": model, "layer": layer,
                                "sample": int(sample), "value": float(value)})

        rows = []
        for a, b in combinations(MODELS, 2):
            (la, xa), (lb, xb) = data[a], data[b]
            pair = pd.concat([xa, xb], axis=1, keys=["a", "b"]).dropna()
            d = (pair["a"] - pair["b"]).to_numpy()
            rows.append({
                "selection": selection,
                "figure": figure, "panel": panel, "source": source,
                "model_a": a, "layer_a": la, "model_b": b, "layer_b": lb,
                "n": len(d),
                "mean_a": pair["a"].mean(), "mean_b": pair["b"].mean(),
                "mean_diff": d.mean(),
                "rank_biserial_r": rank_biserial(d),
                "p_wilcoxon": stats.wilcoxon(pair["a"], pair["b"]).pvalue,
                "p_ttest": stats.ttest_rel(pair["a"], pair["b"]).pvalue,
            })
        panel_df = pd.DataFrame(rows)
        # Holm within this panel: the correction for reading one panel on its own.
        panel_df["p_holm_panel"] = holm(panel_df["p_wilcoxon"].to_numpy())
        tests.append(panel_df)

    tests = pd.concat(tests, ignore_index=True)

    # Study-wide corrections.  Fig. 4 and the k = 5 panels of Fig. 6 test the same
    # tables, so the family is the set of *distinct* comparisons; a duplicated
    # comparison then simply carries the adjusted p-value of its twin.  The two
    # selection rules are alternative analyses of one dataset, not extra
    # hypotheses, so each is corrected within itself.
    tests["comparison"] = tests["source"] + "|" + tests["model_a"] + "|" + tests["model_b"]
    unique = tests.drop_duplicates("comparison").set_index("comparison")
    p = unique["p_wilcoxon"].to_numpy()
    adjusted = pd.DataFrame({"p_holm_all": holm(p), "p_bh_all": benjamini_hochberg(p)},
                            index=unique.index)
    tests = tests.join(adjusted, on="comparison")
    tests["signif"] = tests["p_bh_all"].map(stars)
    print(f"  {selection:<10} n={tests['n'].max()} per comparison, "
          f"{len(unique)} distinct comparisons, "
          f"{(adjusted['p_bh_all'] >= 0.05).sum()} not significant after FDR")
    return pd.DataFrame(samples), tests


def extract():
    frames = [_extract_one(selection) for selection in SELECTIONS]
    samples = pd.concat([f[0] for f in frames], ignore_index=True)
    tests = pd.concat([f[1] for f in frames], ignore_index=True)

    FIG_DATA.mkdir(parents=True, exist_ok=True)
    samples.to_csv(FIG_DATA / "panel_samples.csv", index=False, float_format="%.6g")
    tests.to_csv(FIG_DATA / "pairwise_model_tests.csv", index=False, float_format="%.6g")
    print(f"  samples  ->  {(FIG_DATA / 'panel_samples.csv').relative_to(fio.ROOT).as_posix()}"
          f"  ({len(samples)} rows)")
    print(f"  tests    ->  {(FIG_DATA / 'pairwise_model_tests.csv').relative_to(fio.ROOT).as_posix()}"
          f"  ({len(tests)} comparisons)")
    return samples, tests


# --------------------------------------------------------------------------- #
#  Stage 2 -- figures.                                                          #
# --------------------------------------------------------------------------- #
def _grid(n_panels: int, ncols: int, panel_size=(4.2, 4.0)):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat


def plot_boxes(samples: pd.DataFrame, tests: pd.DataFrame):
    """Distribution over the 50 reference samples, one box per model."""
    rng = np.random.default_rng(0)

    for figure, (title, ncols) in PLOTTED.items():
        panel_names = samples[samples["figure"] == figure]["panel"].unique()
        fig, axes = _grid(len(panel_names), ncols)

        for ax, panel in zip(axes, panel_names):
            sub = samples[(samples["figure"] == figure) & (samples["panel"] == panel)]
            values = [sub[sub["model"] == m]["value"].to_numpy() for m in MODELS]

            bp = ax.boxplot(values, widths=0.6, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.6))
            for patch, model in zip(bp["boxes"], MODELS):
                patch.set_facecolor(MODEL_COLOR[model])
                patch.set_alpha(0.55)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)

            # the 50 paired observations behind each box
            for pos, (vals, model) in enumerate(zip(values, MODELS), start=1):
                jitter = rng.uniform(-0.13, 0.13, len(vals))
                ax.plot(pos + jitter, vals, "o", ms=2.4, mew=0,
                        color=MODEL_COLOR[model], alpha=0.65, zorder=3)

            layers = {m: sub[sub["model"] == m]["layer"].iloc[0] for m in MODELS}
            ax.set_xticks(range(1, len(MODELS) + 1))
            ax.set_xticklabels([DISPLAY[m] if str(layers[m]) == "-" else
                                f"{DISPLAY[m]}\n(L{layers[m]})" for m in MODELS],
                               fontsize=9, rotation=45, ha="right")
            ax.set_title(panel, fontsize=12)
            ax.set_ylabel("Score per reference sample")

        fig.suptitle(f"{title}\ndistribution over the 50 paired reference samples",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save_figure(fig, f"box_{figure}", kind="statistics")
        plt.close(fig)


def plot_heatmaps(tests: pd.DataFrame):
    """Pairwise mean difference (colour) and BH-adjusted significance (annotation)."""
    cmap = plt.get_cmap("RdBu_r")

    for figure, (title, ncols) in PLOTTED.items():
        panel_names = tests[tests["figure"] == figure]["panel"].unique()
        fig, axes = _grid(len(panel_names), ncols, panel_size=(4.4, 4.0))

        for pos, (ax, panel) in enumerate(zip(axes, panel_names)):
            sub = tests[(tests["figure"] == figure) & (tests["panel"] == panel)]
            diff = pd.DataFrame(np.nan, index=MODELS, columns=MODELS)
            ann = pd.DataFrame("", index=MODELS, columns=MODELS)
            for _, row in sub.iterrows():
                a, b = row["model_a"], row["model_b"]
                diff.loc[a, b], diff.loc[b, a] = row["mean_diff"], -row["mean_diff"]
                ann.loc[a, b] = f"{row['mean_diff']:+.3f}\n{stars(row['p_bh_all'])}"
                ann.loc[b, a] = f"{-row['mean_diff']:+.3f}\n{stars(row['p_bh_all'])}"

            lim = np.nanmax(np.abs(diff.to_numpy(dtype=float)))
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
            ax.imshow(diff.to_numpy(dtype=float), cmap=cmap, norm=norm)
            for i in range(len(MODELS)):
                for j in range(len(MODELS)):
                    if i == j:
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                                   color="0.92", zorder=2))
                        continue
                    ax.text(j, i, ann.iloc[i, j], ha="center", va="center",
                            fontsize=7.5, zorder=3,
                            color="white" if abs(diff.iloc[i, j]) > 0.72 * lim else "black")
            ax.set_xticks(range(len(MODELS)))
            ax.set_xticklabels([DISPLAY[m] for m in MODELS], fontsize=8, rotation=45, ha="right")
            ax.set_yticks(range(len(MODELS)))
            ax.set_yticklabels([DISPLAY[m] for m in MODELS] if pos % ncols == 0 else [],
                               fontsize=8)
            ax.set_title(panel, fontsize=11)
            ax.grid(False)
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                                ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("mean difference (row - column)", fontsize=7)

        fig.suptitle(f"{title}\nrow vs column: mean difference over the 50 samples and "
                     f"BH-adjusted significance", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, f"heatmap_{figure}", kind="statistics")
        plt.close(fig)


def plot_paired_diffs(samples: pd.DataFrame, tests: pd.DataFrame):
    """What the test actually looks at: the 50 paired differences against the best model.

    The marginal box plots can overlap while every single sample still ranks the two
    models the same way; these boxes show that directly -- a box that clears zero is a
    model beaten on all 50 reference samples.
    """
    for figure, (title, ncols) in PLOTTED.items():
        panel_names = samples[samples["figure"] == figure]["panel"].unique()
        fig, axes = _grid(len(panel_names), ncols, panel_size=(4.2, 4.0))

        for ax, panel in zip(axes, panel_names):
            sub = samples[(samples["figure"] == figure) & (samples["panel"] == panel)]
            wide = sub.pivot(index="sample", columns="model", values="value")
            ref = wide.mean().idxmax()                      # best model of this panel
            others = [m for m in MODELS if m != ref]

            diffs = [(wide[m] - wide[ref]).to_numpy() for m in others]
            bp = ax.boxplot(diffs, widths=0.6, patch_artist=True, showfliers=True,
                            flierprops=dict(ms=3, mfc="0.4", mec="none"),
                            medianprops=dict(color="black", lw=1.6))
            for patch, model in zip(bp["boxes"], others):
                patch.set_facecolor(MODEL_COLOR[model])
                patch.set_alpha(0.55)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)

            ax.axhline(0, color="crimson", lw=1.2, ls="--")
            look = tests[(tests["figure"] == figure) & (tests["panel"] == panel)]
            for pos, model in enumerate(others, start=1):
                row = look[((look["model_a"] == model) & (look["model_b"] == ref)) |
                           ((look["model_a"] == ref) & (look["model_b"] == model))].iloc[0]
                ax.annotate(stars(row["p_bh_all"]), (pos, 0.96), xycoords=("data", "axes fraction"),
                            ha="center", va="top", fontsize=9, color="crimson")

            ax.set_xticks(range(1, len(others) + 1))
            ax.set_xticklabels([DISPLAY[m] for m in others], fontsize=9, rotation=45, ha="right")
            ax.set_ylabel(f"difference vs {DISPLAY[ref]}")
            ax.set_title(f"{panel}\nbest: {DISPLAY[ref]}", fontsize=11)
            ax.margins(y=0.16)

        fig.suptitle(f"{title}\npaired differences over the 50 reference samples "
                     f"(the quantity the Wilcoxon test ranks)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save_figure(fig, f"paired_{figure}", kind="statistics")
        plt.close(fig)


def plot_pvalue_overview(tests: pd.DataFrame):
    """Every comparison of the study: raw vs BH-adjusted p, with the Wilcoxon floor."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    floor = 2 / 2 ** tests["n"].max()

    for figure, marker in zip(["fig1", "fig4", "fig5", "fig6"], ["o", "s", "^", "d"]):
        sub = tests[tests["figure"] == figure]
        ax.plot(np.maximum(sub["p_wilcoxon"], floor / 2),
                np.maximum(sub["p_bh_all"], floor / 2),
                marker, ms=5, alpha=0.7, label=f"{figure}  (n={len(sub)})")

    lims = [floor / 3, 2]
    ax.plot(lims, lims, "-", color="0.6", lw=1, zorder=0)
    ax.axhline(0.05, ls="--", color="crimson", lw=1)
    ax.axvline(floor, ls=":", color="0.4", lw=1)
    ax.text(floor * 1.4, 1e-12, "exact Wilcoxon floor\n$2/2^{50}$", fontsize=8, color="0.3")
    ax.text(lims[0] * 1.5, 0.065, "0.05", fontsize=8, color="crimson")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("raw Wilcoxon p"); ax.set_ylabel("Benjamini-Hochberg adjusted p")
    unique = tests.drop_duplicates("comparison")
    ax.set_title(f"All {len(unique)} distinct model comparisons\n"
                 f"{(unique['p_bh_all'] >= 0.05).sum()} do not survive FDR correction",
                 fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "pvalue_overview", kind="statistics")
    plt.close(fig)


def plot_selection_effect(tests: pd.DataFrame):
    """Selection optimism: 'oracle' and 'split' score the SAME samples (the odd half).

    They differ only in where the layer was chosen -- on those same samples, or on
    the held-out even half -- so their gap isolates the bias of choosing the layer
    on the data it is then reported on.
    """
    keys = ["figure", "panel", "model_a", "model_b"]
    merged = (tests[tests["selection"] == "oracle"]
              .merge(tests[tests["selection"] == "split"], on=keys,
                     suffixes=("_oracle", "_split")))

    # one row per (panel, model): both models of each pair contribute
    scores = pd.concat([
        merged[["layer_a_oracle", "layer_a_split", "mean_a_oracle", "mean_a_split"]]
        .set_axis(["layer_o", "layer_s", "mean_o", "mean_s"], axis=1),
        merged[["layer_b_oracle", "layer_b_split", "mean_b_oracle", "mean_b_split"]]
        .set_axis(["layer_o", "layer_s", "mean_o", "mean_s"], axis=1),
    ], ignore_index=True).drop_duplicates()

    gap = scores["mean_o"] - scores["mean_s"]
    changed = scores["layer_o"].astype(str) != scores["layer_s"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(scores["mean_o"], scores["mean_s"], s=16, alpha=0.6, color="#0072B2")
    lims = [min(scores["mean_o"].min(), scores["mean_s"].min()) - 0.02,
            max(scores["mean_o"].max(), scores["mean_s"].max()) + 0.02]
    axes[0].plot(lims, lims, color="0.6", lw=1)
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("layer chosen on the scored samples")
    axes[0].set_ylabel("layer chosen on the held-out half")
    axes[0].set_title("Same samples scored, different layer choice")

    axes[1].hist(gap, bins=30, color="#009E73", edgecolor="black", linewidth=0.4)
    axes[1].axvline(0, color="crimson", ls="--", lw=1.2)
    axes[1].set_xlabel("optimism: oracle score  -  split-sample score")
    axes[1].set_ylabel("model x panel")
    axes[1].set_title(f"median optimism {gap.median():+.4f}; "
                      f"layer differs in {changed.mean() * 100:.0f}% of cases")

    fig.suptitle("Bias from choosing the layer on the samples it is scored on",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "selection_effect", kind="statistics")
    plt.close(fig)


def main():
    fio.set_style()
    samples, tests = extract()

    # Figures show the primary analysis: layer chosen on held-out samples.
    primary_samples = samples[samples["selection"] == "split"]
    primary_tests = tests[tests["selection"] == "split"]

    plot_boxes(primary_samples, primary_tests)
    plot_paired_diffs(primary_samples, primary_tests)
    plot_heatmaps(primary_tests)
    plot_pvalue_overview(primary_tests)
    plot_selection_effect(tests)
    return samples, tests


if __name__ == "__main__":
    main()
