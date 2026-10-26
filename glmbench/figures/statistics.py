"""The significance plots, and the tables behind them.

Every model is scored on the same 50 reference samples, so the comparisons are
paired.  What still has to be decided is *which layer* each model is scored at,
and that decision is itself made from data -- so the tables carry it explicitly as
a ``selection`` column:

    reference        layer from the model's reference benchmark, reported on all 50
                     samples.  Primary for the unseen-order panels: the layer is
                     chosen on a different task, never on the numbers reported.
    reference_split  the same, but chosen on even samples and reported on odd ones,
                     so choice and report share no sample at all.
    split            chosen on even samples of this very table, reported on odd.
                     Primary for Fig. 1, which *is* the reference benchmark and so
                     cannot choose from it without circularity.
    in_sample        chosen and reported on all 50 -- reproduces the figures.
    oracle           chosen and reported on the odd samples.  Diagnostic: it scores
                     the same samples as ``split``, so the gap between the two is
                     the selection optimism, uncontaminated by which half is scored.

What these tests do and do not test.  The 50 reference samples are 50 draws of KNOWN
viruses against which the same unseen fragments are scored.  A paired test over them
asks whether model A beats model B *when the reference draw changes*; it says nothing
about whether the result would hold for other viruses than the two unseen orders
studied, nor for other genomes of those orders.  The draws overlap heavily (they come
from one pool of reference genomes), so they are far from 50 independent experiments,
and the exact Wilcoxon p-value hits its floor 2/2^50 whenever one model wins on every
draw -- a statement about the draws, not an estimate of a probability.

Figures: box plots of the per-sample values with significance brackets, box plots
of the paired differences (what the test actually ranks), pairwise heat maps, an
overview of every p-value, and the selection-optimism diagnostic.
"""

import numpy as np
import pandas as pd

from .. import config, selection, statistics
from ..metrics import reference as reference_metrics
from ..metrics import tables
from .style import load_table, save_figure, save_table

SELECTIONS = ("reference", "reference_split", "split", "in_sample", "oracle")
PRIMARY = {"fig1": "split"}          # everything else uses "reference"
PLOTTED = {
    "fig1": ("Reference benchmark", 3),
    "fig4": ("Held-out viruses: Precision@5", 2),
    "fig5": ("RC proximity, forward vs reverse complement", 2),
}
VIRUSES = [("Egovirales_preproc", "Egovirales"), ("Mriyavirales_preproc", "Mriyavirales")]

# Short panel names; the long titles live on the figure axes, not in the tables.
PANEL_TITLES = {"ARI": "ARI", "P5": "Precision@5", "silh": "Silhouette"}

# Column order of the test table: what is being compared, then the result.
TEST_COLUMNS = ["selection", "figure", "panel", "source", "model_a", "layer_a",
                "model_b", "layer_b", "n", "mean_a", "mean_b", "mean_diff",
                "rank_biserial_r", "p_wilcoxon", "p_ttest", "p_holm_panel",
                "comparison", "p_holm_all", "p_bh_all", "signif"]


# ── choosing the layer under each rule ──────────────────────────────────────

def best_row(matrix: pd.DataFrame, rule: str, model=None, metrics_dir=None):
    """(layer label, per-sample values) of one metric table under one rule."""
    even = [column for column in matrix.columns if column % 2 == 0]
    odd = [column for column in matrix.columns if column % 2 == 1]

    if rule.startswith("reference"):
        samples = even if rule == "reference_split" else None
        layer = selection.layer_of(model, matrix, metrics_dir=metrics_dir, samples=samples)
        report = odd if rule == "reference_split" else list(matrix.columns)
        return str(layer), matrix.loc[layer, report]

    choose, report = {"in_sample": (list(matrix.columns), list(matrix.columns)),
                      "split": (even, odd),
                      "oracle": (odd, odd)}[rule]
    layer = matrix[choose].mean(axis=1, skipna=True).idxmax()
    return str(layer), matrix.loc[layer, report]


def _reference_panel(model, metric, rule, metrics_dir=None):
    """The benchmark tables are indexed by position; the panel shows the layer label
    (ViraLM: position + 1), as every other figure does."""
    matrix = reference_metrics.load(model, metric, metrics_dir)
    position, values = best_row(matrix, rule, model, metrics_dir)
    if not model.layered:
        return "-", values
    return str(reference_metrics.layer_label(model, int(position))), values


def _metric_panel(model, filename, rule, metrics_dir=None):
    matrix = tables.load_model_matrix(model, filename, metrics_dir)
    return best_row(matrix, rule, model, metrics_dir)


def panels():
    """(figure, panel label, source table, loader) of every tested panel."""
    for metric, title in PANEL_TITLES.items():
        yield ("fig1", title, metric,
               lambda model, rule, metrics_dir, metric=metric:
               _reference_panel(model, metric, rule, metrics_dir))

    groups = [("Egovirales_preproc", "P@5 Egovirales"),
              ("Mriyavirales_preproc", "P@5 Mriyavirales"),
              ("REVERSECOMPLEMENTEgovirales_preproc", "P@5 Egovirales (RC)"),
              ("REVERSECOMPLEMENTMriyavirales_preproc", "P@5 Mriyavirales (RC)")]
    for virus, label in groups:
        filename = tables.matrix_name("P", config.PK_FOR_FIGURES, virus)
        yield ("fig4", label, filename,
               lambda model, rule, metrics_dir, filename=filename:
               _metric_panel(model, filename, rule, metrics_dir))

    for virus, label in VIRUSES:
        filename = tables.matrix_name("cross_P", config.PK_FOR_FIGURES, virus)
        yield ("fig5", f"RC-proximity P@5 {label}", filename,
               lambda model, rule, metrics_dir, filename=filename:
               _metric_panel(model, filename, rule, metrics_dir))

    for virus, label in VIRUSES:
        for k in config.KS:
            filename = tables.matrix_name("P", k, virus)
            yield ("fig6", f"P@{k} {label}", filename,
                   lambda model, rule, metrics_dir, filename=filename:
                   _metric_panel(model, filename, rule, metrics_dir))


# ── stage 1: per-sample values and the tests built from them ────────────────

def _extract_rule(models, rule, metrics_dir=None):
    samples, tests = [], []
    for figure, panel, source, loader in panels():
        # Fig. 1 is the reference benchmark; choosing its layer from that same
        # benchmark would be circular, so the reference rules skip it.
        if rule.startswith("reference") and figure == "fig1":
            continue

        values, layers = {}, {}
        for model in models:
            try:
                layer, series = loader(model, rule, metrics_dir)
            except FileNotFoundError:
                continue
            values[model.key] = series
            layers[model.key] = layer
            for sample, value in series.items():
                samples.append({"selection": rule, "figure": figure, "panel": panel,
                                "model": model.key, "layer": layer,
                                "sample": int(sample), "value": float(value)})
        if len(values) < 2:
            continue

        frame = statistics.panel_tests(
            values, [model.key for model in models],
            {"selection": rule, "figure": figure, "panel": panel, "source": source})
        frame["layer_a"] = frame["model_a"].map(layers)
        frame["layer_b"] = frame["model_b"].map(layers)
        tests.append(frame)

    if not tests:
        return pd.DataFrame(samples), pd.DataFrame()
    corrected = statistics.correct(pd.concat(tests, ignore_index=True))
    print(f"  {rule:<16} n={corrected['n'].max()} per comparison, "
          f"{corrected['comparison'].nunique()} distinct comparisons, "
          f"{(corrected.drop_duplicates('comparison')['p_bh_all'] >= 0.05).sum()} "
          f"not significant after FDR")
    return pd.DataFrame(samples), corrected


def build(models, metrics_dir=None):
    """Per-sample values and paired tests under every selection rule."""
    frames = [_extract_rule(models, rule, metrics_dir) for rule in SELECTIONS]
    samples = pd.concat([frame[0] for frame in frames], ignore_index=True)
    tests = pd.concat([frame[1] for frame in frames], ignore_index=True)

    tests = tests.reindex(columns=[column for column in TEST_COLUMNS
                                   if column in tests.columns])
    save_table(samples, "panel_samples", float_format="%.6g", directory=config.STATISTICS)
    save_table(tests, "pairwise_model_tests", float_format="%.6g",
               directory=config.STATISTICS)
    return samples, tests


def primary(frame: pd.DataFrame) -> pd.DataFrame:
    """The rule each figure is drawn under."""
    rule = frame["figure"].map(lambda figure: PRIMARY.get(figure, "reference"))
    return frame[frame["selection"] == rule]


# ── stage 2: the figures ────────────────────────────────────────────────────

def _grid(n_panels: int, ncols: int, panel_size=(4.2, 4.0)):
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(n_panels / ncols))
    figure, axes = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axes.ravel()
    for axis in flat[n_panels:]:
        axis.set_visible(False)
    return figure, flat


def _brackets(panel_tests: pd.DataFrame, order: list) -> list:
    """Which comparisons to draw, and on which level.

    Sorted by mean, a bracket between neighbours implies every wider pair, so
    neighbours carry the information -- except for pairs that come out *not*
    significant, which are the interesting ones and are always drawn.
    """
    rank = {model: index for index, model in enumerate(order)}
    chosen = []
    for _, row in panel_tests.iterrows():
        if row["model_a"] not in rank or row["model_b"] not in rank:
            continue
        left, right = sorted((rank[row["model_a"]], rank[row["model_b"]]))
        label = statistics.stars(row["p_bh_all"])
        if right - left == 1 or label == "ns":
            chosen.append({"i": left, "j": right, "label": label, "span": right - left})

    chosen.sort(key=lambda bracket: (bracket["span"], bracket["i"]))
    levels = []
    for bracket in chosen:
        for level, spans in enumerate(levels):
            if all(bracket["j"] <= low or bracket["i"] >= high for low, high in spans):
                spans.append((bracket["i"], bracket["j"]))
                bracket["level"] = level
                break
        else:
            levels.append([(bracket["i"], bracket["j"])])
            bracket["level"] = len(levels) - 1
    return chosen


def _draw_brackets(axis, brackets: list, top: float, step: float) -> None:
    for bracket in brackets:
        y = top + step * (bracket["level"] + 1)
        # ends are inset so brackets over neighbouring boxes stay visibly separate
        left, right = bracket["i"] + 1.16, bracket["j"] + 0.84
        axis.plot([left, left, right, right], [y - step * 0.3, y, y, y - step * 0.3],
                  lw=1.0, color="0.35", clip_on=False, zorder=5)
        significant = bracket["label"] != "ns"
        axis.text((left + right) / 2, y + step * 0.04, bracket["label"],
                  ha="center", va="bottom", zorder=5,
                  fontsize=11 if significant else 8.5,
                  color="black" if significant else "0.45")
    if brackets:
        axis.set_ylim(top=top + step * (max(b["level"] for b in brackets) + 1.9))


def draw_boxes(samples: pd.DataFrame, tests: pd.DataFrame, models) -> None:
    """Distribution over the reference samples, one box per model."""
    import matplotlib.pyplot as plt

    colors = {model.key: model.color for model in models}
    names = {model.key: model.display for model in models}
    rng = np.random.default_rng(0)

    for figure_key, (title, ncols) in PLOTTED.items():
        panel_names = samples[samples["figure"] == figure_key]["panel"].unique()
        if not len(panel_names):
            continue
        figure, axes = _grid(len(panel_names), ncols)

        for axis, panel in zip(axes, panel_names):
            subset = samples[(samples["figure"] == figure_key)
                             & (samples["panel"] == panel)]
            order = (subset.groupby("model")["value"].mean()
                     .sort_values(ascending=False).index.tolist())
            values = [subset[subset["model"] == key]["value"].to_numpy() for key in order]

            boxes = axis.boxplot(values, widths=0.6, patch_artist=True, showfliers=False,
                                 medianprops=dict(color="black", lw=1.6))
            for patch, key in zip(boxes["boxes"], order):
                patch.set_facecolor(colors[key])
                patch.set_alpha(0.55)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)
            for position, (column, key) in enumerate(zip(values, order), start=1):
                jitter = rng.uniform(-0.13, 0.13, len(column))
                axis.plot(position + jitter, column, "o", ms=2.4, mew=0,
                          color=colors[key], alpha=0.65, zorder=3)

            layers = {key: subset[subset["model"] == key]["layer"].iloc[0] for key in order}
            axis.set_xticks(range(1, len(order) + 1))
            axis.set_xticklabels([names[key] if str(layers[key]) == "-"
                                  else f"{names[key]}\n(L{layers[key]})" for key in order],
                                 fontsize=9, rotation=45, ha="right")
            axis.set_title(panel, fontsize=12)
            axis.set_ylabel("Score per reference sample")

            flat = np.concatenate(values)
            spread = flat.max() - flat.min()
            panel_tests = tests[(tests["figure"] == figure_key) & (tests["panel"] == panel)]
            _draw_brackets(axis, _brackets(panel_tests, order),
                           top=flat.max() + spread * 0.04, step=spread * 0.11)

        figure.suptitle(f"{title}\ndistribution over the paired reference samples "
                        f"(brackets: BH-adjusted significance)", fontsize=14, y=1.02)
        figure.tight_layout()
        save_figure(figure, f"box_{figure_key}", kind="statistics")
        plt.close(figure)


def draw_paired(samples: pd.DataFrame, tests: pd.DataFrame, models) -> None:
    """The 50 paired differences against the best model -- what the test ranks."""
    import matplotlib.pyplot as plt

    colors = {model.key: model.color for model in models}
    names = {model.key: model.display for model in models}

    for figure_key, (title, ncols) in PLOTTED.items():
        panel_names = samples[samples["figure"] == figure_key]["panel"].unique()
        if not len(panel_names):
            continue
        figure, axes = _grid(len(panel_names), ncols)

        for axis, panel in zip(axes, panel_names):
            subset = samples[(samples["figure"] == figure_key)
                             & (samples["panel"] == panel)]
            wide = subset.pivot(index="sample", columns="model", values="value")
            best = wide.mean().idxmax()
            others = [key for key in wide.columns if key != best]

            boxes = axis.boxplot([(wide[key] - wide[best]).to_numpy() for key in others],
                                 widths=0.6, patch_artist=True, showfliers=True,
                                 flierprops=dict(ms=3, mfc="0.4", mec="none"),
                                 medianprops=dict(color="black", lw=1.6))
            for patch, key in zip(boxes["boxes"], others):
                patch.set_facecolor(colors[key])
                patch.set_alpha(0.55)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)

            axis.axhline(0, color="crimson", lw=1.2, ls="--")
            panel_tests = tests[(tests["figure"] == figure_key) & (tests["panel"] == panel)]
            for position, key in enumerate(others, start=1):
                row = panel_tests[((panel_tests["model_a"] == key)
                                   & (panel_tests["model_b"] == best))
                                  | ((panel_tests["model_a"] == best)
                                     & (panel_tests["model_b"] == key))]
                if row.empty:
                    continue
                axis.annotate(statistics.stars(row.iloc[0]["p_bh_all"]), (position, 0.96),
                              xycoords=("data", "axes fraction"), ha="center", va="top",
                              fontsize=9, color="crimson")

            axis.set_xticks(range(1, len(others) + 1))
            axis.set_xticklabels([names[key] for key in others], fontsize=9,
                                 rotation=45, ha="right")
            axis.set_ylabel(f"difference vs {names[best]}")
            axis.set_title(f"{panel}\nbest: {names[best]}", fontsize=11)
            axis.margins(y=0.16)

        figure.suptitle(f"{title}\npaired differences over the reference samples "
                        f"(the quantity the Wilcoxon test ranks)", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        save_figure(figure, f"paired_{figure_key}", kind="statistics")
        plt.close(figure)


def draw_heatmaps(tests: pd.DataFrame, models) -> None:
    """Pairwise mean difference (colour) and BH-adjusted significance (annotation)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    cmap = plt.get_cmap("RdBu_r")
    keys = [model.key for model in models]
    names = [model.display for model in models]

    for figure_key, (title, ncols) in PLOTTED.items():
        panel_names = tests[tests["figure"] == figure_key]["panel"].unique()
        if not len(panel_names):
            continue
        figure, axes = _grid(len(panel_names), ncols, panel_size=(4.4, 4.0))

        for position, (axis, panel) in enumerate(zip(axes, panel_names)):
            subset = tests[(tests["figure"] == figure_key) & (tests["panel"] == panel)]
            difference = pd.DataFrame(np.nan, index=keys, columns=keys)
            annotation = pd.DataFrame("", index=keys, columns=keys)
            for _, row in subset.iterrows():
                a, b = row["model_a"], row["model_b"]
                difference.loc[a, b], difference.loc[b, a] = row["mean_diff"], -row["mean_diff"]
                mark = statistics.stars(row["p_bh_all"])
                annotation.loc[a, b] = f"{row['mean_diff']:+.3f}\n{mark}"
                annotation.loc[b, a] = f"{-row['mean_diff']:+.3f}\n{mark}"

            limit = np.nanmax(np.abs(difference.to_numpy(dtype=float)))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            axis.imshow(difference.to_numpy(dtype=float), cmap=cmap, norm=norm)
            for i in range(len(keys)):
                for j in range(len(keys)):
                    if i == j:
                        axis.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                                     color="0.92", zorder=2))
                        continue
                    axis.text(j, i, annotation.iloc[i, j], ha="center", va="center",
                              fontsize=7.5, zorder=3,
                              color="white" if abs(difference.iloc[i, j]) > 0.72 * limit
                              else "black")
            axis.set_xticks(range(len(keys)))
            axis.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
            axis.set_yticks(range(len(keys)))
            axis.set_yticklabels(names if position % ncols == 0 else [], fontsize=8)
            axis.set_title(panel, fontsize=11)
            axis.grid(False)
            bar = figure.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                                  ax=axis, fraction=0.046, pad=0.03)
            bar.ax.tick_params(labelsize=7)
            bar.set_label("mean difference (row - column)", fontsize=7)

        figure.suptitle(f"{title}\nrow vs column: mean difference over the samples "
                        f"and BH-adjusted significance", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(figure, f"heatmap_{figure_key}", kind="statistics")
        plt.close(figure)


def draw_pvalues(tests: pd.DataFrame) -> None:
    """Every comparison: raw against BH-adjusted p, with the exact-Wilcoxon floor."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    floor = statistics.wilcoxon_floor(int(tests["n"].max()))

    for figure_key, marker in zip(["fig1", "fig4", "fig5", "fig6"], ["o", "s", "^", "d"]):
        subset = tests[tests["figure"] == figure_key]
        if subset.empty:
            continue
        axis.plot(np.maximum(subset["p_wilcoxon"], floor / 2),
                  np.maximum(subset["p_bh_all"], floor / 2),
                  marker, ms=5, alpha=0.7, label=f"{figure_key}  (n={len(subset)})")

    limits = [floor / 3, 2]
    axis.plot(limits, limits, "-", color="0.6", lw=1, zorder=0)
    axis.axhline(0.05, ls="--", color="crimson", lw=1)
    axis.axvline(floor, ls=":", color="0.4", lw=1)
    axis.text(floor * 1.4, limits[0] * 6,
              f"exact Wilcoxon floor\n$2/2^{{{int(tests['n'].max())}}}$ = {floor:.1g}",
              fontsize=8, color="0.3")
    axis.text(limits[0] * 1.5, 0.065, "0.05", fontsize=8, color="crimson")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("raw Wilcoxon p")
    axis.set_ylabel("Benjamini-Hochberg adjusted p")
    unique = tests.drop_duplicates("comparison")
    axis.set_title(f"All {len(unique)} distinct model comparisons\n"
                   f"{(unique['p_bh_all'] >= 0.05).sum()} do not survive FDR correction",
                   fontsize=13)
    axis.legend(fontsize=9, loc="lower right")
    figure.tight_layout()
    save_figure(figure, "pvalue_overview", kind="statistics")
    plt.close(figure)


def draw_selection_effect(tests: pd.DataFrame) -> None:
    """Selection optimism: 'oracle' and 'split' score the same samples.

    They differ only in where the layer was chosen, so their gap isolates the bias
    of choosing a layer on the data it is then reported on.
    """
    import matplotlib.pyplot as plt

    keys = ["figure", "panel", "model_a", "model_b"]
    merged = (tests[tests["selection"] == "oracle"]
              .merge(tests[tests["selection"] == "split"], on=keys,
                     suffixes=("_oracle", "_split")))
    if merged.empty:
        return

    scores = pd.concat([
        merged[["layer_a_oracle", "layer_a_split", "mean_a_oracle", "mean_a_split"]]
        .set_axis(["layer_o", "layer_s", "mean_o", "mean_s"], axis=1),
        merged[["layer_b_oracle", "layer_b_split", "mean_b_oracle", "mean_b_split"]]
        .set_axis(["layer_o", "layer_s", "mean_o", "mean_s"], axis=1),
    ], ignore_index=True).drop_duplicates()

    gap = scores["mean_o"] - scores["mean_s"]
    changed = scores["layer_o"].astype(str) != scores["layer_s"].astype(str)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(scores["mean_o"], scores["mean_s"], s=16, alpha=0.6, color="#0072B2")
    limits = [min(scores["mean_o"].min(), scores["mean_s"].min()) - 0.02,
              max(scores["mean_o"].max(), scores["mean_s"].max()) + 0.02]
    axes[0].plot(limits, limits, color="0.6", lw=1)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("layer chosen on the scored samples")
    axes[0].set_ylabel("layer chosen on the held-out half")
    axes[0].set_title("Same samples scored, different layer choice")

    axes[1].hist(gap, bins=30, color="#009E73", edgecolor="black", linewidth=0.4)
    axes[1].axvline(0, color="crimson", ls="--", lw=1.2)
    axes[1].set_xlabel("optimism: oracle score  -  split-sample score")
    axes[1].set_ylabel("model x panel")
    axes[1].set_title(f"median optimism {gap.median():+.4f}; "
                      f"layer differs in {changed.mean() * 100:.0f}% of cases")

    figure.suptitle("Bias from choosing the layer on the samples it is scored on",
                    fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, "selection_effect", kind="statistics")
    plt.close(figure)


def draw_all(models, samples=None, tests=None) -> None:
    if samples is None:
        samples = load_table("panel_samples", directory=config.STATISTICS)
    if tests is None:
        tests = load_table("pairwise_model_tests", directory=config.STATISTICS)
    draw_boxes(primary(samples), primary(tests), models)
    draw_paired(primary(samples), primary(tests), models)
    draw_heatmaps(primary(tests), models)
    draw_pvalues(primary(tests))
    draw_selection_effect(tests)
