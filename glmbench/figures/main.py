"""Fig. 1-7 of the article.

Each figure has a ``build_*`` that turns metric tables into the CSV behind it and a
``draw_*`` that renders that CSV.

Numbering.  The repository numbers its figures 1-7 in the order the pipeline draws
them; the mapping to the article's numbering is the table in ``results/README.md``.
Models without the tables a figure needs (MinHash and Jensen-Shannon have no
reference benchmark yet, so no Fig. 1/2 bars) are left out of that figure only, and
the figure says so.

Terminology.  Two unrelated things were both called "CROSS" in earlier versions:
*RC proximity* (Fig. 5: how close a forward fragment lies to the reverse-complement
fragments of its own genome; higher is better) and *inter-order leakage* (Fig. 7: how
often a fragment of one order retrieves a fragment of the other; lower is better).
Column names of the committed CSVs keep the old names; titles and axes use these.

Where the layer comes from.  Fig. 1 and 2 *are* the reference benchmark, so they
show the best layer of that benchmark itself.  Fig. 4, 5 and 6 report every model
at the layer the benchmark picked (selection.py) -- never at the layer that happens
to score best on the unseen order -- so one layer per model is used across all of
them, and nothing is tuned on the numbers being reported.  Fig. 7 shows every layer
at once, so no choice arises.
"""

import numpy as np
import pandas as pd

from .. import config, selection
from ..metrics import reference, tables
from .. import models as registry
from ..models import Model
from .style import color, display, load_table, save_figure, save_table

VIRUSES = [("Egovirales_preproc", "Egovirales"),
           ("Mriyavirales_preproc", "Mriyavirales")]
FIG4_GROUPS = VIRUSES + [("REVERSECOMPLEMENTEgovirales_preproc", "Egovirales (RC)"),
                         ("REVERSECOMPLEMENTMriyavirales_preproc", "Mriyavirales (RC)")]
PAIRWISE_DIR = "pairwise_Egovirales_vs_Mriyavirales"


def _reference_row(model: Model, filename: str, metrics_dir=None):
    """Per-sample values of one metric table at the reference-chosen layer."""
    matrix = tables.load_model_matrix(model, filename, metrics_dir)
    return selection.row_of(model, matrix, metrics_dir=metrics_dir)


# ── Fig. 1: reference benchmark, best layer of each model ───────────────────

def build_fig1(models, metrics_dir=None) -> pd.DataFrame:
    models = reference.available(models, metrics_dir=metrics_dir)
    frames = [reference.summary(models, metric, metrics_dir)
              for metric in reference.METRICS]
    frame = pd.concat(frames, ignore_index=True)
    save_table(frame, "fig1_reference_benchmark")
    return frame


def draw_fig1(models):
    import matplotlib.pyplot as plt

    frame = load_table("fig1_reference_benchmark")
    models = [model for model in models if model.key in set(frame["model"])]
    keys = [model.key for model in models]
    panels = [(metric, reference.METRIC_TITLES[metric]) for metric in reference.METRICS]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (metric, title) in zip(axes, panels):
        data = frame[frame["metric"] == metric].set_index("model").reindex(keys)
        axis.bar(range(len(keys)), data["mean"], yerr=data["std"], capsize=4,
                 color=[color(key) for key in keys], edgecolor="black", linewidth=0.5)
        labels = [display(key) if np.isnan(data.loc[key, "best_layer"]) or
                  not models[index].layered else
                  f"{display(key)}\n(L{int(data.loc[key, 'best_layer'])})"
                  for index, key in enumerate(keys)]
        axis.set_xticks(range(len(keys)))
        axis.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.margins(y=0.15)
    figure.suptitle("Reference benchmark: best layer of each model", fontsize=15, y=1.04)
    absent = [display(model.key) for model in registry.resolve() if model.key not in keys]
    if absent:
        figure.text(0.5, -0.04, "no reference benchmark computed for: " + ", ".join(absent),
                    ha="center", fontsize=8, color="0.4")
    figure.tight_layout()
    save_figure(figure, "fig1_reference_benchmark", kind="main")
    plt.close(figure)


# ── Fig. 2: layer-wise ARI and P@5 ──────────────────────────────────────────

def build_fig2(models, metrics_dir=None) -> pd.DataFrame:
    models = reference.available(models, metrics_dir=metrics_dir)
    frame = pd.concat([reference.layerwise(models, metric, metrics_dir)
                       for metric in ("ARI", "P5")], ignore_index=True)
    save_table(frame, "fig2_layerwise")
    return frame


def draw_fig2(models):
    import matplotlib.pyplot as plt

    frame = load_table("fig2_layerwise")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, (metric, title) in zip(axes, [("ARI", "Adjusted Rand Index"),
                                            ("P5", "Precision@5")]):
        for model in models:
            data = frame[(frame["metric"] == metric)
                         & (frame["model"] == model.key)].sort_values("layer")
            if data.empty:
                continue
            if not model.layered:
                axis.axhline(data["mean"].iloc[0], ls=":", lw=2,
                             color=model.color, label=model.display)
            else:
                axis.plot(data["layer"], data["mean"], "-o", ms=4,
                          color=model.color, label=model.display)
                axis.fill_between(data["layer"], data["mean"] - data["std"],
                                  data["mean"] + data["std"], color=model.color, alpha=0.12)
        axis.set_xlabel("Layer")
        axis.set_ylabel(title)
        axis.set_title(title)
    axes[0].legend(loc="lower center", ncol=4, fontsize=8)
    figure.suptitle("Clustering quality across transformer layers", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "fig2_layerwise", kind="main")
    plt.close(figure)


# ── Fig. 3: k-mer word-length sweep ─────────────────────────────────────────

def build_fig3(metrics_dir=None) -> pd.DataFrame | None:
    """Copy the committed sweep summary into figure_data, if it is present."""
    path = (metrics_dir or config.METRICS) / "kmer_sweep" / "kmer_sweep_results.xlsx"
    if not path.exists():
        print("  kmer_sweep_results.xlsx not found -> keeping the committed CSV")
        return None
    frame = pd.read_excel(path, sheet_name="summary")
    save_table(frame, "fig3_kmer_sweep")
    return frame


def draw_fig3():
    import matplotlib.pyplot as plt

    if not (config.FIGURE_DATA / "fig3_kmer_sweep.csv").exists():
        print("  fig3: no k-mer sweep summary, not drawn")
        return
    frame = load_table("fig3_kmer_sweep").sort_values("k")
    series = [("ARI_mean", "ARI_std", "Adjusted Rand Index", "#0072B2"),
              ("P@5_mean", "P@5_std", "Precision@5", "#D55E00"),
              ("Silhouette_mean", "Silhouette_std", "Silhouette", "#009E73")]
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    for mean_column, std_column, label, line_color in series:
        axis.errorbar(frame["k"], frame[mean_column], yerr=frame[std_column],
                      marker="o", capsize=3, color=line_color, label=label)
    axis.axvline(4, ls="--", color="gray", alpha=0.7)
    axis.set_xlabel("k (k-mer length)")
    axis.set_ylabel("Score")
    axis.set_title("k-mer spectrum: effect of word length k")
    axis.set_xticks(frame["k"])
    axis.legend()
    figure.tight_layout()
    save_figure(figure, "fig3_kmer_sweep", kind="main")
    plt.close(figure)


# ── Fig. 4 and 5: the unseen orders ─────────────────────────────────────────

def _unseen_table(models, groups, prefix, name, metrics_dir=None, filename=None,
                  optional=False) -> pd.DataFrame:
    """Mean and spread over the reference samples of one table, at the reference layer.

    ``optional`` tables (the order-level and own-RC metrics, which only exist after a
    re-run) skip models that lack the file and are not written at all if none has it.
    """
    rows = []
    for virus, label in groups:
        for model in models:
            table = filename(virus) if filename else tables.matrix_name(
                prefix, config.PK_FOR_FIGURES, virus)
            if optional and not (tables.model_dir(model, metrics_dir) / table).exists():
                continue
            layer, values = _reference_row(model, table, metrics_dir)
            rows.append({"group": label, "model": model.key, "layer": layer,
                         "mean": float(values.mean()), "std": float(values.std(ddof=0))})
    frame = pd.DataFrame(rows)
    if optional and frame.empty:
        return frame
    save_table(frame, name)
    return frame


def build_fig4(models, metrics_dir=None) -> pd.DataFrame:
    return _unseen_table(models, FIG4_GROUPS, "P", "fig4_newvirus_p5", metrics_dir)


def build_fig5(models, metrics_dir=None) -> pd.DataFrame:
    return _unseen_table(models, VIRUSES, "cross_P", "fig5_cross_p5", metrics_dir)


def build_fig4b(models, metrics_dir=None) -> pd.DataFrame:
    """Order-level P@5 (leave-genome-out): success = another genome of the same order."""
    return _unseen_table(models, FIG4_GROUPS, "Porder", "fig4b_order_p5", metrics_dir,
                         optional=True)


def build_fig5b(models, metrics_dir=None) -> pd.DataFrame:
    """Own-RC hit@5: the fragment's own reverse complement is among its 5 neighbours."""
    return _unseen_table(models, VIRUSES, None, "fig5b_own_rc_hit5", metrics_dir,
                         filename=lambda virus: tables.own_rc_names(config.PK_FOR_FIGURES, virus),
                         optional=True)


def build_fig5c(models, metrics_dir=None) -> pd.DataFrame:
    """Mean similarity between a fragment and its own reverse complement."""
    return _unseen_table(models, VIRUSES, None, "fig5c_own_rc_similarity", metrics_dir,
                         filename=tables.own_rc_similarity_name, optional=True)


def _grouped_bars(frame, models, order, ylabel, title, name):
    import matplotlib.pyplot as plt

    models = [model for model in models if model.key in set(frame["model"])]

    figure, axis = plt.subplots(figsize=(10 if len(order) > 2 else 8, 4.8))
    width, positions = 0.8 / len(models), np.arange(len(order))
    for index, model in enumerate(models):
        data = frame[frame["model"] == model.key].set_index("group").reindex(order)
        axis.bar(positions + (index - (len(models) - 1) / 2) * width, data["mean"], width,
                 yerr=data["std"], capsize=2, color=model.color, label=model.display,
                 edgecolor="black", linewidth=0.3)
    axis.set_xticks(positions)
    axis.set_xticklabels(order)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(ncol=min(len(models), 4), fontsize=9, loc="upper center",
                bbox_to_anchor=(0.5, -0.12))
    figure.tight_layout()
    save_figure(figure, name, kind="main")
    plt.close(figure)


def draw_fig4(models):
    frame = load_table("fig4_newvirus_p5")
    _grouped_bars(frame, models, [label for _, label in FIG4_GROUPS],
                  "Mean Precision@5", "Generalisation to unseen viruses (Precision@5)",
                  "fig4_newvirus_p5")


def draw_fig5(models):
    frame = load_table("fig5_cross_p5")
    _grouped_bars(frame, models, [label for _, label in VIRUSES],
                  "Mean RC-proximity Precision@5 (legacy 'CROSS P@5')",
                  "RC proximity: share of the 5 nearest neighbours that are reverse-complement\n"
                  "fragments of the SAME genome (not necessarily the fragment's own RC)",
                  "fig5_cross_p5")


def _draw_optional(name, groups, models, ylabel, title):
    path = config.FIGURE_DATA / f"{name}.csv"
    if path.exists():
        _grouped_bars(load_table(name), models, [label for _, label in groups],
                      ylabel, title, name)


def draw_fig4b(models):
    _draw_optional("fig4b_order_p5", FIG4_GROUPS, models, "Mean order-level Precision@5",
                   "Order-level P@5, leave-genome-out: neighbours from ANOTHER genome of\n"
                   "the same unseen order (own genome excluded)")


def draw_fig5b(models):
    _draw_optional("fig5b_own_rc_hit5", VIRUSES, models, "Mean own-RC hit@5",
                   "Own reverse complement among the 5 nearest neighbours")


def draw_fig5c(models):
    _draw_optional("fig5c_own_rc_similarity", VIRUSES, models,
                   "Mean similarity of x and RC(x)",
                   "Similarity of a fragment and its reverse complement\n"
                   "(cosine / sketch agreement / 1 - JSD, by model)")


# ── Fig. 6: P@k against k ───────────────────────────────────────────────────

def build_fig6(models, metrics_dir=None) -> pd.DataFrame:
    rows = []
    for virus, label in VIRUSES:
        for model in models:
            for k in config.KS:
                layer, values = _reference_row(
                    model, tables.matrix_name("P", k, virus), metrics_dir)
                rows.append({"virus": label, "model": model.key, "k": k,
                             "layer": layer, "mean": float(values.mean())})
    frame = pd.DataFrame(rows)
    save_table(frame, "fig6_pk_vs_k")
    return frame


def draw_fig6(models):
    import matplotlib.pyplot as plt

    frame = load_table("fig6_pk_vs_k")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, (_, label) in zip(axes, VIRUSES):
        for model in models:
            data = frame[(frame["virus"] == label)
                         & (frame["model"] == model.key)].sort_values("k")
            if data.empty:
                continue
            axis.plot(data["k"], data["mean"], "-o", ms=4,
                      color=model.color, label=model.display)
        axis.set_xlabel("k (number of neighbours)")
        axis.set_title(label)
        axis.set_xticks(list(config.KS))
    axes[0].set_ylabel("Precision@k")
    axes[0].legend(fontsize=8, ncol=2)
    figure.suptitle("Precision@k as a function of k (unseen viruses)", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "fig6_pk_vs_k", kind="main")
    plt.close(figure)


# ── Fig. 7: compactness against separation, per layer ───────────────────────

def build_fig7(models, metrics_dir=None) -> pd.DataFrame:
    rows = []
    for model in models:
        directory = tables.model_dir(model, metrics_dir) / PAIRWISE_DIR
        if not directory.is_dir():
            continue

        def layer_mean(filename):
            return pd.read_excel(directory / filename).set_index("layer").mean(axis=1)

        own = (layer_mean("pk_fwd_1_in_2_matrix.xlsx")
               + layer_mean("pk_fwd_2_in_1_matrix.xlsx")) / 2
        cross = layer_mean("cross_1→2_fwd_fwd_matrix.xlsx")
        for layer in own.index:
            rows.append({"model": model.key, "layer": layer,
                         "p_at_10": float(own[layer]),
                         "cross_p_at_10": float(cross[layer])})
    frame = pd.DataFrame(rows)
    save_table(frame, "fig7_pk_vs_cross")
    return frame


def draw_fig7(models):
    import matplotlib.pyplot as plt

    frame = load_table("fig7_pk_vs_cross")
    figure, axis = plt.subplots(figsize=(9, 6))
    flat_markers = iter(["*", "P", "X", "D", "s", "^"])
    for model in models:
        data = frame[frame["model"] == model.key]
        if data.empty:
            continue
        flat = not model.layered
        axis.scatter(data["cross_p_at_10"], data["p_at_10"],
                     marker=next(flat_markers) if flat else "o", s=200 if flat else 55,
                     color=model.color, label=model.display,
                     edgecolor="white", linewidth=0.6, zorder=3)
        for _, row in data.iterrows():
            layer = "".join(char for char in str(row["layer"]) if char.isdigit())
            if layer:
                axis.annotate(layer, (row["cross_p_at_10"], row["p_at_10"]),
                              fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Inter-order leakage: P@10 of Egovirales fragments retrieving "
                    "Mriyavirales (lower = better separation)")
    axis.set_ylabel("Precision@10 within the own order (higher = denser cluster)")
    axis.set_title("Egovirales vs Mriyavirales: cluster compactness vs separation, per layer")
    axis.legend(title="Model")
    figure.tight_layout()
    save_figure(figure, "fig7_pk_vs_cross", kind="main")
    plt.close(figure)


# ── orchestration ───────────────────────────────────────────────────────────

BUILDERS = {
    "fig1": build_fig1, "fig2": build_fig2, "fig4": build_fig4,
    "fig5": build_fig5, "fig6": build_fig6, "fig7": build_fig7,
    "fig4b": build_fig4b, "fig5b": build_fig5b, "fig5c": build_fig5c,
}


def build_all(models, metrics_dir=None, which=None) -> None:
    wanted = set(which or BUILDERS)
    for name, builder in BUILDERS.items():
        if name in wanted:
            builder(models, metrics_dir)
    if "fig3" in wanted or which is None:
        build_fig3(metrics_dir)


def draw_all(models, which=None) -> None:
    wanted = set(which or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7",
                           "fig4b", "fig5b", "fig5c"])
    if "fig1" in wanted:
        draw_fig1(models)
    if "fig2" in wanted:
        draw_fig2(models)
    if "fig3" in wanted:
        draw_fig3()
    if "fig4" in wanted:
        draw_fig4(models)
    if "fig5" in wanted:
        draw_fig5(models)
    if "fig6" in wanted:
        draw_fig6(models)
    if "fig7" in wanted:
        draw_fig7(models)
    if "fig4b" in wanted:
        draw_fig4b(models)
    if "fig5b" in wanted:
        draw_fig5b(models)
    if "fig5c" in wanted:
        draw_fig5c(models)
