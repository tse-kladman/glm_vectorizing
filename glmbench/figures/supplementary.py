"""Supplementary figures and the sensitivity checks of the main results.

Every layer of every metric, one figure per model.  Fig. 1, 4 and 5 summarise a model
by a single layer; these show the full per-layer picture behind that summary, which
is where a reader checks whether the chosen layer sits on a plateau or on a spike.
Files are grouped by metric rather than thrown into one flat folder:

    figures/supplementary/ARI/<model>.png
    figures/supplementary/P5/<model>.png
    figures/supplementary/silhouette/<model>.png
    figures/supplementary/newvirus_p5/<model>_<virus>.png

Two sensitivity checks live here too: feature scaling and distance
(``build_scaling_distance``, drawn as ``figures/main/fig8_scaling_distance``) and an
early layer of Evo2 against the chosen one (``build_early_layer``, table only).

Layers are shown by label (``reference.layer_label``); the dashed line marks the layer
of ``selection.json``, which stores a benchmark position and is translated here.
"""

import numpy as np
import pandas as pd

from .. import config, selection
from .. import models as registry
from ..metrics import reference, tables
from .main import FIG4_GROUPS, VIRUSES as FORWARD_GROUPS
from .style import load_table, save_figure, save_table

FOLDERS = {"ARI": "ARI", "P5": "P5", "silh": "silhouette"}
VIRUSES = ["Mriyavirales_preproc", "Egovirales_preproc",
           "REVERSECOMPLEMENTMriyavirales_preproc", "REVERSECOMPLEMENTEgovirales_preproc"]

UNSEEN_GROUPS = FIG4_GROUPS                 # forward and RC of both unseen orders


def _chosen_label(model):
    """Label of the layer recorded in ``selection.json`` (a position), or None."""
    if not model.layered:
        return None
    position = selection.load().get(model.key, {}).get("layer")
    return None if position is None else selection.layer_label(model, position)


def virus_label(name: str) -> str:
    if name.startswith("REVERSECOMPLEMENT"):
        return name.split("REVERSECOMPLEMENT")[-1].split("_")[0] + " (RC)"
    return name.split("_")[0]


def build_reference(models, metrics_dir=None) -> pd.DataFrame:
    models = reference.available(models, metrics_dir=metrics_dir)
    frame = pd.concat([reference.layerwise(models, metric, metrics_dir)
                       for metric in reference.METRICS], ignore_index=True)
    save_table(frame, "supplementary/layerwise_reference")
    return frame


def build_unseen(models, metrics_dir=None) -> pd.DataFrame:
    rows = []
    for virus in VIRUSES:
        for model in models:
            if not model.layered:
                continue
            path = tables.model_dir(model, metrics_dir) / tables.matrix_name(
                "P", config.PK_FOR_FIGURES, virus)
            if not path.exists():
                continue
            matrix = tables.load_matrix(path)
            mean, std = matrix.mean(axis=1), matrix.std(axis=1)
            for layer in matrix.index:
                rows.append({"virus": virus_label(virus), "model": model.key,
                             "layer": int(layer), "mean": float(mean[layer]),
                             "std": float(std[layer])})
    frame = pd.DataFrame(rows, columns=["virus", "model", "layer", "mean", "std"])
    save_table(frame, "supplementary/layerwise_newvirus_p5")
    return frame


def _bars(data, color, title, ylabel, name, highlight=None):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.bar(data["layer"], data["mean"], yerr=data["std"], capsize=4, color=color,
             error_kw={"elinewidth": 0.6, "alpha": 0.6})
    if highlight is not None:
        axis.axvline(highlight, color="crimson", ls="--", lw=1.2,
                     label="layer chosen on the reference set")
        axis.legend(fontsize=8)
    axis.set_title(title)
    axis.set_xlabel("Layer")
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    save_figure(figure, name, kind="supplementary")
    plt.close(figure)


def draw_reference(models, metrics_dir=None) -> None:
    frame = load_table("supplementary/layerwise_reference")
    for model in models:
        chosen = _chosen_label(model)
        for metric in reference.METRICS:
            data = frame[(frame["model"] == model.key)
                         & (frame["metric"] == metric)].sort_values("layer")
            if data.empty:
                continue
            _bars(data, model.color,
                  f"{reference.METRIC_TITLES[metric]} across layers - {model.display}",
                  "Score", f"{FOLDERS[metric]}/{model.key}",
                  highlight=chosen if metric == config.REFERENCE_METRIC else None)


def draw_unseen(models) -> None:
    frame = load_table("supplementary/layerwise_newvirus_p5")
    if frame.empty:                          # no layered model among those scored
        return
    for virus in frame["virus"].unique():
        for model in models:
            data = frame[(frame["virus"] == virus)
                         & (frame["model"] == model.key)].sort_values("layer")
            if data.empty:
                continue
            slug = virus.replace(" (RC)", "_rc").replace(" ", "_")
            _bars(data, model.color,
                  f"Precision@5 across layers - {model.display} ({virus})",
                  "Precision@5", f"newvirus_p5/{model.key}_{slug}",
                  highlight=_chosen_label(model))


# ── feature scaling and distance, per method ────────────────────────────────
#
# Each method is shown under its own name.  The main tables use standardized
# features and the cosine distance; ``scaling_distance/<variant>/<model>/`` holds the
# same unseen-order tables for the other three combinations of scaling and distance.
# The k-mer profile with the Jensen-Shannon distance is a method of its own and is
# compared with the 4-mer profile in its main setting.

VARIANTS = [("standardized_cosine", "standardized, cosine"),
            ("raw_cosine", "raw, cosine"),
            ("standardized_euclidean", "standardized, Euclidean"),
            ("raw_euclidean", "raw, Euclidean")]
MAIN_VARIANT = "standardized_cosine"
VARIANT_STYLE = {"standardized_cosine": dict(alpha=1.0, hatch=""),
                 "raw_cosine": dict(alpha=0.55, hatch=""),
                 "standardized_euclidean": dict(alpha=1.0, hatch="///"),
                 "raw_euclidean": dict(alpha=0.55, hatch="///")}
SCALING_MODELS = ["kmer", "kmer_js", "dnabert2", "dnabertS", "viralm", "evo2"]
SCALING_GROUPS = FORWARD_GROUPS


def _variant_values(model, variant, virus, metrics_dir=None):
    """(layer, per-sample P@5) of one model under one scaling/distance variant, at the
    layer chosen on the reference set (None when the table does not exist)."""
    name = tables.matrix_name("P", config.PK_FOR_FIGURES, virus)
    base = metrics_dir or config.METRICS
    directory = base if variant == MAIN_VARIANT else base / "scaling_distance" / variant
    path = tables.model_dir(model, directory) / name
    if not path.exists():
        return None, None
    return selection.row_of(model, tables.load_matrix(path), metrics_dir=metrics_dir)


def build_scaling_distance(metrics_dir=None) -> pd.DataFrame:
    """P@5 on the unseen orders of every method under every scaling/distance variant,
    with the test of each variant against the main setting of the same method (and of
    the Jensen-Shannon profile against the 4-mer profile); one BH family."""
    from . import significance

    rows, values = [], {}
    for virus, group in SCALING_GROUPS:
        for key in SCALING_MODELS:
            model = registry.get(key)
            variants = [(MAIN_VARIANT, "Jensen-Shannon")] if key == "kmer_js" else VARIANTS
            for variant, label in variants:
                layer, series = _variant_values(model, variant, virus, metrics_dir)
                if series is None:
                    continue
                values[(group, key, variant)] = series
                rows.append({"group": group, "model": key, "variant": variant,
                             "label": label, "layer": layer,
                             "mean": float(series.mean()), "std": float(series.std(ddof=0))})
    frame = pd.DataFrame(rows)
    raw = {}
    for (group, key, variant), series in values.items():
        if key == "kmer_js":
            reference_key = (group, "kmer", MAIN_VARIANT)
        elif variant != MAIN_VARIANT:
            reference_key = (group, key, MAIN_VARIANT)
        else:
            continue
        if reference_key in values:
            raw[(group, key, variant)] = significance.paired_p(series, values[reference_key])
    adjusted = significance.adjust(raw)
    frame["p_vs_main_bh"] = [adjusted.get((g, m, v), np.nan)
                             for g, m, v in zip(frame["group"], frame["model"], frame["variant"])]
    save_table(frame, "supplementary/scaling_distance")
    return frame


def draw_scaling_distance() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from . import significance

    path = config.FIGURE_DATA / "supplementary" / "scaling_distance.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    keys = [key for key in SCALING_MODELS if key in set(frame["model"])]
    figure, axes = plt.subplots(1, len(SCALING_GROUPS), figsize=(13, 4.6), sharey=True)
    width = 0.8 / len(VARIANTS)
    for axis, (_, group) in zip(axes, SCALING_GROUPS):
        for position, key in enumerate(keys):
            data = frame[(frame["group"] == group) & (frame["model"] == key)]
            for index, (_, row) in enumerate(data.iterrows()):
                offset = 0 if len(data) == 1 else (index - (len(data) - 1) / 2) * width
                x = position + offset
                style = VARIANT_STYLE[row["variant"]]
                axis.bar(x, row["mean"], width, yerr=row["std"], capsize=2,
                         color=registry.color(key), alpha=style["alpha"], hatch=style["hatch"],
                         edgecolor="black", linewidth=0.3)
                significance.put(axis, x, row["mean"] + row["std"] + 0.01,
                                 significance.label(row["p_vs_main_bh"]), fontsize=7)
        axis.set_xticks(range(len(keys)))
        axis.set_xticklabels([registry.display(key) for key in keys], rotation=30, ha="right")
        axis.set_title(group)
        axis.margins(y=0.12)
    axes[0].set_ylabel("Precision@5")
    handles = [Patch(facecolor="0.5", alpha=VARIANT_STYLE[v]["alpha"],
                     hatch=VARIANT_STYLE[v]["hatch"], edgecolor="black", label=label)
               for v, label in VARIANTS]
    figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
                  bbox_to_anchor=(0.5, -0.06))
    figure.suptitle("Effect of feature scaling and distance (Precision@5, held-out viruses)",
                    fontsize=14, y=1.02)
    figure.tight_layout()
    save_figure(figure, "fig8_scaling_distance", kind="main")
    plt.close(figure)


# ── an early layer of a model, chosen on the reference set ──────────────────

def build_early_layer(model_key: str = "evo2", metrics_dir=None) -> pd.DataFrame:
    """The best layer of the first half of the network (by reference P@5) against the
    layer used in the article (best overall), on the unseen orders.

    Both layers are chosen on the reference set, so this is a sensitivity check of the
    layer choice, not a choice made on the unseen viruses.  Positions index the
    metric tables (checked to have as many layers as the benchmark); the
    ``*_layer`` columns hold labels.
    """
    from . import significance

    model = registry.get(model_key)
    means = reference.layer_means(model, config.REFERENCE_METRIC, metrics_dir).to_numpy(dtype=float)
    half = len(means) // 2
    early = int(np.nanargmax(means[:half]))
    chosen = selection.choose_layer(model, metrics_dir=metrics_dir)
    k = config.PK_FOR_FIGURES
    sources = [(f"P@{k}", tables.matrix_name("P", k, virus), group)
               for virus, group in UNSEEN_GROUPS] + \
              [("RC proximity", tables.matrix_name("cross_P", k, virus), group)
               for virus, group in FORWARD_GROUPS]
    rows, raw = [], {}
    for metric, name, group in sources:
        matrix = tables.load_model_matrix(model, name, metrics_dir)
        if len(matrix) != len(means):
            raise ValueError(f"{model.key}/{name}: {len(matrix)} layers, "
                             f"benchmark has {len(means)}")
        early_values, chosen_values = matrix.iloc[early], matrix.iloc[chosen]
        raw[(metric, group)] = significance.paired_p(early_values, chosen_values)
        rows.append({"model": model.key, "metric": metric, "group": group,
                     "early_layer": reference.layer_label(model, early),
                     "early_reference_p5": float(means[early]),
                     "chosen_layer": reference.layer_label(model, chosen),
                     "chosen_reference_p5": float(means[chosen]),
                     "early_mean": float(early_values.mean()),
                     "early_std": float(early_values.std(ddof=0)),
                     "chosen_mean": float(chosen_values.mean()),
                     "chosen_std": float(chosen_values.std(ddof=0))})
    frame = pd.DataFrame(rows)
    adjusted = significance.adjust(raw)
    frame["p_bh"] = [adjusted[(m, g)] for m, g in zip(frame["metric"], frame["group"])]
    save_table(frame, f"supplementary/{model.key}_early_layer")
    return frame


def build_all(models, metrics_dir=None) -> None:
    build_reference(models, metrics_dir)
    build_unseen(models, metrics_dir)
    build_scaling_distance(metrics_dir)
    build_early_layer("evo2", metrics_dir)


def draw_all(models, metrics_dir=None) -> None:
    draw_reference(models, metrics_dir)
    draw_unseen(models)
    draw_scaling_distance()
