"""Every layer of every metric, one figure per model.

Fig. 1, 4 and 5 summarise a model by a single layer; these show the full per-layer
picture behind that summary, which is where a reader checks whether the chosen
layer sits on a plateau or on a spike.

Files are grouped by metric rather than thrown into one flat folder:

    figures/supplementary/ARI/<model>.png
    figures/supplementary/P5/<model>.png
    figures/supplementary/silhouette/<model>.png
    figures/supplementary/newvirus_p5/<model>_<virus>.png
"""

import pandas as pd

from .. import config, selection
from ..metrics import reference, tables
from .style import load_table, save_figure, save_table

FOLDERS = {"ARI": "ARI", "P5": "P5", "silh": "silhouette"}
VIRUSES = ["Mriyavirales_preproc", "Egovirales_preproc",
           "REVERSECOMPLEMENTMriyavirales_preproc", "REVERSECOMPLEMENTEgovirales_preproc"]


def virus_label(name: str) -> str:
    if name.startswith("REVERSECOMPLEMENT"):
        return name.split("REVERSECOMPLEMENT")[-1].split("_")[0] + " (RC)"
    return name.split("_")[0]


def build_reference(models, metrics_dir=None) -> pd.DataFrame:
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
    frame = pd.DataFrame(rows)
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
        chosen = (selection.load().get(model.key, {}).get("layer")
                  if model.layered else None)
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
                  highlight=selection.load().get(model.key, {}).get("layer"))


def build_all(models, metrics_dir=None) -> None:
    build_reference(models, metrics_dir)
    build_unseen(models, metrics_dir)


def draw_all(models, metrics_dir=None) -> None:
    draw_reference(models, metrics_dir)
    draw_unseen(models)
