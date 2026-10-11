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
                  highlight=selection.load().get(model.key, {}).get("layer"))


# ── distance protocols P0..P5 ───────────────────────────────────────────────

PROTOCOLS = [("P0", "standardise + cosine"), ("P1", "raw + cosine"),
             ("P2", "standardise + Euclidean"), ("P3", "raw + Euclidean"),
             ("P4", "MinHash (k=6)"), ("P5", "Jensen-Shannon (k=4)")]
PROTOCOL_GROUPS = [("Egovirales_preproc", "Egovirales"), ("Mriyavirales_preproc", "Mriyavirales"),
                   ("REVERSECOMPLEMENTEgovirales_preproc", "Egovirales (RC)"),
                   ("REVERSECOMPLEMENTMriyavirales_preproc", "Mriyavirales (RC)")]


def _protocol_file(protocol: str, virus: str, metrics_dir=None):
    """Legacy table of the k-mer profile under P0-P3; P4 and P5 are the MinHash and
    Jensen-Shannon models of the current registry (their legacy copies were identical)."""
    base = metrics_dir or config.METRICS
    name = tables.matrix_name("P", config.PK_FOR_FIGURES, virus)
    if protocol == "P4":
        return base / "minhash" / name
    if protocol == "P5":
        return base / "kmer_js" / name
    return base / "distance_protocols" / protocol / "kmer" / name


def build_protocols(metrics_dir=None) -> pd.DataFrame:
    """P@5 of the 4-mer profile under each distance protocol (genome-level, legacy)."""
    rows = []
    for protocol, label in PROTOCOLS:
        for virus, group in PROTOCOL_GROUPS:
            path = _protocol_file(protocol, virus, metrics_dir)
            if not path.exists():
                continue
            values = tables.load_matrix(path).iloc[0]
            rows.append({"protocol": protocol, "label": label, "group": group,
                         "mean": float(values.mean()), "std": float(values.std(ddof=0))})
    frame = pd.DataFrame(rows, columns=["protocol", "label", "group", "mean", "std"])
    save_table(frame, "supplementary/protocol_comparison")
    return frame


def draw_protocols() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    frame = load_table("supplementary/protocol_comparison")
    if frame.empty:
        return
    labels = [f"{p}\n{label}" for p, label in PROTOCOLS if p in set(frame["protocol"])]
    keys = [p for p, _ in PROTOCOLS if p in set(frame["protocol"])]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(PROTOCOL_GROUPS)
    for index, (_, group) in enumerate(PROTOCOL_GROUPS):
        data = frame[frame["group"] == group].set_index("protocol").reindex(keys)
        axis.bar(np.arange(len(keys)) + (index - 1.5) * width, data["mean"], width,
                 yerr=data["std"], capsize=2, label=group, edgecolor="black", linewidth=0.3)
    axis.set_xticks(range(len(keys)))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("Mean Precision@5 (genome level)")
    axis.set_title("4-mer profile under different distance protocols\n"
                   "(P4 = MinHash sketch, P5 = Jensen-Shannon profile: different representations)")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    save_figure(figure, "distance_protocols", kind="supplementary")
    plt.close(figure)


def build_all(models, metrics_dir=None) -> None:
    build_reference(models, metrics_dir)
    build_unseen(models, metrics_dir)
    build_protocols(metrics_dir)


def draw_all(models, metrics_dir=None) -> None:
    draw_reference(models, metrics_dir)
    draw_unseen(models)
    draw_protocols()
