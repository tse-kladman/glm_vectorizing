"""The reference-set benchmark -- the numbers behind Fig. 1 and 2.

Three metrics describe how well a representation organises the reference set, per
layer and per reference sample:

    ARI          adjusted Rand index of a K-medoids clustering against the labels
    P5           precision among the 5 nearest neighbours
    silh         cosine silhouette of that clustering

This module reads those tables.  They are also what chooses the layer and the word
length of every model (selection.py), which is the reason the choice never touches
the unseen orders.

The code that *computes* them from the representations is developed on a separate
branch; the tables shipped with the repository were produced before this package
existed, so recomputing them would change published numbers and is deliberately
not part of the pipeline yet.
"""

import numpy as np
import pandas as pd

from .. import config
from ..models import Model
from . import tables

METRICS = ("ARI", "P5", "silh")
METRIC_TITLES = {"ARI": "Adjusted Rand Index", "P5": "Precision@5", "silh": "Silhouette"}


def load(model: Model | str, metric: str, metrics_dir=None) -> pd.DataFrame:
    """Benchmark table of one model and metric: rows are layers, columns samples."""
    return tables.load_benchmark(model, metric, metrics_dir)


def layer_means(model: Model | str, metric: str, metrics_dir=None, samples=None):
    """Mean over reference samples, one value per layer (positional index)."""
    frame = load(model, metric, metrics_dir)
    columns = list(frame.columns) if samples is None else \
        [column for column in frame.columns if column in set(samples)]
    return frame[columns].mean(axis=1, skipna=True)


def summary(models, metric: str, metrics_dir=None) -> pd.DataFrame:
    """Best layer of each model under one benchmark metric, with mean and spread."""
    rows = []
    for model in models:
        frame = load(model, metric, metrics_dir).to_numpy(dtype=float)
        means = np.nanmean(frame, axis=1)
        best = int(np.nanargmax(means))
        rows.append({"metric": metric,
                     "model": model.key if isinstance(model, Model) else model,
                     "best_layer": best,
                     "mean": float(np.nanmean(frame[best])),
                     "std": float(np.nanstd(frame[best]))})
    return pd.DataFrame(rows)


def layerwise(models, metric: str, metrics_dir=None) -> pd.DataFrame:
    """Every layer of every model under one benchmark metric."""
    rows = []
    for model in models:
        key = model.key if isinstance(model, Model) else model
        frame = load(model, metric, metrics_dir).to_numpy(dtype=float)
        means, stds = np.nanmean(frame, axis=1), np.nanstd(frame, axis=1)
        for layer in range(len(means)):
            rows.append({"metric": metric, "model": key, "layer": layer,
                         "mean": float(means[layer]), "std": float(stds[layer])})
    return pd.DataFrame(rows)


def available(models, metric: str = config.REFERENCE_METRIC, metrics_dir=None) -> list:
    """The subset of models that actually has a benchmark table on disk."""
    return [model for model in models if tables.has_benchmark(model, metric, metrics_dir)]
