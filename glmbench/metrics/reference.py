"""The reference-set benchmark -- the numbers behind Fig. 1 and 2.

Three metrics describe how well a representation organises a reference sample, per
layer and per reference sample (50 independent samples of known viruses, see
``samples.py``):

    ARI    adjusted Rand index between the TRUE class labels and a K-medoids clustering
           (cosine distance in the standardised space, as many clusters as classes,
           fixed random state, ``clustering.py``)
    P5     precision among the 5 nearest neighbours, self excluded by index
    silh   silhouette (cosine) **of the true labels** -- how well separated the known
           classes are in the space.  It is NOT the silhouette of the K-medoids
           clustering; the first version of the documentation said so wrongly.

``compute`` produces the tables ``results/metrics/<model>/{ARI,P5,silh}.xlsx``; the
rest of the module reads them.  They are also what chooses the layer and the word
length of every model (selection.py), which is the reason the choice never touches
the unseen orders.

The committed tables for the original five models pre-date this code (their
computation was not kept in the repository), so recomputing them is expected to give
equivalent but not identical numbers; the README lists what to re-run.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .. import clustering, config, neighbours
from ..models import Model
from ..representations import store
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


# ── computing the benchmark ─────────────────────────────────────────────────

def precision_in_sample(groups: dict, model: Model, k: int = config.PK_FOR_FIGURES) -> float:
    """P@k inside one reference sample: neighbours of every fragment, self excluded.

    The single implementation of the in-sample P@k; ``selection.reference_precision``
    (which chooses the word length) calls this, so the layer and the word length are
    chosen by literally the same number.
    """
    space, labels = neighbours.stack(groups)
    if space is None or len(labels) <= k:
        return float("nan")
    matrix = neighbours.transform(space, model)
    found = neighbours.knn_excluding(matrix, matrix, k, model,
                                     self_index=np.arange(len(matrix)))
    return neighbours.precision_at_ks(neighbours.labels_of(labels, found), labels, [k])[k]


def benchmark_sample(groups: dict, model: Model, random_state: int,
                     k: int = config.PK_FOR_FIGURES) -> dict:
    """ARI, P@k and silhouette of one reference sample under one model (one layer)."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    space, labels = neighbours.stack(groups)
    classes = np.unique(labels)
    if space is None or len(classes) < 2 or len(labels) <= max(k, len(classes)):
        return {"ARI": float("nan"), "P5": float("nan"), "silh": float("nan")}

    matrix = neighbours.transform(space, model)
    distance = neighbours.distance_matrix(matrix, model)

    predicted, _ = clustering.kmedoids(distance, len(classes), random_state)
    found = neighbours.knn_excluding(matrix, matrix, k, model,
                                     self_index=np.arange(len(matrix)))
    return {
        "ARI": float(adjusted_rand_score(labels, predicted)),
        "P5": neighbours.precision_at_ks(neighbours.labels_of(labels, found), labels, [k])[k],
        "silh": float(silhouette_score(distance, labels, metric="precomputed")),
    }


def compute(model: Model, n_ref: int = config.N_REFERENCE, latent_dir=None,
            metrics_dir=None, seed: int = config.SEED, allow_missing: bool = False,
            max_workers: int | None = None, write: bool = True) -> dict:
    """Benchmark one model on its reference samples; write ``{ARI,P5,silh}.xlsx``.

    Rows of the tables are layers by position (sorted by label), columns are the
    reference samples; a sample that is missing or unreadable is counted, reported,
    and makes the call fail unless ``allow_missing`` (its column is then NaN).
    """
    problems, missing = [], []

    def one(index):
        data = store.load_sample(model, index, latent_dir, None, problems)
        if data is None:
            return index, None
        layers = sorted(data) if model.layered else [tables.FLAT_LAYER]
        out = {}
        for position, layer in enumerate(layers):
            groups = data[layer] if model.layered else data
            out[position] = benchmark_sample(groups, model, seed + index)
        return index, out

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers or config.SAMPLE_THREADS) as pool:
        for future in as_completed([pool.submit(one, index) for index in range(n_ref)]):
            index, out = future.result()
            if out is None:
                missing.append(index)
            else:
                results[index] = out

    for name, reason in problems:
        print(f"  WARNING {model.key}: {name}: {reason}")
    if missing:
        message = (f"{model.key}: {len(missing)} of {n_ref} reference samples missing or "
                   f"unreadable ({sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''})")
        if not allow_missing:
            raise SystemExit(message + "; re-run 'extract', or pass --allow-missing")
        print("  WARNING " + message)
    if not results:
        raise SystemExit(f"{model.key}: no reference sample could be loaded")

    n_layers = max(len(out) for out in results.values())
    frames = {}
    for metric in METRICS:
        frame = pd.DataFrame(np.nan, index=range(n_layers), columns=range(n_ref))
        for index, out in results.items():
            for position, scores in out.items():
                frame.loc[position, index] = scores[metric]
        frames[metric] = frame
        if write:
            tables.save_benchmark(frame, model, metric, metrics_dir)
    return frames
