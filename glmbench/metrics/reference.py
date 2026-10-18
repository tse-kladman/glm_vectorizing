"""The reference-set benchmark -- the numbers behind Fig. 1 and 2.

Three metrics describe how well a representation organises a reference sample, per
layer and per reference sample (50 independent samples of known viruses, see
``samples.py``):

    ARI    adjusted Rand index between the TRUE class labels and a K-medoids clustering
           (cosine distance in the standardised space, as many clusters as classes,
           fixed random state, ``clustering.py``)
    P5     precision among the 5 nearest neighbours, self excluded by index
    P5_lgo the same, but every fragment of the query's own GENOME is excluded
           (leave-genome-out).  A sample holds up to 10 fragments of each genome, so
           in P5 the "same class" neighbour is often a piece of the same genome;
           P5_lgo needs another genome of the class.  The genome of each fragment is
           read from ``manifest_sample_<i>.tsv``; samples without a manifest (the
           draws of the first release) give NaN and the table is not written.
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
from pathlib import Path

import numpy as np
import pandas as pd

from .. import clustering, config, neighbours
from ..models import Model
from ..representations import store
from . import tables

METRICS = ("ARI", "P5", "silh")
EXTRA_METRICS = ("P5_lgo",)      # written only when the sample manifests exist
METRIC_TITLES = {"ARI": "Adjusted Rand Index", "P5": "Precision@5", "silh": "Silhouette",
                 "P5_lgo": "Precision@5, leave-genome-out"}


def load(model: Model | str, metric: str, metrics_dir=None) -> pd.DataFrame:
    """Benchmark table of one model and metric: rows are layers, columns samples."""
    return tables.load_benchmark(model, metric, metrics_dir)


def layer_label(model: Model | str, position: int):
    """Label of the layer at benchmark *position*, as the metric tables and the
    extracted vectors number it.

    Benchmark tables are indexed by position 0..n-1 (``compute`` sorts the layer
    labels).  For every model but ViraLM that is the label as well (DNABERT-2/S: 0 =
    embedding output, i = encoder block i; Evo2: i = ``blocks.i``); ViraLM's hook writes
    encoder blocks 1..12 only, so position 2 is block 3.  Everything a reader sees --
    figure axes, the ``layer`` columns of the figure CSVs -- uses the label, so the same
    layer carries the same number in Fig. 1, 2 and 4.
    """
    if isinstance(model, str):
        from ..models import get
        model = get(model)
    if model.extractor == "viralm":
        return list(store.VIRALM_LAYERS)[int(position)]
    return int(position)


def best_position(model: Model | str, metric: str, metrics_dir=None) -> int:
    """Position of the layer with the highest mean over the reference samples (NaN
    samples ignored; a layer that is NaN on every sample is never chosen)."""
    frame = load(model, metric, metrics_dir).to_numpy(dtype=float)
    return int(np.nanargmax(np.nanmean(frame, axis=1)))


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
        best = best_position(model, metric, metrics_dir)
        rows.append({"metric": metric,
                     "model": model.key if isinstance(model, Model) else model,
                     "best_layer": layer_label(model, best),
                     "mean": float(np.nanmean(frame[best])),
                     "std": float(np.nanstd(frame[best]))})
    return pd.DataFrame(rows)


def layerwise(models, metric: str, metrics_dir=None) -> pd.DataFrame:
    """Every layer of every model under one benchmark metric (``layer`` = label,
    see :func:`layer_label`)."""
    rows = []
    for model in models:
        key = model.key if isinstance(model, Model) else model
        frame = load(model, metric, metrics_dir).to_numpy(dtype=float)
        means, stds = np.nanmean(frame, axis=1), np.nanstd(frame, axis=1)
        for position in range(len(means)):
            rows.append({"metric": metric, "model": key,
                         "layer": layer_label(model, position),
                         "mean": float(means[position]), "std": float(stds[position])})
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
                     k: int = config.PK_FOR_FIGURES, genomes: dict | None = None) -> dict:
    """ARI, P@k and silhouette of one reference sample under one model (one layer).

    ``genomes`` (``{class: genome id of every row, in the order of groups[class]}``)
    adds ``P5_lgo``: P@k with all fragments of the query's genome excluded.  Without
    it, or when it does not match the rows, ``P5_lgo`` is NaN.
    """
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    space, labels = neighbours.stack(groups)
    classes = np.unique(labels)
    if space is None or len(classes) < 2 or len(labels) <= max(k, len(classes)):
        return {"ARI": float("nan"), "P5": float("nan"), "silh": float("nan"),
                "P5_lgo": float("nan")}

    matrix = neighbours.transform(space, model)
    distance = neighbours.distance_matrix(matrix, model)

    predicted, _ = clustering.kmedoids(distance, len(classes), random_state)
    found = neighbours.knn_excluding(matrix, matrix, k, model,
                                     self_index=np.arange(len(matrix)))
    out = {
        "ARI": float(adjusted_rand_score(labels, predicted)),
        "P5": neighbours.precision_at_ks(neighbours.labels_of(labels, found), labels, [k])[k],
        "silh": float(silhouette_score(distance, labels, metric="precomputed")),
        "P5_lgo": float("nan"),
    }
    row_genomes = _row_genomes(groups, genomes)
    if row_genomes is not None:
        found = neighbours.knn_excluding(matrix, matrix, k, model,
                                         self_index=np.arange(len(matrix)),
                                         query_groups=row_genomes,
                                         space_groups=row_genomes)
        out["P5_lgo"] = neighbours.precision_at_ks(
            neighbours.labels_of(labels, found), labels, [k])[k]
    return out


def _row_genomes(groups: dict, genomes: dict | None):
    """Genome id of every row of ``neighbours.stack(groups)``, or None if unknown.

    ``stack`` drops empty groups and keeps the order of the dict, so the ids are
    concatenated the same way.  A class whose number of ids differs from its number
    of rows (a fragment missing from the representations) makes the mapping
    unreliable, and then nothing is returned rather than a wrong exclusion.
    """
    if not genomes:
        return None
    ids = []
    for label, rows in groups.items():
        if not len(rows):
            continue
        mine = genomes.get(label)
        if mine is None or len(mine) != len(rows):
            return None
        ids.extend(f"{label}/{genome}" for genome in mine)
    return np.asarray(ids, dtype=object)


def sample_genomes(index: int, directory=None) -> dict | None:
    """``{class: [genome id per fragment, in FASTA order]}`` from a sample manifest.

    ``samples.write_sample`` writes the FASTA in manifest order and every extractor
    keeps FASTA order within a class, so the n-th id of a class belongs to its n-th
    row.  None when the manifest does not exist (samples of the first release).
    """
    path = Path(directory or config.REFERENCE_DIR) / f"manifest_sample_{index}.tsv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    out = {}
    for label, isolate in zip(frame["class"], frame["isolate"]):
        out.setdefault(label, []).append(isolate)
    return out


def compute(model: Model, n_ref: int = config.N_REFERENCE, latent_dir=None,
            metrics_dir=None, seed: int = config.SEED, allow_missing: bool = False,
            max_workers: int | None = None, write: bool = True,
            reference_dir=None) -> dict:
    """Benchmark one model on its reference samples; write ``{ARI,P5,silh}.xlsx``
    and, when the sample manifests exist, ``P5_lgo.xlsx`` (leave-genome-out P@5).

    Rows of the tables are layers by position (sorted by label), columns are the
    reference samples; a sample that is missing or unreadable is counted, reported,
    and makes the call fail unless ``allow_missing`` (its column is then NaN).
    """
    problems, missing = [], []

    def one(index):
        data = store.load_sample(model, index, latent_dir, None, problems)
        if data is None:
            return index, None
        genomes = sample_genomes(index, reference_dir)
        layers = sorted(data) if model.layered else [tables.FLAT_LAYER]
        out = {}
        for position, layer in enumerate(layers):
            groups = data[layer] if model.layered else data
            out[position] = benchmark_sample(groups, model, seed + index, genomes=genomes)
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
    for metric in METRICS + EXTRA_METRICS:
        frame = pd.DataFrame(np.nan, index=range(n_layers), columns=range(n_ref))
        for index, out in results.items():
            for position, scores in out.items():
                frame.loc[position, index] = scores[metric]
        if metric in EXTRA_METRICS and frame.isna().all().all():
            print(f"  {model.key}: no sample manifest found, {metric} not written")
            continue
        frames[metric] = frame
        if write:
            tables.save_benchmark(frame, model, metric, metrics_dir)
    return frames
