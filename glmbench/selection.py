"""Choosing every free parameter -- always on the reference set.

Two parameters are free: which layer of a transformer to report, and how long the
word of a k-mer baseline should be.  Tuning either on the unseen orders would mean
reporting a number that was chosen for being high, so both are chosen here, on the
reference viruses, by the same quantity -- P@5 inside a reference sample -- and
then held fixed for everything downstream.

The choice is written to ``results/metrics/selection.json`` so it is visible,
reviewable and reusable:

    {"evo2": {"layer": 26, "k": null, "score": 0.6126, "chosen_by": "reference P5"}}

Layers are matched to the metric tables **by position**, never by number: ViraLM
numbers its layers 0..11 in the benchmark table and 1..12 in the metric matrices,
so a lookup by number would silently shift it by one.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config, neighbours
from .metrics import reference
from .models import Model
from .representations import kmer as kmer_repr
from .representations import minhash as minhash_repr


# ── the file that records the choice ────────────────────────────────────────

def load(path: Path | None = None) -> dict:
    path = Path(path or config.SELECTION_FILE)
    return json.loads(path.read_text()) if path.exists() else {}


def save(selection: dict, path: Path | None = None) -> Path:
    path = Path(path or config.SELECTION_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    return path


def record(model: Model, layer=None, k=None, score=None, chosen_by="", path=None) -> dict:
    selection = load(path)
    selection[model.key] = {"layer": layer, "k": k,
                            "score": None if score is None else round(float(score), 6),
                            "chosen_by": chosen_by}
    save(selection, path)
    return selection


def chosen_k(model: Model, path=None, default=None):
    entry = load(path).get(model.key, {})
    return entry.get("k") or model.fixed_k or default


# ── layer: taken from the reference benchmark ───────────────────────────────

def choose_layer(model: Model, metric: str = config.REFERENCE_METRIC,
                 metrics_dir=None, samples=None) -> int:
    """Positional index of the layer the reference benchmark ranks first."""
    means = reference.layer_means(model, metric, metrics_dir, samples)
    return int(np.argmax(means.to_numpy()))


def layer_of(model: Model, matrix, metric: str = config.REFERENCE_METRIC,
             metrics_dir=None, samples=None):
    """The label, inside ``matrix``, of the layer chosen on the reference set.

    ``matrix`` is a metric table (rows = layers); its row order is what the
    positional match is against.
    """
    if len(matrix) == 1:
        return matrix.index[0]                     # nothing to choose
    benchmark = reference.load(model, metric, metrics_dir)
    if len(benchmark) != len(matrix):
        key = model.key if isinstance(model, Model) else model
        raise ValueError(f"{key}: benchmark has {len(benchmark)} layers, "
                         f"metric table has {len(matrix)}")
    return matrix.index[choose_layer(model, metric, metrics_dir, samples)]


def row_of(model: Model, matrix, metric: str = config.REFERENCE_METRIC,
           metrics_dir=None):
    """(layer, per-sample values) of a metric table at the reference-chosen layer."""
    layer = layer_of(model, matrix, metric, metrics_dir)
    return layer, matrix.loc[layer]


# ── word length: swept on the reference samples ─────────────────────────────

def reference_precision(groups: dict, model: Model, k_neighbours: int = 5) -> float:
    """P@k inside one reference sample -- the same quantity that chooses the layer."""
    space, labels = neighbours.stack(groups)
    if space is None or len(labels) <= k_neighbours:
        return float("nan")
    matrix = neighbours.transform(space, model)
    found = neighbours.knn(matrix, matrix, k_neighbours + 1, model)[:, 1:]
    return neighbours.precision_at_ks(labels[found], labels, [k_neighbours])[k_neighbours]


def _score_one(packed):
    """One (word length, reference sample) pair; top level, for the process pool."""
    model_key, k, index, reference_dir, k_neighbours = packed
    from .models import get                      # re-imported inside the worker

    model = get(model_key)
    path = Path(reference_dir) / f"sample_{index}.fa"
    if not path.exists():
        return k, index, float("nan")

    if model.extractor == "minhash":
        groups = minhash_repr.from_fasta(path, k, model.sketch_size)
    else:
        groups = kmer_repr.from_fasta(path, k)
    return k, index, reference_precision(groups, model, k_neighbours)


def sweep_k(model: Model, k_values=None, reference_dir=None, n_ref: int = 10,
            k_neighbours: int = 5, max_workers=None):
    """Score every candidate word length on reference samples; return the table."""
    k_values = list(k_values or model.k_values)
    reference_dir = str(reference_dir or config.REFERENCE_DIR)
    tasks = [(model.key, k, index, reference_dir, k_neighbours)
             for k in k_values for index in range(n_ref)]

    scores = {k: {} for k in k_values}
    with ProcessPoolExecutor(max_workers=max_workers or config.MAX_WORKERS) as pool:
        futures = [pool.submit(_score_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks),
                           desc=f"k sweep {model.key}", unit="task"):
            k, index, value = future.result()
            scores[k][index] = value

    table = pd.DataFrame(scores).T.sort_index()
    table.index.name = "k"
    table.columns = [f"sample_{column}" for column in table.columns]
    return table
