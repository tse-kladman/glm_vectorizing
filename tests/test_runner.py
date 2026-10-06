"""End-to-end check of the metric runner on synthetic representations.

Exercises the parts that the regression test cannot reach, because they need
extracted vectors rather than committed tables: the store, the per-sample metrics
and the parallel runner, for all three kinds of representation at once -- layered
embeddings, k-mer distributions and MinHash sketches.
"""

import pickle

import numpy as np
import pytest

from glmbench import models
from glmbench.metrics import runner, tables

N_SAMPLES = 3
LAYERS = 3


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


@pytest.fixture
def latent(tmp_path):
    """A miniature data/latent_dim with one file per model and virus."""
    rng = np.random.default_rng(4)
    root = tmp_path / "latent"

    def embeddings(rows):
        return {layer: {f"ref{j}": rng.normal(size=(rows, 8)) for j in range(3)}
                for layer in range(LAYERS)}

    for index in range(N_SAMPLES):
        _dump(root / "dnabert2" / f"sample_{index}.pickle", embeddings(10))
        _dump(root / "kmer_js" / f"sample_{index}.pickle",
              {f"ref{j}": rng.dirichlet(np.ones(32) * .5, size=10) for j in range(3)})
        _dump(root / "minhash" / f"sample_{index}.pickle",
              {f"ref{j}": rng.integers(0, 2 ** 31, size=(10, 16),
                                       dtype=np.int64).astype(np.uint64) for j in range(3)})

    for name, label in [("V_preproc", "V"),
                        ("REVERSECOMPLEMENTV_preproc", "REVERSECOMPLEMENTV")]:
        _dump(root / "dnabert2" / f"{name}.pickle",
              {layer: {label: rng.normal(size=(8, 8))} for layer in range(LAYERS)})
        _dump(root / "kmer_js" / f"{name}.pickle",
              {label: rng.dirichlet(np.ones(32) * .5, size=8)})
        _dump(root / "minhash" / f"{name}.pickle",
              {label: rng.integers(0, 2 ** 31, size=(8, 16),
                                   dtype=np.int64).astype(np.uint64)})
    return root


def test_runner_scores_every_kind_of_representation(latent, tmp_path):
    metrics_dir = tmp_path / "metrics"
    chosen = models.resolve(["dnabert2", "kmer_js", "minhash"])

    rows = runner.run(chosen, [("V_preproc", "REVERSECOMPLEMENTV_preproc")],
                      n_ref=N_SAMPLES, latent_dir=latent, metrics_dir=metrics_dir,
                      max_workers=1)

    assert rows, "the runner produced no results"
    scored = {row["model"] for row in rows}
    assert scored == {"dnabert2", "kmer_js", "minhash"}

    # every metric is a precision, so it has to land in [0, 1]
    assert all(0.0 <= row["mean"] <= 1.0 for row in rows)

    # forward and cross metrics for every k, for the forward file
    forward = [row for row in rows if row["virus"] == "V_preproc"]
    assert {row["k"] for row in forward} == set(range(1, 11))
    assert {row["type"] for row in forward} == {"forward", "cross"}


def test_runner_writes_tables_of_the_expected_shape(latent, tmp_path):
    metrics_dir = tmp_path / "metrics"
    runner.run(models.resolve(["dnabert2"]), [("V_preproc", "REVERSECOMPLEMENTV_preproc")],
               n_ref=N_SAMPLES, latent_dir=latent, metrics_dir=metrics_dir, max_workers=1)

    path = (tables.model_dir("dnabert2", metrics_dir)
            / tables.matrix_name("P", 5, "V_preproc"))
    assert path.exists()

    matrix = tables.load_matrix(path)
    assert len(matrix) == LAYERS                      # one row per layer
    assert list(matrix.columns) == list(range(N_SAMPLES))   # one column per sample
    assert matrix.to_numpy().min() >= 0


def test_selected_layers_restrict_the_work(latent, tmp_path, monkeypatch):
    """--layers selected must score the chosen layer only."""
    from glmbench import selection

    monkeypatch.setattr(selection, "load", lambda *a, **k: {"dnabert2": {"layer": 1}})
    metrics_dir = tmp_path / "metrics"
    runner.run(models.resolve(["dnabert2"]), [("V_preproc", None)],
               n_ref=N_SAMPLES, latent_dir=latent, metrics_dir=metrics_dir,
               layers_mode="selected", max_workers=1)

    matrix = tables.load_matrix(tables.model_dir("dnabert2", metrics_dir)
                                / tables.matrix_name("P", 5, "V_preproc"))
    assert list(matrix.index) == [1]
