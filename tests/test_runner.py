"""End-to-end check of the metric runner on synthetic representations.

Exercises the parts that the regression test cannot reach, because they need
extracted vectors rather than committed tables: the store, the per-sample metrics
and the parallel runner, for all three kinds of representation at once -- layered
embeddings, k-mer distributions and MinHash sketches -- and what happens when the
inputs are incomplete.
"""

import pickle

import numpy as np
import pytest

from glmbench import models, selection
from glmbench.metrics import runner, tables
from glmbench.representations import store

N_SAMPLES = 3
LAYERS = 3
GENOMES = ("gA", "gB", "gC")
VIRUS, RC = "V_preproc", "REVERSECOMPLEMENTV_preproc"


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def _sketches(rng, rows):
    return rng.integers(0, 2 ** 31, size=(rows, 16), dtype=np.int64).astype(np.uint64)


@pytest.fixture
def latent(tmp_path):
    """A miniature data/latent_dim with one file per model and virus (3 genomes each)."""
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
              {f"ref{j}": _sketches(rng, 10) for j in range(3)})

    for name, prefix in [(VIRUS, ""), (RC, "REVERSECOMPLEMENT")]:
        _dump(root / "dnabert2" / f"{name}.pickle",
              {layer: {f"{prefix}{g}": rng.normal(size=(6, 8)) for g in GENOMES}
               for layer in range(LAYERS)})
        _dump(root / "kmer_js" / f"{name}.pickle",
              {f"{prefix}{g}": rng.dirichlet(np.ones(32) * .5, size=6) for g in GENOMES})
        _dump(root / "minhash" / f"{name}.pickle",
              {f"{prefix}{g}": _sketches(rng, 6) for g in GENOMES})
    return root


def run(chosen, latent, tmp_path, **kwargs):
    return runner.run(models.resolve(chosen), [(VIRUS, RC)], n_ref=N_SAMPLES,
                      latent_dir=latent, metrics_dir=tmp_path / "metrics", max_workers=1,
                      **kwargs)


def test_runner_scores_every_kind_of_representation(latent, tmp_path):
    rows = run(["dnabert2", "kmer_js", "minhash"], latent, tmp_path)

    assert rows, "the runner produced no results"
    assert {row["model"] for row in rows} == {"dnabert2", "kmer_js", "minhash"}

    # every metric is a precision / hit rate, so it has to land in [0, 1]
    assert all(0.0 <= row["mean"] <= 1.0 for row in rows)

    forward = [row for row in rows if row["virus"] == VIRUS]
    assert {row["k"] for row in forward} == set(range(1, 11))
    assert {row["type"] for row in forward} == {"forward", "order", "cross", "own_rc"}


def test_runner_writes_tables_of_the_expected_shape(latent, tmp_path):
    run(["dnabert2"], latent, tmp_path)
    directory = tables.model_dir("dnabert2", tmp_path / "metrics")

    for prefix in ("P", "Porder", "cross_P"):
        matrix = tables.load_matrix(directory / tables.matrix_name(prefix, 5, VIRUS))
        assert len(matrix) == LAYERS                          # one row per layer
        assert list(matrix.columns) == list(range(N_SAMPLES))  # one column per sample
        assert matrix.to_numpy().min() >= 0
    assert (directory / tables.own_rc_names(5, VIRUS)).exists()
    assert (directory / tables.own_rc_similarity_name(VIRUS)).exists()


def test_per_genome_values_are_stored(latent, tmp_path):
    run(["kmer_js"], latent, tmp_path)
    directory = tables.model_dir("kmer_js", tmp_path / "metrics")

    for kind in ("genome", "order", "cross", "ownrc"):
        frame = tables.load_per_genome(directory / tables.per_genome_name(kind, VIRUS))
        assert set(frame["genome"]) == set(GENOMES) or \
            set(frame["genome"]) == {f"REVERSECOMPLEMENT{g}" for g in GENOMES}
        assert set(frame["sample"]) == set(range(N_SAMPLES))
        assert frame["value"].between(0, 1).all()

    # the per-genome values average back to the per-sample table
    frame = tables.load_per_genome(directory / tables.per_genome_name("genome", VIRUS))
    matrix = tables.load_matrix(directory / tables.matrix_name("P", 5, VIRUS))
    mean = frame[frame["metric"] == "P@5"].groupby("sample")["value"].mean()
    np.testing.assert_allclose(mean.to_numpy(), matrix.loc["-"].to_numpy())


def test_protocol_selects_which_tables_are_written(latent, tmp_path):
    run(["kmer_js"], latent, tmp_path, protocol="order")
    directory = tables.model_dir("kmer_js", tmp_path / "metrics")
    assert (directory / tables.matrix_name("Porder", 5, VIRUS)).exists()
    assert not (directory / tables.matrix_name("P", 5, VIRUS)).exists()
    assert not (directory / tables.matrix_name("cross_P", 5, VIRUS)).exists()
    assert (directory / tables.own_rc_names(5, VIRUS)).exists()       # strand metric: always


def test_selected_layers_restrict_the_work(latent, tmp_path, monkeypatch):
    """--layers selected must score the chosen layer only."""
    monkeypatch.setattr(selection, "load", lambda *a, **k: {"dnabert2": {"layer": 1}})
    run(["dnabert2"], latent, tmp_path, layers_mode="selected")
    matrix = tables.load_matrix(tables.model_dir("dnabert2", tmp_path / "metrics")
                                / tables.matrix_name("P", 5, VIRUS))
    assert list(matrix.index) == [1]


def test_selected_layer_is_translated_from_position_to_label_for_viralm():
    """selection.json stores a benchmark POSITION; ViraLM's files are labelled 1..12."""
    viralm = models.get("viralm")
    assert selection.layer_label(viralm, 2) == 3
    assert selection.layer_label(viralm, 0) == 1
    assert selection.layer_label(models.get("dnabert2"), 2) == 2
    assert selection.layer_label(models.get("evo2"), 26) == 26
    assert runner._keep_layers(viralm, "all") is None


# ── nothing is dropped silently ─────────────────────────────────────────────

def test_a_missing_reference_sample_fails_the_run(latent, tmp_path, capsys):
    (latent / "kmer_js" / "sample_1.pickle").unlink()
    with pytest.raises(SystemExit, match="problem"):
        run(["kmer_js"], latent, tmp_path)
    out = capsys.readouterr().out
    assert "1 of 3 reference samples missing" in out
    assert not (tables.model_dir("kmer_js", tmp_path / "metrics")
                / tables.matrix_name("P", 5, VIRUS)).exists()          # nothing half-written


def test_allow_missing_writes_what_loaded_and_says_so(latent, tmp_path, capsys):
    (latent / "kmer_js" / "sample_1.pickle").unlink()
    rows = run(["kmer_js"], latent, tmp_path, allow_missing=True)
    assert rows
    out = capsys.readouterr().out
    assert "WARNING" in out and "reference samples loaded: 2/3" in out
    matrix = tables.load_matrix(tables.model_dir("kmer_js", tmp_path / "metrics")
                                / tables.matrix_name("P", 5, VIRUS))
    assert list(matrix.columns) == [0, 2]


def test_a_corrupt_sample_is_counted_not_swallowed(latent, tmp_path, capsys):
    (latent / "minhash" / "sample_0.pickle").write_bytes(b"garbage")
    with pytest.raises(SystemExit):
        run(["minhash"], latent, tmp_path)
    out = capsys.readouterr().out
    assert "corrupt" in out and "sample_0.pickle" in out


def test_a_missing_virus_file_fails_the_run(latent, tmp_path, capsys):
    (latent / "kmer_js" / f"{RC}.pickle").unlink()
    with pytest.raises(SystemExit):
        run(["kmer_js"], latent, tmp_path)
    assert "missing" in capsys.readouterr().out


def test_misaligned_rc_file_is_reported(latent, tmp_path, capsys):
    path = latent / "kmer_js" / f"{RC}.pickle"
    data = pickle.load(open(path, "rb"))
    data["REVERSECOMPLEMENTgA"] = data["REVERSECOMPLEMENTgA"][:-1]
    _dump(path, data)
    with pytest.raises(SystemExit):
        run(["kmer_js"], latent, tmp_path)
    assert "not aligned" in capsys.readouterr().out


def test_store_reports_corrupt_files_in_the_problem_list(tmp_path):
    root = tmp_path / "latent"
    (root / "kmer").mkdir(parents=True)
    (root / "kmer" / "sample_0.pickle").write_bytes(b"\x80broken")
    problems = []
    assert store.load_sample(models.get("kmer"), 0, root, problems=problems) is None
    assert problems and "corrupt" in problems[0][1]
    assert store.load_sample(models.get("kmer"), 7, root, problems=problems) is None   # absent
    assert len(problems) == 1                                  # absence is not corruption
