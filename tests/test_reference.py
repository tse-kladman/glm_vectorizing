"""The reference benchmark (ARI, P@5, silhouette) on synthetic clusters."""

import pickle

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from glmbench import clustering, neighbours, selection
from glmbench.metrics import reference, tables
from glmbench.models import DNABERT2, KMER, KMER_JS


def blobs(rng, classes=6, per_class=15, dim=10, separation=8.0, spread=1.0):
    centres = rng.normal(0, separation, (classes, dim))
    return {f"c{i}": centres[i] + rng.normal(0, spread, (per_class, dim)) for i in range(classes)}


def noise(rng, classes=6, per_class=15, dim=10):
    return {f"c{i}": rng.normal(0, 1, (per_class, dim)) for i in range(classes)}


# ── K-medoids ───────────────────────────────────────────────────────────────

def test_kmedoids_recovers_well_separated_clusters():
    rng = np.random.default_rng(0)
    groups = blobs(rng)
    x, labels = neighbours.stack(groups)
    distance = cdist(x, x)
    predicted, medoids = clustering.kmedoids(distance, 6, random_state=0)
    assert adjusted_rand_score(labels, predicted) == pytest.approx(1.0)
    assert len(set(medoids)) == 6


@pytest.mark.parametrize("init", ["heuristic", "k-medoids++"])
def test_kmedoids_is_reproducible_for_a_fixed_random_state(init):
    rng = np.random.default_rng(1)
    x, _ = neighbours.stack(blobs(rng, separation=3.0))
    distance = cdist(x, x)
    a, _ = clustering.kmedoids(distance, 6, random_state=5, init=init)
    b, _ = clustering.kmedoids(distance, 6, random_state=5, init=init)
    assert np.array_equal(a, b)


def test_kmedoids_medoid_never_increases_the_cost():
    rng = np.random.default_rng(2)
    x, _ = neighbours.stack(blobs(rng, separation=2.0, spread=2.0))
    distance = cdist(x, x)
    labels, medoids = clustering.kmedoids(distance, 6, random_state=0)
    cost = distance[np.arange(len(x)), medoids[labels]].sum()
    start = clustering._init_heuristic(distance, 6)
    start_cost = distance[:, start].min(axis=1).sum()
    assert cost <= start_cost + 1e-9


def test_kmedoids_rejects_impossible_requests():
    with pytest.raises(ValueError):
        clustering.kmedoids(np.zeros((3, 3)), 5, random_state=0)


# ── the three metrics ───────────────────────────────────────────────────────

def test_structured_data_scores_high_and_noise_scores_low():
    rng = np.random.default_rng(3)
    good = reference.benchmark_sample(blobs(rng), KMER, random_state=0)
    bad = reference.benchmark_sample(noise(rng), KMER, random_state=0)

    assert good["ARI"] > 0.9 and good["P5"] > 0.95 and good["silh"] > 0.4
    assert abs(bad["ARI"]) < 0.15
    assert bad["P5"] < 0.3                       # chance is ~ 14/89
    assert bad["silh"] < 0.1


def test_silhouette_is_computed_against_the_true_labels():
    rng = np.random.default_rng(4)
    groups = blobs(rng, separation=2.0, spread=2.0)           # clusters overlap
    x, labels = neighbours.stack(groups)
    scores = reference.benchmark_sample(groups, KMER, random_state=0)

    matrix = StandardScaler().fit_transform(x)
    expected = silhouette_score(matrix, labels, metric="cosine")
    assert scores["silh"] == pytest.approx(expected, abs=1e-9)


def test_p5_agrees_with_an_independent_brute_force_computation():
    rng = np.random.default_rng(5)
    groups = blobs(rng, separation=2.0, spread=2.0)
    x, labels = neighbours.stack(groups)
    matrix = StandardScaler().fit_transform(x)

    distance = cdist(matrix, matrix, metric="cosine")
    np.fill_diagonal(distance, np.inf)                       # a point is not its own neighbour
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :5]
    expected = np.mean((labels[nearest] == labels[:, None]).mean(axis=1))

    assert reference.precision_in_sample(groups, KMER, 5) == pytest.approx(expected)
    assert reference.benchmark_sample(groups, KMER, 0)["P5"] == pytest.approx(expected)
    assert selection.reference_precision(groups, KMER, 5) == pytest.approx(expected)


def test_p5_agrees_with_the_neighbours_module_and_unseen_path():
    """Same question, asked through the unseen-order machinery, gives the same number."""
    from glmbench.metrics import unseen

    rng = np.random.default_rng(6)
    groups = blobs(rng, separation=2.0, spread=2.0)
    held_out = "c0"
    rest = {k: v for k, v in groups.items() if k != held_out}
    # genome-level P@5 of c0 inside (rest + c0) is the in-sample P@5 of that class alone
    by_genome = unseen.precision_by_genome(rest, {held_out: groups[held_out]}, [5], KMER)
    space, labels = neighbours.stack({**rest, held_out: groups[held_out]})
    matrix = neighbours.transform(space, KMER)
    queries = np.flatnonzero(labels == held_out)
    found = neighbours.knn_excluding(matrix[queries], matrix, 5, KMER, self_index=queries)
    manual = np.mean(labels[found] == held_out)
    assert by_genome[held_out][5] == pytest.approx(manual)


def test_degenerate_samples_give_nan_not_a_crash():
    scores = reference.benchmark_sample({"only": np.random.default_rng(0).normal(size=(30, 4))},
                                        KMER, 0)
    assert all(np.isnan(value) for value in scores.values())


def test_jensenshannon_models_are_benchmarked_too():
    rng = np.random.default_rng(7)
    groups = {f"c{i}": rng.dirichlet(np.random.default_rng(i).dirichlet(np.ones(16)) * 300,
                                     size=12) for i in range(5)}
    scores = reference.benchmark_sample(groups, KMER_JS, 0)
    assert scores["P5"] > 0.9 and scores["ARI"] > 0.8


# ── writing the tables ──────────────────────────────────────────────────────

def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


@pytest.fixture
def latent(tmp_path):
    rng = np.random.default_rng(8)
    root = tmp_path / "latent"
    for index in range(4):
        _dump(root / "dnabert2" / f"sample_{index}.pickle",
              {layer: blobs(rng, classes=4, per_class=10, dim=6, separation=3.0 + layer)
               for layer in range(3)})
    return root


def test_compute_writes_tables_in_the_format_the_pipeline_reads(latent, tmp_path):
    out = tmp_path / "metrics"
    frames = reference.compute(DNABERT2, n_ref=4, latent_dir=latent, metrics_dir=out, seed=1)

    for metric in reference.METRICS:
        assert frames[metric].shape == (3, 4)                  # layers x samples
        table = tables.load_benchmark(DNABERT2, metric, out)
        assert table.shape == (3, 4) and list(table.columns) == [0, 1, 2, 3]
        np.testing.assert_allclose(table.to_numpy(), frames[metric].to_numpy())
    # layers separate better and better by construction
    means = reference.layer_means(DNABERT2, "P5", out)
    assert means.iloc[2] >= means.iloc[0]
    assert selection.choose_layer(DNABERT2, "P5", out) in (1, 2)


def test_compute_is_reproducible(latent, tmp_path):
    a = reference.compute(DNABERT2, 4, latent, tmp_path / "a", seed=3, write=False)
    b = reference.compute(DNABERT2, 4, latent, tmp_path / "b", seed=3, write=False)
    for metric in reference.METRICS:
        assert a[metric].equals(b[metric])


def test_missing_reference_samples_fail_unless_allowed(latent, tmp_path):
    (latent / "dnabert2" / "sample_2.pickle").unlink()
    with pytest.raises(SystemExit, match="missing"):
        reference.compute(DNABERT2, 4, latent, tmp_path / "m", seed=1, write=False)

    frames = reference.compute(DNABERT2, 4, latent, tmp_path / "m", seed=1,
                               allow_missing=True, write=False)
    assert np.isnan(frames["P5"][2]).all()                    # the lost sample is an empty column
    assert frames["P5"][[0, 1, 3]].notna().all().all()


def test_corrupt_reference_samples_are_counted_as_missing(latent, tmp_path):
    (latent / "dnabert2" / "sample_1.pickle").write_bytes(b"not a pickle")
    with pytest.raises(SystemExit, match="1 of 4"):
        reference.compute(DNABERT2, 4, latent, tmp_path / "c", seed=1, write=False)
