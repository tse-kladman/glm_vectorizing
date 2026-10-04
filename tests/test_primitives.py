"""Unit tests of the pieces every metric is built from.

These are the claims the rest of the code relies on and that are cheap to check
exactly: that the distances are the distances they claim to be, that the layer of a
model is matched by position rather than by number, and that reading P@1..P@10 off
one neighbour list gives the same answer as ten separate searches.

    python -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon

from glmbench import neighbours, selection, statistics
from glmbench.models import KMER, KMER_JS, MINHASH, get
from glmbench.representations import kmer, minhash

RNG = np.random.default_rng(0)


def random_sequence(length: int) -> str:
    return "".join(RNG.choice(list("ACGT"), length))


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# ── Jensen-Shannon ──────────────────────────────────────────────────────────

def test_jensenshannon_matches_scipy():
    """Our blockwise search must rank neighbours exactly as scipy's distance does."""
    query = RNG.dirichlet(np.ones(24) * 0.3, size=20)
    space = RNG.dirichlet(np.ones(24) * 0.3, size=200)

    expected = np.array([[jensenshannon(q, s, base=2) ** 2 for s in space] for q in query])
    found = neighbours.knn(query, space, 7, KMER_JS)

    np.testing.assert_allclose(np.take_along_axis(expected, found, axis=1),
                               np.sort(expected, axis=1)[:, :7], atol=1e-6)


def test_jensenshannon_rejects_embeddings():
    """A metric defined on the simplex must refuse arbitrary vectors, not guess."""
    with pytest.raises(ValueError, match="non-negative"):
        neighbours.transform(RNG.normal(size=(10, 8)), KMER_JS)      # negative entries
    unnormalised = np.abs(RNG.normal(size=(10, 8)))
    with pytest.raises(ValueError, match="distributions"):
        neighbours.transform(unnormalised, KMER_JS)                  # rows do not sum to 1


# ── MinHash ─────────────────────────────────────────────────────────────────

def test_sketch_is_strand_invariant():
    """Canonical k-mers make a sketch identical for a fragment and its RC."""
    a, b = minhash.hash_functions(256)
    sequence = random_sequence(2000)
    assert np.array_equal(minhash.sketch(sequence, 8, a, b),
                          minhash.sketch(reverse_complement(sequence), 8, a, b))


def test_jaccard_estimate_is_accurate():
    """The fraction of agreeing sketch positions estimates the true Jaccard index."""
    a, b = minhash.hash_functions(2048)
    left = random_sequence(2000)
    right = left[:1000] + random_sequence(1000)

    sets = [set(minhash._canonical_codes(sequence, 8).tolist()) for sequence in (left, right)]
    exact = len(sets[0] & sets[1]) / len(sets[0] | sets[1])
    estimate = minhash.jaccard(minhash.sketch(left, 8, a, b), minhash.sketch(right, 8, a, b))
    assert abs(exact - estimate) < 0.01


def test_minhash_rejects_float_vectors():
    with pytest.raises(ValueError, match="sketches"):
        neighbours.transform(RNG.random((10, 8)), MINHASH)


# ── k-mer profiles ──────────────────────────────────────────────────────────

def test_profile_is_a_distribution():
    profile = kmer.vector(random_sequence(500), 4)
    assert profile.shape == (256,)
    assert profile.min() >= 0
    assert abs(profile.sum() - 1.0) < 1e-6


def test_profile_counts_the_right_words():
    """A hand-checkable case: every 2-mer of ACGT, one of each."""
    profile = kmer.vector("ACGT", 2)
    assert profile.sum() == pytest.approx(1.0)
    assert (profile > 0).sum() == 3                      # AC, CG, GT
    assert profile[0 * 4 + 1] == pytest.approx(1 / 3)    # AC


def test_profile_skips_ambiguous_bases():
    assert kmer.vector("NNNN", 3).sum() == 0


# ── precision at k ──────────────────────────────────────────────────────────

def test_prefixes_match_separate_searches():
    """P@1..P@10 read off one neighbour list == ten independent computations."""
    labels = np.array(["a"] * 30 + ["b"] * 30)
    space = np.vstack([RNG.normal(0, 1, (30, 6)), RNG.normal(3, 1, (30, 6))])
    found = neighbours.knn(space, space, 11, KMER)[:, 1:]

    prefixes = neighbours.precision_at_ks(labels[found], labels, range(1, 11))
    for k in range(1, 11):
        single = neighbours.precision_at_ks(labels[found[:, :k]], labels, [k])[k]
        assert prefixes[k] == pytest.approx(single)


# ── layer selection ─────────────────────────────────────────────────────────

def test_layer_is_matched_by_position(monkeypatch):
    """ViraLM numbers layers 0..11 in the benchmark and 1..12 in the metric tables.

    Matching by number would silently shift the result; matching by position must
    return the metric table's own label for the winning row.
    """
    benchmark = pd.DataFrame({0: [0.1, 0.2, 0.9, 0.3], 1: [0.1, 0.2, 0.9, 0.3]})
    matrix = pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0]}, index=[1, 2, 3, 4])

    monkeypatch.setattr(selection.reference, "load", lambda *a, **k: benchmark)
    monkeypatch.setattr(selection.reference, "layer_means",
                        lambda *a, **k: benchmark.mean(axis=1))
    assert selection.layer_of(get("viralm"), matrix) == 3      # position 2 -> label 3


def test_flat_model_needs_no_benchmark():
    matrix = pd.DataFrame({0: [0.5], 1: [0.6]}, index=["-"])
    assert selection.layer_of(KMER, matrix) == "-"


# ── corrections ─────────────────────────────────────────────────────────────

def test_holm_and_bh_are_monotone_and_bounded():
    raw = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    for adjusted in (statistics.holm(raw), statistics.benjamini_hochberg(raw)):
        assert np.all(adjusted >= raw - 1e-12)
        assert np.all(np.diff(adjusted) >= -1e-12)
        assert adjusted.max() <= 1.0


def test_holm_is_more_conservative_than_bh():
    raw = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    assert np.all(statistics.holm(raw) >= statistics.benjamini_hochberg(raw) - 1e-12)


def test_rank_biserial_saturates_when_one_side_always_wins():
    assert statistics.rank_biserial(np.array([0.1, 0.2, 0.3])) == pytest.approx(1.0)
    assert statistics.rank_biserial(np.array([-0.1, -0.2])) == pytest.approx(-1.0)
