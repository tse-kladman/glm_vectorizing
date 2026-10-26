"""The unseen-order metrics, on synthetic worlds where the right answer is known.

The point of most of these tests is the gap between what a metric is *called* and what
it *measures*: a genome-level score that is perfect for genomes that have nothing in
common, and an RC-proximity score that is perfect although no fragment is paired with
its own reverse complement.
"""

import numpy as np

from glmbench import neighbours
from glmbench.metrics import unseen
from glmbench.models import KMER, KMER_JS, MINHASH

KS = [1, 5]


def reference_world(rng, classes=20, per_class=10, dim=8, scale=5.0, spread=0.3):
    centres = rng.normal(0, scale, (classes, dim))
    return {f"class{c}": centres[c] + rng.normal(0, spread, (per_class, dim))
            for c in range(classes)}


def scattered_order(rng, genomes=10, per_genome=8, dim=8, scale=5.0, spread=0.3):
    """Every genome tight, genomes unrelated to one another (and to the reference)."""
    return {f"g{g}": rng.normal(0, scale, dim) + rng.normal(0, spread, (per_genome, dim))
            for g in range(genomes)}


# ── genome level ────────────────────────────────────────────────────────────

def test_genome_level_is_perfect_for_tight_but_unrelated_genomes():
    """Success = same genome: a genome that is one tight clump scores ~1, whatever its
    relation to the other genomes of its order."""
    rng = np.random.default_rng(0)
    reference, order = reference_world(rng), scattered_order(rng)
    genome = unseen.summarise(unseen.precision_by_genome(reference, order, KS, KMER), KS)
    assert genome[1] > 0.95 and genome[5] > 0.95


def test_per_genome_values_are_returned_for_every_genome():
    rng = np.random.default_rng(4)
    reference, order = reference_world(rng), scattered_order(rng, genomes=5)
    result = unseen.precision_by_genome(reference, order, KS, KMER)
    assert set(result) == set(order)
    assert all(set(values) == set(KS) for values in result.values())
    assert np.isnan(unseen.summarise({}, KS)[5])


def test_genome_level_runs_for_every_kind_of_representation():
    rng = np.random.default_rng(5)

    def dirichlet(centre_seed, rows):
        base = np.random.default_rng(centre_seed).dirichlet(np.ones(16) * .3)
        return rng.dirichlet(base * 200 + 0.01, size=rows)

    reference = {f"c{i}": dirichlet(i, 8) for i in range(6)}
    order = {f"g{i}": dirichlet(100 + i, 5) for i in range(4)}
    for model in (KMER, KMER_JS):
        result = unseen.precision_by_genome(reference, order, KS, model)
        assert all(0.0 <= v[5] <= 1.0 for v in result.values())

    sketches = lambda: rng.integers(0, 2 ** 31, size=(5, 12), dtype=np.int64).astype(np.uint64)
    reference = {f"c{i}": rng.integers(0, 2 ** 31, size=(8, 12), dtype=np.int64).astype(np.uint64)
                 for i in range(6)}
    order = {f"g{i}": sketches() for i in range(4)}
    assert unseen.precision_by_genome(reference, order, KS, MINHASH)


# ── RC proximity ────────────────────────────────────────────────────────────

def test_rc_proximity_is_perfect_without_any_fragment_paired_to_its_own_rc():
    """RC fragments near the right genome, but none is the reverse complement of a
    particular forward fragment: RC proximity still scores ~1."""
    rng = np.random.default_rng(7)
    reference = reference_world(rng)
    centres = rng.normal(0, 5, (6, 8))
    forward = {f"g{g}": centres[g] + rng.normal(0, 0.3, (8, 8)) for g in range(6)}
    reverse = {f"REVERSECOMPLEMENT{g}": centres[i] + rng.normal(0, 0.3, rows.shape)
               for i, (g, rows) in enumerate(forward.items())}
    proximity = unseen.summarise(
        unseen.precision_rc_proximity(reference, forward, reverse, [5], KMER), [5])
    assert proximity[5] > 0.95


def test_rc_proximity_skips_a_genome_without_rc_fragments():
    rng = np.random.default_rng(8)
    reference, forward = reference_world(rng), scattered_order(rng, genomes=3)
    reverse = {"REVERSECOMPLEMENTg0": forward["g0"].copy()}
    result = unseen.precision_rc_proximity(reference, forward, reverse, KS, KMER)
    assert set(result) == {"g0"}


# ── neighbour exclusion by index ────────────────────────────────────────────

def test_self_is_removed_by_index_even_when_a_duplicate_ties_with_it():
    rng = np.random.default_rng(10)
    rows = rng.normal(size=(30, 6))
    rows[29] = rows[0]                         # a duplicate of row 0, later in the space
    found = neighbours.knn_excluding(rows[[29]], rows, 3, KMER, self_index=np.array([29]))
    assert 29 not in found[0]                  # never returns itself ...
    assert 0 in found[0]                       # ... but its twin is a legitimate neighbour


def test_positional_removal_would_have_returned_the_query_itself():
    """Document the failure mode: with ties, column 0 need not be the query."""
    rng = np.random.default_rng(11)
    rows = rng.normal(size=(20, 5))
    rows[19] = rows[0]
    raw = neighbours.knn(rows[[19]], rows, 4, KMER)
    positional = raw[0, 1:]
    by_index = neighbours.knn_excluding(rows[[19]], rows, 3, KMER, self_index=np.array([19]))[0]
    assert 19 not in by_index
    # if the tie put row 0 first, the positional rule keeps row 19 (itself)
    if raw[0, 0] == 0:
        assert 19 in positional


def test_zero_vectors_do_not_break_exclusion():
    rows = np.vstack([np.zeros((4, 5)), np.random.default_rng(12).normal(size=(10, 5))])
    queries = np.arange(4)
    found = neighbours.knn_excluding(rows[queries], rows, 3, KMER, self_index=queries)
    assert found.shape == (4, 3)
    for q in queries:
        assert q not in found[q]


def test_exclusion_pads_when_candidates_run_out():
    rows = np.random.default_rng(14).normal(size=(3, 3))
    found = neighbours.knn_excluding(rows[:1], rows, 4, KMER, self_index=np.array([0]))
    assert set(found[0, :2]) == {1, 2} and (found[0, 2:] == neighbours.PAD).all()
    labels = neighbours.labels_of(np.array(list("abc")), found)
    assert labels[0, 3] == neighbours.PAD_LABEL            # a pad never matches a real label
