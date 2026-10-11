"""The unseen-order metrics, on synthetic worlds where the right answer is known.

The point of most of these tests is the gap between what a metric is *called* and what
it *measures*: a genome-level score that looks perfect while the order-level score is
at chance, and an RC-proximity score that looks perfect while a fragment never finds
its own reverse complement.
"""

import numpy as np
import pytest

from glmbench import neighbours
from glmbench.metrics import unseen
from glmbench.models import KMER, KMER_JS, MINHASH
from glmbench.representations import kmer

KS = [1, 5]


def reference_world(rng, classes=20, per_class=10, dim=8, scale=5.0, spread=0.3):
    centres = rng.normal(0, scale, (classes, dim))
    return {f"class{c}": centres[c] + rng.normal(0, spread, (per_class, dim))
            for c in range(classes)}


def scattered_order(rng, genomes=10, per_genome=8, dim=8, scale=5.0, spread=0.3):
    """Every genome tight, genomes unrelated to one another (and to the reference)."""
    return {f"g{g}": rng.normal(0, scale, dim) + rng.normal(0, spread, (per_genome, dim))
            for g in range(genomes)}


def coherent_order(rng, genomes=10, per_genome=8, dim=8, spread=0.3, offset=40.0):
    """The genomes of one order share a centre, far from every reference class."""
    centre = np.full(dim, offset) + rng.normal(0, 1, dim)
    return {f"g{g}": centre + rng.normal(0, 2 * spread, dim)
            + rng.normal(0, spread, (per_genome, dim)) for g in range(genomes)}


# ── genome level against order level ────────────────────────────────────────

def test_genome_level_looks_perfect_while_order_level_is_chance():
    rng = np.random.default_rng(0)
    reference, order = reference_world(rng), scattered_order(rng)

    genome = unseen.summarise(unseen.precision_by_genome(reference, order, KS, KMER), KS)
    by_order = unseen.summarise(unseen.precision_by_order(reference, order, KS, KMER), KS)
    chance = np.mean(list(unseen.chance_order_precision(reference, order).values()))

    assert genome[5] > 0.95                    # each genome is a tight clump: "perfect"
    assert by_order[5] < 0.5                   # but no genome finds a relative
    assert abs(by_order[5] - chance) < 0.2     # ... which is about what chance gives
    assert by_order[5] < genome[5] - 0.4


def test_order_level_is_high_when_the_genomes_of_an_order_belong_together():
    rng = np.random.default_rng(1)
    reference, order = reference_world(rng), coherent_order(rng)
    by_order = unseen.summarise(unseen.precision_by_order(reference, order, KS, KMER), KS)
    assert by_order[1] > 0.95
    assert by_order[5] > 0.95


def test_order_level_excludes_the_query_genome():
    """If a genome's own fragments were allowed, this world would score ~1, not chance."""
    rng = np.random.default_rng(2)
    reference, order = reference_world(rng), scattered_order(rng, genomes=6, per_genome=20)
    per_genome = unseen.precision_by_order(reference, order, [5], KMER)
    assert len(per_genome) == 6
    assert max(values[5] for values in per_genome.values()) < 0.6


def test_order_level_needs_two_genomes():
    rng = np.random.default_rng(3)
    reference = reference_world(rng)
    assert unseen.precision_by_order(reference, scattered_order(rng, genomes=1), KS, KMER) == {}
    assert np.isnan(unseen.summarise({}, KS)[5])


def test_per_genome_values_are_returned_for_every_genome():
    rng = np.random.default_rng(4)
    reference, order = reference_world(rng), scattered_order(rng, genomes=5)
    for function in (unseen.precision_by_genome, unseen.precision_by_order):
        result = function(reference, order, KS, KMER)
        assert set(result) == set(order)
        assert all(set(values) == set(KS) for values in result.values())


def test_both_levels_run_for_every_kind_of_representation():
    rng = np.random.default_rng(5)

    def dirichlet(centre_seed, rows):
        base = np.random.default_rng(centre_seed).dirichlet(np.ones(16) * .3)
        return rng.dirichlet(base * 200 + 0.01, size=rows)

    reference = {f"c{i}": dirichlet(i, 8) for i in range(6)}
    order = {f"g{i}": dirichlet(100 + i, 5) for i in range(4)}
    for model in (KMER, KMER_JS):
        for function in (unseen.precision_by_genome, unseen.precision_by_order):
            result = function(reference, order, KS, model)
            assert all(0.0 <= v[5] <= 1.0 for v in result.values())

    sketches = lambda: rng.integers(0, 2 ** 31, size=(5, 12), dtype=np.int64).astype(np.uint64)
    reference = {f"c{i}": rng.integers(0, 2 ** 31, size=(8, 12), dtype=np.int64).astype(np.uint64)
                 for i in range(6)}
    order = {f"g{i}": sketches() for i in range(4)}
    for function in (unseen.precision_by_genome, unseen.precision_by_order):
        assert function(reference, order, KS, MINHASH)


# ── own reverse complement against RC proximity ─────────────────────────────

def strand_world(rng, genomes=6, per_genome=8, dim=8):
    reference = reference_world(rng)
    centres = rng.normal(0, 5, (genomes, dim))
    forward = {f"g{g}": centres[g] + rng.normal(0, 0.3, (per_genome, dim))
               for g in range(genomes)}
    return reference, forward, centres


def test_a_perfectly_invariant_model_finds_every_own_rc():
    rng = np.random.default_rng(6)
    reference, forward, _ = strand_world(rng)
    reverse = {f"REVERSECOMPLEMENT{g}": rows.copy() for g, rows in forward.items()}

    hits, similarity = unseen.own_rc(reference, forward, reverse, [1, 5], KMER)
    assert all(values[1] == 1.0 and values[5] == 1.0 for values in hits.values())
    assert all(value == pytest.approx(1.0) for value in similarity.values())


def test_rc_proximity_can_look_perfect_while_the_own_rc_is_never_singled_out():
    """RC fragments of the right genome, but each paired with the wrong forward fragment."""
    rng = np.random.default_rng(7)
    reference, forward, centres = strand_world(rng)
    reverse = {f"REVERSECOMPLEMENT{g}": centres[i] + rng.normal(0, 0.3, rows.shape)
               for i, (g, rows) in enumerate(forward.items())}

    legacy = unseen.summarise(
        unseen.precision_rc_proximity(reference, forward, reverse, [5], KMER), [5])
    hits, similarity = unseen.own_rc(reference, forward, reverse, [1, 5], KMER)
    own = unseen.summarise(hits, [1, 5])

    assert legacy[5] > 0.95                    # "the neighbours are RC fragments of my genome"
    assert own[1] < 0.35                       # ... but mine is not singled out
    assert own[5] < 0.7
    # the mean cosine is no better: fragments of one genome are all close in cosine, so
    # similarity(x, RC(x)) stays high however the pairing is scrambled. hit@k is the test.
    assert np.mean(list(similarity.values())) > 0.9


def test_own_rc_refuses_misaligned_files():
    rng = np.random.default_rng(8)
    reference, forward, _ = strand_world(rng)
    reverse = {f"REVERSECOMPLEMENT{g}": rows[:-1] for g, rows in forward.items()}
    with pytest.raises(ValueError, match="not aligned"):
        unseen.own_rc(reference, forward, reverse, [1], KMER)
    with pytest.raises(ValueError, match="not aligned"):
        unseen.own_rc(reference, forward, {}, [1], KMER)


def test_a_canonical_kmer_profile_is_found_by_its_own_rc():
    """End to end with real sequences: canonical k-mers are strand-invariant, plain are not."""
    rng = np.random.default_rng(9)

    def sequence(n):
        return "".join(rng.choice(list("ACGT"), n))

    def rc(text):
        return text.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    genomes = {f"g{i}": [sequence(600) for _ in range(6)] for i in range(5)}
    reference_seqs = {f"c{i}": [sequence(600) for _ in range(8)] for i in range(10)}

    for canonical, expected in ((True, 1.0), (False, None)):
        profile = lambda texts: np.vstack([kmer.vector(t, 4, canonical) for t in texts])
        reference = {k: profile(v) for k, v in reference_seqs.items()}
        forward = {g: profile(v) for g, v in genomes.items()}
        reverse = {f"REVERSECOMPLEMENT{g}": profile([rc(t) for t in v])
                   for g, v in genomes.items()}
        hits, similarity = unseen.own_rc(reference, forward, reverse, [1], KMER)
        value = unseen.summarise(hits, [1])[1]
        if expected is not None:
            assert value == expected
            assert np.mean(list(similarity.values())) == pytest.approx(1.0)
        else:
            assert value < 1.0                  # plain 4-mer profiles are NOT invariant


# ── neighbour exclusion by index ────────────────────────────────────────────

def test_self_is_removed_by_index_even_when_a_duplicate_ties_with_it():
    rng = np.random.default_rng(10)
    rows = rng.normal(size=(30, 6))
    rows[29] = rows[0]                         # a duplicate of row 0, later in the space
    found = neighbours.knn_excluding(rows[[29]], rows, 3, KMER, self_index=np.array([29]))
    assert 29 not in found[0]                  # never returns itself ...
    assert 0 in found[0]                       # ... but its twin is a legitimate neighbour


def test_old_positional_removal_would_have_returned_the_query_itself():
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


def test_group_exclusion_removes_every_fragment_of_the_query_genome():
    rng = np.random.default_rng(13)
    rows = rng.normal(size=(40, 4))
    groups = np.repeat(np.arange(8), 5)
    queries = np.arange(40)
    found = neighbours.knn_excluding(rows, rows, 6, KMER, self_index=queries,
                                     query_groups=groups, space_groups=groups)
    assert (found >= 0).all()
    assert (groups[found] != groups[:, None]).all()


def test_exclusion_pads_when_candidates_run_out():
    rows = np.random.default_rng(14).normal(size=(6, 3))
    groups = np.array([0, 0, 0, 1, 1, 1])
    found = neighbours.knn_excluding(rows[:1], rows, 5, KMER, self_index=np.array([0]),
                                     query_groups=groups[:1], space_groups=groups)
    assert (found[0, :3] >= 3).all() and (found[0, 3:] == neighbours.PAD).all()
    labels = neighbours.labels_of(np.array(list("abcdef")), found)
    assert labels[0, 4] == neighbours.PAD_LABEL            # a pad never matches a real label


def test_chance_level_formula():
    rng = np.random.default_rng(15)
    reference = {"c": rng.normal(size=(30, 3))}
    order = {"a": rng.normal(size=(10, 3)), "b": rng.normal(size=(20, 3))}
    chance = unseen.chance_order_precision(reference, order)
    assert chance["a"] == pytest.approx(20 / (30 + 20))
    assert chance["b"] == pytest.approx(10 / (30 + 10))


# ── canonical k-mers ────────────────────────────────────────────────────────

def test_canonical_profile_is_identical_for_a_fragment_and_its_reverse_complement():
    rng = np.random.default_rng(16)
    for _ in range(10):
        text = "".join(rng.choice(list("ACGT"), 500))
        reverse = text.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        a, b = kmer.vector(text, 4, canonical=True), kmer.vector(reverse, 4, canonical=True)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(kmer.vector(text, 4), kmer.vector(reverse, 4))
        assert a.shape == (136,) and a.sum() == pytest.approx(1.0)


def test_canonical_width_formula():
    assert kmer.width(4, True) == 136                       # (4^4 + 4^2) / 2
    assert kmer.width(6, True) == (4 ** 6 + 4 ** 3) // 2
    assert kmer.width(4, False) == 256


def test_canonical_counts_agree_with_a_hand_checked_case():
    # AC and GT are reverse complements of each other: one canonical word, count 2
    profile = kmer.vector("ACGT", 2, canonical=True)
    assert profile.sum() == pytest.approx(1.0)
    assert profile.max() == pytest.approx(2 / 3)            # AC, GT -> same word; CG alone
