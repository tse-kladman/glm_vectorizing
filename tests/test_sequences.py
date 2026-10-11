"""Fragmenting: complete pieces only, a seeded random subset, unchanged headers."""

import numpy as np
import pytest

from glmbench import sequences


def genome(length: int, seed: int = 0) -> str:
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), length))


def test_incomplete_tail_is_dropped():
    pieces = sequences.chunk("A" * 4500, 2000, None)
    assert [len(piece) for piece in pieces] == [2000, 2000]          # the 500 bp tail is gone


def test_min_length_is_configurable():
    pieces = sequences.chunk("A" * 4500, 2000, None, min_length=500)
    assert [len(piece) for piece in pieces] == [2000, 2000, 500]


def test_sequence_shorter_than_one_piece_gives_nothing():
    assert sequences.chunk("ACGT" * 100, 2000, 10) == []


def test_random_subset_is_not_the_first_pieces():
    sequence = genome(60 * 2000)
    rng = np.random.default_rng(1)
    starts = []
    sequences.chunk(sequence, 2000, 10, rng=rng, positions=starts)
    assert len(starts) == 10
    assert starts != list(range(0, 10 * 2000, 2000))        # not "the first ten"
    assert starts == sorted(starts)                         # original order kept
    assert len(set(starts)) == 10                           # no piece twice
    assert all(start % 2000 == 0 for start in starts)       # non-overlapping, on the grid


def test_choice_without_a_seed_is_refused():
    with pytest.raises(ValueError, match="seed"):
        sequences.chunk(genome(30 * 2000), 2000, 10)


def test_same_seed_same_fragments_different_seed_different_fragments():
    sequence = genome(60 * 2000)
    a, b, c = ([] for _ in range(3))
    sequences.chunk(sequence, 2000, 10, rng=sequences.genome_rng(42, "g1"), positions=a)
    sequences.chunk(sequence, 2000, 10, rng=sequences.genome_rng(42, "g1"), positions=b)
    sequences.chunk(sequence, 2000, 10, rng=sequences.genome_rng(43, "g1"), positions=c)
    assert a == b
    assert a != c


def test_choice_depends_on_the_genome_not_on_its_neighbours(tmp_path):
    """Adding a genome to the file must not change the fragments of the others."""
    from Bio import SeqIO

    one, two = genome(40 * 2000, 1), genome(40 * 2000, 2)

    def run(records, name):
        source = tmp_path / f"{name}.fa"
        source.write_text("".join(f">{rid}\n{seq}\n" for rid, seq in records))
        out = tmp_path / f"{name}_preproc.fa"
        sequences.preprocess(source, out, seed=7)
        return {record.id: str(record.seq) for record in SeqIO.parse(str(out), "fasta")}

    alone = run([("g2", two)], "alone")
    both = run([("g1", one), ("g2", two)], "both")
    assert {k: v for k, v in both.items() if k.startswith("g2_")} == alone


def test_header_format_and_sidecar(tmp_path):
    from Bio import SeqIO

    source = tmp_path / "order.fa"
    source.write_text(f">gA\n{genome(25 * 2000 + 700)}\n>short\n{genome(900)}\n")
    out = tmp_path / "order_preproc.fa"
    sequences.preprocess(source, out, seed=3)

    records = list(SeqIO.parse(str(out), "fasta"))
    assert [record.id for record in records] == [f"gA_{i}" for i in range(10)]
    assert all(len(record.seq) == 2000 for record in records)          # no stubs, no tail
    assert all(sequences.label_of(record.id) == "gA" for record in records)
    sidecar = out.with_suffix(".chunks.tsv").read_text().splitlines()
    assert sidecar[0].startswith("# seed=3")
    assert len(sidecar) == 2 + 10                                       # comment, header, rows


def test_reverse_complement_file_is_index_aligned(tmp_path):
    from Bio import SeqIO

    source = tmp_path / "order.fa"
    source.write_text(f">gA\n{genome(30 * 2000)}\n")
    forward = sequences.preprocess(source, tmp_path / "a_preproc.fa", seed=1)
    reverse = sequences.reverse_complement(forward, tmp_path / "rc.fa")

    fwd = list(SeqIO.parse(str(forward), "fasta"))
    rc = list(SeqIO.parse(str(reverse), "fasta"))
    assert len(fwd) == len(rc)
    for f, r in zip(fwd, rc):
        assert r.id == f"REVERSECOMPLEMENT{f.id}"
        assert str(r.seq) == str(f.seq.reverse_complement())
