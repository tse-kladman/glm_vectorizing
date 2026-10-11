"""Building the reference samples: which classes qualify, and independent draws."""

import numpy as np
import pandas as pd
import pytest

from glmbench import samples, sequences


def make_vmr(species_per_class: dict, additional: int = 0):
    """A VMR-shaped table: one exemplar isolate per species (plus extra isolates)."""
    rows, serial = [], 0
    for cls, n_species in species_per_class.items():
        for s in range(n_species):
            for extra in range(1 + additional):
                serial += 1
                rows.append({"Isolate ID": serial, "Class": cls,
                             "Species": f"{cls} species {s}",
                             "Exemplar or additional isolate": "E" if extra == 0 else "A",
                             "Virus GENBANK accession": f"AB{100000 + serial}"})
    return pd.DataFrame(rows)


def write_genomes(directory, vmr, length=30_000, seed=0):
    rng = np.random.default_rng(seed)
    with open(directory / "genomes.fa", "w") as handle:
        for accession in vmr["Virus GENBANK accession"]:
            handle.write(f">{accession}.1 test genome\n{''.join(rng.choice(list('ACGT'), length))}\n")
    return directory


@pytest.fixture
def world(tmp_path):
    vmr = make_vmr({"Alphaviricetes": 14, "Betaviricetes": 12, "Gammaviricetes": 10,
                    "Deltaviricetes": 3, "Epsilonviricetes": 25})
    fasta = tmp_path / "fasta"
    fasta.mkdir()
    write_genomes(fasta, vmr)
    return vmr, samples.FastaIndex.from_dir(fasta)


# ── which classes qualify ───────────────────────────────────────────────────

def test_qualifying_classes_need_MORE_than_ten_species():
    vmr = make_vmr({"Alphaviricetes": 14, "Betaviricetes": 12, "Gammaviricetes": 10,
                    "Deltaviricetes": 3, "Epsilonviricetes": 25}, additional=2)
    classes = samples.qualifying_classes(vmr)
    # 10 species does not qualify; extra isolates of a species do not add species
    assert list(classes.index) == ["Alphaviricetes", "Betaviricetes", "Epsilonviricetes"]
    assert classes["Epsilonviricetes"] == 25


def test_report_warns_when_the_count_differs_from_the_article(capsys):
    vmr = make_vmr({"A": 11, "B": 12})
    samples.report_classes(vmr, expected=39)
    out = capsys.readouterr().out
    assert "2 of 2" in out and "WARNING" in out and "39" in out


def test_accessions_are_read_from_multi_segment_cells():
    cell = "RNA1: AB009656; RNA2: AB010376.2; RNA3: NC_001234"
    assert samples.accessions_of(cell) == ["AB009656", "AB010376", "NC_001234"]
    assert samples.accessions_of(float("nan")) == []


def test_the_real_vmr_has_the_documented_class_count():
    """Regression for the number printed in the README (and the discrepancy with the paper)."""
    from glmbench import config

    path = next(iter(sorted(config.ICTV_DIR.glob("VMR*.xlsx"))), None)
    if path is None:
        pytest.skip("VMR spreadsheet not present")
    classes = samples.qualifying_classes(samples.load_vmr(path))
    assert len(classes) == 36


# ── drawing ─────────────────────────────────────────────────────────────────

def test_a_sample_has_the_documented_structure(world, tmp_path):
    vmr, index = world
    paths = samples.build_samples(vmr, index, tmp_path / "ref", n_ref=1, seed=42)
    from Bio import SeqIO

    records = list(SeqIO.parse(str(paths[0]), "fasta"))
    labels = {sequences.label_of(record.id) for record in records}
    assert labels == {"Alphaviricetes", "Betaviricetes", "Epsilonviricetes"}   # >10 species only
    assert all(len(record.seq) == 2000 for record in records)
    per_class = {label: sum(sequences.label_of(r.id) == label for r in records) for label in labels}
    assert all(count == 10 * 10 for count in per_class.values())              # 10 genomes x 10 fragments

    manifest = pd.read_csv(tmp_path / "ref" / "manifest_sample_0.tsv", sep="\t", comment="#")
    assert (manifest.groupby("class")["isolate"].nunique() == 10).all()
    # fragments of one genome never overlap
    for _, group in manifest.groupby("accession"):
        starts = sorted(group["start"])
        assert all(b - a >= 2000 for a, b in zip(starts, starts[1:]))


def test_draws_are_independent_and_reproducible(world, tmp_path):
    vmr, index = world
    first = samples.build_samples(vmr, index, tmp_path / "a", n_ref=4, seed=42)
    again = samples.build_samples(vmr, index, tmp_path / "b", n_ref=4, seed=42)
    other = samples.build_samples(vmr, index, tmp_path / "c", n_ref=1, seed=1000)

    def genomes(directory, i):
        table = pd.read_csv(directory / f"manifest_sample_{i}.tsv", sep="\t", comment="#")
        return frozenset(table["isolate"])

    def text(path):
        return path.read_text()

    # reproducible: the same seed gives byte-identical samples
    assert all(text(a) == text(b) for a, b in zip(first, again))
    # independent: different draws choose different genomes (the old random_state=42
    # for every draw would have given the same set each time)
    drawn = [genomes(tmp_path / "a", i) for i in range(4)]
    assert len(set(drawn)) == 4
    assert len(set(drawn[0]) & set(drawn[1])) < len(drawn[0])
    # and a different seed gives different samples again
    assert text(other[0]) != text(first[0])
    # draw i of seed s is exactly draw 0 of seed s + i
    solo = samples.build_samples(vmr, index, tmp_path / "d", n_ref=1, seed=42 + 2)
    assert text(solo[0]) == text(first[2])


def test_fewer_available_genomes_than_requested_is_reported(world, tmp_path, capsys):
    vmr, index = world
    samples.build_samples(vmr, index, tmp_path / "e", n_ref=1, seed=1, n_genomes=20)
    assert "supplied" in capsys.readouterr().out


def test_genomes_without_a_complete_fragment_are_replaced(tmp_path):
    vmr = make_vmr({"A": 12})
    fasta = tmp_path / "fasta"
    fasta.mkdir()
    rng = np.random.default_rng(0)
    with open(fasta / "g.fa", "w") as handle:
        for i, accession in enumerate(vmr["Virus GENBANK accession"]):
            length = 500 if i < 4 else 8000                     # four genomes too short
            handle.write(f">{accession}\n{''.join(rng.choice(list('ACGT'), length))}\n")
    index = samples.FastaIndex.from_dir(fasta)
    paths = samples.build_samples(vmr, index, tmp_path / "ref", n_ref=1, seed=1, n_genomes=8)
    manifest = pd.read_csv(tmp_path / "ref" / "manifest_sample_0.tsv", sep="\t", comment="#")
    assert manifest["isolate"].nunique() == 8
    assert (manifest["start"] % 2000 == 0).all()


def test_existing_samples_are_not_overwritten_without_force(world, tmp_path):
    vmr, index = world
    paths = samples.build_samples(vmr, index, tmp_path / "ref", n_ref=1, seed=1)
    paths[0].write_text(">keep_0\nACGT\n")
    samples.build_samples(vmr, index, tmp_path / "ref", n_ref=1, seed=1)
    assert paths[0].read_text().startswith(">keep_0")
    samples.build_samples(vmr, index, tmp_path / "ref", n_ref=1, seed=1, force=True)
    assert not paths[0].read_text().startswith(">keep_0")


def test_missing_genomes_stop_the_build(tmp_path):
    vmr = make_vmr({"A": 12})
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no genome"):
        samples.build_samples(vmr, samples.FastaIndex.from_dir(empty), tmp_path / "ref", n_ref=1)
