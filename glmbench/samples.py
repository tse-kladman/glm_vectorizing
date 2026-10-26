"""Building the 50 reference samples from the ICTV VMR and a directory of genomes.

A reference sample is a small, labelled set of fragments of KNOWN viruses in which the
unseen orders are later embedded.  Each of the ``n_ref`` samples is an independent draw:

    1. classes   every taxonomic *Class* of the VMR with MORE than 10 species
                 (``config.REFERENCE_MIN_SPECIES``);
    2. genomes   10 genomes per class, drawn at random (one isolate per species: the
                 exemplar isolate, unless ``include_additional``);
    3. fragments up to 10 random non-overlapping 2 kb fragments per genome (incomplete
                 tails are dropped; a genome with no complete fragment is replaced by
                 the next one in the same random order);
    4. output    ``sample_<i>.fa`` with headers ``<Class>_<n>`` -- the class is the
                 label, as for every other FASTA in the pipeline -- and
                 ``manifest_sample_<i>.tsv`` recording which genome and which offset
                 every fragment came from.

Draw ``i`` uses ``numpy.random.default_rng(seed + i)``, so each draw is reproducible
alone.  The draws come from one pool of genomes, so they overlap: they are 50 draws,
not 50 independent sets of viruses.

The genomes themselves are not part of the repository.  Point ``fasta_dir`` at a
directory of FASTA files whose record ids start with a GenBank accession (a trailing
``.version`` is ignored), e.g. the output of NCBI Datasets / Entrez for the VMR
accessions, and optionally give an ``accession_map`` (TSV: ``accession<TAB>file``) for
accessions that live in files elsewhere.  A genome made of several segments
(``RNA1: AB000001; RNA2: AB000002``) is used only if every segment is available, and
its fragments are drawn from all segments together.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, sequences

VMR_SHEET = "VMR MSL40"
ACCESSION = re.compile(r"[A-Z]{1,2}_?\d{5,9}")
FASTA_GLOBS = ("*.fa", "*.fasta", "*.fna", "*.txt", "*.fas")


# ── which classes qualify ───────────────────────────────────────────────────

def load_vmr(path: Path | None = None, sheet: str = VMR_SHEET) -> pd.DataFrame:
    path = Path(path) if path else next(iter(sorted(config.ICTV_DIR.glob("VMR*.xlsx"))), None)
    if path is None or not Path(path).exists():
        raise SystemExit(f"no VMR spreadsheet found in {config.relative(config.ICTV_DIR)}; "
                         f"pass --vmr")
    return pd.read_excel(path, sheet_name=sheet)


def species_per_class(vmr: pd.DataFrame, rank: str = "Class") -> pd.Series:
    """Number of distinct species in every taxon of ``rank`` (unassigned rows ignored)."""
    return (vmr.dropna(subset=[rank, "Species"]).groupby(rank)["Species"]
            .nunique().sort_values(ascending=False))


def qualifying_classes(vmr: pd.DataFrame, min_species: int = config.REFERENCE_MIN_SPECIES,
                       rank: str = "Class") -> pd.Series:
    """Classes with MORE than ``min_species`` species, with their species counts."""
    counts = species_per_class(vmr, rank)
    return counts[counts > min_species].sort_index()


def report_classes(vmr: pd.DataFrame, expected: int | None = None, rank: str = "Class",
                   min_species: int = config.REFERENCE_MIN_SPECIES) -> pd.Series:
    """Print how many classes qualify and which; warn if it differs from ``expected``."""
    classes = qualifying_classes(vmr, min_species, rank)
    total = species_per_class(vmr, rank)
    print(f"{len(classes)} of {len(total)} {rank.lower()}es have more than {min_species} "
          f"species in this VMR:")
    for name, count in classes.items():
        print(f"  {name:<28} {count:>5} species")
    if expected is not None and len(classes) != expected:
        print(f"\n  WARNING: {len(classes)} qualifying classes, but {expected} were "
              f"expected (the article's figure). The VMR release, the rank or the "
              f"threshold differs from the one used for the article; check before "
              f"relying on the sample composition.")
    return classes


# ── where the genomes are ───────────────────────────────────────────────────

def accessions_of(text) -> list:
    """All accessions in a VMR cell, e.g. ``"RNA1: AB009656; RNA2: AB010376"``."""
    if pd.isna(text):
        return []
    return [match.split(".")[0] for match in ACCESSION.findall(str(text))]


class FastaIndex:
    """Random access to genomes by accession, over a directory of FASTA files."""

    def __init__(self):
        self._where = {}                     # accession -> (file, record id)
        self._handles = {}

    @classmethod
    def from_dir(cls, fasta_dir=None, accession_map=None):
        index = cls()
        files = []
        if fasta_dir:
            directory = Path(fasta_dir)
            if not directory.is_dir():
                raise SystemExit(f"--fasta-dir {directory} is not a directory")
            for pattern in FASTA_GLOBS:
                files.extend(sorted(directory.glob(pattern)))
        for file in files:
            index._scan(file)
        if accession_map:
            for line in Path(accession_map).read_text(encoding="utf8").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                accession, file = line.split("\t")[:2]
                index._scan(Path(file), only=accession.split(".")[0])
        return index

    def _scan(self, file: Path, only=None):
        """Register the accessions of one FASTA file (ids only; sequences stay on disk)."""
        from Bio import SeqIO
        handle = self._handles.get(file)
        if handle is None:
            handle = self._handles[file] = SeqIO.index(str(file), "fasta")
        for record_id in handle.keys():
            accession = record_id.split()[0].split(".")[0]
            if only and accession != only:
                continue
            self._where.setdefault(accession, (file, record_id))

    def __contains__(self, accession):
        return accession in self._where

    def __len__(self):
        return len(self._where)

    def fetch(self, accession: str) -> str:
        file, record_id = self._where[accession]
        return str(self._handles[file][record_id].seq).upper()


def candidate_genomes(vmr: pd.DataFrame, classes, index: FastaIndex | None,
                      include_additional: bool = False) -> dict:
    """``{class: [genome, ...]}`` -- the genomes a class can contribute.

    A genome is a VMR isolate (exemplar only, by default) whose accession(s) are all
    available in ``index``.  With ``index=None`` every isolate that has an accession
    counts (planning only).  Genomes are ordered by isolate id so a draw depends on the
    seed alone.
    """
    frame = vmr[vmr["Class"].isin(list(classes))]
    if not include_additional and "Exemplar or additional isolate" in frame:
        frame = frame[frame["Exemplar or additional isolate"] == "E"]
    out = {name: [] for name in classes}
    for row in frame.sort_values(["Class", "Isolate ID"]).to_dict("records"):
        accessions = accessions_of(row.get("Virus GENBANK accession"))
        if not accessions or (index is not None and not all(a in index for a in accessions)):
            continue
        out[row["Class"]].append({"class": row["Class"], "species": row["Species"],
                                  "isolate": row["Isolate ID"], "accessions": accessions})
    return out


# ── drawing ─────────────────────────────────────────────────────────────────

def _fragments_of(genome: dict, index: FastaIndex, size: int, n_chunks: int,
                  rng: np.random.Generator):
    """Up to ``n_chunks`` random non-overlapping fragments from all segments together."""
    pool = []
    for accession in genome["accessions"]:
        sequence = index.fetch(accession)
        starts = []
        pieces = sequences.chunk(sequence, size, None, positions=starts)
        pool.extend((accession, start, piece) for start, piece in zip(starts, pieces))
    if not pool:
        return []
    if len(pool) > n_chunks:
        keep = np.sort(rng.choice(len(pool), size=n_chunks, replace=False))
        pool = [pool[i] for i in keep]
    return pool


def draw_sample(candidates: dict, index: FastaIndex, seed: int,
                n_genomes: int = config.REFERENCE_GENOMES_PER_CLASS,
                n_chunks: int = config.REFERENCE_CHUNKS_PER_GENOME,
                size: int = config.CHUNK_SIZE):
    """One reference sample: ``(records, manifest)`` for a draw seeded with ``seed``."""
    rng = np.random.default_rng(seed)
    records, manifest = [], []

    for label in sorted(candidates):
        pool = candidates[label]
        taken, counter = 0, 0
        for position in rng.permutation(len(pool)):        # random order; skip the unusable
            if taken == n_genomes:
                break
            genome = pool[position]
            pieces = _fragments_of(genome, index, size, n_chunks, rng)
            if not pieces:
                continue
            taken += 1
            for accession, start, piece in pieces:
                name = f"{label}_{counter}"
                counter += 1
                records.append((name, piece))
                manifest.append((name, label, genome["species"], genome["isolate"],
                                 accession, start))
        if taken < n_genomes:
            print(f"  WARNING: class {label} supplied {taken} genomes, not {n_genomes}")
    return records, manifest


def write_sample(records, manifest, directory: Path, i: int, seed: int) -> Path:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sample_{i}.fa"
    SeqIO.write([SeqRecord(Seq(sequence), id=name, description="")
                 for name, sequence in records], str(path), "fasta")
    with open(directory / f"manifest_sample_{i}.tsv", "w", encoding="utf8") as handle:
        handle.write(f"# draw={i} seed={seed}\n")
        handle.write("fragment\tclass\tspecies\tisolate\taccession\tstart\n")
        handle.writelines("\t".join(map(str, row)) + "\n" for row in manifest)
    return path


def build_samples(vmr: pd.DataFrame, index: FastaIndex, directory: Path | None = None,
                  n_ref: int = config.N_REFERENCE, seed: int = config.SEED,
                  n_genomes: int = config.REFERENCE_GENOMES_PER_CLASS,
                  n_chunks: int = config.REFERENCE_CHUNKS_PER_GENOME,
                  size: int = config.CHUNK_SIZE, include_additional: bool = False,
                  min_species: int = config.REFERENCE_MIN_SPECIES,
                  force: bool = False, expected_classes: int | None = None) -> list:
    """Draw and write ``n_ref`` independent samples; returns the written paths."""
    directory = Path(directory or config.REFERENCE_DIR)
    classes = report_classes(vmr, expected_classes, min_species=min_species)
    candidates = candidate_genomes(vmr, list(classes.index), index, include_additional)

    usable = {label: pool for label, pool in candidates.items() if pool}
    for label in set(candidates) - set(usable):
        print(f"  WARNING: no genome of class {label} is available in the FASTA directory")
    if not usable:
        raise SystemExit("no genome of any qualifying class was found; check --fasta-dir "
                         "and the accession column of the VMR")

    paths = []
    for i in range(n_ref):
        path = directory / f"sample_{i}.fa"
        if path.exists() and not force:
            paths.append(path)
            continue
        records, manifest = draw_sample(usable, index, seed + i, n_genomes, n_chunks, size)
        paths.append(write_sample(records, manifest, directory, i, seed + i))
        print(f"  sample_{i}: {len(records)} fragments from {len(set(m[3] for m in manifest))} "
              f"genomes in {len(set(m[1] for m in manifest))} classes")
    return paths
