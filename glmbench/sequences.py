"""From whole genomes to the fragments everything else works on.

Each input genome is cut into fixed-length chunks (incomplete tail pieces are
dropped; if there are more than ``max_chunks``, a seeded random subset is kept) and a
reverse-complement copy of every chunk is written alongside it.  Both are needed: the forward fragments are
what the metrics score, and the reverse-complement ones are what reveals whether a
model reads the two strands of the same sequence as the same thing.

Fragment headers carry the class in the name -- ``<label>_<index>`` -- which is how
every representation recovers labels without a separate annotation file.
"""

import zlib
from pathlib import Path

import numpy as np

from . import config

FASTA_SUFFIXES = {".fa", ".fasta", ".txt"}
RC_PREFIX = "REVERSECOMPLEMENT"


def label_of(fragment_id: str) -> str:
    """``<label>_<index>`` -> ``<label>``; the convention shared by all extractors."""
    return fragment_id.rsplit("_", 1)[0]


def rc_name(name: str) -> str:
    """Name of the reverse-complement file that belongs to a preprocessed file."""
    return f"{RC_PREFIX}{name.replace('_preproc', '')}_preproc"


def discover(input_dir: Path | None = None) -> list[Path]:
    directory = Path(input_dir or config.UNSEEN_DIR)
    if not directory.is_dir():
        return []
    return [path for path in sorted(directory.iterdir())
            if path.suffix.lower() in FASTA_SUFFIXES]


def genome_rng(seed: int, record_id: str) -> np.random.Generator:
    """A generator that depends only on (seed, genome id).

    Seeding per genome, from a stable hash of its id, makes the choice of fragments
    independent of the order genomes are read in and of how many other genomes are in
    the file: re-running with one genome added or removed changes nothing for the
    others.  (``hash()`` would not do: it is salted per process.)
    """
    return np.random.default_rng([int(seed), zlib.crc32(record_id.encode("utf8"))])


def chunk(sequence: str, size: int, max_chunks: int | None,
          min_length: int | None = None, rng: np.random.Generator | None = None,
          positions: list | None = None) -> list:
    """Cut a sequence into non-overlapping pieces and keep at most ``max_chunks``.

    * A piece shorter than ``min_length`` (default: ``size``) is dropped -- the tail
      of a genome is almost never a full fragment, and a 300 bp stub is not
      comparable to a 2 kb fragment.
    * When more than ``max_chunks`` pieces remain, ``max_chunks`` of them are chosen
      **at random** (without replacement, from ``rng``) and returned in their
      original order.  Taking the first ones instead would sample only the 5' end of
      every genome.  Asking for a random choice without a generator is an error:
      every random choice in this package takes an explicit seed.

    If ``positions`` is a list, the 0-based start of every kept piece is appended.
    """
    min_length = size if min_length is None else min_length
    pieces = [(start, sequence[start:start + size])
              for start in range(0, len(sequence), size)]
    pieces = [(start, piece) for start, piece in pieces if len(piece) >= max(1, min_length)]

    if max_chunks is not None and len(pieces) > max_chunks:
        if rng is None:
            raise ValueError("choosing fragments at random needs an explicit rng/seed")
        keep = np.sort(rng.choice(len(pieces), size=max_chunks, replace=False))
        pieces = [pieces[index] for index in keep]

    if positions is not None:
        positions.extend(start for start, _ in pieces)
    return [piece for _, piece in pieces]


def preprocess(input_path: Path, output_path: Path, size=config.CHUNK_SIZE,
               max_chunks=config.MAX_CHUNKS, seed: int = config.SEED,
               min_length: int | None = None) -> Path:
    """Cut a FASTA into chunks named ``<original id>_<index>``.

    The index counts the kept fragments 0, 1, 2, ... in genome order (so the header
    format is the same as ever); where each one came from is written next to the
    FASTA, in ``<output>.chunks.tsv``, together with the seed.
    """
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    records, origins, too_short = [], [], []
    for record in SeqIO.parse(str(input_path), "fasta"):
        starts = []
        pieces = chunk(str(record.seq).upper(), size, max_chunks, min_length,
                       genome_rng(seed, record.id), starts)
        if not pieces:
            too_short.append(record.id)
        for index, (piece, start) in enumerate(zip(pieces, starts)):
            records.append(SeqRecord(Seq(piece), id=f"{record.id}_{index}", description=""))
            origins.append((f"{record.id}_{index}", record.id, start))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(output_path), "fasta")
    sidecar = output_path.with_suffix(".chunks.tsv")
    with open(sidecar, "w", encoding="utf8") as handle:
        handle.write(f"# seed={seed} size={size} max_chunks={max_chunks} "
                     f"min_length={size if min_length is None else min_length}\n")
        handle.write("fragment\tgenome\tstart\n")
        handle.writelines(f"{name}\t{genome}\t{start}\n" for name, genome, start in origins)

    print(f"  chunks  -> {config.relative(output_path)}  ({len(records)} fragments)")
    if too_short:
        print(f"  WARNING: {len(too_short)} sequence(s) shorter than one fragment were "
              f"dropped: {', '.join(too_short[:5])}{' ...' if len(too_short) > 5 else ''}")
    return output_path


def reverse_complement(preprocessed: Path, output_path: Path) -> Path:
    """Write the reverse complement of every fragment, prefixing its id."""
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord

    records = [SeqRecord(record.seq.reverse_complement(),
                         id=f"{RC_PREFIX}{record.id}", description="")
               for record in SeqIO.parse(str(preprocessed), "fasta")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(output_path), "fasta")
    print(f"  RC      -> {config.relative(output_path)}  ({len(records)} fragments)")
    return output_path


def prepare(input_dir=None, preproc_dir=None, size=config.CHUNK_SIZE,
            max_chunks=config.MAX_CHUNKS, force=False, seed: int = config.SEED,
            min_length: int | None = None) -> dict:
    """Chunk every input genome and write its reverse complement.

    Returns ``{label: (forward name, rc name)}`` -- the file names the rest of the
    pipeline refers to.
    """
    preproc_dir = Path(preproc_dir or config.PREPROC_DIR)
    prepared = {}

    for path in discover(input_dir):
        label = path.stem
        forward = f"{label}_preproc"
        reverse = rc_name(forward)
        forward_path = preproc_dir / f"{forward}.fa"
        reverse_path = preproc_dir / f"{reverse}.fa"

        if force or not forward_path.exists():
            preprocess(path, forward_path, size, max_chunks, seed, min_length)
        if force or not reverse_path.exists():
            reverse_complement(forward_path, reverse_path)
        prepared[label] = (forward, reverse)

    return prepared


def existing(preproc_dir=None) -> dict:
    """The already-prepared files, for stages that run without redoing the prep."""
    directory = Path(preproc_dir or config.PREPROC_DIR)
    prepared = {}
    if not directory.is_dir():
        return prepared
    for path in sorted(directory.glob("*.fa")):
        name = path.stem
        if name.startswith(RC_PREFIX):
            continue
        label = name.replace("_preproc", "")
        reverse = rc_name(name)
        prepared[label] = (name, reverse if (directory / f"{reverse}.fa").exists() else None)
    return prepared
