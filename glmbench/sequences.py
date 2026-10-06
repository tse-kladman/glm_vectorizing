"""From whole genomes to the fragments everything else works on.

Each input genome is cut into fixed-length chunks and a reverse-complement copy of
every chunk is written alongside it.  Both are needed: the forward fragments are
what the metrics score, and the reverse-complement ones are what reveals whether a
model reads the two strands of the same sequence as the same thing.

Fragment headers carry the class in the name -- ``<label>_<index>`` -- which is how
every representation recovers labels without a separate annotation file.
"""

from pathlib import Path

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


def chunk(sequence: str, size: int, max_chunks: int | None):
    pieces = [sequence[start:start + size] for start in range(0, len(sequence), size)]
    pieces = [piece for piece in pieces if piece]
    return pieces[:max_chunks] if max_chunks is not None else pieces


def preprocess(input_path: Path, output_path: Path, size=config.CHUNK_SIZE,
               max_chunks=config.MAX_CHUNKS) -> Path:
    """Cut a FASTA into chunks named ``<original id>_<index>``."""
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    records = []
    for record in SeqIO.parse(str(input_path), "fasta"):
        for index, piece in enumerate(chunk(str(record.seq).upper(), size, max_chunks)):
            records.append(SeqRecord(Seq(piece), id=f"{record.id}_{index}", description=""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(output_path), "fasta")
    print(f"  chunks  -> {config.relative(output_path)}  ({len(records)} fragments)")
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
            max_chunks=config.MAX_CHUNKS, force=False) -> dict:
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
            preprocess(path, forward_path, size, max_chunks)
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
