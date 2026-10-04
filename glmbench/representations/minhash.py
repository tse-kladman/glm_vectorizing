"""MinHash sketches -- a fragment as the *set* of its k-mers.

Unlike every other representation here, MinHash does not describe a fragment by
how often each word occurs but by which words occur at all, and compares two
fragments by the Jaccard index of those sets.

Each k-mer is packed two bits per letter, folded to its canonical form
``min(kmer, revcomp(kmer))`` -- which makes a sketch identical for a fragment and
its reverse complement, by construction -- mixed with a 64-bit hash, and passed
through s independent universal hash functions ``h_i(x) = (a_i*x + b_i) mod p``.
The sketch is the vector of s minima.  The probability that two sets share a
minimum under one hash function equals their Jaccard index, so the fraction of
agreeing positions estimates it without bias.

All files of one run must share ``seed``: sketches are only comparable under the
same hash functions.
"""

from collections import defaultdict

import numpy as np

MERSENNE = np.uint64((1 << 31) - 1)
BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
DEFAULT_SEED = 42


def _canonical_codes(sequence: str, k: int) -> np.ndarray:
    """The distinct canonical k-mers of a fragment, as packed integers."""
    codes = np.full(len(sequence), -1, dtype=np.int64)
    raw = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    for base, value in BASE_CODE.items():
        codes[raw == ord(base)] = value
    if len(codes) < k:
        return np.empty(0, dtype=np.uint64)

    windows = np.lib.stride_tricks.sliding_window_view(codes, k)
    windows = windows[(windows >= 0).all(axis=1)]
    if not len(windows):
        return np.empty(0, dtype=np.uint64)

    shifts = 2 * np.arange(k - 1, -1, -1, dtype=np.int64)
    forward = (windows.astype(np.int64) << shifts).sum(axis=1)
    # reverse complement: complement each letter (3 - code) and read backwards
    reverse = ((3 - windows[:, ::-1]).astype(np.int64) << shifts).sum(axis=1)
    return np.unique(np.minimum(forward, reverse).astype(np.uint64))


def _mix64(values: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, so neighbouring k-mers do not get neighbouring hashes."""
    with np.errstate(over="ignore"):
        x = values.astype(np.uint64)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
    return x & MERSENNE


def hash_functions(sketch_size: int, seed: int = DEFAULT_SEED):
    """s pairs (a, b) for universal hashing modulo a Mersenne prime."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(MERSENNE), size=sketch_size, dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, int(MERSENNE), size=sketch_size, dtype=np.int64).astype(np.uint64)
    return a, b


def sketch(sequence: str, k: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash sketch of one fragment: s minima over its canonical k-mers."""
    codes = _canonical_codes(sequence, k)
    if not len(codes):
        return np.full(len(a), MERSENNE, dtype=np.uint64)
    mixed = _mix64(codes)
    with np.errstate(over="ignore"):
        hashed = (a[:, None] * mixed[None, :] + b[:, None]) % MERSENNE
    return hashed.min(axis=1).astype(np.uint64)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Estimated Jaccard index of two sketches: the fraction of agreeing positions."""
    return float(np.mean(left == right))


def from_fasta(path, k: int, sketch_size: int, seed: int = DEFAULT_SEED) -> dict:
    """FASTA -> {label: matrix of sketches}, one row per fragment."""
    from Bio import SeqIO

    a, b = hash_functions(sketch_size, seed)
    groups = defaultdict(list)
    for record in SeqIO.parse(str(path), "fasta"):
        groups[record.id.rsplit("_", 1)[0]].append(sketch(str(record.seq), k, a, b))
    return {label: np.vstack(rows) for label, rows in groups.items()}
