"""k-mer frequency profiles -- the classical baseline, at any word length.

One fragment becomes a vector of length 4^k holding the frequency of every word,
normalised to sum to one.  That normalisation is what makes the profile a
probability distribution and so lets Jensen-Shannon be used on it.

With ``canonical=True`` every word is first replaced by ``min(word, reverse
complement of word)``, and the profile has one entry per canonical word (136 for
k = 4, ``(4^k + 4^(k/2)) / 2`` in general for even k).  A fragment and its reverse
complement then have *identical* profiles by construction -- the RC-invariant
baseline for the learned models.

Words containing anything other than A/C/G/T are skipped.  A fragment with no
valid word at all yields a zero vector, which the metrics treat as an empty
fragment rather than an error.
"""

from collections import defaultdict
from functools import lru_cache

import numpy as np

BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def _base_codes(sequence: str) -> np.ndarray:
    """Sequence -> per-letter codes, with -1 marking anything that is not ACGT."""
    codes = np.full(len(sequence), -1, dtype=np.int64)
    raw = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    for base, value in BASE_CODE.items():
        codes[raw == ord(base)] = value
    return codes


@lru_cache(maxsize=None)
def canonical_table(k: int):
    """(map from word code to canonical column, number of canonical words)."""
    codes = np.arange(4 ** k, dtype=np.int64)
    reverse = np.zeros_like(codes)
    rest = codes.copy()
    for _ in range(k):                      # read the word backwards, complemented
        reverse = reverse * 4 + (3 - rest % 4)
        rest //= 4
    canonical = np.minimum(codes, reverse)
    unique, column = np.unique(canonical, return_inverse=True)
    return column.astype(np.int64), len(unique)


def width(k: int, canonical: bool = False) -> int:
    """Length of a profile."""
    return canonical_table(k)[1] if canonical else 4 ** k


def vector(sequence: str, k: int, canonical: bool = False) -> np.ndarray:
    """Normalised k-mer frequency profile of one fragment (length 4^k, or the number
    of canonical words)."""
    profile = np.zeros(width(k, canonical), dtype=np.float32)
    codes = _base_codes(sequence)
    if len(codes) < k:
        return profile

    windows = np.lib.stride_tricks.sliding_window_view(codes, k)
    windows = windows[(windows >= 0).all(axis=1)]
    if not len(windows):
        return profile

    place_values = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    words = windows @ place_values
    if canonical:
        words = canonical_table(k)[0][words]
    np.add.at(profile, words, 1.0)
    total = profile.sum()
    return profile / total if total else profile


def from_fasta(path, k: int, canonical: bool = False) -> dict:
    """FASTA -> {label: matrix of profiles}, one row per fragment."""
    from Bio import SeqIO                     # only needed when reading sequences

    groups = defaultdict(list)
    for record in SeqIO.parse(str(path), "fasta"):
        groups[record.id.rsplit("_", 1)[0]].append(vector(str(record.seq), k, canonical))
    return {label: np.vstack(rows) for label, rows in groups.items()}
