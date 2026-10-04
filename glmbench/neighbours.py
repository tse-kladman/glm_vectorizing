"""The nearest-neighbour core shared by every metric.

Every metric in this study is some variation on one question: *of the k nearest
neighbours of a fragment, how many are the kind we hoped for?*  Only two things
change between models -- how the matrix is transformed, and how distance is
measured -- and both come from the model registry, so this module is the only
place that implements neighbour search.

The distances that are not in scikit-learn are computed blockwise here:

    jensenshannon   JSD(P,Q) = H((P+Q)/2) - (H(P)+H(Q))/2, so the row entropies are
                    computed once and only the entropy of the mixture is pairwise.
                    Ranking uses JSD directly: the square root that makes it a
                    metric is monotone and cannot reorder neighbours.
    minhash         the fraction of sketch positions where two sketches agree is an
                    unbiased estimate of the Jaccard index, so 1 - that fraction
                    orders neighbours by set dissimilarity.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config
from .models import Model


# ── transforms ──────────────────────────────────────────────────────────────

def _check_distribution(matrix: np.ndarray, model: Model) -> None:
    """Jensen-Shannon needs rows that really are distributions: x >= 0, sum(x) = 1.

    A row of zeros is a fragment with no valid k-mer at all; those occur in the
    data and are left alone rather than treated as an error.
    """
    if not matrix.size:
        return
    if matrix.min() < -1e-9:
        raise ValueError(f"{model.key} needs non-negative vectors, found {matrix.min():.4g}")
    sums = matrix[::max(1, len(matrix) // 32)].sum(axis=1)
    bad = sums[(np.abs(sums - 1.0) > 1e-6) & (np.abs(sums) > 1e-9)]
    if bad.size:
        raise ValueError(f"{model.key} needs distributions (sum = 1), found {bad[0]:.4g}")


def _check_sketch(matrix: np.ndarray, model: Model) -> None:
    if matrix.size and not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError(f"{model.key} needs integer MinHash sketches, got {matrix.dtype}")


def transform(matrix: np.ndarray, model: Model) -> np.ndarray:
    """Prepare a stacked matrix for distance computation, as the model requires."""
    if model.kind == "sketch":
        _check_sketch(matrix, model)
        return matrix                      # a sketch is not a vector; never scaled
    if model.kind == "distribution" and model.metric == "jensenshannon":
        _check_distribution(matrix, model)
    if model.transform == "standardize":
        return StandardScaler().fit_transform(matrix)
    if model.transform == "none":
        return matrix
    raise ValueError(f"unknown transform {model.transform!r} for {model.key}")


# ── distances ───────────────────────────────────────────────────────────────

def _xlog2x(x: np.ndarray) -> np.ndarray:
    """x*log2(x) with the convention 0*log0 = 0, without taking log of zero."""
    return x * np.log2(np.where(x > 0, x, 1.0))


def entropy(matrix: np.ndarray) -> np.ndarray:
    """Row-wise entropy in bits."""
    return -_xlog2x(matrix).sum(axis=1)


def _block_size(n_space: int, width: int, item_bytes: int) -> int:
    """Query rows per block, so the intermediate tensor fits the memory budget."""
    return max(1, int(config.MEMORY_BUDGET / max(1, n_space * width * item_bytes)))


def _select(distance: np.ndarray, n_neighbours: int) -> np.ndarray:
    """Indices of the n smallest distances per row, in increasing order."""
    part = np.argpartition(distance, n_neighbours - 1, axis=1)[:, :n_neighbours]
    order = np.take_along_axis(distance, part, axis=1).argsort(axis=1)
    return np.take_along_axis(part, order, axis=1)


def _jensenshannon_knn(query: np.ndarray, space: np.ndarray, n_neighbours: int) -> np.ndarray:
    query = np.ascontiguousarray(query, dtype=np.float32)
    space = np.ascontiguousarray(space, dtype=np.float32)
    h_query, h_space = entropy(query), entropy(space)

    out = np.empty((len(query), n_neighbours), dtype=np.int64)
    block = _block_size(len(space), space.shape[1], 4)
    for start in range(0, len(query), block):
        stop = min(start + block, len(query))
        mixture = (query[start:stop, None, :] + space[None, :, :]) * 0.5
        divergence = (-_xlog2x(mixture).sum(axis=2)
                      - 0.5 * (h_query[start:stop, None] + h_space[None, :]))
        out[start:stop] = _select(divergence, n_neighbours)
    return out


def _minhash_knn(query: np.ndarray, space: np.ndarray, n_neighbours: int) -> np.ndarray:
    query = np.ascontiguousarray(query, dtype=np.uint64)
    space = np.ascontiguousarray(space, dtype=np.uint64)
    width = space.shape[1]

    out = np.empty((len(query), n_neighbours), dtype=np.int64)
    block = _block_size(len(space), width, 1)
    for start in range(0, len(query), block):
        stop = min(start + block, len(query))
        agree = (query[start:stop, None, :] == space[None, :, :]).sum(axis=2)
        out[start:stop] = _select(1.0 - agree / width, n_neighbours)
    return out


def knn(query: np.ndarray, space: np.ndarray, n_neighbours: int, model: Model) -> np.ndarray:
    """Indices of the nearest neighbours of each query row inside ``space``."""
    n_neighbours = min(n_neighbours, len(space))
    if model.metric == "jensenshannon":
        return _jensenshannon_knn(query, space, n_neighbours)
    if model.metric == "minhash":
        return _minhash_knn(query, space, n_neighbours)
    finder = NearestNeighbors(n_neighbors=n_neighbours, metric=model.metric)
    finder.fit(space)
    return finder.kneighbors(query, return_distance=False)


# ── precision at k ──────────────────────────────────────────────────────────

def precision_at_ks(neighbour_labels: np.ndarray, targets, ks) -> dict:
    """P@k for every k at once: prefixes of one neighbour list.

    Asking for the 10 nearest neighbours once and reading prefixes off it gives the
    same answer as ten separate searches, for a tenth of the work.
    """
    targets = np.asarray(targets)
    if targets.ndim == 0:
        targets = np.repeat(targets, len(neighbour_labels))
    hits = (neighbour_labels == targets[:, None]).astype(np.float64)
    cumulative = np.cumsum(hits, axis=1)
    available = neighbour_labels.shape[1]
    return {k: float(np.mean(cumulative[:, min(k, available) - 1] / min(k, available)))
            for k in ks}


def stack(groups: dict):
    """{label: rows} -> (matrix, labels); empty groups are dropped."""
    matrices, labels = [], []
    for label, rows in groups.items():
        if len(rows):
            matrices.append(rows)
            labels.extend([label] * len(rows))
    if not matrices:
        return None, np.array([])
    return np.vstack(matrices), np.asarray(labels)
