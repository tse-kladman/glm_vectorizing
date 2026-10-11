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


# ── neighbour search with exclusions ────────────────────────────────────────

PAD = -1            # index used to fill a neighbour list that ran out of candidates
PAD_LABEL = "\x00no-neighbour"


def knn_excluding(query: np.ndarray, space: np.ndarray, n_wanted: int, model: Model,
                  self_index=None, query_groups=None, space_groups=None) -> np.ndarray:
    """Nearest neighbours of each query row, with named rows and groups left out.

    Removing a query's own entry from its neighbour list by *position* (``[:, 1:]``)
    assumes the entry is first.  It is not, when the space holds a duplicate of the
    query (a repeated fragment, an identical sequence) or when distances tie -- and a
    zero vector ties with every other zero vector.  Then the wrong row is dropped and
    the query is silently scored against itself.  So exclusion is done **by index**:

    ``self_index``     per query row, its own row number inside ``space`` (or ``-1``)
    ``query_groups`` / ``space_groups``
                       a neighbour is dropped when its group equals the query's;
                       used by the leave-genome-out protocol, where *every* fragment
                       of the query's own genome is off limits, not just the fragment
                       itself

    Returns ``(n_query, n_wanted)`` indices into ``space``, nearest first, padded with
    ``PAD`` when fewer than ``n_wanted`` admissible neighbours exist.
    """
    n_query = len(query)
    extra = 0
    if query_groups is not None:
        query_groups = np.asarray(query_groups)
        space_groups = np.asarray(space_groups)
        # at most this many candidates can be excluded for any one query
        sizes = {group: int((space_groups == group).sum()) for group in np.unique(query_groups)}
        extra = max(sizes.values()) if sizes else 0
    if self_index is not None:
        self_index = np.asarray(self_index)
        extra = max(extra, 1)

    found = knn(query, space, n_wanted + extra, model)
    keep = np.ones(found.shape, dtype=bool)
    if self_index is not None:
        keep &= found != self_index[:, None]
    if query_groups is not None:
        keep &= space_groups[found] != query_groups[:, None]

    # first n_wanted admissible entries of every row, in their original order
    rank = np.cumsum(keep, axis=1)
    keep &= rank <= n_wanted
    out = np.full((n_query, n_wanted), PAD, dtype=np.int64)
    rows, columns = np.nonzero(keep)
    out[rows, rank[rows, columns] - 1] = found[rows, columns]
    return out


def labels_of(labels: np.ndarray, found: np.ndarray) -> np.ndarray:
    """Labels of neighbour indices; padded entries get a label that matches nothing."""
    labels = np.asarray(labels).astype(object)
    padded = np.append(labels, PAD_LABEL)               # index -1 -> the pad label
    return padded[found]


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


# ── similarity of paired rows ───────────────────────────────────────────────

def similarity_kind(model: Model) -> str:
    """What :func:`rowwise_similarity` means for a model, for table headers."""
    return {"cosine": "cosine", "minhash": "sketch agreement",
            "jensenshannon": "1 - JSD (bits)"}.get(model.metric, model.metric)


def rowwise_similarity(left: np.ndarray, right: np.ndarray, model: Model) -> np.ndarray:
    """Similarity between ``left[i]`` and ``right[i]`` under the model's own notion.

    cosine         cosine of the (already transformed) vectors
    minhash        fraction of sketch positions that agree (the Jaccard estimate)
    jensenshannon  1 - JSD in bits, so 1 means identical distributions

    NaN where a cosine is undefined (a zero vector).
    """
    if model.metric == "cosine":
        left, right = left.astype(np.float64), right.astype(np.float64)
        norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(norm > 0, (left * right).sum(axis=1) / norm, np.nan)
    if model.metric == "minhash":
        return (left == right).mean(axis=1)
    if model.metric == "jensenshannon":
        left, right = left.astype(np.float64), right.astype(np.float64)
        divergence = (entropy((left + right) / 2)
                      - 0.5 * (entropy(left) + entropy(right)))
        return 1.0 - divergence
    raise ValueError(f"no row-wise similarity for metric {model.metric!r}")


# ── full distance matrix ────────────────────────────────────────────────────

def distance_matrix(matrix: np.ndarray, model: Model) -> np.ndarray:
    """All pairwise distances of a (transformed) matrix, under the model's metric.

    Used where a whole matrix is needed (clustering, silhouette).  For the cosine and
    Jensen-Shannon / MinHash metrics the ranking is the same as in :func:`knn`; the
    values are cosine distance, JSD in bits, and 1 - sketch agreement.
    """
    if model.metric == "cosine":
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(matrix, metric="cosine")
    elif model.metric == "jensenshannon":
        data = np.ascontiguousarray(matrix, dtype=np.float32)
        h = entropy(data)
        distances = np.empty((len(data), len(data)), dtype=np.float64)
        block = _block_size(len(data), data.shape[1], 4)
        for start in range(0, len(data), block):
            stop = min(start + block, len(data))
            mixture = (data[start:stop, None, :] + data[None, :, :]) * 0.5
            distances[start:stop] = (-_xlog2x(mixture).sum(axis=2)
                                     - 0.5 * (h[start:stop, None] + h[None, :]))
        distances = np.maximum(distances, 0.0)
    elif model.metric == "minhash":
        data = np.ascontiguousarray(matrix, dtype=np.uint64)
        distances = np.empty((len(data), len(data)), dtype=np.float64)
        block = _block_size(len(data), data.shape[1], 1)
        for start in range(0, len(data), block):
            stop = min(start + block, len(data))
            agree = (data[start:stop, None, :] == data[None, :, :]).mean(axis=2)
            distances[start:stop] = 1.0 - agree
    else:
        raise ValueError(f"no distance matrix for metric {model.metric!r}")
    distances = (distances + distances.T) / 2          # remove float asymmetry
    np.fill_diagonal(distances, 0.0)
    return distances
