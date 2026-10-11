"""K-medoids on a precomputed distance matrix -- numpy only.

The reference benchmark clusters each reference sample into as many clusters as it has
classes and scores the partition against the true labels (adjusted Rand index).  The
original tables used K-medoids from ``scikit-learn-extra``, which does not build on
current Python/NumPy; this is a small re-implementation of the same "alternate" (Voronoi
iteration) algorithm, with the same two initialisations, so the package depends on
numpy and scikit-learn only.

It is a re-implementation, not the original code: ties, iteration order and the
random ``k-medoids++`` seeding can differ in the last decimals, so ARI values
recomputed with it are equivalent in kind but not guaranteed to be bit-identical to
the committed tables (those pre-date this module; see the README).

    init="k-medoids++"   seeded D^2-style spreading of the initial medoids; the best of
                         ``n_init`` restarts (lowest total distance to the medoids) is
                         kept.  The default here.
    init="heuristic"     the ``n_clusters`` points with the smallest total distance to
                         every other point -- deterministic, ``random_state`` unused.
                         This is scikit-learn-extra's default, but all its starting
                         medoids lie in the densest region, and on well-separated
                         clusters the alternate iteration often stays in a poor local
                         optimum, so it is not the default here.
"""

import numpy as np


def _init_heuristic(distance: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.argpartition(distance.sum(axis=1), n_clusters - 1)[:n_clusters]


def _init_plusplus(distance: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n = len(distance)
    medoids = [int(rng.integers(n))]
    closest = distance[medoids[0]].copy()
    for _ in range(1, n_clusters):
        weights = closest ** 2
        total = weights.sum()
        if total <= 0:                                  # all remaining points coincide
            choices = np.setdiff1d(np.arange(n), medoids)
            pick = int(rng.choice(choices))
        else:
            pick = int(rng.choice(n, p=weights / total))
        medoids.append(pick)
        closest = np.minimum(closest, distance[pick])
    return np.asarray(medoids)


def _alternate(distance, medoids, max_iter):
    n_clusters = len(medoids)
    medoids = np.sort(medoids)
    labels = np.argmin(distance[:, medoids], axis=1)
    for _ in range(max_iter):
        previous = medoids.copy()
        for cluster in range(n_clusters):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                continue
            costs = distance[np.ix_(members, members)].sum(axis=1)
            best = members[int(np.argmin(costs))]
            if costs.min() < distance[medoids[cluster], members].sum():
                medoids[cluster] = best
        labels = np.argmin(distance[:, medoids], axis=1)
        if np.array_equal(np.sort(previous), np.sort(medoids)):
            break
    # a medoid is always in its own cluster, even when another medoid coincides with it
    labels[medoids] = np.arange(n_clusters)
    return labels, medoids


def kmedoids(distance: np.ndarray, n_clusters: int, random_state: int, max_iter: int = 300,
             init: str = "k-medoids++", n_init: int = 5):
    """Cluster by medoids.  Returns ``(labels, medoid indices)``.

    ``distance`` is a symmetric (n, n) matrix with a zero diagonal.  ``random_state`` is
    required (it matters for ``init="k-medoids++"``) so that no call is ever implicitly
    random.
    """
    distance = np.asarray(distance, dtype=np.float64)
    n = len(distance)
    if not 1 <= n_clusters <= n:
        raise ValueError(f"cannot form {n_clusters} clusters from {n} points")
    rng = np.random.default_rng(random_state)

    if init == "heuristic":
        starts = [_init_heuristic(distance, n_clusters)]
    elif init == "k-medoids++":
        starts = [_init_plusplus(distance, n_clusters, rng) for _ in range(max(1, n_init))]
    else:
        raise ValueError(f"unknown init {init!r}")

    best = None
    for start in starts:
        labels, medoids = _alternate(distance, start.copy(), max_iter)
        cost = distance[np.arange(n), medoids[labels]].sum()
        if best is None or cost < best[0]:
            best = (cost, labels, medoids)
    return best[1], best[2]
