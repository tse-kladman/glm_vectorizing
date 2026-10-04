"""Metrics on the unseen virus orders -- the numbers behind Fig. 4, 5 and 6.

Both metrics put the fragments of an unseen order into a space built from one
reference sample and ask what the nearest neighbours are:

    precision        space = reference + this order.  How many of a fragment's k
                     neighbours belong to its own order?  Higher is better: the
                     order stays together even though the model never saw it.

    cross_precision  space = reference + the reverse complement of this order,
                     query = the forward fragments, which are not in the space.
                     How many neighbours are the fragment's own reverse
                     complement?  High means the model reads both strands alike.

Both are averaged per label first and then over labels, so a large label cannot
dominate a small one.
"""

import numpy as np

from .. import neighbours
from ..models import Model


def precision(reference: dict, virus: dict, ks, model: Model) -> dict:
    """P@k of one unseen order inside reference + itself, for every k at once."""
    ks = sorted(ks)
    per_label = []

    for label, rows in virus.items():
        if not len(rows):
            continue
        space, labels = neighbours.stack({**reference, label: rows})
        if space is None:
            continue
        queries = np.flatnonzero(labels == label)
        if not len(queries) or len(queries) == len(labels):
            continue

        matrix = neighbours.transform(space, model)
        # the query rows are part of the space, so drop each fragment's own match
        found = neighbours.knn(matrix[queries], matrix, max(ks) + 1, model)[:, 1:]
        per_label.append(neighbours.precision_at_ks(labels[found], labels[queries], ks))

    if not per_label:
        return {k: 0.0 for k in ks}
    return {k: float(np.mean([p[k] for p in per_label])) for k in ks}


def cross_precision(reference: dict, forward: dict, reverse: dict, ks, model: Model) -> dict:
    """CROSS P@k: how often a forward fragment retrieves its reverse complement."""
    ks = sorted(ks)
    per_label = []

    for label, rows in forward.items():
        if not len(rows):
            continue
        rc_label = f"REVERSECOMPLEMENT{label}"
        rc_rows = reverse.get(rc_label)
        if rc_rows is None or not len(rc_rows):
            continue

        space, labels = neighbours.stack({**reference, rc_label: rc_rows})
        if space is None:
            continue

        # scale space and query together, then split: the forward fragments are the
        # query only, and must not join the space they are searched in
        combined = neighbours.transform(np.vstack([space, rows]), model)
        found = neighbours.knn(combined[len(space):], combined[:len(space)],
                               max(ks), model)
        per_label.append(neighbours.precision_at_ks(labels[found], rc_label, ks))

    if not per_label:
        return {k: 0.0 for k in ks}
    return {k: float(np.mean([p[k] for p in per_label])) for k in ks}
