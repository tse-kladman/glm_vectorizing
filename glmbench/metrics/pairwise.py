"""One unseen order against the other -- the numbers behind Fig. 7.

Two complementary questions about a pair of orders A and B:

    cross_between      space = reference + B, query = A.  How often does an A
                       fragment retrieve a B fragment instead of its own kind?
                       Lower is better: the two orders are well separated.

    precision_with_other  space = reference + B + A, query = one label of A.  Does
                       A still find itself when B shares the space?  Higher is
                       better: a dense own-order cluster.

Together they place a model on the compactness/separation plane of Fig. 7; the
ideal sits top-left.  Unlike unseen.precision, cross_between treats all fragments
of the target as a single class, because the question is about orders, not labels.
"""

import numpy as np

from .. import neighbours
from ..models import Model

# the twelve strand/direction combinations written per pair of orders
COMBINATIONS = [
    "cross_1→2_fwd_rc", "cross_1→2_rc_fwd", "cross_1→2_fwd_fwd", "cross_1→2_rc_rc",
    "cross_2→1_fwd_rc", "cross_2→1_rc_fwd", "cross_2→1_fwd_fwd", "cross_2→1_rc_rc",
    "pk_fwd_1_in_2", "pk_rc_1_in_2", "pk_fwd_2_in_1", "pk_rc_2_in_1",
]


def cross_between(reference: dict, query: dict, target: dict, k: int, model: Model) -> float:
    """Fraction of a query order's k neighbours that belong to the target order."""
    query_rows = [rows for rows in query.values() if len(rows)]
    if not query_rows:
        return 0.0
    queries = np.vstack(query_rows)

    space_rows, n_target = [], 0
    for rows in reference.values():
        if len(rows):
            space_rows.append(rows)
    for rows in target.values():
        if len(rows):
            space_rows.append(rows)
            n_target += len(rows)
    if not space_rows or not n_target:
        return 0.0

    space = np.vstack(space_rows)
    is_target = np.zeros(len(space), dtype=bool)
    is_target[len(space) - n_target:] = True

    combined = neighbours.transform(np.vstack([space, queries]), model)
    found = neighbours.knn(combined[len(space):], combined[:len(space)], k, model)
    return float(is_target[found].mean())


def precision_with_other(reference: dict, query: dict, other: dict, k: int,
                         model: Model) -> float:
    """P@k of one order when the other order also occupies the space."""
    per_label = []
    for label, rows in query.items():
        if not len(rows):
            continue
        space, labels = neighbours.stack({**reference, **other, label: rows})
        if space is None:
            continue
        queries = np.flatnonzero(labels == label)
        if not len(queries) or len(queries) == len(labels):
            continue

        matrix = neighbours.transform(space, model)
        found = neighbours.knn(matrix[queries], matrix, k + 1, model)[:, 1:]
        per_label.append(neighbours.precision_at_ks(labels[found], labels[queries], [k])[k])

    return float(np.mean(per_label)) if per_label else 0.0


def all_combinations(reference: dict, forward_a: dict, reverse_a: dict,
                     forward_b: dict, reverse_b: dict, k: int, model: Model) -> dict:
    """Every strand/direction combination of one pair, for one reference sample."""
    return {
        "cross_1→2_fwd_rc": cross_between(reference, forward_a, reverse_b, k, model),
        "cross_1→2_rc_fwd": cross_between(reference, reverse_a, forward_b, k, model),
        "cross_1→2_fwd_fwd": cross_between(reference, forward_a, forward_b, k, model),
        "cross_1→2_rc_rc": cross_between(reference, reverse_a, reverse_b, k, model),
        "cross_2→1_fwd_rc": cross_between(reference, forward_b, reverse_a, k, model),
        "cross_2→1_rc_fwd": cross_between(reference, reverse_b, forward_a, k, model),
        "cross_2→1_fwd_fwd": cross_between(reference, forward_b, forward_a, k, model),
        "cross_2→1_rc_rc": cross_between(reference, reverse_b, reverse_a, k, model),
        "pk_fwd_1_in_2": precision_with_other(reference, forward_a, forward_b, k, model),
        "pk_rc_1_in_2": precision_with_other(reference, reverse_a, reverse_b, k, model),
        "pk_fwd_2_in_1": precision_with_other(reference, forward_b, forward_a, k, model),
        "pk_rc_2_in_1": precision_with_other(reference, reverse_b, reverse_a, k, model),
    }
