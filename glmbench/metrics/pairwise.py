"""One unseen order against the other -- the numbers behind Fig. 7.

Two complementary questions about a pair of orders A and B:

    cross_between      space = reference + B, query = A.  How often does an A
                       fragment retrieve a B fragment?  Lower is better: the two
                       orders are well separated.  (Called "inter-order leakage" in
                       the docs; unrelated to the fwd<->RC "RC proximity" of
                       unseen.precision_rc_proximity, for which HIGHER is better.)

    precision_with_other  space = reference + B + A, query = one genome (label) of A.
                       Does the genome still find itself when B shares the space?
                       Success = a fragment of the SAME GENOME, exactly as in
                       unseen.precision_by_genome -- a within-genome compactness,
                       not an order-level quantity.  Higher is better.

Together they place a model on the compactness/separation plane of Fig. 7; the
ideal sits top-left.  Unlike unseen.precision_by_genome, cross_between treats all
fragments of the target as a single class, because the question is about orders, not
genomes.
"""

import numpy as np

from .. import neighbours
from ..models import Model

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
        found = neighbours.knn_excluding(matrix[queries], matrix, k, model,
                                         self_index=queries)
        per_label.append(neighbours.precision_at_ks(
            neighbours.labels_of(labels, found), labels[queries], [k])[k])

    return float(np.mean(per_label)) if per_label else 0.0


def all_combinations(reference: dict, forward_a: dict, reverse_a: dict,
                     forward_b: dict, reverse_b: dict, k: int, model: Model) -> dict:
    """The twelve strand/direction combinations of one pair, for one reference sample."""
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
