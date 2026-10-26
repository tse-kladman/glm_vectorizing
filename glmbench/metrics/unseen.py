"""Metrics on the unseen virus orders -- the numbers behind Fig. 4, 5 and 6.

Every metric puts fragments of an unseen order into a space built from ONE
reference sample (one of 50 draws of known viruses) and asks what the nearest
neighbours of a fragment are.  They differ in what counts as a success, so each is
spelled out here.

Both functions take ``reference`` (``{class: rows}``) and the fragments of the order
as ``{genome id: rows}``, and return per-genome values ``{genome: {k: value}}``;
:func:`summarise` averages them over genomes.

``precision_by_genome``   -- genome level ("P@k" in the tables on disk)
    space = reference + the fragments of ONE genome; query = that genome's
    fragments, each without itself.  Success = a neighbour is a fragment of the
    **same genome**.  This measures how tightly the fragments of one genome stick
    together, *not* whether they group with other genomes of the same order.

``precision_rc_proximity`` -- RC proximity ("CROSS P@k" / ``cross_P`` in the tables on disk)
    space = reference + the RC fragments of ONE genome; query = its forward
    fragments (not in the space).  Success = a neighbour is **any** RC fragment of
    the same genome -- not necessarily the fragment's own RC.  It therefore also
    rewards a model that merely maps all fragments of a genome close together.

Neighbour exclusion is by index, never by position in the neighbour list
(``neighbours.knn_excluding``).  Scaling (``StandardScaler`` for vector models) is
fitted on the whole space of each query set -- reference plus the order's fragments --
so it is transductive (the scaler sees the query fragments, not their labels).
"""

import numpy as np

from .. import neighbours
from ..models import Model

RC_PREFIX = "REVERSECOMPLEMENT"


def summarise(per_genome: dict, ks) -> dict:
    """Mean over genomes of ``{genome: {k: value}}``; NaN when there is no genome."""
    if not per_genome:
        return {k: float("nan") for k in ks}
    return {k: float(np.mean([values[k] for values in per_genome.values()])) for k in ks}


def _nonempty(groups: dict) -> dict:
    return {label: rows for label, rows in groups.items() if len(rows)}


# ── genome level ────────────────────────────────────────────────────────────

def precision_by_genome(reference: dict, virus: dict, ks, model: Model) -> dict:
    """P@k with success = same genome.  Returns ``{genome: {k: P@k}}``."""
    ks = sorted(ks)
    result = {}

    for label, rows in _nonempty(virus).items():
        space, labels = neighbours.stack({**reference, label: rows})
        queries = np.flatnonzero(labels == label)
        if not len(queries) or len(queries) == len(labels):
            continue

        matrix = neighbours.transform(space, model)
        found = neighbours.knn_excluding(matrix[queries], matrix, max(ks), model,
                                         self_index=queries)
        result[label] = neighbours.precision_at_ks(
            neighbours.labels_of(labels, found), labels[queries], ks)
    return result


# ── reverse-complement proximity ────────────────────────────────────────

def precision_rc_proximity(reference: dict, forward: dict, reverse: dict, ks,
                           model: Model) -> dict:
    """RC proximity ("CROSS P@k"): neighbours that are ANY RC fragment of the same genome.

    Returns ``{genome: {k: value}}``.  Not a test of strand invariance -- see the
    module docstring.
    """
    ks = sorted(ks)
    result = {}

    for label, rows in _nonempty(forward).items():
        rc_label = f"{RC_PREFIX}{label}"
        rc_rows = reverse.get(rc_label)
        if rc_rows is None or not len(rc_rows):
            continue

        space, labels = neighbours.stack({**reference, rc_label: rc_rows})
        # scale space and query together, then split: the forward fragments are the
        # query only, and must not join the space they are searched in
        combined = neighbours.transform(np.vstack([space, rows]), model)
        found = neighbours.knn(combined[len(space):], combined[:len(space)],
                               max(ks), model)
        result[label] = neighbours.precision_at_ks(labels[found], rc_label, ks)
    return result
