"""Metrics on the unseen virus orders -- the numbers behind Fig. 4, 5 and 6.

Every metric puts fragments of an unseen order into a space built from ONE
reference sample (50 independent draws of known viruses) and asks what the nearest
neighbours of a fragment are.  They differ in what counts as a success, and the
differences matter, so each is spelled out here.

All functions take ``reference`` (``{class: rows}``) and the fragments of the order
as ``{genome id: rows}``, and return **per-genome** values ``{genome: {k: value}}``;
:func:`summarise` averages them over genomes.  The unit of replication of a result
is the *genome*, not the fragment: fragments of one genome are not independent, which
is why the per-genome values are kept (see ``statistics.bootstrap_ci_by_genome``).

``precision_by_genome``   -- legacy ("P@k" in the tables on disk)
    space = reference + the fragments of ONE genome; query = that genome's
    fragments, each without itself.  Success = a neighbour is a fragment of the
    **same genome**.  This measures how tightly the fragments of one genome stick
    together, *not* whether they group with other genomes of the same order: the
    "own order" in the first version of this docstring was a misnomer.  A genome
    whose fragments are mutually similar scores ~1 however unrelated it is to every
    other genome of its order.

``precision_by_order``    -- the order-level question
    space = reference + ALL fragments of the order, labelled with the order's name;
    query = every fragment of the order.  Neighbours that belong to the **same
    genome as the query are excluded** (leave-genome-out).  Success = a neighbour is
    a fragment of *another genome of the same order*.  The chance level is the share
    of the space that is such a fragment (see :func:`chance_order_precision`).
    Undefined with a single genome.

``own_rc``                -- strand invariance, per fragment
    for each forward fragment x of the order: space = reference + all forward
    fragments of the order + all reverse-complement (RC) fragments of the order; x
    itself is excluded.  ``hit@k`` = the RC of x is among x's k nearest neighbours;
    ``similarity`` = similarity between x and its own RC (cosine of the transformed
    vectors, sketch agreement for MinHash, 1 - Jensen-Shannon divergence for
    profiles).  Rows of the forward and RC arrays are index-aligned (the RC file is
    written in the same order as the forward file); this is asserted.

``precision_rc_proximity`` -- legacy ("CROSS P@k" / ``cross_P`` in the tables on disk)
    space = reference + the RC fragments of ONE genome; query = its forward
    fragments (not in the space).  Success = a neighbour is **any** RC fragment of
    the same genome -- not necessarily the fragment's own RC.  It therefore also
    rewards a model that merely maps all fragments of a genome close together, which
    is what ``precision_by_genome`` measures; use ``own_rc`` for strand invariance.

Neighbour exclusion is by index, never by position in the neighbour list
(``neighbours.knn_excluding``).  Scaling (``StandardScaler`` for vector models) is
fitted on the whole space of each query set -- reference plus the order's fragments --
so it is transductive; the README ("Methods notes") says what that implies.
"""

import numpy as np

from .. import neighbours
from ..models import Model

ORDER_LABEL = "\x00order"          # label of the order's own fragments in the space
RC_PREFIX = "REVERSECOMPLEMENT"


def summarise(per_genome: dict, ks) -> dict:
    """Mean over genomes of ``{genome: {k: value}}``; NaN when there is no genome."""
    if not per_genome:
        return {k: float("nan") for k in ks}
    return {k: float(np.mean([values[k] for values in per_genome.values()])) for k in ks}


def _nonempty(groups: dict) -> dict:
    return {label: rows for label, rows in groups.items() if len(rows)}


# ── genome level (legacy) ───────────────────────────────────────────────────

def precision_by_genome(reference: dict, virus: dict, ks, model: Model) -> dict:
    """P@k with success = same genome (legacy).  Returns ``{genome: {k: P@k}}``."""
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


# ── order level ─────────────────────────────────────────────────────────────

def _order_space(reference: dict, virus: dict):
    """(rows, labels, genome group of every row, query indices) of reference + order."""
    reference = _nonempty(reference)
    genomes = _nonempty(virus)
    ref_rows, ref_labels = neighbours.stack(reference)
    n_ref = 0 if ref_rows is None else len(ref_rows)

    order_rows = np.vstack(list(genomes.values()))
    groups = np.concatenate([np.full(len(rows), index)
                             for index, rows in enumerate(genomes.values())])
    rows = order_rows if ref_rows is None else np.vstack([ref_rows, order_rows])
    labels = np.concatenate([np.asarray(ref_labels, dtype=object),
                             np.full(len(order_rows), ORDER_LABEL, dtype=object)])
    space_groups = np.concatenate([np.full(n_ref, -1), groups])
    return rows, labels, space_groups, np.arange(n_ref, len(rows)), list(genomes)


def precision_by_order(reference: dict, virus: dict, ks, model: Model) -> dict:
    """P@k with success = another genome of the same order (leave-genome-out).

    Returns ``{genome: {k: P@k}}``: for each genome, the mean over its fragments of
    the fraction of the k nearest *admissible* neighbours that are fragments of some
    other genome of the order.  Empty if the order has fewer than two genomes.
    """
    ks = sorted(ks)
    genomes = _nonempty(virus)
    if len(genomes) < 2:
        return {}

    rows, labels, space_groups, queries, names = _order_space(reference, virus)
    matrix = neighbours.transform(rows, model)
    found = neighbours.knn_excluding(matrix[queries], matrix, max(ks), model,
                                     self_index=queries,
                                     query_groups=space_groups[queries],
                                     space_groups=space_groups)
    neighbour_labels = neighbours.labels_of(labels, found)

    result = {}
    for index, genome in enumerate(names):
        mine = space_groups[queries] == index
        result[genome] = neighbours.precision_at_ks(neighbour_labels[mine], ORDER_LABEL, ks)
    return result


def chance_order_precision(reference: dict, virus: dict) -> dict:
    """What ``precision_by_order`` would score if neighbours were drawn at random.

    Per genome: the share of the admissible candidates (everything except the
    genome's own fragments) that are fragments of other genomes of the order.  A
    model must beat this, not zero, to have grouped the order together.
    """
    genomes = _nonempty(virus)
    n_ref = sum(len(rows) for rows in _nonempty(reference).values())
    total = sum(len(rows) for rows in genomes.values())
    return {genome: (total - len(rows)) / (n_ref + total - len(rows))
            for genome, rows in genomes.items()}


# ── strand invariance ───────────────────────────────────────────────────────

def own_rc(reference: dict, forward: dict, reverse: dict, ks, model: Model):
    """Does a fragment find its OWN reverse complement?  See the module docstring.

    Returns ``(hits, similarity)``: ``hits = {genome: {k: hit@k}}`` and
    ``similarity = {genome: mean similarity between x and RC(x)}``.
    """
    ks = sorted(ks)
    genomes = [g for g, rows in forward.items() if len(rows)]
    forward_rows, reverse_rows = [], []
    for genome in genomes:
        rc = reverse.get(f"{RC_PREFIX}{genome}")
        if rc is None or len(rc) != len(forward[genome]):
            raise ValueError(
                f"{genome}: forward and reverse-complement fragments are not aligned "
                f"({len(forward[genome])} vs {0 if rc is None else len(rc)}); the RC "
                f"file must be written from the forward file, fragment for fragment")
        forward_rows.append(forward[genome])
        reverse_rows.append(rc)
    if not genomes:
        return {}, {}

    ref_rows, _ = neighbours.stack(_nonempty(reference))
    n_ref = 0 if ref_rows is None else len(ref_rows)
    fwd, rc = np.vstack(forward_rows), np.vstack(reverse_rows)
    parts = ([ref_rows] if n_ref else []) + [fwd, rc]
    matrix = neighbours.transform(np.vstack(parts), model)

    n = len(fwd)
    queries = np.arange(n_ref, n_ref + n)
    partners = queries + n                      # row of each query's own RC
    found = neighbours.knn_excluding(matrix[queries], matrix, max(ks), model,
                                     self_index=queries)
    is_partner = np.cumsum(found == partners[:, None], axis=1) > 0
    similarity = neighbours.rowwise_similarity(matrix[queries], matrix[partners], model)

    owner = np.concatenate([np.full(len(rows), index) for index, rows in enumerate(forward_rows)])
    hits, sims = {}, {}
    for index, genome in enumerate(genomes):
        mine = owner == index
        hits[genome] = {k: float(is_partner[mine, min(k, is_partner.shape[1]) - 1].mean())
                        for k in ks}
        sims[genome] = float(np.nanmean(similarity[mine])) if mine.any() else float("nan")
    return hits, sims


# ── reverse-complement proximity (legacy) ───────────────────────────────────

def precision_rc_proximity(reference: dict, forward: dict, reverse: dict, ks,
                           model: Model) -> dict:
    """Legacy "CROSS P@k": neighbours that are ANY RC fragment of the same genome.

    Returns ``{genome: {k: value}}``.  Not a test of strand invariance -- see the
    module docstring and ``own_rc``.
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
