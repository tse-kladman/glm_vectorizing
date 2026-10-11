"""Running the metrics in parallel and writing the tables.

Two levels of parallelism, chosen to match where the work actually is:

    processes   one task = one (model, virus file); tasks are independent and each
                holds its own copy of that model's representation
    threads     the 50 reference samples inside a task; these are I/O bound on
                loading pickles and release the GIL inside numpy

Memory, not cores, is the limit: every process imports numpy/scipy and keeps one
model's vectors resident, so ``--max-workers`` defaults to 8 rather than to the
core count, and the blockwise distances respect ``--memory-budget``.  Layered
models can be restricted to the chosen layer, which is 32x less work for Evo2.

Nothing is dropped silently.  A reference sample that is missing or unreadable, a
virus file that is absent, a representation the metric cannot take: each is counted
and reported, the affected tables are NOT written, and the run ends with a non-zero
exit unless ``allow_missing`` is set (``--allow-missing``), in which case the tables
are written from the samples that did load and the loss is stated in the summary.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import config, selection
from ..models import Model, get
from ..representations import store
from . import pairwise, tables, unseen

PAIRWISE_K = 10


def _layers_of(reference_data, virus_data, layered: bool):
    """The layers present in both the reference sample and the virus."""
    if not layered:
        return [tables.FLAT_LAYER]
    return [layer for layer in reference_data if layer in virus_data]


def _split(model: Model, data, layer):
    """The ``{label: rows}`` view of one layer (or of a layer-less model)."""
    if data is None:
        return None
    return data.get(layer) if model.layered else data


def _sample_scores(model: Model, reference_data, virus, virus_rc, ks, keep_layers,
                   protocol: str):
    """All unseen-order scores of one reference sample: ``{kind: {layer: ...}}``."""
    scores = defaultdict(dict)
    for layer in _layers_of(reference_data, virus, model.layered):
        if keep_layers is not None and layer not in keep_layers:
            continue
        ref_groups = _split(model, reference_data, layer)
        virus_groups = _split(model, virus, layer)
        rc_groups = _split(model, virus_rc, layer) if virus_rc else None
        if not ref_groups or not virus_groups:
            continue

        if protocol in ("genome", "both"):
            scores["genome"][layer] = unseen.precision_by_genome(
                ref_groups, virus_groups, ks, model)
            if rc_groups:
                scores["cross"][layer] = unseen.precision_rc_proximity(
                    ref_groups, virus_groups, rc_groups, ks, model)
        if protocol in ("order", "both"):
            scores["order"][layer] = unseen.precision_by_order(
                ref_groups, virus_groups, ks, model)
        if rc_groups:
            scores["ownrc"][layer] = unseen.own_rc(ref_groups, virus_groups,
                                                   rc_groups, ks, model)
    return scores


def _run_task(packed):
    """One (model, virus file) over all reference samples; top level, for the pool."""
    (model_key, virus_name, rc_name, n_ref, ks, latent_dir, keep_layers, protocol) = packed
    model = get(model_key)
    latent_dir = Path(latent_dir) if latent_dir else None
    problems = []
    key = (model_key, virus_name)

    virus = store.load_virus(model, virus_name, latent_dir, problems)
    if virus is None:
        return key, {"fatal": f"virus file for {virus_name} is missing or unreadable",
                     "problems": problems}
    virus_rc = store.load_virus(model, rc_name, latent_dir, problems) if rc_name else None
    if rc_name and virus_rc is None:
        problems.append((rc_name, "reverse-complement file is missing"))
    layers = list(virus.keys()) if model.extractor == "viralm" else None

    def one_sample(index):
        reference_data = store.load_sample(model, index, latent_dir, layers, problems)
        if reference_data is None:
            return index, None
        try:
            return index, _sample_scores(model, reference_data, virus, virus_rc,
                                         ks, keep_layers, protocol)
        except ValueError as error:                 # representation/metric mismatch
            return index, error

    matrices = defaultdict(lambda: {k: defaultdict(dict) for k in ks})
    similarity = defaultdict(dict)                  # layer -> sample -> value
    per_genome = defaultdict(list)                  # kind -> rows
    loaded, missing, failure = [], [], None

    with ThreadPoolExecutor(max_workers=config.SAMPLE_THREADS) as pool:
        futures = [pool.submit(one_sample, index) for index in range(n_ref)]
        for future in as_completed(futures):
            index, result = future.result()
            if result is None:
                missing.append(index)
                continue
            if isinstance(result, ValueError):
                failure = str(result)
                continue
            loaded.append(index)
            for kind, by_layer in result.items():
                for layer, payload in by_layer.items():
                    _absorb(kind, layer, index, payload, ks, matrices, similarity,
                            per_genome)

    return key, {"matrices": {kind: {k: {layer: dict(v) for layer, v in per_k.items()}
                                     for k, per_k in by_k.items()}
                              for kind, by_k in matrices.items()},
                 "similarity": {layer: dict(v) for layer, v in similarity.items()},
                 "per_genome": dict(per_genome),
                 "loaded": sorted(loaded), "missing": sorted(missing),
                 "problems": problems, "error": failure}


def _absorb(kind, layer, sample, payload, ks, matrices, similarity, per_genome) -> None:
    """Fold one sample's per-genome payload into the running tables."""
    if kind == "ownrc":
        hits, sims = payload
        means = unseen.summarise(hits, ks)
        for k in ks:
            matrices[kind][k][layer][sample] = means[k]
        if sims:
            similarity[layer][sample] = float(np.nanmean(list(sims.values())))
        for genome, by_k in hits.items():
            for k in config.PER_GENOME_KS:
                if k in by_k:
                    per_genome[kind].append((layer, sample, genome, f"hit@{k}", by_k[k]))
        for genome, value in sims.items():
            per_genome[kind].append((layer, sample, genome, "similarity", value))
        return

    means = unseen.summarise(payload, ks)
    for k in ks:
        matrices[kind][k][layer][sample] = means[k]
    for genome, by_k in payload.items():
        for k in config.PER_GENOME_KS:
            if k in by_k:
                per_genome[kind].append((layer, sample, genome, f"P@{k}", by_k[k]))


def _run_pairwise_task(packed):
    """One (model, pair of orders) over all reference samples."""
    (model_key, first, second, n_ref, latent_dir, keep_layers) = packed
    model = get(model_key)
    latent_dir = Path(latent_dir) if latent_dir else None
    problems = []
    key = (model_key, first[0], second[0])

    loaded_files = [store.load_virus(model, name, latent_dir, problems)
                    for name in (first[0], first[1], second[0], second[1])]
    if any(item is None for item in loaded_files):
        return key, {"fatal": "forward/RC files of the pair are missing", "problems": problems}
    forward_a, reverse_a, forward_b, reverse_b = loaded_files
    layers = list(forward_a.keys()) if model.extractor == "viralm" else None

    def one_sample(index):
        reference_data = store.load_sample(model, index, latent_dir, layers, problems)
        if reference_data is None:
            return index, None
        out = {}
        for layer in _layers_of(reference_data, forward_a, model.layered):
            if keep_layers is not None and layer not in keep_layers:
                continue
            views = [_split(model, data, layer)
                     for data in (reference_data, forward_a, reverse_a, forward_b, reverse_b)]
            if not all(views):
                continue
            out[layer] = pairwise.all_combinations(*views, PAIRWISE_K, model)
        return index, out

    matrices = defaultdict(lambda: defaultdict(dict))
    loaded, missing = [], []
    with ThreadPoolExecutor(max_workers=config.SAMPLE_THREADS) as pool:
        for future in as_completed([pool.submit(one_sample, i) for i in range(n_ref)]):
            index, out = future.result()
            if out is None:
                missing.append(index)
                continue
            loaded.append(index)
            for layer, values in out.items():
                for name, value in values.items():
                    matrices[name][layer][index] = value
    return key, {"matrices": {name: {layer: dict(v) for layer, v in by_layer.items()}
                              for name, by_layer in matrices.items()},
                 "loaded": sorted(loaded), "missing": sorted(missing), "problems": problems}


# ── building the task list ──────────────────────────────────────────────────

def _keep_layers(model: Model, layers_mode: str):
    """Layer labels to score; ``None`` means all of them.

    ``selection.json`` holds the chosen layer as a **position** in the benchmark
    table; the labels of the metric tables differ for ViraLM (1..12 against 0..11),
    so the position is translated, never used as a label.
    """
    if layers_mode != "selected" or not model.layered:
        return None
    entry = selection.load().get(model.key, {})
    if entry.get("layer") is None:
        return None
    return {selection.layer_label(model, entry["layer"])}


def build_tasks(models, viruses, n_ref, ks, latent_dir, layers_mode, metrics_dir,
                protocol="both"):
    """One task per (model, virus file); RC files are scored as viruses of their own."""
    tasks = []
    for model in models:
        keep = _keep_layers(model, layers_mode)
        shared = (n_ref, tuple(ks), str(latent_dir) if latent_dir else None, keep, protocol)
        for name, rc_name in viruses:
            tasks.append((model.key, name, rc_name, *shared))
            if rc_name:
                tasks.append((model.key, rc_name, None, *shared))
    return tasks


def build_pairwise_tasks(models, viruses, n_ref, latent_dir, layers_mode):
    """One task per (model, pair of orders that both have a forward and an RC file)."""
    complete = sorted((name, rc) for name, rc in viruses if rc)
    tasks = []
    for model in models:
        keep = _keep_layers(model, layers_mode)
        for first, second in combinations(complete, 2):
            tasks.append((model.key, first, second, n_ref,
                          str(latent_dir) if latent_dir else None, keep))
    return tasks


# ── writing ─────────────────────────────────────────────────────────────────

MATRIX_PREFIX = {"genome": "P", "order": "Porder", "cross": "cross_P", "ownrc": "ownrc_hit"}


def write(result: dict, model_key: str, virus: str, ks, metrics_dir=None) -> None:
    directory = tables.model_dir(model_key, metrics_dir)
    for kind, by_k in result["matrices"].items():
        for k in ks:
            if by_k.get(k):
                tables.save_matrix(by_k[k], directory / tables.matrix_name(
                    MATRIX_PREFIX[kind], k, virus))
    if result["similarity"]:
        tables.save_matrix(result["similarity"],
                           directory / tables.own_rc_similarity_name(virus))
    for kind, rows in result["per_genome"].items():
        if rows:
            tables.save_per_genome(rows, directory / tables.per_genome_name(kind, virus))


def summarise(result: dict, model_key: str, virus: str, ks) -> list:
    """One row per (k, kind): the best layer by mean and its spread."""
    rows = []
    names = {"genome": "forward", "order": "order", "cross": "cross", "ownrc": "own_rc"}
    for kind, by_k in result["matrices"].items():
        for k in ks:
            matrix = by_k.get(k)
            if not matrix:
                continue
            means = {layer: float(np.mean(list(values.values())))
                     for layer, values in matrix.items() if values}
            means = {layer: mean for layer, mean in means.items() if np.isfinite(mean)}
            if not means:
                continue
            best = max(means, key=means.get)
            rows.append({"model": model_key, "virus": virus, "type": names[kind], "k": k,
                         "best_layer": best, "mean": round(means[best], 6),
                         "std": round(float(np.std(list(matrix[best].values()))), 6)})
    return rows


def write_pairwise(result: dict, model_key: str, first: str, second: str,
                   metrics_dir=None) -> None:
    label = lambda name: name.replace("_preproc", "")
    directory = (tables.model_dir(model_key, metrics_dir)
                 / f"pairwise_{label(first)}_vs_{label(second)}")
    for name, matrix in result["matrices"].items():
        tables.save_matrix(matrix, directory / f"{name}_matrix.xlsx")


def _report(problems: list, allow_missing: bool) -> None:
    for line in problems:
        print(("  WARNING " if allow_missing else "  PROBLEM ") + line)


def run(models, viruses, n_ref=config.N_REFERENCE, ks=config.KS, latent_dir=None,
        metrics_dir=None, layers_mode="all", max_workers=None, dry_run=False,
        protocol="both", allow_missing=False, with_pairwise=True):
    """Compute every unseen-order metric and write the tables; returns summary rows.

    Raises ``SystemExit`` (non-zero) if anything was missing, corrupt or skipped,
    unless ``allow_missing``; the tables of an affected task are then not written.
    """
    ks = tuple(ks)
    tasks = build_tasks(models, viruses, n_ref, ks, latent_dir, layers_mode,
                        metrics_dir, protocol)
    pair_tasks = (build_pairwise_tasks(models, viruses, n_ref, latent_dir, layers_mode)
                  if with_pairwise else [])
    print(f"tasks: {len(tasks)} per-order + {len(pair_tasks)} pairwise  "
          f"({len(models)} models, n_ref={n_ref}, layers={layers_mode}, "
          f"protocol={protocol})")
    if dry_run:
        for task in tasks:
            print(f"  {task[0]:<10} {task[1]}")
        for task in pair_tasks:
            print(f"  {task[0]:<10} pairwise {task[1][0]} vs {task[2][0]}")
        return []

    rows, problems = [], []
    totals = {}                                        # model -> [fewest loaded, wanted]

    def vet(label, result, model_key=None):
        """True if the result may be written; records every problem."""
        if "fatal" in result:
            problems.append(f"{label}: {result['fatal']}")
            return False
        if result.get("error"):
            problems.append(f"{label}: skipped, {result['error']}")
            return False
        for name, reason in result["problems"]:
            problems.append(f"{label}: {name}: {reason}")
        if model_key:
            seen = totals.setdefault(model_key, [n_ref, n_ref])
            seen[0] = min(seen[0], len(result["loaded"]))
        if result["missing"]:
            problems.append(f"{label}: {len(result['missing'])} of {n_ref} reference "
                            f"samples missing or unreadable "
                            f"({_ranges(result['missing'])})")
            if not allow_missing:
                return False
        return bool(result["loaded"])

    with ProcessPoolExecutor(max_workers=max_workers or config.MAX_WORKERS) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        progress = tqdm(as_completed(futures), total=len(tasks), desc="metrics", unit="task")
        for future in progress:
            (model_key, virus), result = future.result()
            if not vet(f"{model_key}/{virus}", result, model_key):
                continue
            write(result, model_key, virus, ks, metrics_dir)
            rows.extend(summarise(result, model_key, virus, ks))
            progress.set_postfix_str(f"{model_key}/{virus}")

        futures = [pool.submit(_run_pairwise_task, task) for task in pair_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="pairwise",
                           unit="task"):
            (model_key, first, second), result = future.result()
            if vet(f"{model_key}/{first}-vs-{second}", result):
                write_pairwise(result, model_key, first, second, metrics_dir)

    for model_key, (loaded, wanted) in sorted(totals.items()):
        print(f"  {model_key:<10} reference samples loaded: {loaded}/{wanted}")
    if problems:
        _report(problems, allow_missing)
        if not allow_missing:
            raise SystemExit(
                f"{len(problems)} problem(s) above: nothing was written for the affected "
                f"tasks. Fix the inputs (re-run 'extract'), or pass --allow-missing to "
                f"accept results computed from the samples that did load.")
    return rows


def _ranges(indices) -> str:
    """[0,1,2,7] -> '0-2, 7' for readable messages."""
    out, start = [], None
    indices = sorted(indices)
    for position, value in enumerate(indices):
        if start is None:
            start = previous = value
        elif value == previous + 1:
            previous = value
        else:
            out.append((start, previous))
            start = previous = value
        if position == len(indices) - 1:
            out.append((start, previous))
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in out)
