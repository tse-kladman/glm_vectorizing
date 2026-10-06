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
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import config, selection
from ..models import Model, get
from ..representations import store
from . import tables, unseen


def _layers_of(reference_data, virus_data, layered: bool):
    """The layers present in both the reference sample and the virus."""
    if not layered:
        return [tables.FLAT_LAYER]
    return [layer for layer in reference_data if layer in virus_data]


def _sample_scores(model: Model, reference_data, virus, virus_rc, ks, keep_layers):
    """P@k and CROSS P@k of one reference sample, per layer."""
    precision, cross = {}, {}
    for layer in _layers_of(reference_data, virus, model.layered):
        if keep_layers is not None and layer not in keep_layers:
            continue
        if model.layered:
            ref_groups, virus_groups = reference_data[layer], virus[layer]
            rc_groups = virus_rc.get(layer) if virus_rc else None
        else:
            ref_groups, virus_groups, rc_groups = reference_data, virus, virus_rc
        if not ref_groups or not virus_groups:
            continue

        precision[layer] = unseen.precision(ref_groups, virus_groups, ks, model)
        if rc_groups:
            cross[layer] = unseen.cross_precision(ref_groups, virus_groups,
                                                  rc_groups, ks, model)
    return precision, cross


def _run_task(packed):
    """One (model, virus file) over all reference samples; top level, for the pool."""
    model_key, virus_name, rc_name, n_ref, ks, latent_dir, keep_layers = packed
    model = get(model_key)
    latent_dir = Path(latent_dir) if latent_dir else None

    virus = store.load_virus(model, virus_name, latent_dir)
    if virus is None:
        return (model_key, virus_name), None
    virus_rc = store.load_virus(model, rc_name, latent_dir) if rc_name else None
    layers = list(virus.keys()) if model.extractor == "viralm" else None

    def one_sample(index):
        reference_data = store.load_sample(model, index, latent_dir, layers)
        if reference_data is None:
            return index, None
        try:
            return index, _sample_scores(model, reference_data, virus, virus_rc,
                                         ks, keep_layers)
        except ValueError as error:                 # representation/metric mismatch
            return index, error

    precision = {k: defaultdict(dict) for k in ks}
    cross = {k: defaultdict(dict) for k in ks}
    failure = None

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(one_sample, index) for index in range(n_ref)]
        for future in as_completed(futures):
            index, result = future.result()
            if result is None:
                continue
            if isinstance(result, ValueError):
                failure = str(result)
                break
            per_layer_precision, per_layer_cross = result
            for layer, values in per_layer_precision.items():
                for k, value in values.items():
                    precision[k][layer][index] = value
            for layer, values in per_layer_cross.items():
                for k, value in values.items():
                    cross[k][layer][index] = value

    if failure:
        return (model_key, virus_name), {"error": failure}
    if not any(precision[k] for k in ks):
        return (model_key, virus_name), None
    return (model_key, virus_name), {
        "precision": {k: dict(v) for k, v in precision.items()},
        "cross": {k: dict(v) for k, v in cross.items()},
    }


def build_tasks(models, viruses, n_ref, ks, latent_dir, layers_mode, metrics_dir):
    """One task per (model, virus file); RC files are scored as viruses of their own."""
    tasks = []
    for model in models:
        keep = None
        if layers_mode == "selected" and model.layered:
            entry = selection.load().get(model.key, {})
            if entry.get("layer") is not None:
                keep = {entry["layer"]}
        for name, rc_name in viruses:
            tasks.append((model.key, name, rc_name, n_ref, tuple(ks),
                          str(latent_dir) if latent_dir else None, keep))
            if rc_name:
                tasks.append((model.key, rc_name, None, n_ref, tuple(ks),
                              str(latent_dir) if latent_dir else None, keep))
    return tasks


def write(result: dict, model_key: str, virus: str, ks, metrics_dir=None) -> None:
    directory = tables.model_dir(model_key, metrics_dir)
    for k in ks:
        if result["precision"].get(k):
            tables.save_matrix(result["precision"][k],
                               directory / tables.matrix_name("P", k, virus))
        if result["cross"].get(k):
            tables.save_matrix(result["cross"][k],
                               directory / tables.matrix_name("cross_P", k, virus))


def summarise(result: dict, model_key: str, virus: str, ks) -> list:
    """One row per (k, kind): the best layer by mean and its spread."""
    rows = []
    for kind, key in (("forward", "precision"), ("cross", "cross")):
        for k in ks:
            matrix = result[key].get(k)
            if not matrix:
                continue
            means = {layer: float(np.mean(list(values.values())))
                     for layer, values in matrix.items() if values}
            if not means:
                continue
            best = max(means, key=means.get)
            rows.append({"model": model_key, "virus": virus, "type": kind, "k": k,
                         "best_layer": best, "mean": round(means[best], 6),
                         "std": round(float(np.std(list(matrix[best].values()))), 6)})
    return rows


def run(models, viruses, n_ref=config.N_REFERENCE, ks=config.KS, latent_dir=None,
        metrics_dir=None, layers_mode="all", max_workers=None, dry_run=False):
    """Compute every unseen-order metric and write the tables; returns summary rows."""
    tasks = build_tasks(models, viruses, n_ref, ks, latent_dir, layers_mode, metrics_dir)
    print(f"tasks: {len(tasks)}  ({len(models)} models x {len(tasks) // max(1, len(models))} "
          f"files, n_ref={n_ref}, layers={layers_mode})")
    if dry_run:
        for task in tasks:
            print(f"  {task[0]:<10} {task[1]}")
        return []

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers or config.MAX_WORKERS) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        progress = tqdm(as_completed(futures), total=len(tasks), desc="metrics", unit="task")
        for future in progress:
            (model_key, virus), result = future.result()
            if result is None:
                continue
            if "error" in result:
                print(f"  SKIPPED {model_key}/{virus}: {result['error']}")
                continue
            write(result, model_key, virus, ks, metrics_dir)
            rows.extend(summarise(result, model_key, virus, ks))
            progress.set_postfix_str(f"{model_key}/{virus}")
    return rows
