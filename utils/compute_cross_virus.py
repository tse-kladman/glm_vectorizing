#!/usr/bin/env python
"""
compute_cross_virus.py — попарный cross-virus P@K между НОВЫМИ вирусами.

В отличие от cross P@K в compute_metrics.py (Fwd↔RC одного вируса),
здесь считается, насколько фрагменты одного вируса находят фрагменты ДРУГОГО.

Для каждой пары вирусов (A ≠ B) и каждой комбинации цепей считаем P@K:
  - Fwd(A) vs Fwd(B):  query=Fwd(A), target=Fwd(B)
  - Fwd(A) vs RC(B):   query=Fwd(A), target=RC(B)
  - RC(A) vs Fwd(B):   query=RC(A),  target=Fwd(B)
  - RC(A) vs RC(B):    query=RC(A),  target=RC(B)

Два уровня параллелизма:
  - ProcessPoolExecutor: разные пары вирусов
  - ThreadPoolExecutor: разные референсные сэмплы

Использование:
    # Как модуль:
    from utils.compute_cross_virus import compute_cross_virus_all
    results = compute_cross_virus_all(preproc_files, rc_files, models, ...)

    # Standalone (запускать из корня репозитория):
    python utils/compute_cross_virus.py --latent_dir data/latent_dim \
        --metrics_dir data/metrics \
        --names Egovirales_preproc Mriyavirales_preproc \
        --models kmer dnabert2 dnabertS evo2 viralm

Результат: data/metrics/cross_virus/{Fwd_vs_Fwd,Fwd_vs_RC,RC_vs_Fwd,RC_vs_RC}/
           <model>/P@{k}_matrix_<query>_vs_<target>.xlsx  +  summary_cross_virus.xlsx
"""

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Позволяет запускать файл и как модуль (python -m / import), и как скрипт
# (python utils/compute_cross_virus.py) — во втором случае корень репозитория
# сам по себе в sys.path не попадает.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.compute_metrics import (calc_pair_cross_pk, _safe_pickle_load,
                                   _save_pk_matrix, _extract_best, viralm_get)

# ─────────────────────────── КОНФИГУРАЦИЯ ────────────────────────────────────

STRAND_COMBOS = ['Fwd_vs_Fwd', 'Fwd_vs_RC', 'RC_vs_Fwd', 'RC_vs_RC']


def _load_virus_embeddings(model_key: str, name: str, latent_dir: Path):
    """Загружает эмбеддинги одного вируса для заданной модели."""
    if model_key == 'viralm':
        # ViraLM: отдельные .pkl на каждый слой
        ref_dir = latent_dir / model_key
        virus_by_layer = {}
        for l in range(1, 13):
            data = _safe_pickle_load(ref_dir / f'{name}_layer_{l}.pkl')
            if data is not None:
                virus_by_layer[l] = viralm_get(data)
        return virus_by_layer
    else:
        # K-mer, DNABERT, Evo2: один .pickle
        ref_dir = latent_dir / model_key
        return _safe_pickle_load(ref_dir / f'{name}.pickle')


def _compute_cross_virus_kmer(query_name: str, target_name: str,
                               n_ref: int, k: int, latent_dir: Path):
    """Cross-virus P@K для k-mer (без слоёв)."""
    ref_dir = latent_dir / 'kmer'
    virus_query = _safe_pickle_load(ref_dir / f'{query_name}.pickle')
    virus_target = _safe_pickle_load(ref_dir / f'{target_name}.pickle')
    if virus_query is None or virus_target is None:
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, None
        return i, calc_pair_cross_pk(ref, virus_query, virus_target, k=k)

    pk_by_sample = [None] * n_ref
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  cross-virus kmer {query_name}->{target_name}',
                      leave=False):
            i, val = f.result()
            pk_by_sample[i] = val

    pk_by_sample = [v for v in pk_by_sample if v is not None]
    if not pk_by_sample:
        return None
    return {'best_layer': '-',
            'mean_pk': float(np.mean(pk_by_sample)),
            'std_pk': float(np.std(pk_by_sample)),
            'pk_by_sample': pk_by_sample}


def _compute_cross_virus_layered(model_key: str, query_name: str, target_name: str,
                                  n_ref: int, k: int, latent_dir: Path):
    """Cross-virus P@K для моделей со слоями (DNABERT, Evo2)."""
    ref_dir = latent_dir / model_key
    virus_query = _safe_pickle_load(ref_dir / f'{query_name}.pickle')
    virus_target = _safe_pickle_load(ref_dir / f'{target_name}.pickle')
    if virus_query is None or virus_target is None:
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, {}
        result = {}
        for l in ref:
            vq, vt = virus_query.get(l), virus_target.get(l)
            if vq is None or vt is None:
                continue
            if len(ref[l]) == 0 or len(vq) == 0 or len(vt) == 0:
                continue
            result[l] = calc_pair_cross_pk(ref[l], vq, vt, k=k)
        return i, result

    pk_matrix = defaultdict(dict)
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  cross-virus {model_key} {query_name}->{target_name}',
                      leave=False):
            i, layer_pks = f.result()
            for l, pk in layer_pks.items():
                pk_matrix[l][i] = pk

    if not pk_matrix:
        return None
    bl, bm, bs = _extract_best(pk_matrix)
    return {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
            'pk_matrix': dict(pk_matrix)}


def _compute_cross_virus_viralm(query_name: str, target_name: str,
                                 n_ref: int, k: int, latent_dir: Path):
    """Cross-virus P@K для ViraLM."""
    ref_dir = latent_dir / 'viralm'

    query_by_layer, target_by_layer = {}, {}
    for l in range(1, 13):
        qd = _safe_pickle_load(ref_dir / f'{query_name}_layer_{l}.pkl')
        td = _safe_pickle_load(ref_dir / f'{target_name}_layer_{l}.pkl')
        if qd is not None and td is not None:
            query_by_layer[l] = viralm_get(qd)
            target_by_layer[l] = viralm_get(td)
    if not query_by_layer:
        return None

    def _worker(i):
        result = {}
        for l in query_by_layer:
            data = _safe_pickle_load(ref_dir / f'sample_{i}_layer_{l}.pkl')
            if data is None:
                continue
            result[l] = calc_pair_cross_pk(viralm_get(data),
                                            query_by_layer[l],
                                            target_by_layer[l], k=k)
        return i, result

    pk_matrix = defaultdict(dict)
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  cross-virus viralm {query_name}->{target_name}',
                      leave=False):
            i, layer_pks = f.result()
            for l, pk in layer_pks.items():
                pk_matrix[l][i] = pk

    if not pk_matrix:
        return None
    bl, bm, bs = _extract_best(pk_matrix)
    return {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
            'pk_matrix': dict(pk_matrix)}


# ─────────────────────────── ОРКЕСТРАТОР ─────────────────────────────────────

def _compute_pair(model: str, query_name: str, target_name: str,
                  n_ref: int, k: int, latent_dir: Path):
    """Вычисляет cross-virus P@K для одной пары (query, target)."""
    if model == 'kmer':
        return _compute_cross_virus_kmer(query_name, target_name, n_ref, k, latent_dir)
    elif model == 'viralm':
        return _compute_cross_virus_viralm(query_name, target_name, n_ref, k, latent_dir)
    else:
        return _compute_cross_virus_layered(model, query_name, target_name,
                                             n_ref, k, latent_dir)


def _run_cross_virus_task(packed):
    """Top-level функция для ProcessPoolExecutor."""
    (model, combo, query_name, target_name), n_ref, k, latent_dir = packed
    result = _compute_pair(model, query_name, target_name, n_ref, k, latent_dir)
    return (model, combo, query_name, target_name), result


def compute_cross_virus_all(preproc_files: dict, rc_files: dict,
                            models: list, n_ref: int, k: int,
                            latent_dir: Path, metrics_dir: Path,
                            max_workers: int = None):
    """
    Попарный cross-virus P@K между всеми новыми вирусами.

    Параметры:
        preproc_files: {label: (path, name)} — forward файлы
        rc_files:      {label: (path, name)} — RC файлы
        models:        список моделей
        n_ref, k:      параметры P@K
        latent_dir:    базовая папка эмбеддингов
        metrics_dir:   выходная папка метрик
        max_workers:   максимум процессов (None = auto)

    Возвращает: словарь {(model, combo, pair): result}
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(preproc_files.keys())
    if len(labels) < 2:
        print("Нужно минимум 2 вируса для попарного сравнения.")
        return {}

    # ── Собираем задачи ───────────────────────────────────────────────────
    # Для каждой пары (A, B) и каждой комбинации цепей
    tasks = []
    for virus_a, virus_b in combinations(labels, 2):
        fwd_a = preproc_files[virus_a][1]
        fwd_b = preproc_files[virus_b][1]
        rc_a = rc_files.get(virus_a, (None, None))[1]
        rc_b = rc_files.get(virus_b, (None, None))[1]

        for model in models:
            # A → B; каждая задача требует только те цепи, которые сама использует
            if fwd_a and fwd_b:
                tasks.append((model, 'Fwd_vs_Fwd', fwd_a, fwd_b))
            if fwd_a and rc_b:
                tasks.append((model, 'Fwd_vs_RC', fwd_a, rc_b))
            if rc_a and fwd_b:
                tasks.append((model, 'RC_vs_Fwd', rc_a, fwd_b))
            if rc_a and rc_b:
                tasks.append((model, 'RC_vs_RC', rc_a, rc_b))

            # Обратное направление (B → A)
            if fwd_b and fwd_a:
                tasks.append((model, 'Fwd_vs_Fwd', fwd_b, fwd_a))
            if fwd_b and rc_a:
                tasks.append((model, 'Fwd_vs_RC', fwd_b, rc_a))
            if rc_b and fwd_a:
                tasks.append((model, 'RC_vs_Fwd', rc_b, fwd_a))
            if rc_b and rc_a:
                tasks.append((model, 'RC_vs_RC', rc_b, rc_a))

    print(f"Cross-virus tasks: {len(tasks)}  "
          f"({len(labels)} viruses × {len(models)} models × {len(list(combinations(labels, 2)))} pairs)")

    # ── Параллельный запуск ──────────────────────────────────────────────
    all_results = {}
    full_tasks = [(t, n_ref, k, latent_dir) for t in tasks]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_cross_virus_task, ft): ft[0] for ft in full_tasks}
        pbar = tqdm(as_completed(futures), total=len(tasks),
                    desc='Cross-virus metrics', unit='task')
        for f in pbar:
            key, result = f.result()
            if result is None:
                continue
            model, combo, query_name, target_name = key
            all_results[key] = result

            # Сохраняем матрицу
            pair_label = f'{query_name}_vs_{target_name}'
            out_dir = metrics_dir / 'cross_virus' / combo / model
            out_dir.mkdir(parents=True, exist_ok=True)
            if 'pk_matrix' in result:
                _save_pk_matrix(result['pk_matrix'],
                               out_dir / f'P@{k}_matrix_{pair_label}.xlsx')
            elif 'pk_by_sample' in result:
                matrix = {'-': {i: v for i, v in enumerate(result['pk_by_sample'])}}
                _save_pk_matrix(matrix, out_dir / f'P@{k}_matrix_{pair_label}.xlsx')

            pbar.set_postfix_str(f'{combo}/{model}/{pair_label}')

    return all_results


# ─────────────────────────── ВЫВОД ────────────────────────────────────────────

def print_cross_virus_summary(all_results: dict, k: int):
    """Печатает сводную таблицу cross-virus P@K."""
    print(f"\n{'=' * 70}")
    print(f"  Cross-virus P@{k} summary")
    print(f"{'=' * 70}")
    print(f"  {'Combo':<14} {'Model':<10} {'Pair':<40} {'Layer':>6}   P@K (mean ± std)")
    print(f"  {'-' * 66}")

    for (model, combo, query, target), r in sorted(all_results.items()):
        layer = str(r['best_layer'])
        print(f"  {combo:<14} {model:<10} {query} -> {target:<30} "
              f"{layer:>6}   {r['mean_pk']:.4f} ± {r['std_pk']:.4f}")

    print()


def save_cross_virus_summary(all_results: dict, metrics_dir: Path):
    """Сохраняет сводную таблицу cross-virus P@K в Excel."""
    rows = []
    for (model, combo, query, target), r in all_results.items():
        rows.append({
            'combo': combo,
            'model': model,
            'query_virus': query,
            'target_virus': target,
            'best_layer': r['best_layer'],
            'mean_P@K': round(r['mean_pk'], 4),
            'std_P@K': round(r['std_pk'], 4),
        })

    if rows:
        df = pd.DataFrame(rows)
        out = Path(metrics_dir) / 'cross_virus' / 'summary_cross_virus.xlsx'
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out, index=False)
        print(f"Cross-virus summary saved -> {out}")
    return rows


# ─────────────────────────── STANDALONE ───────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Pairwise cross-virus P@K computation')
    p.add_argument('--latent_dir', required=True, type=Path)
    p.add_argument('--metrics_dir', required=True, type=Path)
    p.add_argument('--names', nargs='+', required=True,
                   help='Forward file names (e.g. Egovirales_preproc Mriyavirales_preproc)')
    p.add_argument('--rc_names', nargs='*', default=None,
                   help='RC file names (default: REVERSECOMPLEMENT{name})')
    p.add_argument('--models', nargs='+',
                   default=['kmer', 'dnabert2', 'dnabertS', 'evo2', 'viralm'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--n_ref', type=int, default=50)
    p.add_argument('--max_workers', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    preproc_files = {}
    rc_files = {}
    for name in args.names:
        label = name.replace('_preproc', '')
        preproc_files[label] = (None, name)
        if args.rc_names:
            rc_name = args.rc_names[args.names.index(name)]
        else:
            rc_name = f'REVERSECOMPLEMENT{label}_preproc'
        rc_files[label] = (None, rc_name)

    all_results = compute_cross_virus_all(
        preproc_files=preproc_files,
        rc_files=rc_files,
        models=args.models,
        n_ref=args.n_ref,
        k=args.k,
        latent_dir=args.latent_dir,
        metrics_dir=args.metrics_dir,
        max_workers=args.max_workers,
    )

    print_cross_virus_summary(all_results, args.k)
    save_cross_virus_summary(all_results, args.metrics_dir)
