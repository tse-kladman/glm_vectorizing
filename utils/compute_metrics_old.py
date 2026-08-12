#!/usr/bin/env python
"""
compute_metrics.py — параллельный расчёт метрик P@K и cross P@K.

Два уровня параллелизма:
  - Внешний (ProcessPoolExecutor): разные вирусные файлы
  - Внутренний (ThreadPoolExecutor): разные референсные сэмплы

Использование из run_pipeline.py:
    from compute_metrics import compute_all_metrics
    results = compute_all_metrics(preproc_files, rc_files, ...)

Или standalone:
    python compute_metrics.py --latent_dir data/latent_dim --metrics_dir data/metrics \\
        --names Egovirales_preproc Mriyavirales_preproc --models kmer dnabert2
"""

import argparse
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─────────────────────────── БАЗОВЫЕ МЕТРИКИ ─────────────────────────────────

def precision_K_new(all_emb, all_labels, new_indices, k=5, metric='cosine'):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(all_emb)
    _, indices = nn.kneighbors(all_emb)
    all_labels = np.array(all_labels)
    precisions = []
    for idx in new_indices:
        nbrs = all_labels[indices[idx][1:]]
        precisions.append(np.mean(nbrs == all_labels[idx]))
    return float(np.mean(precisions)) if precisions else 0.0


def calc_metrics_with_new(reference: dict, virus: dict, k=5):
    """P@K: для каждого вирусного лейбла — пространство ref + только этот лейбл."""
    per_label_pks = []
    for vlabel, vembs in virus.items():
        if len(vembs) == 0:
            continue
        all_emb, all_labels = [], []
        for rlabel, rembs in reference.items():
            if len(rembs) == 0:
                continue
            all_emb.append(rembs)
            all_labels.extend([rlabel] * len(rembs))

        start = len(all_labels)
        all_emb.append(vembs)
        all_labels.extend([vlabel] * len(vembs))

        if not all_emb or start == 0:
            continue

        all_emb = np.vstack(all_emb)
        new_indices = np.arange(start, len(all_labels))
        all_emb = StandardScaler().fit_transform(all_emb)
        per_label_pks.append(precision_K_new(all_emb, all_labels, new_indices, k=k))

    return float(np.mean(per_label_pks)) if per_label_pks else 0.0


def calc_cross_pk(reference: dict, virus_fwd: dict, virus_rc: dict, k=5):
    """Cross P@K: доля RC-соседей среди k ближайших для каждого прямого фрагмента.

    Пространство поиска: ref + RC (без прямых фрагментов вируса).
    Прямые фрагменты — только query.
    """
    per_label_pks = []
    for vlabel, vembs in virus_fwd.items():
        if len(vembs) == 0:
            continue
        rc_label = f'REVERSECOMPLEMENT{vlabel}'
        rc_embs = virus_rc.get(rc_label)
        if rc_embs is None or len(rc_embs) == 0:
            continue

        # Пространство: ref + RC (без forward virus)
        space_emb, space_labels = [], []
        for rlabel, rembs in reference.items():
            if len(rembs) == 0:
                continue
            space_emb.append(rembs)
            space_labels.extend([rlabel] * len(rembs))

        space_emb.append(rc_embs)
        space_labels.extend([rc_label] * len(rc_embs))

        if not space_emb:
            continue

        space_emb = np.vstack(space_emb)
        fwd_emb = np.vstack([vembs])

        # Масштабируем совместно, затем разделяем
        combined = np.vstack([space_emb, fwd_emb])
        combined = StandardScaler().fit_transform(combined)
        space_scaled = combined[:len(space_labels)]
        fwd_scaled = combined[len(space_labels):]

        # Ищем k соседей fwd в пространстве ref+RC (self нет → n_neighbors=k)
        nn = NearestNeighbors(n_neighbors=k, metric='cosine')
        nn.fit(space_scaled)
        _, indices = nn.kneighbors(fwd_scaled)
        space_labels_arr = np.array(space_labels)

        precisions = []
        for idx_row in indices:
            nbrs = space_labels_arr[idx_row]
            precisions.append(float(np.mean(nbrs == rc_label)))
        per_label_pks.append(float(np.mean(precisions)))

    return float(np.mean(per_label_pks)) if per_label_pks else 0.0

# ─────────────────────────── CROSS-VIRUS P@K ──────────────────────────────────

def calc_cross_virus_pk(reference: dict, virus_query: dict, virus_target: dict, k=5):
    """Cross-virus P@K: насколько фрагменты одного вируса находят фрагменты другого.

    Пространство поиска: reference + все фрагменты virus_target.
    Запрос: все фрагменты virus_query.
    P@K: доля соседей из virus_target среди k ближайших.

    Используется для попарного сравнения НОВЫХ вирусов между собой
    (не путать с calc_cross_pk, который сравнивает Fwd↔RC одного вируса).
    """
    # Все эмбеддинги target-вируса
    target_all = []
    for vembs in virus_target.values():
        if len(vembs) > 0:
            target_all.append(vembs)
    if not target_all:
        return 0.0
    target_all = np.vstack(target_all)

    # Все эмбеддинги query-вируса
    query_all = []
    for vembs in virus_query.values():
        if len(vembs) > 0:
            query_all.append(vembs)
    if not query_all:
        return 0.0
    query_all = np.vstack(query_all)

    # Пространство поиска: reference + target
    space_emb, space_labels = [], []
    for rlabel, rembs in reference.items():
        if len(rembs) == 0:
            continue
        space_emb.append(rembs)
        space_labels.extend([rlabel] * len(rembs))

    space_emb.append(target_all)
    space_labels.extend(['TARGET'] * len(target_all))
    space_emb = np.vstack(space_emb)

    # Совместное масштабирование space + query, затем разделяем
    combined = np.vstack([space_emb, query_all])
    combined = StandardScaler().fit_transform(combined)
    space_scaled = combined[:len(space_labels)]
    query_scaled = combined[len(space_labels):]

    # Ищем k соседей
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(space_scaled)
    _, indices = nn.kneighbors(query_scaled)
    space_labels_arr = np.array(space_labels)

    # P@K: доля TARGET среди k ближайших
    precisions = []
    for idx_row in indices:
        nbrs = space_labels_arr[idx_row]
        precisions.append(float(np.mean(nbrs == 'TARGET')))

    return float(np.mean(precisions)) if precisions else 0.0


# ─────────────────────────── УТИЛИТЫ ─────────────────────────────────────────

def viralm_get(d: dict):
    """Ключи вида {label}_{chunk}_{start}_{end} → {label: [arrays]}."""
    res = defaultdict(list)
    for key, val in d.items():
        res[key.rsplit('_', 3)[0]].append(val)
    return dict(res)


def _safe_pickle_load(path: Path):
    """Загружает pickle, возвращает None если файл повреждён или пуст."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"  WARNING: не удалось прочитать {path.name}: {e}")
        return None


def _save_pk_matrix(pk_matrix: dict, out_path: Path):
    """Сохраняет матрицу P@K: строки = слои, столбцы = номера сэмплов."""
    df = pd.DataFrame(pk_matrix).T.sort_index()
    df.index.name = 'layer'
    df.columns = [f'sample_{c}' for c in df.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path)


def _extract_best(pk_matrix: dict):
    """Из pk_matrix {layer: {sample_i: pk}} → best_layer, mean, std."""
    layer_means = {l: np.mean(list(pk_matrix[l].values())) for l in pk_matrix}
    best_layer = max(layer_means, key=layer_means.get)
    best_mean = float(layer_means[best_layer])
    best_std = float(np.std(list(pk_matrix[best_layer].values())))
    return best_layer, best_mean, best_std

# ─────────────────────────── P@K ПО МОДЕЛЯМ ──────────────────────────────────

def _compute_kmer(name: str, n_ref: int, k: int, latent_dir: Path):
    ref_dir = latent_dir / 'kmer'
    virus = _safe_pickle_load(ref_dir / f'{name}.pickle')
    if virus is None:
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, None
        return i, calc_metrics_with_new(ref, virus, k=k)

    pk_by_sample = [None] * n_ref
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  kmer {name}', leave=False):
            i, val = f.result()
            pk_by_sample[i] = val

    pk_by_sample = [v for v in pk_by_sample if v is not None]
    if not pk_by_sample:
        return None
    return {'best_layer': '-',
            'mean_pk': float(np.mean(pk_by_sample)),
            'std_pk': float(np.std(pk_by_sample)),
            'pk_by_sample': pk_by_sample}


def _compute_layered(model_key: str, name: str, n_ref: int, k: int,
                     latent_dir: Path):
    ref_dir = latent_dir / model_key
    virus = _safe_pickle_load(ref_dir / f'{name}.pickle')
    if virus is None:
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, {}
        result = {}
        for l in ref:
            vl = virus.get(l)
            if vl is None or len(ref[l]) == 0 or len(vl) == 0:
                continue
            result[l] = calc_metrics_with_new(ref[l], vl, k=k)
        return i, result

    pk_matrix = defaultdict(dict)
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  {model_key} {name}', leave=False):
            i, layer_pks = f.result()
            for l, pk in layer_pks.items():
                pk_matrix[l][i] = pk

    if not pk_matrix:
        return None
    bl, bm, bs = _extract_best(pk_matrix)
    return {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
            'pk_matrix': dict(pk_matrix)}


def _compute_viralm(name: str, n_ref: int, k: int,
                    latent_dir: Path, n_layers=12):
    ref_dir = latent_dir / 'viralm'

    virus_by_layer = {}
    for l in range(1, n_layers + 1):
        data = _safe_pickle_load(ref_dir / f'{name}_layer_{l}.pkl')
        if data is not None:
            virus_by_layer[l] = viralm_get(data)
    if not virus_by_layer:
        return None

    def _worker(i):
        result = {}
        for l, virus in virus_by_layer.items():
            data = _safe_pickle_load(ref_dir / f'sample_{i}_layer_{l}.pkl')
            if data is None:
                continue
            result[l] = calc_metrics_with_new(viralm_get(data), virus, k=k)
        return i, result

    pk_matrix = defaultdict(dict)
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  viralm {name}', leave=False):
            i, layer_pks = f.result()
            for l, pk in layer_pks.items():
                pk_matrix[l][i] = pk

    if not pk_matrix:
        return None
    bl, bm, bs = _extract_best(pk_matrix)
    return {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
            'pk_matrix': dict(pk_matrix)}

# ─────────────────────────── CROSS P@K ПО МОДЕЛЯМ ────────────────────────────

def _compute_cross_kmer(fwd_name: str, rc_name: str, n_ref: int, k: int,
                        latent_dir: Path):
    ref_dir = latent_dir / 'kmer'
    virus_fwd = _safe_pickle_load(ref_dir / f'{fwd_name}.pickle')
    virus_rc = _safe_pickle_load(ref_dir / f'{rc_name}.pickle')
    if virus_fwd is None or virus_rc is None:
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, None
        return i, calc_cross_pk(ref, virus_fwd, virus_rc, k=k)

    pk_by_sample = [None] * n_ref
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  cross kmer {fwd_name}', leave=False):
            i, val = f.result()
            pk_by_sample[i] = val

    pk_by_sample = [v for v in pk_by_sample if v is not None]
    if not pk_by_sample:
        return None
    return {'best_layer': '-',
            'mean_pk': float(np.mean(pk_by_sample)),
            'std_pk': float(np.std(pk_by_sample)),
            'pk_by_sample': pk_by_sample}


def _compute_cross_layered(model_key: str, fwd_name: str, rc_name: str,
                           n_ref: int, k: int, latent_dir: Path):
    ref_dir = latent_dir / model_key
    virus_fwd = _safe_pickle_load(ref_dir / f'{fwd_name}.pickle')
    virus_rc = _safe_pickle_load(ref_dir / f'{rc_name}.pickle')
    if virus_fwd is None or virus_rc is None:
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, {}
        result = {}
        for l in ref:
            vf, vr = virus_fwd.get(l), virus_rc.get(l)
            if vf is None or vr is None:
                continue
            if len(ref[l]) == 0 or len(vf) == 0 or len(vr) == 0:
                continue
            result[l] = calc_cross_pk(ref[l], vf, vr, k=k)
        return i, result

    pk_matrix = defaultdict(dict)
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  cross {model_key} {fwd_name}', leave=False):
            i, layer_pks = f.result()
            for l, pk in layer_pks.items():
                pk_matrix[l][i] = pk

    if not pk_matrix:
        return None
    bl, bm, bs = _extract_best(pk_matrix)
    return {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
            'pk_matrix': dict(pk_matrix)}


def _compute_cross_viralm(fwd_name: str, rc_name: str, n_ref: int, k: int,
                          latent_dir: Path, n_layers=12):
    ref_dir = latent_dir / 'viralm'

    fwd_by_layer, rc_by_layer = {}, {}
    for l in range(1, n_layers + 1):
        fd = _safe_pickle_load(ref_dir / f'{fwd_name}_layer_{l}.pkl')
        rd = _safe_pickle_load(ref_dir / f'{rc_name}_layer_{l}.pkl')
        if fd is not None and rd is not None:
            fwd_by_layer[l] = viralm_get(fd)
            rc_by_layer[l] = viralm_get(rd)
    if not fwd_by_layer:
        return None

    def _worker(i):
        result = {}
        for l in fwd_by_layer:
            data = _safe_pickle_load(ref_dir / f'sample_{i}_layer_{l}.pkl')
            if data is None:
                continue
            result[l] = calc_cross_pk(viralm_get(data),
                                      fwd_by_layer[l], rc_by_layer[l], k=k)
        return i, result

    pk_matrix = defaultdict(dict)
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  cross viralm {fwd_name}', leave=False):
            i, layer_pks = f.result()
            for l, pk in layer_pks.items():
                pk_matrix[l][i] = pk

    if not pk_matrix:
        return None
    bl, bm, bs = _extract_best(pk_matrix)
    return {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
            'pk_matrix': dict(pk_matrix)}

# ─────────────────────────── ОРКЕСТРАТОРЫ ─────────────────────────────────────

def _run_model(model: str, name: str, n_ref: int, k: int, latent_dir: Path):
    """Считает P@K одной модели для одного файла (вызывается в отдельном процессе)."""
    if model == 'kmer':
        return _compute_kmer(name, n_ref, k, latent_dir)
    elif model == 'viralm':
        return _compute_viralm(name, n_ref, k, latent_dir)
    else:
        return _compute_layered(model, name, n_ref, k, latent_dir)


def _run_cross_model(model: str, fwd_name: str, rc_name: str,
                     n_ref: int, k: int, latent_dir: Path):
    """Считает cross P@K одной модели для одного файла."""
    if model == 'kmer':
        return _compute_cross_kmer(fwd_name, rc_name, n_ref, k, latent_dir)
    elif model == 'viralm':
        return _compute_cross_viralm(fwd_name, rc_name, n_ref, k, latent_dir)
    else:
        return _compute_cross_layered(model, fwd_name, rc_name,
                                      n_ref, k, latent_dir)


def _run_task(packed):
    """Top-level функция для ProcessPoolExecutor (должна быть picklable)."""
    task, n_ref, k, latent_dir = packed
    kind, model, name, rc_name = task
    if kind in ('fwd', 'rc'):
        return task, _run_model(model, name, n_ref, k, latent_dir)
    else:
        return task, _run_cross_model(model, name, rc_name,
                                      n_ref, k, latent_dir)


def _save_result(r: dict, model: str, name: str, k: int,
                 metrics_dir: Path, prefix: str = 'P'):
    """Сохраняет матрицу P@K / cross P@K в Excel."""
    out_dir = metrics_dir / model
    out_dir.mkdir(parents=True, exist_ok=True)
    if 'pk_matrix' in r:
        _save_pk_matrix(r['pk_matrix'],
                        out_dir / f'{prefix}@{k}_matrix_{name}.xlsx')
    elif 'pk_by_sample' in r:
        matrix = {'-': {i: v for i, v in enumerate(r['pk_by_sample'])}}
        _save_pk_matrix(matrix, out_dir / f'{prefix}@{k}_matrix_{name}.xlsx')


def compute_all_metrics(preproc_files: dict, rc_files: dict,
                        models: list, n_ref: int, k: int,
                        latent_dir: Path, metrics_dir: Path,
                        max_workers: int = None):
    """
    Параллельный расчёт всех метрик для всех файлов.

    Параметры:
        preproc_files: {label: (path, name)} — forward файлы
        rc_files:      {label: (path, name)} — RC файлы
        models:        ['kmer', 'dnabert2', ...] — список моделей
        n_ref:         количество референсных сэмплов
        k:             k для P@K
        latent_dir:    базовая папка эмбеддингов
        metrics_dir:   выходная папка метрик
        max_workers:   максимум параллельных процессов (None = auto)

    Возвращает: (all_fwd, all_rc, all_cross) — три словаря {name: {model: result}}
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Собираем все задачи ───────────────────────────────────────────────
    # Каждая задача: (тип, model, name, [rc_name])
    tasks = []

    for label, (_, name) in preproc_files.items():
        for model in models:
            tasks.append(('fwd', model, name, None))

    for label, (_, rc_name) in rc_files.items():
        for model in models:
            tasks.append(('rc', model, rc_name, None))

    for label in preproc_files:
        if label not in rc_files:
            continue
        fwd_name = preproc_files[label][1]
        rc_name = rc_files[label][1]
        for model in models:
            tasks.append(('cross', model, fwd_name, rc_name))

    # ── Параллельный запуск ───────────────────────────────────────────────
    all_fwd = defaultdict(dict)    # name → {model: result}
    all_rc = defaultdict(dict)
    all_cross = defaultdict(dict)

    # Упаковываем общие параметры в каждую задачу (для pickling)
    full_tasks = [(t, n_ref, k, latent_dir) for t in tasks]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_task, ft): ft[0] for ft in full_tasks}
        pbar = tqdm(as_completed(futures), total=len(tasks),
                    desc='All metrics', unit='task')
        for f in pbar:
            task, result = f.result()
            kind, model, name, rc_name = task

            if result is None:
                continue

            if kind == 'fwd':
                all_fwd[name][model] = result
                _save_result(result, model, name, k, metrics_dir, prefix='P')
            elif kind == 'rc':
                all_rc[name][model] = result
                _save_result(result, model, name, k, metrics_dir, prefix='P')
            else:
                all_cross[name][model] = result
                _save_result(result, model, name, k, metrics_dir, prefix='cross_P')

            pbar.set_postfix_str(f'{kind}/{model}/{name}')

    return dict(all_fwd), dict(all_rc), dict(all_cross)


def print_summary(name: str, results: dict, k: int, prefix: str = ''):
    """Печатает таблицу результатов для одного файла."""
    label = f'{prefix}P@{k}' if prefix else f'P@{k}'
    print(f"\n{'=' * 60}")
    print(f"  {label}: {name}")
    print(f"{'=' * 60}")
    print(f"  {'Модель':<12} {'Лучший слой':>12}   {label} (mean +/- std)")
    print(f"  {'-' * 52}")
    for model, r in results.items():
        layer = str(r['best_layer'])
        print(f"  {model:<12} {layer:>12}   "
              f"{r['mean_pk']:.4f} +/- {r['std_pk']:.4f}")
    print()


def save_summary_table(all_fwd: dict, all_rc: dict, all_cross: dict,
                       metrics_dir: Path):
    """Сводная таблица по всем вирусам в Excel."""
    rows = []
    for vname, results in all_fwd.items():
        for model, r in results.items():
            rows.append({'virus': vname, 'type': 'forward', 'model': model,
                         'best_layer': r['best_layer'],
                         'mean_P@K': round(r['mean_pk'], 4),
                         'std_P@K': round(r['std_pk'], 4)})
    for vname, results in all_rc.items():
        for model, r in results.items():
            rows.append({'virus': vname, 'type': 'RC', 'model': model,
                         'best_layer': r['best_layer'],
                         'mean_P@K': round(r['mean_pk'], 4),
                         'std_P@K': round(r['std_pk'], 4)})
    for vname, results in all_cross.items():
        for model, r in results.items():
            rows.append({'virus': vname, 'type': 'cross', 'model': model,
                         'best_layer': r['best_layer'],
                         'mean_P@K': round(r['mean_pk'], 4),
                         'std_P@K': round(r['std_pk'], 4)})

    if rows:
        df = pd.DataFrame(rows)
        out = metrics_dir / 'summary_new_viruses.xlsx'
        df.to_excel(out, index=False)
        print(f"Сводная таблица сохранена → {out}")
    return rows

# ─────────────────────────── STANDALONE ───────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Parallel metrics computation')
    p.add_argument('--latent_dir', required=True, type=Path)
    p.add_argument('--metrics_dir', required=True, type=Path)
    p.add_argument('--names', nargs='+', required=True,
                   help='Имена forward файлов (например Egovirales_preproc)')
    p.add_argument('--rc_names', nargs='*', default=None,
                   help='Имена RC файлов (по умолчанию REVERSECOMPLEMENT{name})')
    p.add_argument('--models', nargs='+',
                   default=['kmer', 'dnabert2', 'dnabertS', 'evo2', 'viralm'])
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--n_ref', type=int, default=50)
    p.add_argument('--max_workers', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # Строим preproc_files и rc_files из аргументов
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

    all_fwd, all_rc, all_cross = compute_all_metrics(
        preproc_files=preproc_files,
        rc_files=rc_files,
        models=args.models,
        n_ref=args.n_ref,
        k=args.k,
        latent_dir=args.latent_dir,
        metrics_dir=args.metrics_dir,
        max_workers=args.max_workers,
    )

    # Вывод
    for name, results in all_fwd.items():
        print_summary(name, results, args.k)
    for name, results in all_rc.items():
        print_summary(name, results, args.k, prefix='RC ')
    for name, results in all_cross.items():
        print_summary(name, results, args.k, prefix='CROSS ')

    save_summary_table(all_fwd, all_rc, all_cross, args.metrics_dir)
