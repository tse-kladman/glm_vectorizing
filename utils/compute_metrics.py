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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─────────────────────────── ПРОТОКОЛЫ ───────────────────────────────────────
#
# Метрика неотделима от представления: JS определён только для распределений, а
# MinHash вообще работает не с вектором частот, а со множеством k-меров.  Единицей
# расчёта поэтому служит протокол = (преобразование, метрика, требуемое представление).
#
#   P0  стандартизация + косинус     — исторический протокол всех текущих таблиц
#   P1  без преобразования + косинус — контроль: отделяет эффект масштабирования
#   P2  стандартизация + евклид
#   P3  без преобразования + евклид
#   P4  MinHash: оценка Жаккара по скетчу      (только скетчи, см. minhash_sketch.py)
#   P5  sqrt(JSD) = Jensen-Shannon             (только распределения)

@dataclass(frozen=True)
class Protocol:
    name: str
    transform: str   # 'standardize' | 'none' | 'sqrt'
    metric: str      # 'cosine' | 'euclidean' | 'jensenshannon' | 'minhash'
    requires: str    # 'any' | 'distribution' | 'sketch'
    label: str


PROTOCOLS = {p.name: p for p in [
    Protocol('P0', 'standardize', 'cosine',        'any',          'standardize + cosine'),
    Protocol('P1', 'none',        'cosine',        'any',          'raw + cosine'),
    Protocol('P2', 'standardize', 'euclidean',     'any',          'standardize + euclidean'),
    Protocol('P3', 'none',        'euclidean',     'any',          'raw + euclidean'),
    Protocol('P4', 'none',        'minhash',       'sketch',       'MinHash (Jaccard)'),
    Protocol('P5', 'none',        'jensenshannon', 'distribution', 'Jensen-Shannon'),
]}

DEFAULT_PROTOCOL = PROTOCOLS['P0']

# Блок памяти для векторных расчётов JS/MinHash (байты на промежуточный тензор).
_JS_BLOCK_BYTES = 128 << 20


def _as_protocol(protocol) -> Protocol:
    if protocol is None:
        return DEFAULT_PROTOCOL
    return protocol if isinstance(protocol, Protocol) else PROTOCOLS[str(protocol)]


def _check_distribution(X: np.ndarray, protocol: Protocol):
    """P5 и sqrt-преобразование требуют вектора-распределения: x >= 0 и sum(x) = 1."""
    if not X.size:
        return
    if X.min() < -1e-9:
        raise ValueError(f'{protocol.name} ({protocol.label}) требует неотрицательных '
                         f'векторов, найдено {X.min():.4g}')
    # Нормировку достаточно проверить на выборке строк: все они из одного источника.
    # Нулевая строка — это пустой фрагмент (не нашлось ни одного валидного k-мера);
    # такие встречаются и раньше молча проходили, поэтому их не считаем ошибкой.
    sums = X[::max(1, len(X) // 32)].sum(axis=1)
    bad = sums[(np.abs(sums - 1.0) > 1e-6) & (np.abs(sums) > 1e-9)]
    if bad.size:
        raise ValueError(f'{protocol.name} ({protocol.label}) требует распределений '
                         f'(sum = 1), найдена сумма {bad[0]:.4g}')


def _check_sketch(X: np.ndarray, protocol: Protocol):
    """P4 работает только со скетчами MinHash, а не с векторами признаков."""
    if X.size and not np.issubdtype(X.dtype, np.integer):
        raise ValueError(f'{protocol.name} ({protocol.label}) требует целочисленных '
                         f'скетчей MinHash (см. utils/minhash_sketch.py), '
                         f'получено {X.dtype}')


def _apply_transform(X: np.ndarray, protocol: Protocol) -> np.ndarray:
    if protocol.requires == 'sketch':
        _check_sketch(X, protocol)
        return X                      # скетч нельзя масштабировать: это не вектор
    if protocol.transform == 'standardize':
        return StandardScaler().fit_transform(X)
    if protocol.transform == 'sqrt':
        _check_distribution(X, protocol)
        return np.sqrt(X)
    if protocol.transform == 'none':
        if protocol.requires == 'distribution':
            _check_distribution(X, protocol)
        return X
    raise ValueError(f'unknown transform: {protocol.transform}')


# ─────────────────────────── ПОИСК СОСЕДЕЙ ───────────────────────────────────

def _xlog2x(X: np.ndarray) -> np.ndarray:
    """x*log2(x) с соглашением 0*log0 = 0, без вычисления log от нуля."""
    return X * np.log2(np.where(X > 0, X, 1.0))


def _entropy(X: np.ndarray) -> np.ndarray:
    """Построчная энтропия в битах (нули трактуются как 0 log 0 = 0)."""
    return -_xlog2x(X).sum(axis=1)


def _js_knn(query: np.ndarray, space: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Индексы n ближайших соседей по расстоянию sqrt(JSD).

    JSD(P,Q) = H((P+Q)/2) - (H(P)+H(Q))/2, поэтому энтропии строк считаются один
    раз, а попарно нужна только энтропия смеси.  Корень — монотонная функция, на
    порядок соседей он не влияет, поэтому ранжируем прямо по JSD.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    space = np.ascontiguousarray(space, dtype=np.float32)
    h_query, h_space = _entropy(query), _entropy(space)

    n_space, dim = space.shape
    block = max(1, int(_JS_BLOCK_BYTES / max(1, n_space * dim * 4)))

    out = np.empty((len(query), n_neighbors), dtype=np.int64)
    for start in range(0, len(query), block):
        stop = min(start + block, len(query))
        mixture = (query[start:stop, None, :] + space[None, :, :]) * 0.5
        h_mixture = -_xlog2x(mixture).sum(axis=2)
        jsd = h_mixture - 0.5 * (h_query[start:stop, None] + h_space[None, :])
        part = np.argpartition(jsd, n_neighbors - 1, axis=1)[:, :n_neighbors]
        order = np.take_along_axis(jsd, part, axis=1).argsort(axis=1)
        out[start:stop] = np.take_along_axis(part, order, axis=1)
    return out


def _minhash_knn(query: np.ndarray, space: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Индексы ближайших соседей по расстоянию MinHash (1 - оценка Жаккара).

    Скетч — это s независимых минимумов хэшей множества k-меров, построенных
    minhash_sketch.py.  Для двух множеств вероятность совпадения минимума по
    одной хэш-функции равна их коэффициенту Жаккара, поэтому доля совпавших
    позиций скетча — несмещённая оценка J, а 1 - J задаёт ранжирование.
    """
    query = np.ascontiguousarray(query, dtype=np.uint64)
    space = np.ascontiguousarray(space, dtype=np.uint64)

    n_space, sketch_size = space.shape
    block = max(1, int(_JS_BLOCK_BYTES / max(1, n_space * sketch_size)))

    out = np.empty((len(query), n_neighbors), dtype=np.int64)
    for start in range(0, len(query), block):
        stop = min(start + block, len(query))
        agree = (query[start:stop, None, :] == space[None, :, :]).sum(axis=2)
        distance = 1.0 - agree / sketch_size
        part = np.argpartition(distance, n_neighbors - 1, axis=1)[:, :n_neighbors]
        order = np.take_along_axis(distance, part, axis=1).argsort(axis=1)
        out[start:stop] = np.take_along_axis(part, order, axis=1)
    return out


def _knn_indices(query: np.ndarray, space: np.ndarray, n_neighbors: int,
                 metric: str) -> np.ndarray:
    n_neighbors = min(n_neighbors, len(space))
    if metric == 'jensenshannon':
        return _js_knn(query, space, n_neighbors)
    if metric == 'minhash':
        return _minhash_knn(query, space, n_neighbors)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(space)
    return nn.kneighbors(query, return_distance=False)


def _prefix_precisions(neighbour_labels: np.ndarray, targets, ks) -> dict:
    """P@k для всех k сразу: префиксы одного и того же списка соседей."""
    targets = np.asarray(targets)
    if targets.ndim == 0:
        targets = np.repeat(targets, len(neighbour_labels))
    hits = (neighbour_labels == targets[:, None]).astype(np.float64)
    cumulative = np.cumsum(hits, axis=1)
    got = neighbour_labels.shape[1]
    return {k: float(np.mean(cumulative[:, min(k, got) - 1] / min(k, got))) for k in ks}


# ─────────────────────────── БАЗОВЫЕ МЕТРИКИ ─────────────────────────────────

def precision_K_new(all_emb, all_labels, new_indices, k=5, metric='cosine'):
    """P@K для точек new_indices в общем пространстве all_emb (self исключается)."""
    all_labels = np.asarray(all_labels)
    new_indices = np.asarray(new_indices)
    if len(new_indices) == 0:
        return 0.0
    # Соседи нужны только для новых точек — остальные строки не запрашиваем.
    idx = _knn_indices(all_emb[new_indices], all_emb, k + 1, metric)[:, 1:]
    return _prefix_precisions(all_labels[idx], all_labels[new_indices], [k])[k]


def calc_metrics_with_new(reference: dict, virus: dict, k=5, protocol=None, ks=None):
    """P@K: для каждого вирусного лейбла — пространство ref + только этот лейбл.

    ``ks`` задаёт набор k, считаемых за один проход (Fig. 6); тогда возвращается
    словарь {k: P@k}, иначе — одно число для ``k``.
    """
    protocol = _as_protocol(protocol)
    wanted = sorted(ks) if ks else [k]
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

        all_emb = _apply_transform(np.vstack(all_emb), protocol)
        all_labels = np.asarray(all_labels)
        new_indices = np.arange(start, len(all_labels))

        idx = _knn_indices(all_emb[new_indices], all_emb, max(wanted) + 1,
                           protocol.metric)[:, 1:]
        per_label_pks.append(_prefix_precisions(all_labels[idx],
                                                all_labels[new_indices], wanted))

    if not per_label_pks:
        return {k: 0.0 for k in wanted} if ks else 0.0
    averaged = {k: float(np.mean([p[k] for p in per_label_pks])) for k in wanted}
    return averaged if ks else averaged[k]


def calc_cross_pk(reference: dict, virus_fwd: dict, virus_rc: dict, k=5,
                  protocol=None, ks=None):
    """Cross P@K: доля RC-соседей среди k ближайших для каждого прямого фрагмента.

    Пространство поиска: ref + RC (без прямых фрагментов вируса).
    Прямые фрагменты — только query.
    """
    protocol = _as_protocol(protocol)
    wanted = sorted(ks) if ks else [k]
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

        # Преобразуем совместно, затем разделяем
        combined = _apply_transform(np.vstack([space_emb, fwd_emb]), protocol)
        space_scaled = combined[:len(space_labels)]
        fwd_scaled = combined[len(space_labels):]

        # Ищем соседей fwd в пространстве ref+RC (self нет → n_neighbors=k)
        idx = _knn_indices(fwd_scaled, space_scaled, max(wanted), protocol.metric)
        per_label_pks.append(_prefix_precisions(np.asarray(space_labels)[idx],
                                                rc_label, wanted))

    if not per_label_pks:
        return {k: 0.0 for k in wanted} if ks else 0.0
    averaged = {k: float(np.mean([p[k] for p in per_label_pks])) for k in wanted}
    return averaged if ks else averaged[k]


def calc_pair_cross_pk(reference, query_virus, target_virus, k=5, protocol=None):
    """Cross P@K between two different viruses.

    Space = all reference fragments + all target_virus fragments.
    Query = all query_virus fragments (flattened across labels).
    Metric = fraction of k nearest neighbours that belong to target_virus.

    Does NOT do per-label matching — treats all target fragments as one class.
    """
    query_parts = [v for v in query_virus.values() if len(v) > 0]
    if not query_parts:
        return 0.0
    query_embs = np.vstack(query_parts)

    space_parts = []
    n_target = 0
    for rembs in reference.values():
        if len(rembs) > 0:
            space_parts.append(rembs)
    for tembs in target_virus.values():
        if len(tembs) > 0:
            space_parts.append(tembs)
            n_target += len(tembs)

    if not space_parts or n_target == 0:
        return 0.0

    space_embs = np.vstack(space_parts)
    space_is_target = np.array([False] * (len(space_embs) - n_target)
                               + [True] * n_target)

    protocol = _as_protocol(protocol)
    combined = _apply_transform(np.vstack([space_embs, query_embs]), protocol)
    space_scaled = combined[:len(space_embs)]
    query_scaled = combined[len(space_embs):]

    indices = _knn_indices(query_scaled, space_scaled, k, protocol.metric)

    precisions = space_is_target[indices].mean(axis=1)
    return float(precisions.mean())


def calc_pk_with_other(reference, query_virus, other_virus, k=5, protocol=None):
    """P@K for query_virus in a space that also contains other_virus.

    Per-label: space = ref + all of other_virus, query = one label of query_virus.
    Measures whether query fragments still find their own label when another
    virus occupies the embedding space.
    """
    protocol = _as_protocol(protocol)
    per_label_pks = []
    for vlabel, vembs in query_virus.items():
        if len(vembs) == 0:
            continue

        all_emb, all_labels = [], []
        for rlabel, rembs in reference.items():
            if len(rembs) == 0:
                continue
            all_emb.append(rembs)
            all_labels.extend([rlabel] * len(rembs))

        for olabel, oembs in other_virus.items():
            if len(oembs) == 0:
                continue
            all_emb.append(oembs)
            all_labels.extend([olabel] * len(oembs))

        start = len(all_labels)
        all_emb.append(vembs)
        all_labels.extend([vlabel] * len(vembs))

        if not all_emb or start == 0:
            continue

        all_emb = _apply_transform(np.vstack(all_emb), protocol)
        new_indices = np.arange(start, len(all_labels))
        per_label_pks.append(precision_K_new(all_emb, all_labels, new_indices,
                                             k=k, metric=protocol.metric))

    return float(np.mean(per_label_pks)) if per_label_pks else 0.0

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

# ─────────────────────────── PAIRWISE CROSS-VIRUS ────────────────────────────

PAIRWISE_METRICS = [
    'cross_1→2_fwd_rc', 'cross_1→2_rc_fwd', 'cross_1→2_fwd_fwd', 'cross_1→2_rc_rc',
    'cross_2→1_fwd_rc', 'cross_2→1_rc_fwd', 'cross_2→1_fwd_fwd', 'cross_2→1_rc_rc',
    'pk_fwd_1_in_2', 'pk_rc_1_in_2', 'pk_fwd_2_in_1', 'pk_rc_2_in_1',
]


def _compute_pairwise_cross_kmer(fwd_name_a, rc_name_a, fwd_name_b, rc_name_b,
                                 n_ref, k, latent_dir):
    ref_dir = latent_dir / 'kmer'
    fwd_a = _safe_pickle_load(ref_dir / f'{fwd_name_a}.pickle')
    rc_a  = _safe_pickle_load(ref_dir / f'{rc_name_a}.pickle')
    fwd_b = _safe_pickle_load(ref_dir / f'{fwd_name_b}.pickle')
    rc_b  = _safe_pickle_load(ref_dir / f'{rc_name_b}.pickle')
    if any(v is None for v in [fwd_a, rc_a, fwd_b, rc_b]):
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, {}
        return i, {
            'cross_1→2_fwd_rc':  calc_pair_cross_pk(ref, fwd_a, rc_b, k),
            'cross_1→2_rc_fwd':  calc_pair_cross_pk(ref, rc_a, fwd_b, k),
            'cross_1→2_fwd_fwd': calc_pair_cross_pk(ref, fwd_a, fwd_b, k),
            'cross_1→2_rc_rc':   calc_pair_cross_pk(ref, rc_a, rc_b, k),
            'cross_2→1_fwd_rc':  calc_pair_cross_pk(ref, fwd_b, rc_a, k),
            'cross_2→1_rc_fwd':  calc_pair_cross_pk(ref, rc_b, fwd_a, k),
            'cross_2→1_fwd_fwd': calc_pair_cross_pk(ref, fwd_b, fwd_a, k),
            'cross_2→1_rc_rc':   calc_pair_cross_pk(ref, rc_b, rc_a, k),
            'pk_fwd_1_in_2':     calc_pk_with_other(ref, fwd_a, fwd_b, k),
            'pk_rc_1_in_2':      calc_pk_with_other(ref, rc_a, rc_b, k),
            'pk_fwd_2_in_1':     calc_pk_with_other(ref, fwd_b, fwd_a, k),
            'pk_rc_2_in_1':      calc_pk_with_other(ref, rc_b, rc_a, k),
        }

    samples = {}
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  pairwise kmer {fwd_name_a}<> {fwd_name_b}', leave=False):
            i, metrics = f.result()
            if metrics:
                samples[i] = metrics

    if not samples:
        return None
    result = {}
    for mname in PAIRWISE_METRICS:
        vals = [samples[i][mname] for i in sorted(samples) if mname in samples[i]]
        result[mname] = {
            'best_layer': '-',
            'mean_pk': float(np.mean(vals)) if vals else 0.0,
            'std_pk': float(np.std(vals)) if vals else 0.0,
            'pk_by_sample': vals,
        }
    return result


def _compute_pairwise_cross_layered(model_key, fwd_name_a, rc_name_a,
                                    fwd_name_b, rc_name_b, n_ref, k, latent_dir):
    ref_dir = latent_dir / model_key
    fwd_a = _safe_pickle_load(ref_dir / f'{fwd_name_a}.pickle')
    rc_a  = _safe_pickle_load(ref_dir / f'{rc_name_a}.pickle')
    fwd_b = _safe_pickle_load(ref_dir / f'{fwd_name_b}.pickle')
    rc_b  = _safe_pickle_load(ref_dir / f'{rc_name_b}.pickle')
    if any(v is None for v in [fwd_a, rc_a, fwd_b, rc_b]):
        return None

    def _worker(i):
        ref = _safe_pickle_load(ref_dir / f'sample_{i}.pickle')
        if ref is None:
            return i, {}
        result = {}
        for l in ref:
            if not all(l in v for v in [fwd_a, rc_a, fwd_b, rc_b]):
                continue
            ra, fa, rca = ref[l], fwd_a[l], rc_a[l]
            fb, rcb = fwd_b[l], rc_b[l]
            if any(len(v) == 0 for v in [ra, fa, rca, fb, rcb]):
                continue
            result[l] = {
                'cross_1→2_fwd_rc':  calc_pair_cross_pk(ra, fa, rcb, k),
                'cross_1→2_rc_fwd':  calc_pair_cross_pk(ra, rca, fb, k),
                'cross_1→2_fwd_fwd': calc_pair_cross_pk(ra, fa, fb, k),
                'cross_1→2_rc_rc':   calc_pair_cross_pk(ra, rca, rcb, k),
                'cross_2→1_fwd_rc':  calc_pair_cross_pk(ra, fb, rca, k),
                'cross_2→1_rc_fwd':  calc_pair_cross_pk(ra, rcb, fa, k),
                'cross_2→1_fwd_fwd': calc_pair_cross_pk(ra, fb, fa, k),
                'cross_2→1_rc_rc':   calc_pair_cross_pk(ra, rcb, rca, k),
                'pk_fwd_1_in_2':     calc_pk_with_other(ra, fa, fb, k),
                'pk_rc_1_in_2':      calc_pk_with_other(ra, rca, rcb, k),
                'pk_fwd_2_in_1':     calc_pk_with_other(ra, fb, fa, k),
                'pk_rc_2_in_1':      calc_pk_with_other(ra, rcb, rca, k),
            }
        return i, result

    pk_matrices = {m: defaultdict(dict) for m in PAIRWISE_METRICS}
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  pairwise {model_key} {fwd_name_a}<> {fwd_name_b}', leave=False):
            i, layer_results = f.result()
            for l, metrics in layer_results.items():
                for mname, mval in metrics.items():
                    pk_matrices[mname][l][i] = mval

    result = {}
    for mname in PAIRWISE_METRICS:
        if pk_matrices[mname]:
            bl, bm, bs = _extract_best(pk_matrices[mname])
            result[mname] = {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
                             'pk_matrix': dict(pk_matrices[mname])}
    return result if result else None


def _compute_pairwise_cross_viralm(fwd_name_a, rc_name_a, fwd_name_b, rc_name_b,
                                   n_ref, k, latent_dir, n_layers=12):
    ref_dir = latent_dir / 'viralm'

    def _load_per_layer(name):
        by_layer = {}
        for l in range(1, n_layers + 1):
            data = _safe_pickle_load(ref_dir / f'{name}_layer_{l}.pkl')
            if data is not None:
                by_layer[l] = viralm_get(data)
        return by_layer

    fwd_a = _load_per_layer(fwd_name_a)
    rc_a  = _load_per_layer(rc_name_a)
    fwd_b = _load_per_layer(fwd_name_b)
    rc_b  = _load_per_layer(rc_name_b)
    if not all([fwd_a, rc_a, fwd_b, rc_b]):
        return None

    def _worker(i):
        result = {}
        for l in range(1, n_layers + 1):
            if l not in fwd_a or l not in rc_a or l not in fwd_b or l not in rc_b:
                continue
            ref_data = _safe_pickle_load(ref_dir / f'sample_{i}_layer_{l}.pkl')
            if ref_data is None:
                continue
            ref = viralm_get(ref_data)
            fa, rca = fwd_a[l], rc_a[l]
            fb, rcb = fwd_b[l], rc_b[l]
            if any(len(v) == 0 for d in [fa, rca, fb, rcb, ref]
                   for v in d.values()):
                continue
            result[l] = {
                'cross_1→2_fwd_rc':  calc_pair_cross_pk(ref, fa, rcb, k),
                'cross_1→2_rc_fwd':  calc_pair_cross_pk(ref, rca, fb, k),
                'cross_1→2_fwd_fwd': calc_pair_cross_pk(ref, fa, fb, k),
                'cross_1→2_rc_rc':   calc_pair_cross_pk(ref, rca, rcb, k),
                'cross_2→1_fwd_rc':  calc_pair_cross_pk(ref, fb, rca, k),
                'cross_2→1_rc_fwd':  calc_pair_cross_pk(ref, rcb, fa, k),
                'cross_2→1_fwd_fwd': calc_pair_cross_pk(ref, fb, fa, k),
                'cross_2→1_rc_rc':   calc_pair_cross_pk(ref, rcb, rca, k),
                'pk_fwd_1_in_2':     calc_pk_with_other(ref, fa, fb, k),
                'pk_rc_1_in_2':      calc_pk_with_other(ref, rca, rcb, k),
                'pk_fwd_2_in_1':     calc_pk_with_other(ref, fb, fa, k),
                'pk_rc_2_in_1':      calc_pk_with_other(ref, rcb, rca, k),
            }
        return i, result

    pk_matrices = {m: defaultdict(dict) for m in PAIRWISE_METRICS}
    with ThreadPoolExecutor() as pool:
        futs = [pool.submit(_worker, i) for i in range(n_ref)]
        for f in tqdm(as_completed(futs), total=n_ref,
                      desc=f'  pairwise viralm {fwd_name_a}<> {fwd_name_b}', leave=False):
            i, layer_results = f.result()
            for l, metrics in layer_results.items():
                for mname, mval in metrics.items():
                    pk_matrices[mname][l][i] = mval

    result = {}
    for mname in PAIRWISE_METRICS:
        if pk_matrices[mname]:
            bl, bm, bs = _extract_best(pk_matrices[mname])
            result[mname] = {'best_layer': bl, 'mean_pk': bm, 'std_pk': bs,
                             'pk_matrix': dict(pk_matrices[mname])}
    return result if result else None


def _run_pairwise_model(model, fwd_name_a, rc_name_a, fwd_name_b, rc_name_b,
                        n_ref, k, latent_dir):
    if model == 'kmer':
        return _compute_pairwise_cross_kmer(fwd_name_a, rc_name_a,
                                            fwd_name_b, rc_name_b,
                                            n_ref, k, latent_dir)
    elif model == 'viralm':
        return _compute_pairwise_cross_viralm(fwd_name_a, rc_name_a,
                                              fwd_name_b, rc_name_b,
                                              n_ref, k, latent_dir)
    else:
        return _compute_pairwise_cross_layered(model, fwd_name_a, rc_name_a,
                                               fwd_name_b, rc_name_b,
                                               n_ref, k, latent_dir)


def _run_pairwise_task(packed):
    model, fwd_na, rc_na, fwd_nb, rc_nb, n_ref, k, latent_dir = packed
    return (fwd_na, fwd_nb, model), _run_pairwise_model(
        model, fwd_na, rc_na, fwd_nb, rc_nb, n_ref, k, latent_dir)


def compute_pairwise_cross_metrics(preproc_files, rc_files, models,
                                   n_ref, k, latent_dir, metrics_dir,
                                   max_workers=None):
    """Pairwise cross-virus metrics for all pairs of different viruses.

    Returns a list of dict rows (one per metric/model/pair) and saves Excel.
    """
    labels = sorted(preproc_files.keys())
    if len(labels) < 2:
        print("Pairwise cross: need at least 2 viruses, skipping.")
        return []

    # Build all unordered pairs
    pairs = [(labels[i], labels[j]) for i in range(len(labels))
             for j in range(i + 1, len(labels))]

    tasks = []
    for a, b in pairs:
        fwd_na = preproc_files[a][1]
        rc_na  = rc_files.get(a, (None, f'REVERSECOMPLEMENT{a}_preproc'))[1]
        fwd_nb = preproc_files[b][1]
        rc_nb  = rc_files.get(b, (None, f'REVERSECOMPLEMENT{b}_preproc'))[1]
        for model in models:
            tasks.append((model, fwd_na, rc_na, fwd_nb, rc_nb,
                          n_ref, k, latent_dir))

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_pairwise_task, t): t for t in tasks}
        pbar = tqdm(as_completed(futures), total=len(tasks),
                    desc='Pairwise cross', unit='task')
        for f in pbar:
            (fwd_na, fwd_nb, model), result = f.result()
            if result is None:
                continue
            virus_a = fwd_na.replace('_preproc', '')
            virus_b = fwd_nb.replace('_preproc', '')
            pair_dir = metrics_dir / model / f'pairwise_{virus_a}_vs_{virus_b}'
            pair_dir.mkdir(parents=True, exist_ok=True)
            for mname in PAIRWISE_METRICS:
                if mname not in result:
                    continue
                r = result[mname]
                rows.append({
                    'virus_a': virus_a, 'virus_b': virus_b,
                    'model': model, 'metric': mname,
                    'best_layer': r['best_layer'],
                    'mean_P@K': round(r['mean_pk'], 4),
                    'std_P@K': round(r['std_pk'], 4),
                })
                # Сохраняем подробную матрицу: слои × сэмплы
                if 'pk_matrix' in r:
                    _save_pk_matrix(r['pk_matrix'],
                                    pair_dir / f'{mname}_matrix.xlsx')
                elif 'pk_by_sample' in r:
                    matrix = {'-': {i: v for i, v in enumerate(r['pk_by_sample'])}}
                    _save_pk_matrix(matrix,
                                    pair_dir / f'{mname}_matrix.xlsx')

    if rows:
        df = pd.DataFrame(rows)
        out = metrics_dir / 'pairwise_cross_viruses.xlsx'
        df.to_excel(out, index=False)
        print(f"Pairwise cross summary saved → {out}")

    return rows


def print_pairwise_summary(rows):
    """Print pairwise cross-virus results as a readable table."""
    if not rows:
        print("No pairwise cross results.")
        return
    df = pd.DataFrame(rows)
    print(f"\n{'=' * 80}")
    print("  PAIRWISE CROSS-VIRUS METRICS")
    print(f"{'=' * 80}")
    for (va, vb), grp in df.groupby(['virus_a', 'virus_b']):
        print(f"\n  Pair: {va}  <->  {vb}")
        print(f"  {'Model':<12} {'Metric':<22} {'Layer':>8}   mean +/- std")
        print(f"  {'-' * 60}")
        for _, row in grp.iterrows():
            print(f"  {row['model']:<12} {row['metric']:<22} "
                  f"{str(row['best_layer']):>8}   "
                  f"{row['mean_P@K']:.4f} +/- {row['std_P@K']:.4f}")
    print()

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
