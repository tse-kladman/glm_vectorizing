#!/usr/bin/env python
"""
metric_axis.py — прогон метрики как отдельной оси (данные Fig. 4-6).

Метрика неотделима от преобразования вектора, поэтому единицей расчёта служит
протокол ``(transform, metric)`` из ``compute_metrics.PROTOCOLS``:

    P0  стандартизация + косинус       исторический протокол всех текущих таблиц
    P1  без преобразования + косинус   контроль: отделяет эффект масштабирования
    P2  стандартизация + евклид
    P3  без преобразования + евклид
    P4  sqrt + евклид                  расстояние Хеллингера    | только для
    P5  sqrt(JSD)                      Jensen-Shannon           | распределений

P4 и P5 определены лишь для векторов-распределений, поэтому применяются только к
k-mer-профилям; для эмбеддингов gLM они пропускаются (``--strict`` превращает
пропуск в ошибку).  Это ограничение математическое, а не техническое: привести
эмбеддинг к симплексу можно лишь произвольным преобразованием.

За один проход по соседям считаются сразу P@1..P@10 и CROSS P@1..P@10, то есть
данные Fig. 4, Fig. 5 и Fig. 6 получаются одним прогоном.

Результат (те же имена файлов, что у основного пайплайна, но с папкой протокола):

    <metrics_dir>/<protocol>/<model>/P@{k}_matrix_<name>.xlsx
    <metrics_dir>/<protocol>/<model>/cross_P@{k}_matrix_<name>.xlsx
    <metrics_dir>/metric_axis_summary.xlsx

Запуск из корня репозитория:

    python utils/metric_axis.py --latent_dir data/latent_dim --metrics_dir data/metrics \
        --names Egovirales_preproc Mriyavirales_preproc

    python utils/metric_axis.py ... --protocols P0 P5 --models kmer   # подмножество
    python utils/metric_axis.py ... --n_ref 1 --dry_run               # замер времени
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.compute_metrics import (PROTOCOLS, _as_protocol, _safe_pickle_load,
                                   _save_pk_matrix, calc_cross_pk,
                                   calc_metrics_with_new, viralm_get)

KS = list(range(1, 11))          # P@1..P@10 за один проход по соседям
VIRALM_LAYERS = range(1, 13)
COMPOSITIONAL_MODELS = {'kmer'}  # единственное представление-распределение

# Каждый процесс загружает numpy/scipy/sklearn и свою копию эмбеддингов, поэтому
# упираемся мы в память, а не в ядра.  На машинах с малым файлом подкачки спавн
# десятка процессов падает на импорте scipy — задавайте --max_workers вручную.
DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 1)


# ─────────────────────────── ЗАГРУЗКА ЭМБЕДДИНГОВ ────────────────────────────

def _load(model: str, name: str, latent_dir: Path):
    """Эмбеддинги одного файла: {layer: {label: [vectors]}} либо {label: [vectors]}."""
    ref_dir = latent_dir / model
    if model == 'viralm':
        by_layer = {}
        for layer in VIRALM_LAYERS:
            data = _safe_pickle_load(ref_dir / f'{name}_layer_{layer}.pkl')
            if data is not None:
                by_layer[layer] = viralm_get(data)
        return by_layer or None
    return _safe_pickle_load(ref_dir / f'{name}.pickle')


def _load_sample(model: str, index: int, latent_dir: Path, layers=None):
    ref_dir = latent_dir / model
    if model == 'viralm':
        out = {}
        for layer in layers or VIRALM_LAYERS:
            data = _safe_pickle_load(ref_dir / f'sample_{index}_layer_{layer}.pkl')
            if data is not None:
                out[layer] = viralm_get(data)
        return out or None
    return _safe_pickle_load(ref_dir / f'sample_{index}.pickle')


# ─────────────────────────── РАСЧЁТ ОДНОЙ ЗАДАЧИ ─────────────────────────────

def _metrics_for_sample(model, protocol, reference, virus, virus_rc):
    """P@k и CROSS P@k одного сэмпла: {'pk': {layer: {k: v}}, 'cross': {...}}."""
    layered = model != 'kmer'
    pk, cross = {}, {}

    layers = reference.keys() if layered else ['-']
    for layer in layers:
        ref = reference[layer] if layered else reference
        vir = virus.get(layer) if layered else virus
        if vir is None or not len(ref) or not len(vir):
            continue
        pk[layer] = calc_metrics_with_new(ref, vir, ks=KS, protocol=protocol)

        rc = (virus_rc.get(layer) if layered else virus_rc) if virus_rc else None
        if rc:
            cross[layer] = calc_cross_pk(ref, vir, rc, ks=KS, protocol=protocol)

    return {'pk': pk, 'cross': cross}


def _run_one(model, protocol_name, fwd_name, rc_name, n_ref, latent_dir):
    """Одна задача: модель × протокол × файл, все 50 сэмплов и все k."""
    protocol = _as_protocol(protocol_name)
    virus = _load(model, fwd_name, latent_dir)
    if virus is None:
        return None
    virus_rc = _load(model, rc_name, latent_dir) if rc_name else None

    layers = list(virus.keys()) if model == 'viralm' else None

    def worker(index):
        reference = _load_sample(model, index, latent_dir, layers)
        if reference is None:
            return index, None
        try:
            return index, _metrics_for_sample(model, protocol, reference, virus, virus_rc)
        except ValueError as exc:            # протокол неприменим к этим векторам
            return index, {'error': str(exc)}

    pk_matrix = {k: defaultdict(dict) for k in KS}
    cross_matrix = {k: defaultdict(dict) for k in KS}
    error = None

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(worker, i) for i in range(n_ref)]
        for future in tqdm(as_completed(futures), total=n_ref, leave=False,
                           desc=f'  {protocol_name} {model} {fwd_name}'):
            index, result = future.result()
            if result is None:
                continue
            if 'error' in result:
                error = result['error']
                break
            for layer, per_k in result['pk'].items():
                for k, value in per_k.items():
                    pk_matrix[k][layer][index] = value
            for layer, per_k in result['cross'].items():
                for k, value in per_k.items():
                    cross_matrix[k][layer][index] = value

    if error:
        return {'error': error}
    if not any(pk_matrix[k] for k in KS):
        return None
    return {'pk': {k: dict(v) for k, v in pk_matrix.items()},
            'cross': {k: dict(v) for k, v in cross_matrix.items()}}


def _run_task(packed):
    model, protocol_name, fwd_name, rc_name, n_ref, latent_dir = packed
    started = time.time()
    result = _run_one(model, protocol_name, fwd_name, rc_name, n_ref, latent_dir)
    return (model, protocol_name, fwd_name), result, time.time() - started


# ─────────────────────────── ОРКЕСТРАТОР ─────────────────────────────────────

def build_tasks(names, rc_names, models, protocols, n_ref, latent_dir, strict):
    tasks, skipped = [], []
    for model in models:
        for protocol_name in protocols:
            protocol = PROTOCOLS[protocol_name]
            if protocol.compositional and model not in COMPOSITIONAL_MODELS:
                message = (f'{protocol.name} ({protocol.label}) неприменим к {model}: '
                           f'эмбеддинги не являются распределениями')
                if strict:
                    raise SystemExit(f'ОШИБКА: {message}')
                skipped.append(message)
                continue
            for fwd_name, rc_name in zip(names, rc_names):
                tasks.append((model, protocol_name, fwd_name, rc_name, n_ref, latent_dir))
                if rc_name:                       # RC-файл как отдельный forward-прогон
                    tasks.append((model, protocol_name, rc_name, None, n_ref, latent_dir))
    return tasks, skipped


def save_result(result, model, protocol_name, name, metrics_dir):
    out_dir = Path(metrics_dir) / protocol_name / model
    for k in KS:
        if result['pk'].get(k):
            _save_pk_matrix(result['pk'][k], out_dir / f'P@{k}_matrix_{name}.xlsx')
        if result['cross'].get(k):
            _save_pk_matrix(result['cross'][k], out_dir / f'cross_P@{k}_matrix_{name}.xlsx')


def summarise(result, model, protocol_name, name):
    """Строки сводки: лучший слой по среднему, для каждого k и типа метрики."""
    rows = []
    for kind, key in [('forward', 'pk'), ('cross', 'cross')]:
        for k in KS:
            matrix = result[key].get(k)
            if not matrix:
                continue
            means = {layer: float(np.mean(list(values.values())))
                     for layer, values in matrix.items() if values}
            if not means:
                continue
            best = max(means, key=means.get)
            rows.append({
                'protocol': protocol_name, 'metric': PROTOCOLS[protocol_name].label,
                'model': model, 'virus': name, 'type': kind, 'k': k,
                'best_layer': best, 'mean_P@K': round(means[best], 6),
                'std_P@K': round(float(np.std(list(matrix[best].values()))), 6),
            })
    return rows


def check_p0_regression(metrics_dir: Path, models, tolerance=1e-12):
    """Сверка протокола P0 с историческими таблицами пайплайна.

    P0 — это ровно тот протокол, которым посчитаны закоммиченные
    ``data/metrics/<model>/*.xlsx``.  Любое расхождение означает, что таблицы
    статьи порождены не тем кодом, что лежит в репозитории, — тогда прогон
    оси метрик останавливается до выяснения причины.
    """
    metrics_dir = Path(metrics_dir)
    rows, worst = [], 0.0
    for model in models:
        new_dir, old_dir = metrics_dir / 'P0' / model, metrics_dir / model
        if not new_dir.is_dir():
            continue
        for new_path in sorted(new_dir.glob('*.xlsx')):
            old_path = old_dir / new_path.name
            if not old_path.exists():
                continue
            new = pd.read_excel(new_path).set_index('layer').sort_index(axis=1)
            old = pd.read_excel(old_path).set_index('layer').sort_index(axis=1)
            shared = sorted(set(new.columns) & set(old.columns))
            if not shared or not new.index.equals(old.index):
                rows.append({'model': model, 'file': new_path.name,
                             'status': 'структура не совпадает', 'max_abs_diff': np.nan})
                continue
            diff = float(np.nanmax(np.abs(new[shared].to_numpy(float)
                                          - old[shared].to_numpy(float))))
            worst = max(worst, diff)
            rows.append({'model': model, 'file': new_path.name,
                         'status': 'OK' if diff <= tolerance else 'РАСХОЖДЕНИЕ',
                         'max_abs_diff': diff})

    report = pd.DataFrame(rows)
    if report.empty:
        print('  Нечего сверять: таблицы P0 не найдены.')
        return report, True

    bad = report[report['status'] != 'OK']
    print(f'\n  Сверка P0 с историческими таблицами: {len(report)} файлов, '
          f'макс. расхождение {worst:.3g}')
    if bad.empty:
        print('  Все совпали.')
    else:
        print(f'  РАСХОЖДЕНИЙ: {len(bad)}')
        print(bad.to_string(index=False))
    return report, bad.empty


def main():
    args = parse_args()
    latent_dir, metrics_dir = Path(args.latent_dir), Path(args.metrics_dir)

    if args.check_p0_only:
        _, ok = check_p0_regression(metrics_dir, args.models)
        raise SystemExit(0 if ok else 1)

    rc_names = args.rc_names or [f'REVERSECOMPLEMENT{n}' for n in args.names]
    tasks, skipped = build_tasks(args.names, rc_names, args.models, args.protocols,
                                 args.n_ref, latent_dir, args.strict)

    for message in skipped:
        print(f'  ПРОПУСК: {message}')
    print(f'\nЗадач: {len(tasks)}  '
          f'({len(args.models)} моделей × {len(args.protocols)} протоколов, n_ref={args.n_ref})')
    if args.dry_run:
        for task in tasks:
            print(f'  {task[1]}  {task[0]:<9} {task[2]}')
        return

    rows, timings = [], []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        progress = tqdm(as_completed(futures), total=len(tasks),
                        desc='Metric axis', unit='task')
        for future in progress:
            (model, protocol_name, name), result, elapsed = future.result()
            timings.append({'model': model, 'protocol': protocol_name,
                            'virus': name, 'seconds': round(elapsed, 1)})
            if result is None:
                continue
            if 'error' in result:
                print(f'  ОШИБКА {protocol_name}/{model}/{name}: {result["error"]}')
                continue
            save_result(result, model, protocol_name, name, metrics_dir)
            rows.extend(summarise(result, model, protocol_name, name))
            progress.set_postfix_str(f'{protocol_name}/{model}/{name}  {elapsed:.0f}s')

    if rows:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = metrics_dir / 'metric_axis_summary.xlsx'
        with pd.ExcelWriter(out) as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name='summary', index=False)
            pd.DataFrame(timings).to_excel(writer, sheet_name='timings', index=False)
        print(f'\nСводка сохранена → {out}  ({len(rows)} строк)')
        total = sum(t['seconds'] for t in timings)
        print(f'Суммарное время задач: {total / 3600:.2f} ч '
              f'(wall-clock меньше за счёт параллелизма)')

    if 'P0' in args.protocols and not args.skip_check:
        _, ok = check_p0_regression(metrics_dir, args.models)
        if not ok:
            raise SystemExit('P0 не воспроизводит закоммиченные таблицы — '
                             'остановка до выяснения причины (--skip_check чтобы продолжить)')


def parse_args():
    parser = argparse.ArgumentParser(description='Metric axis sweep (Fig. 4-6)')
    parser.add_argument('--latent_dir', required=True)
    parser.add_argument('--metrics_dir', required=True)
    parser.add_argument('--names', nargs='+', required=True,
                        help='forward-файлы, например Egovirales_preproc')
    parser.add_argument('--rc_names', nargs='*', default=None)
    parser.add_argument('--models', nargs='+',
                        default=['kmer', 'dnabert2', 'dnabertS', 'evo2', 'viralm'])
    parser.add_argument('--protocols', nargs='+', default=list(PROTOCOLS),
                        choices=list(PROTOCOLS))
    parser.add_argument('--n_ref', type=int, default=50)
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'процессов (по умолчанию {DEFAULT_MAX_WORKERS}); '
                             f'уменьшите при нехватке памяти')
    parser.add_argument('--strict', action='store_true',
                        help='падать, а не пропускать, если протокол неприменим к модели')
    parser.add_argument('--dry_run', action='store_true',
                        help='только показать список задач')
    parser.add_argument('--skip_check', action='store_true',
                        help='не сверять P0 с историческими таблицами после прогона')
    parser.add_argument('--check_p0_only', action='store_true',
                        help='только сверить уже посчитанный P0 и выйти')
    return parser.parse_args()


if __name__ == '__main__':
    main()
