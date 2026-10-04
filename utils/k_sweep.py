#!/usr/bin/env python
"""
k_sweep.py — выбор длины слова k на РЕФЕРЕНСНЫХ вирусах.

У k-mer-методов длина слова играет ту же роль, что номер слоя у трансформера: это
свободный параметр, и подбирать его по новым вирусам — значит подглядывать в
данные, по которым потом отчитываешься.  Поэтому k выбирается здесь, на
референсном наборе, той же метрикой, что и слой (``P@5`` внутри референсного
сэмпла), а дальше по пайплайну используется уже зафиксированным.

Методы:
    js       частотный профиль + sqrt(JSD)        → протокол P5
    minhash  скетч множества k-меров + Жаккар     → протокол P4

Диапазоны по умолчанию выбраны из насыщения: при малых k множества/профили почти
совпадают у всех фрагментов, при больших — почти не пересекаются, и метрика в обоих
случаях вырождается.

Результат:
    data/metrics/k_sweep/{method}_k_sweep.xlsx   матрица k × сэмплы
    data/metrics/k_sweep/best_k.json             {"js": 6, "minhash": 8}

Запуск из корня репозитория:

    python utils/k_sweep.py --method js --n_ref 10
    python utils/k_sweep.py --method minhash --n_ref 10 --k_values 6 7 8 9 10
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.compute_metrics import (PROTOCOLS, _apply_transform, _knn_indices,
                                   _prefix_precisions)
from utils.kmer_profiles import profiles_fasta
from utils.minhash_sketch import hash_functions, sketch_sequence

METHOD_PROTOCOL = {'js': 'P5', 'minhash': 'P4'}
DEFAULT_K = {'js': [3, 4, 5, 6, 7], 'minhash': [6, 7, 8, 9, 10]}
BEST_K_FILE = 'best_k.json'

# Каждый процесс поднимает свою копию numpy/scipy, поэтому упираемся в память, а не
# в ядра; на машинах с малым файлом подкачки спавн десятка процессов падает на импорте.
DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 1)


def _sketches_fasta(path: Path, k: int, sketch_size: int, seed: int) -> dict:
    """FASTA → {label: матрица скетчей} (аналог profiles_fasta для MinHash)."""
    from collections import defaultdict

    from Bio import SeqIO

    a, b = hash_functions(sketch_size, seed)
    out = defaultdict(list)
    for record in SeqIO.parse(str(path), 'fasta'):
        out[record.id.rsplit('_', 1)[0]].append(sketch_sequence(str(record.seq), k, a, b))
    return {label: np.vstack(values) for label, values in out.items()}


def reference_precision(data: dict, protocol, k_neighbours: int = 5) -> float:
    """P@K внутри одного референсного сэмпла: соседи того же класса, self исключён.

    Та же величина, что в ``data/metrics/<model>/P5.xlsx``, которой выбирается слой, —
    поэтому k и слой выбираются по одному и тому же критерию.
    """
    embeddings, labels = [], []
    for label, vectors in data.items():
        if len(vectors):
            embeddings.append(vectors)
            labels.extend([label] * len(vectors))
    if len(labels) <= k_neighbours:
        return float('nan')

    matrix = _apply_transform(np.vstack(embeddings), protocol)
    labels = np.asarray(labels)
    indices = _knn_indices(matrix, matrix, k_neighbours + 1, protocol.metric)[:, 1:]
    return _prefix_precisions(labels[indices], labels, [k_neighbours])[k_neighbours]


def _one(packed):
    """Одна пара (k, сэмпл) — верхнеуровневая функция для ProcessPoolExecutor."""
    method, k, index, ref_dir, sketch_size, seed, k_neighbours = packed
    path = Path(ref_dir) / f'sample_{index}.fa'
    if not path.exists():
        return k, index, float('nan')
    data = (profiles_fasta(path, k) if method == 'js'
            else _sketches_fasta(path, k, sketch_size, seed))
    protocol = PROTOCOLS[METHOD_PROTOCOL[method]]
    return k, index, reference_precision(data, protocol, k_neighbours)


def sweep(method: str, k_values, ref_dir: Path, n_ref: int,
          sketch_size: int, seed: int, k_neighbours: int, max_workers=None):
    tasks = [(method, k, i, str(ref_dir), sketch_size, seed, k_neighbours)
             for k in k_values for i in range(n_ref)]
    scores = {k: {} for k in k_values}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks),
                           desc=f'k-sweep {method}', unit='task'):
            k, index, value = future.result()
            scores[k][index] = value

    table = pd.DataFrame(scores).T.sort_index()
    table.index.name = 'k'
    table.columns = [f'sample_{c}' for c in table.columns]
    return table


def main():
    args = parse_args()
    k_values = args.k_values or DEFAULT_K[args.method]
    outdir = Path(args.metrics_dir) / 'k_sweep'
    outdir.mkdir(parents=True, exist_ok=True)

    table = sweep(args.method, k_values, Path(args.ref_dir), args.n_ref,
                  args.sketch_size, args.seed, args.k_neighbours, args.max_workers)
    means = table.mean(axis=1, skipna=True)
    best_k = int(means.idxmax())

    table.to_excel(outdir / f'{args.method}_k_sweep.xlsx')
    print(f"\n  Референсный P@{args.k_neighbours} по {args.n_ref} сэмплам "
          f"({args.method}, протокол {METHOD_PROTOCOL[args.method]}):")
    for k, value in means.items():
        mark = '  <- лучший' if k == best_k else ''
        print(f'    k={k:<3} {value:.4f} ± {table.loc[k].std(ddof=0):.4f}{mark}')

    path = outdir / BEST_K_FILE
    chosen = json.loads(path.read_text()) if path.exists() else {}
    chosen[args.method] = best_k
    path.write_text(json.dumps(chosen, indent=2))
    print(f'\n  Лучший k = {best_k} → {path}')
    print(f'  Дальше: python utils/'
          f'{"kmer_profiles" if args.method == "js" else "minhash_sketch"}.py '
          f'--k {best_k} ...')
    return best_k


def parse_args():
    parser = argparse.ArgumentParser(description='Choose k on the reference viruses')
    parser.add_argument('--method', choices=list(METHOD_PROTOCOL), required=True)
    parser.add_argument('--k_values', nargs='*', type=int, default=None)
    parser.add_argument('--ref_dir', default='data/random_samples')
    parser.add_argument('--metrics_dir', default='data/metrics')
    parser.add_argument('--n_ref', type=int, default=50,
                        help='сколько референсных сэмплов использовать для выбора k')
    parser.add_argument('--k_neighbours', type=int, default=5, help='K в P@K')
    parser.add_argument('--sketch_size', type=int, default=128, help='только для minhash')
    parser.add_argument('--seed', type=int, default=42, help='только для minhash')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'процессов (по умолчанию {DEFAULT_MAX_WORKERS})')
    return parser.parse_args()


if __name__ == '__main__':
    main()
