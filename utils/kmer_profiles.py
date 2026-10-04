#!/usr/bin/env python
"""
kmer_profiles.py — k-mer частотные профили произвольной длины слова.

Основной пайплайн строит профили только для k = 4 (``run_pipeline.extract_kmer``).
Чтобы длину слова можно было выбрать по референсу, а потом считать с ней все
метрики дальше, нужен экстрактор с параметром k, пишущий в отдельную «модель»:

    data/latent_dim/kmer_k{K}/sample_{i}.pickle
    data/latent_dim/kmer_k{K}/<virus_name>.pickle

Формат тот же, что у остальных представлений — ``{label: [векторы]}``, — поэтому
metric_axis.py читает их как обычную бесслойную модель ``kmer_k{K}``.

Вектор нормирован на сумму 1, то есть является распределением; это требование
протокола P5 (Jensen-Shannon).

Запуск из корня репозитория:

    python utils/kmer_profiles.py --k 6 --ref_dir data/random_samples \
        --virus_dir data/preproc_unk_viruses \
        --names Egovirales_preproc Mriyavirales_preproc --rc
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE_CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def kmer_vector(seq: str, k: int) -> np.ndarray:
    """Частотный профиль длины 4^k, нормированный на сумму 1 (не-ACGT пропускаются)."""
    codes = np.full(len(seq), -1, dtype=np.int64)
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    for base, value in BASE_CODE.items():
        codes[arr == ord(base)] = value

    vector = np.zeros(4 ** k, dtype=np.float32)
    if len(codes) < k:
        return vector

    windows = np.lib.stride_tricks.sliding_window_view(codes, k)
    windows = windows[(windows >= 0).all(axis=1)]
    if not len(windows):
        return vector

    shifts = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    indices = windows @ shifts
    np.add.at(vector, indices, 1.0)
    total = vector.sum()
    return vector / total if total else vector


def profiles_fasta(path: Path, k: int) -> dict:
    """FASTA → {label: матрица профилей}; label как в run_pipeline."""
    from Bio import SeqIO          # ленивый импорт: ядро обходится без него

    out = defaultdict(list)
    for record in SeqIO.parse(str(path), 'fasta'):
        out[record.id.rsplit('_', 1)[0]].append(kmer_vector(str(record.seq), k))
    return {label: np.vstack(values) for label, values in out.items()}


def targets_from_args(args):
    """(путь к FASTA, имя пикла) для референсных сэмплов и вирусных файлов."""
    targets = []
    if args.ref_dir:
        ref_dir = Path(args.ref_dir)
        targets += [(ref_dir / f'sample_{i}.fa', f'sample_{i}') for i in range(args.n_ref)]
    if args.names:
        virus_dir = Path(args.virus_dir)
        for name in args.names:
            targets.append((virus_dir / f'{name}.fa', name))
            if args.rc:
                label = name.replace('_preproc', '')
                rc_name = f'REVERSECOMPLEMENT{label}_preproc'
                targets.append((virus_dir / f'{rc_name}.fa', rc_name))
    return targets


def main():
    args = parse_args()
    outdir = Path(args.outdir or f'data/latent_dim/kmer_k{args.k}')
    outdir.mkdir(parents=True, exist_ok=True)

    targets = targets_from_args(args)
    print(f'k={args.k} (размерность {4 ** args.k}), файлов: {len(targets)} → {outdir}')
    done = 0
    for path, name in tqdm(targets, desc=f'k-mer k={args.k}', unit='file'):
        if not path.exists():
            print(f'  ПРОПУСК: нет файла {path}')
            continue
        data = profiles_fasta(path, args.k)
        if not data:
            print(f'  ПРОПУСК: пустой файл {path}')
            continue
        with open(outdir / f'{name}.pickle', 'wb') as handle:
            pickle.dump(data, handle)
        done += 1
    print(f'Готово: {done} файлов → {outdir}')


def parse_args():
    parser = argparse.ArgumentParser(description='k-mer frequency profiles for any k')
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--outdir', default=None,
                        help='по умолчанию data/latent_dim/kmer_k{K}')
    parser.add_argument('--ref_dir', default='data/random_samples')
    parser.add_argument('--n_ref', type=int, default=50)
    parser.add_argument('--virus_dir', default='data/preproc_unk_viruses')
    parser.add_argument('--names', nargs='*', default=None)
    parser.add_argument('--rc', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
