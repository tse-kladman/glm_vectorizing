#!/usr/bin/env python
"""
minhash_sketch.py — MinHash-скетчи фрагментов (представление для протокола P4).

MinHash — это не ещё одна метрика поверх вектора частот, а другое представление.
Он описывает фрагмент *множеством* его k-меров и оценивает коэффициент Жаккара —
долю общих слов у двух фрагментов.  Поэтому скетчи строятся прямо из FASTA, а не
из готовых пиклов k-mer-профилей.

Как устроен скетч.  Каждый k-mer кодируется двумя битами на букву, приводится к
каноническому виду ``min(kmer, revcomp(kmer))`` (так представление не зависит от
цепи), перемешивается 64-битным хэшом и прогоняется через s независимых
универсальных хэш-функций ``h_i(x) = (a_i*x + b_i) mod (2^31-1)``.  Скетч — вектор
из s минимумов.  Вероятность совпадения минимума по одной функции равна Жаккару
двух множеств, поэтому доля совпавших позиций — несмещённая оценка J.

Выбор k.  Классический Mash берёт k = 21, но это для целых геномов: на фрагменте
2 кб два разных вируса не делят ни одного 21-мера, Жаккар схлопывается в ноль и
метрика вырождается.  Для фрагментов осмысленный диапазон — примерно k = 7..10,
где множества ещё пересекаются, но уже не покрывают весь алфавит (при k = 4 все
256 слов присутствуют почти везде, и Жаккар вырождается в единицу).  Отсюда
значение по умолчанию k = 8; это параметр, его стоит просвипировать.

Результат пишется в формате остальных представлений — ``{label: [массивы]}``,
один pickle на файл, — поэтому metric_axis.py читает его как обычную «модель»:

    data/latent_dim/minhash/sample_{i}.pickle
    data/latent_dim/minhash/<virus_name>.pickle

Запуск из корня репозитория:

    python utils/minhash_sketch.py --ref_dir data/random_samples \
        --virus_dir data/preproc_unk_viruses \
        --names Egovirales_preproc Mriyavirales_preproc --rc \
        --outdir data/latent_dim/minhash
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

MERSENNE = np.uint64((1 << 31) - 1)      # модуль универсального хэширования
BASE_CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
DEFAULT_K = 8
DEFAULT_SKETCH = 128


# ─────────────────────────── КОДИРОВАНИЕ K-МЕРОВ ─────────────────────────────

def _encode(seq: str) -> np.ndarray:
    """Последовательность → коды букв; не-ACGT помечаются как -1."""
    codes = np.full(len(seq), -1, dtype=np.int64)
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    for base, value in BASE_CODE.items():
        codes[arr == ord(base)] = value
    return codes


def _kmer_codes(seq: str, k: int) -> np.ndarray:
    """Канонические коды всех валидных k-меров последовательности (уникальные)."""
    bases = _encode(seq)
    if len(bases) < k:
        return np.empty(0, dtype=np.uint64)

    windows = np.lib.stride_tricks.sliding_window_view(bases, k)
    valid = (windows >= 0).all(axis=1)
    windows = windows[valid]
    if not len(windows):
        return np.empty(0, dtype=np.uint64)

    shifts = 2 * np.arange(k - 1, -1, -1, dtype=np.int64)
    forward = (windows.astype(np.int64) << shifts).sum(axis=1)
    # обратный комплемент: дополнение кода буквы до 3 и обратный порядок
    reverse = ((3 - windows[:, ::-1]).astype(np.int64) << shifts).sum(axis=1)

    canonical = np.minimum(forward, reverse).astype(np.uint64)
    return np.unique(canonical)


def _mix64(values: np.ndarray) -> np.ndarray:
    """Перемешивание (splitmix64), чтобы соседние коды не давали соседние хэши."""
    with np.errstate(over='ignore'):
        x = values.astype(np.uint64)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xff51afd7ed558ccd)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xc4ceb9fe1a85ec53)
        x ^= x >> np.uint64(33)
    return x & MERSENNE


# ─────────────────────────── СКЕТЧ ───────────────────────────────────────────

def hash_functions(sketch_size: int, seed: int = 42):
    """s пар (a, b) для универсального хэширования по модулю Мерсенна."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(MERSENNE), size=sketch_size, dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, int(MERSENNE), size=sketch_size, dtype=np.int64).astype(np.uint64)
    return a, b


def sketch_sequence(seq: str, k: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash-скетч одной последовательности: s минимумов хэшей её k-меров."""
    codes = _kmer_codes(seq, k)
    if not len(codes):
        return np.full(len(a), MERSENNE, dtype=np.uint64)
    mixed = _mix64(codes)
    with np.errstate(over='ignore'):
        hashed = (a[:, None] * mixed[None, :] + b[:, None]) % MERSENNE
    return hashed.min(axis=1).astype(np.uint64)


def sketch_fasta(path: Path, k: int, sketch_size: int, seed: int = 42) -> dict:
    """FASTA → {label: [скетчи]}; label берётся так же, как в run_pipeline."""
    from Bio import SeqIO          # ленивый импорт: ядро скетчей обходится без него

    a, b = hash_functions(sketch_size, seed)
    out = defaultdict(list)
    for record in SeqIO.parse(str(path), 'fasta'):
        label = record.id.rsplit('_', 1)[0]
        out[label].append(sketch_sequence(str(record.seq), k, a, b))
    return {label: np.vstack(values) for label, values in out.items()}


def jaccard(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    """Оценка коэффициента Жаккара по двум скетчам (доля совпавших позиций)."""
    return float(np.mean(sketch_a == sketch_b))


def mash_distance(sketch_a: np.ndarray, sketch_b: np.ndarray, k: int) -> float:
    """Расстояние Mash: оценка доли различающихся позиций, D = -ln(2J/(1+J))/k.

    Монотонно по J, поэтому на порядок соседей не влияет; полезно для отчётов.
    """
    j = jaccard(sketch_a, sketch_b)
    if j <= 0:
        return 1.0
    return float(-np.log(2 * j / (1 + j)) / k)


# ─────────────────────────── CLI ─────────────────────────────────────────────

def _dump(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def main():
    args = parse_args()
    outdir = Path(args.outdir or f'data/latent_dim/minhash_k{args.k}')

    targets = []                                   # (fasta-путь, имя пикла)
    if args.ref_dir:
        ref_dir = Path(args.ref_dir)
        for index in range(args.n_ref):
            targets.append((ref_dir / f'sample_{index}.fa', f'sample_{index}'))
    if args.names:
        virus_dir = Path(args.virus_dir)
        for name in args.names:
            targets.append((virus_dir / f'{name}.fa', name))
            if args.rc:
                label = name.replace('_preproc', '')
                rc_name = f'REVERSECOMPLEMENT{label}_preproc'
                targets.append((virus_dir / f'{rc_name}.fa', rc_name))

    print(f'Скетчей: k={args.k}, размер скетча={args.sketch_size}, файлов={len(targets)}')
    missing = 0
    for path, name in tqdm(targets, desc='MinHash', unit='file'):
        if not path.exists():
            print(f'  ПРОПУСК: нет файла {path}')
            missing += 1
            continue
        data = sketch_fasta(path, args.k, args.sketch_size, args.seed)
        if not data:
            print(f'  ПРОПУСК: пустой файл {path}')
            missing += 1
            continue
        _dump(data, outdir / f'{name}.pickle')

    done = len(targets) - missing
    print(f'Готово: {done} файлов → {outdir}')
    if done:
        print(f'Дальше:  python utils/metric_axis.py --latent_dir {outdir.parent} '
              f'--metrics_dir data/metrics --names {" ".join(args.names or [])} '
              f'--models {outdir.name} --protocols P4')


def parse_args():
    parser = argparse.ArgumentParser(description='MinHash sketches for protocol P4')
    parser.add_argument('--outdir', default=None,
                        help='по умолчанию data/latent_dim/minhash_k{K}')
    parser.add_argument('--ref_dir', default='data/random_samples',
                        help='папка с референсными сэмплами sample_<i>.fa')
    parser.add_argument('--n_ref', type=int, default=50)
    parser.add_argument('--virus_dir', default='data/preproc_unk_viruses')
    parser.add_argument('--names', nargs='*', default=None,
                        help='имена вирусных файлов, например Egovirales_preproc')
    parser.add_argument('--rc', action='store_true',
                        help='дополнительно скетчить REVERSECOMPLEMENT-версии')
    parser.add_argument('--k', type=int, default=DEFAULT_K,
                        help=f'длина k-мера (по умолчанию {DEFAULT_K}; '
                             f'21 из Mash рассчитан на целые геномы, не на фрагменты)')
    parser.add_argument('--sketch_size', type=int, default=DEFAULT_SKETCH,
                        help=f'число хэш-функций (по умолчанию {DEFAULT_SKETCH})')
    parser.add_argument('--seed', type=int, default=42,
                        help='общий сид хэш-функций; обязан совпадать у всех файлов')
    return parser.parse_args()


if __name__ == '__main__':
    main()
