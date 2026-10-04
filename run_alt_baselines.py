#!/usr/bin/env python
"""
run_alt_baselines.py — альтернативные бейзлайны одной командой.

Считает на НОВЫХ вирусах два представления, у которых длина слова k выбрана на
РЕФЕРЕНСНЫХ вирусах, и ничего лишнего:

    1. k_sweep       выбор k на референсе по P@5            (js и minhash)
    2. kmer_profiles k-mer профили при выбранном k           → kmer_k{K}
       minhash_sketch MinHash-скетчи при выбранном k         → minhash_k{K}
    3. metric_axis   P@1..P@10 и CROSS P@1..P@10 на новых вирусах, один проход
    4. metric_axis_figures  фигуры и статистика рядом с опубликованными моделями

Экономия вычислений:
  * свип идёт по подвыборке референсных сэмплов (--sweep_n_ref, по умолчанию 10)
    и по короткому списку k — выбор параметра не требует всех 50 сэмплов;
  * извлекается только выигравший k, промежуточные не сохраняются;
  * P@1..P@10 и CROSS получаются из одного запроса соседей, а не из десяти;
  * уже посчитанные шаги пропускаются, если не передан --force.

Запуск из корня репозитория:

    python run_alt_baselines.py
    python run_alt_baselines.py --methods js            # только Jensen-Shannon
    python run_alt_baselines.py --dry_run               # показать команды
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
METHOD_PROTOCOL = {'js': 'P5', 'minhash': 'P4'}


def run(command, dry_run: bool) -> None:
    printable = ' '.join(str(c) for c in command)
    print(f'\n$ {printable}\n')
    if dry_run:
        return
    result = subprocess.run([sys.executable] + [str(c) for c in command], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f'ОШИБКА: шаг завершился с кодом {result.returncode}')


def best_k(metrics_dir: Path, method: str):
    path = metrics_dir / 'k_sweep' / 'best_k.json'
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(method)


def model_dir(method: str, k: int) -> str:
    return f'kmer_k{k}' if method == 'js' else f'minhash_k{k}'


def main():
    args = parse_args()
    metrics_dir, latent_dir = Path(args.metrics_dir), Path(args.latent_dir)
    names = args.names

    models = []
    for method in args.methods:
        # ── 1. выбор k на референсных вирусах ────────────────────────────────
        k = best_k(metrics_dir, method)
        if k is None or args.force:
            run(['utils/k_sweep.py', '--method', method,
                 '--ref_dir', args.ref_dir, '--metrics_dir', metrics_dir,
                 '--n_ref', args.sweep_n_ref,
                 '--max_workers', args.max_workers], args.dry_run)
            k = best_k(metrics_dir, method) or args.fallback_k[method]
        else:
            print(f'\n[{method}] k={k} уже выбран, свип пропущен (--force чтобы повторить)')

        # ── 2. извлечение представления при выбранном k ──────────────────────
        model = model_dir(method, k)
        script = 'utils/kmer_profiles.py' if method == 'js' else 'utils/minhash_sketch.py'
        target = latent_dir / model
        if args.force or not target.is_dir():
            command = [script, '--k', k, '--ref_dir', args.ref_dir, '--n_ref', args.n_ref,
                       '--virus_dir', args.virus_dir, '--outdir', target, '--rc',
                       '--names'] + names
            run(command, args.dry_run)
        else:
            print(f'[{method}] {target} уже существует, извлечение пропущено')
        models.append((method, model))

    # ── 3. метрики на новых вирусах ──────────────────────────────────────────
    for method, model in models:
        run(['utils/metric_axis.py', '--latent_dir', latent_dir,
             '--metrics_dir', metrics_dir, '--models', model,
             '--protocols', METHOD_PROTOCOL[method], '--n_ref', args.n_ref,
             '--max_workers', args.max_workers, '--names'] + names, args.dry_run)

    # ── 4. фигуры и статистика ───────────────────────────────────────────────
    protocols = [METHOD_PROTOCOL[m] for m, _ in models]
    run(['visualization/metric_axis_figures.py', '--metrics_dir', metrics_dir,
         '--extra_protocols'] + protocols, args.dry_run)

    print('\nГотово. Выбранные k:',
          {method: best_k(metrics_dir, method) for method, _ in models})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--methods', nargs='+', default=['js', 'minhash'],
                        choices=list(METHOD_PROTOCOL))
    parser.add_argument('--names', nargs='+',
                        default=['Egovirales_preproc', 'Mriyavirales_preproc'])
    parser.add_argument('--ref_dir', default='data/random_samples')
    parser.add_argument('--virus_dir', default='data/preproc_unk_viruses')
    parser.add_argument('--latent_dir', default='data/latent_dim')
    parser.add_argument('--metrics_dir', default='data/metrics')
    parser.add_argument('--n_ref', type=int, default=50,
                        help='референсных сэмплов в расчёте метрик')
    parser.add_argument('--sweep_n_ref', type=int, default=10,
                        help='референсных сэмплов для выбора k (хватает подвыборки)')
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--force', action='store_true',
                        help='пересчитать шаги, даже если результат уже есть')
    parser.add_argument('--dry_run', action='store_true',
                        help='только показать команды')
    args = parser.parse_args()
    args.fallback_k = {'js': 5, 'minhash': 8}
    return args


if __name__ == '__main__':
    main()
