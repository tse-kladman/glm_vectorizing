#!/usr/bin/env python
"""
run_pipeline.py — полный пайплайн для новых вирусов.

Шаги:
  1. Предобработка: чанкование FASTA-файлов из INPUT_DIR → PREPROC_DIR
  2. Извлечение эмбеддингов (K-mer, DNABERT-2, DNABERT-S в текущем окружении;
     Evo2 и ViraLM — через subprocess в отдельных conda-окружениях)
  3. Расчёт метрики P@K по 50 референсным сэмплам

Запуск:
    python run_pipeline.py
    python run_pipeline.py --models kmer dnabert2 dnabertS
    python run_pipeline.py --skip_extraction          # только пересчёт метрик
    python run_pipeline.py --skip_preproc             # пропустить предобработку
"""

import argparse
import gc
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from tqdm import tqdm

# ─────────────────────────── КОНФИГУРАЦИЯ ────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent.resolve()

INPUT_DIR    = 'data/unk_viruses'          # .fa / .fasta / .txt файлы
PREPROC_DIR  = 'data/preproc_unk_viruses'  # предобработанные FASTA
LATENT_DIR   = 'data/latent_dim'           # базовая папка эмбеддингов
REF_DIR      = 'data/random_samples'       # 50 референсных сэмплов
METRICS_DIR  = 'data/metrics'             # выходные метрики

CHUNK_SIZE   = 2000   # размер чанка в bp
MAX_CHUNKS   = 10   # максимум чанков с одной последовательности (None = все)
K_METRIC     = 10      # k для P@K
N_REF        = 50     # количество референсных сэмплов

ALL_MODELS   = ['kmer', 'dnabert2', 'dnabertS', 'evo2', 'viralm']
EVO2_ENV     = 'evo2_env'
VIRALM_ENV   = 'viralm'

# ─────────────────────────── ПРЕДОБРАБОТКА ───────────────────────────────────

def chunk_sequence(seq: str, chunk_size: int, max_chunks=None):
    chunks = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]
    chunks = [c for c in chunks if c]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return chunks


def preprocess_fasta(input_path: Path, output_path: Path,
                     chunk_size=2000, max_chunks=None):
    """Чанкует FASTA и сохраняет в формате >original_header_idx."""
    records_out = []
    for record in SeqIO.parse(str(input_path), 'fasta'):
        seq = str(record.seq).upper()
        orig_id = record.id
        chunks = chunk_sequence(seq, chunk_size, max_chunks)
        for idx, chunk in enumerate(chunks):
            rec_id = f"{orig_id}_{idx}"
            records_out.append(SeqRecord(Seq(chunk), id=rec_id, description=''))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records_out, str(output_path), 'fasta')
    print(f"  Preprocessed → {output_path}  ({len(records_out)} chunks)")
    return output_path


def generate_rc_fasta(preproc_path: Path, rc_path: Path):
    """Создаёт reverse-complement FASTA из предобработанного файла.

    Заголовки: REVERSECOMPLEMENT{orig_header}_{idx}
    """
    records_out = []
    for record in SeqIO.parse(str(preproc_path), 'fasta'):
        rc_seq = record.seq.reverse_complement()
        rc_id = f"REVERSECOMPLEMENT{record.id}"
        records_out.append(SeqRecord(rc_seq, id=rc_id, description=''))

    rc_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records_out, str(rc_path), 'fasta')
    print(f"  RC → {rc_path}  ({len(records_out)} chunks)")
    return rc_path


def discover_input_files(input_dir: Path):
    """Возвращает список .fa / .fasta / .txt файлов из папки."""
    exts = {'.fa', '.fasta', '.txt'}
    return [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]

# ─────────────────────────── K-MER ───────────────────────────────────────────

def _all_kmers(k=4):
    return [''.join(p) for p in product('ACGT', repeat=k)]


def _kmer_vector(seq: str, k=4):
    from collections import Counter
    seq = seq.upper()
    kmers = _all_kmers(k)
    counts = Counter(seq[i:i+k] for i in range(len(seq)-k+1)
                     if set(seq[i:i+k]) <= {'A','C','G','T'})
    vec = np.array([counts.get(km, 0) for km in kmers], dtype=float)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def extract_kmer(fasta_path: Path, name: str, outdir: Path):
    final_dict = defaultdict(list)
    for record in SeqIO.parse(str(fasta_path), 'fasta'):
        label = record.id.rsplit('_', 1)[0]
        final_dict[label].append(_kmer_vector(str(record.seq)))

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'{name}.pickle'
    with open(out, 'wb') as f:
        pickle.dump(dict(final_dict), f)
    print(f"  K-mer → {out}")

# ─────────────────────────── DNABERT-2 / DNABERT-S ───────────────────────────

def _build_bert_dataset(fasta_path: Path, tokenizer, chunk_size=4096, max_length=512):
    from datasets import Dataset

    rows = []
    for record in SeqIO.parse(str(fasta_path), 'fasta'):
        label = record.id.rsplit('_', 1)[0]
        seq = str(record.seq)
        for i in range(0, len(seq), chunk_size):
            rows.append({'seq': seq[i:i+chunk_size], 'label': label})

    def tokenize(batch):
        return tokenizer(batch['seq'], padding='max_length',
                         truncation=True, max_length=max_length)

    ds = Dataset.from_list(rows)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return ds


def _latent_dim_bert(model, dataset, device, n_layers=14):
    layers = {i: defaultdict(list) for i in range(n_layers)}
    model.eval()
    model.to(device)

    for sent in tqdm(dataset, leave=False):
        input_ids     = sent['input_ids'].to(device)
        attention_mask = sent['attention_mask'].to(device)
        label = sent['label']

        with torch.no_grad():
            emb = model.embeddings(input_ids.unsqueeze(0),
                                   attention_mask.unsqueeze(0))
            enc = model.encoder(emb, attention_mask.unsqueeze(0))

        layers[0][label].append(emb.mean(dim=1)[0].cpu().numpy())
        for i in range(2, n_layers):
            layers[i-1][label].append(enc[i-2].mean(axis=0).cpu().numpy())

    return layers


def extract_dnabert(fasta_path: Path, name: str, model_name: str, outdir: Path):
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    dataset = _build_bert_dataset(fasta_path, tokenizer)
    data    = _latent_dim_bert(model, dataset, device)

    # освобождаем GPU-память
    del model
    torch.cuda.empty_cache()
    gc.collect()

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'{name}.pickle'
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"  {model_name} → {out}")

# ─────────────────────────── SUBPROCESS (EVO2 / VIRALM) ──────────────────────

# Путь к ViraLM-скрипту (относительно SCRIPT_DIR)
VIRALM_HOOK_RELDIR = 'envs_for_glms/ViraLM'   # рабочая директория для viralm_hook.py
VIRALM_HOOK_SCRIPT = 'viralm_hook.py'


def run_in_conda(env_name: str, script: str, fasta: Path, name: str, outdir: Path):
    """Запускает extract_evo2.py (или аналог) в conda-окружении env_name."""
    cmd = [
        'conda', 'run', '--no-capture-output', '-n', env_name,
        'python', str(SCRIPT_DIR / script),
        '--fasta',  str(fasta),
        '--name',   name,
        '--outdir', str(outdir),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"  WARNING: {script} exited with code {result.returncode}")
    return result.returncode == 0


def run_viralm_hook(env_name: str, fasta: Path, outdir: Path,
                    layers: list = None, batch_size: int = 1):
    """
    Запускает viralm_hook.py из envs_for_glms/ViraLM/ — так, как это делал
    оригинальный The_one_ring.ipynb.

    viralm_hook.py выводит файлы вида: {outdir}/{fasta.stem}_layer_{l}.pkl
    """
    if layers is None:
        layers = list(range(1, 13))

    hook_dir = SCRIPT_DIR / VIRALM_HOOK_RELDIR

    env = os.environ.copy()
    env['MKL_THREADING_LAYER'] = 'GNU'
    env['OMP_NUM_THREADS']     = '1'

    cmd = [
        'conda', 'run', '--no-capture-output', '-n', env_name,
        'python', VIRALM_HOOK_SCRIPT,
        '--input_pth',  str(fasta),
        '--output_pth', str(outdir),
        '--batch_size', str(batch_size),
        '--layers_to_monitor',
    ] + [str(l) for l in layers]

    print(f"  Running: {' '.join(cmd)}")
    print(f"  cwd:     {hook_dir}")
    result = subprocess.run(cmd, cwd=str(hook_dir), env=env)
    if result.returncode != 0:
        print(f"  WARNING: viralm_hook.py exited with code {result.returncode}")
    return result.returncode == 0

# ─────────────────────────── МЕТРИКИ (из compute_metrics.py) ──────────────────

from utils.compute_metrics import (compute_all_metrics, print_summary,
                             save_summary_table,
                             compute_pairwise_cross_metrics,
                             print_pairwise_summary)

# ─────────────────────────── MAIN ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Viral genome embedding pipeline')
    p.add_argument('--input_dir',       default=INPUT_DIR)
    p.add_argument('--preproc_dir',     default=PREPROC_DIR)
    p.add_argument('--latent_dir',      default=LATENT_DIR)
    p.add_argument('--ref_dir',         default=REF_DIR)
    p.add_argument('--metrics_dir',     default=METRICS_DIR)
    p.add_argument('--models',          nargs='+', default=ALL_MODELS,
                   choices=ALL_MODELS,
                   help='Модели для запуска')
    p.add_argument('--chunk_size',      type=int, default=CHUNK_SIZE)
    p.add_argument('--max_chunks',      type=int, default=MAX_CHUNKS)
    p.add_argument('--k',               type=int, default=K_METRIC)
    p.add_argument('--n_ref',           type=int, default=N_REF)
    p.add_argument('--evo2_env',        default=EVO2_ENV)
    p.add_argument('--viralm_env',      default=VIRALM_ENV)
    p.add_argument('--skip_preproc',    action='store_true',
                   help='Пропустить предобработку (файлы уже в preproc_dir)')
    p.add_argument('--skip_extraction', action='store_true',
                   help='Пропустить извлечение эмбеддингов')
    p.add_argument('--skip_metrics',    action='store_true',
                   help='Пропустить расчёт метрик')
    p.add_argument('--pairwise_cross',  action='store_true',
                   help='Посчитать попарные cross-virus метрики (A↔B для всех пар)')
    return p.parse_args()


def main():
    args = parse_args()

    input_dir   = SCRIPT_DIR / args.input_dir
    preproc_dir = SCRIPT_DIR / args.preproc_dir
    latent_dir  = SCRIPT_DIR / args.latent_dir
    metrics_dir = SCRIPT_DIR / args.metrics_dir

    # ── 1. Обнаруживаем входные файлы ─────────────────────────────────────
    input_files = discover_input_files(input_dir)
    if not input_files:
        print(f"Файлы не найдены в {input_dir}")
        sys.exit(1)

    print(f"\nНайдено файлов: {len(input_files)}")
    for f in input_files:
        print(f"  {f.name}")

    # ── 2. Предобработка ───────────────────────────────────────────────────
    preproc_files = {}   # label → (path, name)
    rc_files      = {}   # label → (path, name)

    if not args.skip_preproc:
        print("\n── Предобработка ─────────────────────────────────────────────")
        for fpath in input_files:
            label       = fpath.stem
            out_name    = f'{label}_preproc'
            out_path    = preproc_dir / f'{out_name}.fa'
            preprocess_fasta(fpath, out_path,
                             chunk_size=args.chunk_size,
                             max_chunks=args.max_chunks)
            preproc_files[label] = (out_path, out_name)

            # Reverse complement
            rc_name = f'REVERSECOMPLEMENT{label}_preproc'
            rc_path = preproc_dir / f'{rc_name}.fa'
            generate_rc_fasta(out_path, rc_path)
            rc_files[label] = (rc_path, rc_name)
    else:
        for fpath in input_files:
            label    = fpath.stem
            out_name = f'{label}_preproc'
            out_path = preproc_dir / f'{out_name}.fa'
            rc_name  = f'REVERSECOMPLEMENT{label}_preproc'
            rc_path  = preproc_dir / f'{rc_name}.fa'
            if out_path.exists():
                preproc_files[label] = (out_path, out_name)
                print(f"  Найден: {out_path}")
            else:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: не найден {out_path}, пропускаем")
            if rc_path.exists():
                rc_files[label] = (rc_path, rc_name)
                print(f"  Найден RC: {rc_path}")

    if not preproc_files:
        print("Нет предобработанных файлов. Завершение.")
        sys.exit(1)

    # ── 3. Извлечение эмбеддингов ──────────────────────────────────────────
    # Объединяем forward и RC файлы для единого цикла извлечения
    all_extract = {}  # key → (path, name)
    for label, (fpath, name) in preproc_files.items():
        all_extract[f'fwd_{label}'] = (fpath, name)
    for label, (rc_path, rc_name) in rc_files.items():
        all_extract[f'rc_{label}'] = (rc_path, rc_name)

    if not args.skip_extraction:
        # K-mer
        if 'kmer' in args.models:
            print("\n── K-mer ─────────────────────────────────────────────────────")
            for key, (fpath, name) in all_extract.items():
                extract_kmer(fpath, name, latent_dir / 'kmer')

        # DNABERT-2
        if 'dnabert2' in args.models:
            print("\n── DNABERT-2 ─────────────────────────────────────────────────")
            for key, (fpath, name) in all_extract.items():
                extract_dnabert(fpath, name, 'zhihan1996/DNABERT-2-117M',
                                latent_dir / 'dnabert2')

        # DNABERT-S
        if 'dnabertS' in args.models:
            print("\n── DNABERT-S ─────────────────────────────────────────────────")
            for key, (fpath, name) in all_extract.items():
                extract_dnabert(fpath, name, 'zhihan1996/DNABERT-S',
                                latent_dir / 'dnabertS')

        # Evo2 — subprocess
        if 'evo2' in args.models:
            print("\n── Evo2 ──────────────────────────────────────────────────────")
            for key, (fpath, name) in all_extract.items():
                run_in_conda(args.evo2_env, 'extract_evo2.py',
                             fpath, name, latent_dir / 'evo2')

        # ViraLM — subprocess через viralm_hook.py
        if 'viralm' in args.models:
            print("\n── ViraLM ────────────────────────────────────────────────────")
            (latent_dir / 'viralm').mkdir(parents=True, exist_ok=True)
            for key, (fpath, name) in all_extract.items():
                run_viralm_hook(args.viralm_env, fpath, latent_dir / 'viralm')

    # ── 4. Метрики (параллельно по файлам и моделям) ─────────────────────
    if not args.skip_metrics:
        print("\n── Метрики (параллельный расчёт) ─────────────────────────────")

        all_fwd, all_rc, all_cross = compute_all_metrics(
            preproc_files=preproc_files,
            rc_files=rc_files,
            models=args.models,
            n_ref=args.n_ref,
            k=args.k,
            latent_dir=latent_dir,
            metrics_dir=metrics_dir,
        )

        # Вывод результатов
        for name, results in all_fwd.items():
            print_summary(name, results, args.k)
        for name, results in all_rc.items():
            print_summary(name, results, args.k, prefix='RC ')
        for name, results in all_cross.items():
            print_summary(name, results, args.k, prefix='CROSS ')

        save_summary_table(all_fwd, all_rc, all_cross, metrics_dir)

    # ── 5. Попарные cross-virus метрики ─────────────────────────────────
    if args.pairwise_cross:
        print("\n── Попарные cross-virus метрики ───────────────────────────────")
        rows = compute_pairwise_cross_metrics(
            preproc_files=preproc_files,
            rc_files=rc_files,
            models=args.models,
            n_ref=args.n_ref,
            k=args.k,
            latent_dir=latent_dir,
            metrics_dir=metrics_dir,
        )
        print_pairwise_summary(rows)


if __name__ == '__main__':
    main()
