"""
ViraLM embedding extractor — запускать в окружении viralm_env.

Использование:
    python extract_viralm.py --fasta data/my_virus.fa --name my_virus
    python extract_viralm.py --fasta data/my_virus.fa --name my_virus --outdir data/latent_dim/viralm

Формат FASTA: заголовки вида  >НАЗВАНИЕ_idx
Выходные файлы: {outdir}/{name}_layer_{l}.pkl  для l = 1..n_layers
  Структура каждого файла: {label_idx: np.array}
  (ключ — record.id целиком, значение — эмбеддинг; viralm_get() разберёт метку по '_')
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

# ── TODO: замените на фактический импорт ViraLM ──────────────────────────────
# from viralm import ViraLM           # вариант 1: специализированный пакет
# from transformers import AutoModel, AutoTokenizer  # вариант 2: HuggingFace
# MODEL_PATH = "path/to/viralm"       # локальный путь или HuggingFace Hub id
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Extract ViraLM embeddings from FASTA")
    p.add_argument("--fasta",     required=True,  help="Input FASTA file")
    p.add_argument("--name",      required=True,  help="Output file name (no extension)")
    p.add_argument("--outdir",    default="data/latent_dim/viralm")
    p.add_argument("--n_layers",  type=int, default=12, help="Number of ViraLM layers")
    p.add_argument("--max_length",type=int, default=512)
    p.add_argument("--chunk_size",type=int, default=512,
                   help="Split sequences into chunks before tokenisation")
    return p.parse_args()


def extract_label(record_id: str) -> str:
    """НАЗВАНИЕ_idx  →  НАЗВАНИЕ."""
    if "_" in record_id:
        return record_id.rsplit("_", 1)[0]
    return record_id


def split_into_chunks(seq: str, chunk_size: int):
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size) if seq[i:i + chunk_size]]


def load_model_and_tokenizer(n_layers: int, max_length: int):
    """
    TODO: Замените этот блок на загрузку вашей ViraLM модели.
    Функция должна вернуть (model, tokenizer).
    """
    # ── Пример для HuggingFace-совместимой модели ─────────────────────────
    # from transformers import AutoTokenizer, AutoModel
    # MODEL_PATH = "author/viralm-model"      # <- поставьте правильный путь
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # model     = AutoModel.from_pretrained(MODEL_PATH, output_hidden_states=True)
    # return model, tokenizer
    # ─────────────────────────────────────────────────────────────────────
    raise NotImplementedError(
        "Заполните load_model_and_tokenizer() в extract_viralm.py"
    )


def get_hidden_states(model, tokenizer, sequence: str, device, max_length: int, n_layers: int):
    """
    TODO: Адаптируйте под API вашей ViraLM.
    Должна вернуть список np.array длиной n_layers — по одному вектору на слой.

    Пример для HuggingFace BertModel-style:
    """
    inputs = tokenizer(
        sequence,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple длиной n_layers+1 (embedding + каждый encoder-слой)
    # каждый элемент: (batch=1, seq_len, hidden_size)
    hidden_states = outputs.hidden_states  # tuple

    # Mean-pooling по seq_len для каждого слоя (слои 1..n_layers)
    layer_vecs = []
    for layer_idx in range(1, n_layers + 1):
        vec = hidden_states[layer_idx][0].mean(dim=0).cpu().numpy()  # (hidden_size,)
        layer_vecs.append(vec)

    return layer_vecs  # list of n_layers np.arrays


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.n_layers, args.max_length)
    model.eval().to(device)

    # layers[l][record_id] = np.array  (l = 1..n_layers, 1-based)
    layers = {l: {} for l in range(1, args.n_layers + 1)}

    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"Sequences to process: {len(records)}", flush=True)

    for record in tqdm(records, file=sys.stdout):
        seq   = str(record.seq)
        label = extract_label(record.id)
        if not seq:
            continue
        try:
            chunks = split_into_chunks(seq, args.chunk_size)
            chunk_vecs = {l: [] for l in range(1, args.n_layers + 1)}

            for chunk in chunks:
                layer_vecs = get_hidden_states(
                    model, tokenizer, chunk, device, args.max_length, args.n_layers
                )
                for l, vec in enumerate(layer_vecs, start=1):
                    chunk_vecs[l].append(vec)

            # Mean-pool чанки → один вектор на запись, на слой
            for l in range(1, args.n_layers + 1):
                mean_vec = np.mean(chunk_vecs[l], axis=0)
                # Ключ = record.id (label_idx), чтобы viralm_get() работал корректно
                layers[l][record.id] = mean_vec

        except Exception as e:
            print(f"  Error [{record.id}]: {e}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    for l in range(1, args.n_layers + 1):
        out_path = os.path.join(args.outdir, f"{args.name}_layer_{l}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(layers[l], f)
    print(f"Saved {args.n_layers} layer files → {args.outdir}/{args.name}_layer_*.pkl",
          flush=True)


if __name__ == "__main__":
    main()
