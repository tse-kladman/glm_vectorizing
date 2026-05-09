"""
Evo2 embedding extractor — запускать в окружении evo2_env.

Использование:
    python extract_evo2.py --fasta data/my_virus.fa --name my_virus
    python extract_evo2.py --fasta data/my_virus.fa --name my_virus --outdir data/latent_dim/evo2

Формат FASTA: заголовки вида  >НАЗВАНИЕ_idx
Выходной файл: {outdir}/{name}.pickle
  Структура: {block_idx: {label: [np.array, ...]}}  (32 блока)
"""

import argparse
import gc
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Extract Evo2 embeddings from FASTA")
    p.add_argument("--fasta",     required=True,  help="Input FASTA file")
    p.add_argument("--name",      required=True,  help="Output file name (no extension)")
    p.add_argument("--outdir",    default="data/latent_dim/evo2")
    p.add_argument("--n_blocks",  type=int, default=32, help="Number of Evo2 blocks")
    p.add_argument("--model_id",  default="evo2_7b", help="Evo2 model identifier")
    return p.parse_args()


def extract_label(record_id: str) -> str:
    """НАЗВАНИЕ_idx  →  НАЗВАНИЕ (берём всё до последнего подчёркивания)."""
    if "_" in record_id:
        return record_id.rsplit("_", 1)[0]
    return record_id


def main():
    args = parse_args()

    os.environ["VORTEX_FP8_DISABLED"] = "1"

    print(f"Loading Evo2 model: {args.model_id} ...", flush=True)
    from evo2 import Evo2
    model = Evo2(args.model_id)

    layer_names = [f"blocks.{i}.mlp.l3" for i in range(args.n_blocks)]
    layer_emb   = {i: defaultdict(list) for i in range(args.n_blocks)}

    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"Sequences to process: {len(records)}", flush=True)

    for record in tqdm(records, file=sys.stdout):
        seq   = str(record.seq)
        label = extract_label(record.id)
        if not seq:
            continue
        try:
            ids = torch.tensor(
                model.tokenizer.tokenize(seq), dtype=torch.int
            ).unsqueeze(0).to("cuda")

            with torch.no_grad():
                _, embeddings = model(
                    ids, return_embeddings=True, layer_names=layer_names
                )

            for i, name in enumerate(layer_names):
                emb = embeddings[name].detach().cpu().float().numpy()
                layer_emb[i][label].append(np.mean(emb, axis=1).squeeze())

        except Exception as e:
            print(f"  Error [{record.id}]: {e}", flush=True)
        finally:
            try:
                del ids, embeddings
            except NameError:
                pass
            torch.cuda.empty_cache()
            gc.collect()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{args.name}.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(dict(layer_emb), f)
    print(f"Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
