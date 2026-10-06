"""Extracting embeddings from the genomic language models.

DNABERT-2 and DNABERT-S run in this process: the fragments are tokenised, pushed
through the encoder, and every layer's token states are mean-pooled into one vector
per fragment.  Layer 0 is the embedding layer, before any attention block.

Evo2 and ViraLM cannot be installed next to the rest -- their dependencies
conflict -- so they run in their own conda environments through a subprocess, and
write the same pickle format back.  That is the only reason this module shells out.
"""

import gc
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from .. import config
from ..models import Model
from ..sequences import label_of
from . import store

VIRALM_HOOK_DIR = "envs_for_glms/ViraLM"
VIRALM_HOOK_SCRIPT = "viralm_hook.py"
BERT_LAYERS = 14            # embedding layer + 13 encoder outputs
MAX_TOKENS = 512


def _dataset(fasta_path: Path, tokenizer, chunk_size=4096, max_length=MAX_TOKENS):
    from datasets import Dataset

    rows = []
    for record in _records(fasta_path):
        sequence = str(record.seq)
        for start in range(0, len(sequence), chunk_size):
            rows.append({"seq": sequence[start:start + chunk_size],
                         "label": label_of(record.id)})

    def tokenize(batch):
        return tokenizer(batch["seq"], padding="max_length",
                         truncation=True, max_length=max_length)

    dataset = Dataset.from_list(rows).map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def _records(fasta_path: Path):
    from Bio import SeqIO
    return SeqIO.parse(str(fasta_path), "fasta")


def _pool_layers(model, dataset, device, n_layers=BERT_LAYERS):
    """Mean-pooled hidden state of every layer, grouped by label."""
    import torch
    from tqdm import tqdm

    layers = {index: defaultdict(list) for index in range(n_layers)}
    model.eval().to(device)

    for item in tqdm(dataset, leave=False, desc="  fragments"):
        input_ids = item["input_ids"].to(device)
        attention_mask = item["attention_mask"].to(device)
        with torch.no_grad():
            embedded = model.embeddings(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
            encoded = model.encoder(embedded, attention_mask.unsqueeze(0))

        layers[0][item["label"]].append(embedded.mean(dim=1)[0].cpu().numpy())
        for index in range(2, n_layers):
            layers[index - 1][item["label"]].append(
                encoded[index - 2].mean(axis=0).cpu().numpy())

    return layers


def extract_bert(model: Model, fasta_path: Path, name: str, latent_dir=None) -> Path:
    """DNABERT-2 / DNABERT-S, in this process, on GPU when one is available."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading {model.hf_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model.hf_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(model.hf_name, trust_remote_code=True)
    data = _pool_layers(encoder, _dataset(fasta_path, tokenizer), device)

    del encoder
    torch.cuda.empty_cache()
    gc.collect()

    path = store.model_dir(model, latent_dir) / f"{name}.pickle"
    store.save({layer: dict(groups) for layer, groups in data.items()}, path)
    print(f"  {model.key} -> {config.relative(path)}")
    return path


def _conda(env: str, arguments: list, cwd: Path, extra_env=None) -> bool:
    command = ["conda", "run", "--no-capture-output", "-n", env, "python"] + arguments
    print(f"  running: {' '.join(str(part) for part in command)}")
    environment = {**os.environ, **(extra_env or {})}
    result = subprocess.run(command, cwd=str(cwd), env=environment)
    if result.returncode != 0:
        print(f"  WARNING: {env} extractor exited with code {result.returncode}")
    return result.returncode == 0


def extract_evo2(model: Model, fasta_path: Path, name: str, latent_dir=None) -> bool:
    """Evo2 through its own conda environment; writes the pickle itself."""
    outdir = store.model_dir(model, latent_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    return _conda(model.conda_env,
                  [str(config.ROOT / "extract_evo2.py"),
                   "--fasta", str(fasta_path), "--name", name, "--outdir", str(outdir)],
                  cwd=config.ROOT)


def extract_viralm(model: Model, fasta_path: Path, name: str, latent_dir=None,
                   layers=None, batch_size=1) -> bool:
    """ViraLM through its own hook, which writes one pickle per layer."""
    outdir = store.model_dir(model, latent_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    layers = list(layers or store.VIRALM_LAYERS)
    return _conda(model.conda_env,
                  [VIRALM_HOOK_SCRIPT,
                   "--input_pth", str(fasta_path), "--output_pth", str(outdir),
                   "--batch_size", str(batch_size), "--layers_to_monitor",
                   *[str(layer) for layer in layers]],
                  cwd=config.ROOT / VIRALM_HOOK_DIR,
                  extra_env={"MKL_THREADING_LAYER": "GNU", "OMP_NUM_THREADS": "1"})
