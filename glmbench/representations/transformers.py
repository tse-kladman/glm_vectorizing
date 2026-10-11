"""Extracting embeddings from the genomic language models.

DNABERT-2 and DNABERT-S run in this process: the fragments are tokenised, pushed
through the encoder, and every layer's token states are pooled into one vector per
fragment (masked mean by default, see pooling.py -- padding is never averaged in).
Layer 0 is the embedding layer, before any attention block; layers 1..12 are the
encoder blocks.

Evo2 and ViraLM cannot be installed next to the rest -- their dependencies
conflict -- so they run in their own conda environments through a subprocess, and
write the same pickle format back.  That is the only reason this module shells out.
"""

import gc
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config
from ..models import Model
from ..sequences import label_of
from . import pooling as pooling_module
from . import store

VIRALM_HOOK_DIR = "envs_for_glms/ViraLM"
VIRALM_HOOK_SCRIPT = "viralm_hook.py"
BERT_LAYERS = 13            # embedding layer + 12 encoder blocks
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


def _pooled(states, mask, pooling):
    """Pool one layer's states of one fragment, whatever shape the encoder returns.

    The embedding output is padded to the full length (one row per position, mask
    applies as is).  DNABERT-2's remote code, however, hands back the encoder layers
    *unpadded* -- one row per real token only -- in which case every row is real and
    the mask must not be applied again (it would not even line up).
    """
    array = states.float().cpu().numpy()
    if array.ndim == 3:                      # (1, length, width)
        array = array[0]
    if len(array) == len(mask):
        used = mask
    elif len(array) == int(mask.sum()):
        used = np.ones(len(array), dtype=mask.dtype)
    else:
        raise ValueError(f"hidden states of length {len(array)} fit neither the padded "
                         f"length {len(mask)} nor the {int(mask.sum())} real tokens")
    return pooling_module.pool(array, used, pooling).astype("float32")


def _pool_layers(model, dataset, device, n_layers=BERT_LAYERS, pooling=pooling_module.DEFAULT_MODE):
    """Pooled hidden state of every layer, grouped by label.

    Layer 0 is the embedding output, layers 1..n-1 the encoder blocks.  Pooling uses
    the attention mask, so the padding up to 512 tokens never enters a vector.
    """
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

        mask = attention_mask.cpu().numpy()
        label = item["label"]
        layers[0][label].append(_pooled(embedded[0], mask, pooling))
        for index in range(1, n_layers):
            layers[index][label].append(_pooled(encoded[index - 1], mask, pooling))

    return layers


def extract_bert(model: Model, fasta_path: Path, name: str, latent_dir=None,
                 pooling: str = pooling_module.DEFAULT_MODE) -> Path:
    """DNABERT-2 / DNABERT-S, in this process, on GPU when one is available."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading {model.hf_name} on {device}")

    revision = {"revision": model.revision} if model.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model.hf_name, trust_remote_code=True, **revision)
    encoder = AutoModel.from_pretrained(model.hf_name, trust_remote_code=True, **revision)
    data = _pool_layers(encoder, _dataset(fasta_path, tokenizer), device, pooling=pooling)

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
    """Evo2 through its own conda environment; writes the pickle itself.

    Returns False if the extractor failed or reported failed fragments (it exits
    non-zero then, and lists the ids in ``<name>.failed.txt`` next to the pickle).
    """
    outdir = store.model_dir(model, latent_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    arguments = [str(config.CODE_ROOT / "extract_evo2.py"),
                 "--fasta", str(fasta_path), "--name", name, "--outdir", str(outdir)]
    if model.revision:
        arguments += ["--model_id", model.revision]
    return _conda(model.conda_env, arguments, cwd=config.CODE_ROOT)


def viralm_hook_path() -> Path:
    return config.CODE_ROOT / VIRALM_HOOK_DIR / VIRALM_HOOK_SCRIPT


def check_viralm_hook() -> Path:
    """Fail early, with instructions, if the ViraLM hook script is not installed.

    The hook is *not* part of this repository (it wraps a third-party model with its
    own licence); see the README section "ViraLM" for where to get the model and for
    exactly what the script must accept and write.
    """
    hook = viralm_hook_path()
    if not hook.is_file():
        raise SystemExit(
            f"ViraLM hook script not found: {hook}\n"
            f"  ViraLM is not bundled with this repository. Install ViraLM from "
            f"https://github.com/ChengPENG-wolf/ViraLM (check its licence), put the "
            f"hook that dumps per-layer embeddings at\n"
            f"  {VIRALM_HOOK_DIR}/{VIRALM_HOOK_SCRIPT}\n"
            f"  and make the 'viralm' conda environment run it. The command line it "
            f"must accept and the files it must write are specified in the README "
            f"(section 'ViraLM').")
    return hook


def extract_viralm(model: Model, fasta_path: Path, name: str, latent_dir=None,
                   layers=None, batch_size=1) -> bool:
    """ViraLM through its own hook, which writes one pickle per layer."""
    hook = check_viralm_hook()
    outdir = store.model_dir(model, latent_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    layers = list(layers or store.VIRALM_LAYERS)
    return _conda(model.conda_env,
                  [hook.name,
                   "--input_pth", str(fasta_path), "--output_pth", str(outdir),
                   "--batch_size", str(batch_size), "--layers_to_monitor",
                   *[str(layer) for layer in layers]],
                  cwd=hook.parent,
                  extra_env={"MKL_THREADING_LAYER": "GNU", "OMP_NUM_THREADS": "1"})
