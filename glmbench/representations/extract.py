"""Driving the extractors: FASTA files in, ``data/latent_dim/<model>/`` out.

The reference samples and the unseen orders go through exactly the same code, so a
fragment is turned into a vector the same way no matter which set it belongs to --
otherwise the two would not be comparable in one space.

For the two baselines whose word length is tuned, the length chosen on the
reference set (selection.py) is used here; passing ``k`` overrides it.
"""

import json
from pathlib import Path

from tqdm import tqdm

from .. import config, selection
from ..models import Model
from . import kmer, minhash, store, transformers
from . import pooling as pooling_module


def targets(virus_files: dict, reference_dir=None, n_ref=config.N_REFERENCE,
            preproc_dir=None) -> list:
    """(FASTA path, name) of every file a model has to encode."""
    reference_dir = Path(reference_dir or config.REFERENCE_DIR)
    preproc_dir = Path(preproc_dir or config.PREPROC_DIR)

    items = [(reference_dir / f"sample_{index}.fa", f"sample_{index}")
             for index in range(n_ref)]
    for forward, reverse in virus_files.values():
        items.append((preproc_dir / f"{forward}.fa", forward))
        if reverse:
            items.append((preproc_dir / f"{reverse}.fa", reverse))
    return items


def extract_model(model: Model, items: list, latent_dir=None, k=None, force=False,
                  pooling: str = pooling_module.DEFAULT_MODE, seed: int = config.SEED) -> int:
    """Encode every file with one model; returns how many files were written.

    A file that fails (extractor exit code, empty input) is *counted and reported*,
    its name is written to ``<model dir>/failed.txt``, and the stage ends with a
    non-zero exit once every other file has been tried -- a half-extracted model
    must never look like a finished one.
    """
    directory = store.model_dir(model, latent_dir)
    directory.mkdir(parents=True, exist_ok=True)
    failed = []

    if model.extractor == "viralm":
        transformers.check_viralm_hook()                    # before any work is done
    word = None
    if model.extractor in ("kmer", "minhash"):
        word = k or selection.chosen_k(model, default=model.fixed_k)
        if word is None:
            raise SystemExit(
                f"{model.key}: no word length chosen yet -- run the 'select' stage "
                f"first, or pass --k")
        print(f"  {model.key}: k={word}"
              + (f", sketch size {model.sketch_size}" if model.extractor == "minhash" else ""))

    # what the vectors in this folder were made with (read by hand; pickles carry no metadata)
    (directory / "extraction.json").write_text(json.dumps(
        {"extractor": model.extractor, "k": word, "seed": seed,
         "pooling": pooling if model.extractor == "bert" else None,
         "hf_name": model.hf_name or None, "revision": model.revision},
        indent=2), encoding="utf8")

    written = 0
    for path, name in tqdm(items, desc=f"extract {model.key}", unit="file"):
        if not path.exists():
            print(f"  MISSING input: {config.relative(path)}")
            failed.append(name)
            continue

        if model.extractor in ("kmer", "minhash"):
            destination = directory / f"{name}.pickle"
            if destination.exists() and not force:
                continue
            groups = (kmer.from_fasta(path, word, model.canonical) if model.extractor == "kmer"
                      else minhash.from_fasta(path, word, model.sketch_size, seed))
            if not groups:
                print(f"  EMPTY input (no fragments): {config.relative(path)}")
                failed.append(name)
                continue
            store.save(groups, destination)
        elif model.extractor == "bert":
            if (directory / f"{name}.pickle").exists() and not force:
                continue
            try:
                transformers.extract_bert(model, path, name, latent_dir, pooling)
            except Exception as error:                      # report, keep going
                print(f"  FAILED {name}: {error}")
                failed.append(name)
                continue
        elif model.extractor == "evo2":
            if (directory / f"{name}.pickle").exists() and not force:
                continue
            if not transformers.extract_evo2(model, path, name, latent_dir):
                failed.append(name)
                continue
        elif model.extractor == "viralm":
            if (directory / f"{name}_layer_1.pkl").exists() and not force:
                continue
            if not transformers.extract_viralm(model, path, name, latent_dir):
                failed.append(name)
                continue
        else:
            raise SystemExit(f"unknown extractor {model.extractor!r} for {model.key}")
        written += 1

    print(f"  {model.key}: {written} files written to {config.relative(directory)}")
    if failed:
        (directory / "failed.txt").write_text("\n".join(failed) + "\n", encoding="utf8")
        raise SystemExit(f"{model.key}: {len(failed)} of {len(items)} files FAILED "
                         f"({', '.join(failed[:5])}{' ...' if len(failed) > 5 else ''}); "
                         f"listed in {config.relative(directory / 'failed.txt')}")
    return written


def run(models, virus_files: dict, latent_dir=None, reference_dir=None,
        preproc_dir=None, n_ref=config.N_REFERENCE, k=None, force=False,
        dry_run=False, pooling: str = pooling_module.DEFAULT_MODE,
        seed: int = config.SEED) -> None:
    """Extract every requested model, one after another.

    Deliberately sequential: a transformer already saturates the GPU, and the two
    cheap baselines finish in seconds, so running them in parallel would only risk
    running out of memory.
    """
    items = targets(virus_files, reference_dir, n_ref, preproc_dir)
    print(f"files per model: {len(items)}  ({n_ref} reference samples + "
          f"{len(items) - n_ref} virus files)")
    if dry_run:
        for model in models:
            print(f"  {model.key:<10} -> {config.relative(store.model_dir(model, latent_dir))}")
        return
    problems = []
    for model in models:
        try:
            extract_model(model, items, latent_dir, k, force, pooling, seed)
        except SystemExit as error:                     # try the other models first
            problems.append(str(error))
    if problems:
        raise SystemExit("extraction incomplete:\n  " + "\n  ".join(problems))
