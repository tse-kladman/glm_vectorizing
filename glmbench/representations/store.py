"""Reading and writing ``data/latent_dim/`` -- the extracted representations.

Layered models store one pickle per FASTA file holding ``{layer: {label: rows}}``;
layer-less ones store ``{label: rows}`` directly.  ViraLM is the exception: its
hook writes one file per layer, which this module folds into the same shape, so
callers never have to care.

``load_virus`` and ``load_sample`` therefore always return the same structure:

    layered     {layer: {label: array}}
    flat        {label: array}
"""

import pickle
from collections import defaultdict
from pathlib import Path

from .. import config
from ..models import Model

VIRALM_LAYERS = range(1, 13)


def model_dir(model: Model, latent_dir: Path | None = None) -> Path:
    """Where a model keeps its representations; the tuned word length is in the name."""
    base = Path(latent_dir or config.LATENT_DIR)
    return base / model.key


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def _load_pickle(path: Path, problems: list | None = None):
    """Load a pickle.  A missing file gives ``None``; a corrupt one also gives ``None``
    but is *reported*: printed, and appended to ``problems`` when a list is passed,
    so that the caller can count what was lost instead of silently scoring on less.
    """
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        return None
    except (EOFError, pickle.UnpicklingError, OSError, ValueError, AttributeError,
            ImportError, IndexError) as error:
        print(f"  WARNING: cannot read {path.name}: {error}")
        if problems is not None:
            problems.append((path.name, f"corrupt: {error}"))
        return None


def _viralm_groups(raw: dict) -> dict:
    """ViraLM keys look like ``{label}_{chunk}_{start}_{end}``; fold back to labels."""
    groups = defaultdict(list)
    for key, value in raw.items():
        groups[key.rsplit("_", 3)[0]].append(value)
    return dict(groups)


def load_virus(model: Model, name: str, latent_dir: Path | None = None,
               problems: list | None = None):
    directory = model_dir(model, latent_dir)
    if model.extractor == "viralm":
        by_layer = {}
        for layer in VIRALM_LAYERS:
            raw = _load_pickle(directory / f"{name}_layer_{layer}.pkl", problems)
            if raw is not None:
                by_layer[layer] = _viralm_groups(raw)
        return by_layer or None
    return _load_pickle(directory / f"{name}.pickle", problems)


def load_sample(model: Model, index: int, latent_dir: Path | None = None, layers=None,
                problems: list | None = None):
    """One reference sample, or ``None`` if it is missing or unreadable.

    ``problems`` (a list) collects ``(file, reason)`` for corrupt files and for
    ViraLM samples that lack some of their layer files.
    """
    directory = model_dir(model, latent_dir)
    if model.extractor == "viralm":
        wanted = list(layers or VIRALM_LAYERS)
        by_layer = {}
        for layer in wanted:
            raw = _load_pickle(directory / f"sample_{index}_layer_{layer}.pkl", problems)
            if raw is not None:
                by_layer[layer] = _viralm_groups(raw)
        if by_layer and len(by_layer) < len(wanted) and problems is not None:
            lost = sorted(set(wanted) - set(by_layer))
            problems.append((f"sample_{index}", f"missing ViraLM layers {lost}"))
        return by_layer or None
    return _load_pickle(directory / f"sample_{index}.pickle", problems)


def available(model: Model, latent_dir: Path | None = None) -> bool:
    directory = model_dir(model, latent_dir)
    return directory.is_dir() and any(directory.iterdir())
