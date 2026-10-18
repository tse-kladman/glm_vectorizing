"""Reading and writing ``data/latent_dim/`` -- the extracted representations.

Layered models store one pickle per FASTA file holding ``{layer: {label: rows}}``;
layer-less ones store ``{label: rows}`` directly.  ViraLM is the exception: its
hook writes one file per layer and one entry per 1,000-nt *window* (two per 2,000-nt
fragment); this module averages the windows of a fragment and folds the files into
the same shape, so callers never have to care (see ``VIRALM_WINDOWS``).

``load_virus`` and ``load_sample`` therefore always return the same structure:

    layered     {layer: {label: array}}
    flat        {label: array}
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config
from ..models import Model

VIRALM_LAYERS = range(1, 13)


def model_dir(model: Model, latent_dir: Path | None = None) -> Path:
    """Where a model keeps its representations (one folder per model key; the word
    length a k-mer/MinHash folder was made with is in its ``extraction.json``)."""
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


# How the windows of one fragment are combined.  The ViraLM hook cuts every input
# record into 1,000-nt windows, so a 2,000-nt fragment arrives as TWO entries,
# ``{label}_{chunk}_0_1000`` and ``{label}_{chunk}_1000_2000``.  The first release kept
# each window as its own point of the space: the sister window of the same fragment was
# then always an admissible "same class" neighbour (self-exclusion removes only the
# window itself), which inflated every P@k, CROSS P@k, ARI and silhouette of ViraLM
# and slowed the decay of P@k with k.
#     "mean"      one point per fragment: the mean of its window vectors (default)
#     "windows"   one point per window, as in the first release (to reproduce it)
VIRALM_WINDOWS = "mean"


def _viralm_groups(raw: dict, windows: str | None = None) -> dict:
    """ViraLM keys look like ``{label}_{chunk}_{start}_{end}``; fold back to labels.

    With ``windows="mean"`` the windows of one fragment (same ``{label}_{chunk}``) are
    averaged into one vector, so ViraLM contributes one row per fragment like every
    other model.  A window the hook skipped (a letter other than A, C, G, T) is simply
    absent, and the fragment is then represented by its remaining window.  Fragments
    keep the order of their first window, i.e. FASTA order, which the RC matching
    relies on.
    """
    windows = windows or VIRALM_WINDOWS
    if windows not in ("mean", "windows"):
        raise ValueError(f"unknown ViraLM window mode {windows!r}")
    groups = defaultdict(list)
    if windows == "windows":
        for key, value in raw.items():
            groups[key.rsplit("_", 3)[0]].append(value)
        return dict(groups)

    by_fragment = defaultdict(list)
    for key, value in raw.items():
        by_fragment[key.rsplit("_", 2)[0]].append(np.asarray(value))
    for fragment, rows in by_fragment.items():
        groups[fragment.rsplit("_", 1)[0]].append(np.mean(np.stack(rows), axis=0))
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
