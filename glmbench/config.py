"""Paths and shared defaults.

The repository separates three kinds of content, and nothing crosses the line:

    data/       inputs only -- what came from outside (ICTV metadata, genomes) plus
                the extracted representations, which are too large to ship
    results/    every number the analysis produces
    figures/    every plot

Import the paths from here rather than writing them out; this module is the only
place that knows the layout.
"""

import os
from pathlib import Path


def _find_root(start: Path) -> Path:
    """Walk upwards to the repository root, so scripts work from any directory."""
    for candidate in (start, *start.parents):
        if (candidate / "glmbench").is_dir() and (candidate / "data").is_dir():
            return candidate
    return start


ROOT = _find_root(Path(__file__).resolve().parent)

# ── inputs ──────────────────────────────────────────────────────────────────
DATA = ROOT / "data"
ICTV_DIR = DATA / "ictv_data"
UNSEEN_DIR = DATA / "unk_viruses"              # genomes of the unseen orders
PREPROC_DIR = DATA / "preproc_unk_viruses"     # the same, chunked, plus RC
REFERENCE_DIR = DATA / "random_samples"        # the 50 reference samples
LATENT_DIR = DATA / "latent_dim"               # extracted representations

# ── numbers ─────────────────────────────────────────────────────────────────
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"                  # per-sample metric tables (xlsx)
FIGURE_DATA = RESULTS / "figure_data"          # the numbers plotted in each figure
STATISTICS = RESULTS / "statistics"            # significance tests
SELECTION_FILE = METRICS / "selection.json"    # chosen layer / word length per model

# ── plots ───────────────────────────────────────────────────────────────────
FIGURES = ROOT / "figures"
FIG_MAIN = FIGURES / "main"                    # Fig. 1-7 of the article
FIG_SUPP = FIGURES / "supplementary"           # every layer of every metric
FIG_STATS = FIGURES / "statistics"             # significance plots
FIG_EXPLORATORY = FIGURES / "exploratory"

FIGURE_DIRS = {
    "main": FIG_MAIN,
    "supplementary": FIG_SUPP,
    "statistics": FIG_STATS,
    "exploratory": FIG_EXPLORATORY,
}

# ── analysis defaults ───────────────────────────────────────────────────────
CHUNK_SIZE = 2000        # bp per fragment
MAX_CHUNKS = 10          # fragments kept per input sequence
N_REFERENCE = 50         # reference samples
KS = tuple(range(1, 11))  # the k of P@k, computed in one pass
PK_FOR_FIGURES = 5       # the P@k reported in Fig. 1, 4 and 5
REFERENCE_METRIC = "P5"  # reference table that chooses the layer and the word length

# Every worker process loads its own copy of numpy/scipy and of the embeddings, so
# the limit is memory rather than cores; MEMORY_BUDGET caps the intermediate tensor
# of the blockwise distances (Jensen-Shannon, MinHash).
MAX_WORKERS = min(8, os.cpu_count() or 1)
MEMORY_BUDGET = 128 << 20


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    """Repository-relative path for log lines."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
