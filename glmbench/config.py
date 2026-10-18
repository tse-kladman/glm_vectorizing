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


ROOT_ENV = "GLMBENCH_ROOT"

# Where the *code* lives (extractor scripts, the ViraLM hook): fixed, whatever the
# data root is.
CODE_ROOT = Path(__file__).resolve().parent.parent


def _find_root(start: Path) -> Path:
    """The repository root, or ``$GLMBENCH_ROOT`` when that is set.

    Without the variable, walk upwards from this file so scripts work from any
    directory.  With it, *every* path below -- inputs, tables, figures, the run
    manifest -- lives under that directory instead, which is what lets the test
    suite run the whole pipeline in a scratch folder without touching the repository.
    """
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "glmbench").is_dir() and (candidate / "data").is_dir():
            return candidate
    return start


def set_root(root) -> Path:
    """Re-point every path in this module at ``root`` (and export it for workers).

    The paths are module-level constants, so changing the root means rebinding them
    all.  The environment variable is set as well so that worker processes started
    with ``spawn`` (Windows, macOS) import this module with the same root.
    """
    global ROOT, DATA, ICTV_DIR, UNSEEN_DIR, PREPROC_DIR, REFERENCE_DIR, LATENT_DIR
    global RESULTS, METRICS, FIGURE_DATA, STATISTICS, SELECTION_FILE, MANIFEST_FILE, SCALING_DISTANCE_DIR
    global FIGURES, FIG_MAIN, FIG_SUPP, FIG_STATS, FIG_EXPLORATORY, FIGURE_DIRS

    ROOT = Path(root).resolve()
    os.environ[ROOT_ENV] = str(ROOT)

    # ── inputs ──────────────────────────────────────────────────────────────
    DATA = ROOT / "data"
    ICTV_DIR = DATA / "ictv_data"
    UNSEEN_DIR = DATA / "unk_viruses"              # genomes of the unseen orders
    PREPROC_DIR = DATA / "preproc_unk_viruses"     # the same, chunked, plus RC
    REFERENCE_DIR = DATA / "random_samples"        # the 50 reference samples
    LATENT_DIR = DATA / "latent_dim"               # extracted representations

    # ── numbers ─────────────────────────────────────────────────────────────
    RESULTS = ROOT / "results"
    METRICS = RESULTS / "metrics"                  # per-sample metric tables (xlsx)
    FIGURE_DATA = RESULTS / "figure_data"          # the numbers plotted in each figure
    STATISTICS = RESULTS / "statistics"            # significance tests
    SELECTION_FILE = METRICS / "selection.json"    # chosen layer / word length per model
    SCALING_DISTANCE_DIR = METRICS / "scaling_distance"  # same models, other scaling/distance
    MANIFEST_FILE = RESULTS / "run_manifest.json"  # what produced the numbers

    # ── plots ───────────────────────────────────────────────────────────────
    FIGURES = ROOT / "figures"
    FIG_MAIN = FIGURES / "main"                    # Fig. 1, 2, 4-8 of the article
    FIG_SUPP = FIGURES / "supplementary"           # every layer of every metric
    FIG_STATS = FIGURES / "statistics"             # significance plots
    FIG_EXPLORATORY = FIGURES / "exploratory"
    FIGURE_DIRS = {
        "main": FIG_MAIN,
        "supplementary": FIG_SUPP,
        "statistics": FIG_STATS,
        "exploratory": FIG_EXPLORATORY,
    }
    return ROOT


set_root(_find_root(Path(__file__).resolve().parent))

# ── analysis defaults ───────────────────────────────────────────────────────
SEED = 42                # the one seed every random choice is derived from
CHUNK_SIZE = 2000        # bp per fragment
MAX_CHUNKS = 10          # fragments kept per input sequence
N_REFERENCE = 50         # reference samples
REFERENCE_GENOMES_PER_CLASS = 10     # genomes drawn per taxonomic class
REFERENCE_MIN_SPECIES = 10           # a class qualifies with MORE than this many species
REFERENCE_CHUNKS_PER_GENOME = 10     # non-overlapping fragments per reference genome
PER_GENOME_KS = (5, 10)              # the k whose per-genome values are stored
KS = tuple(range(1, 11))  # the k of P@k, computed in one pass
PK_FOR_FIGURES = 5       # the P@k reported in Fig. 1, 4 and 5
REFERENCE_METRIC = "P5"  # reference table that chooses the layer and the word length

# Every worker process loads its own copy of numpy/scipy and of the embeddings, so
# the limit is memory rather than cores; MEMORY_BUDGET caps the intermediate tensor
# of the blockwise distances (Jensen-Shannon, MinHash).
MAX_WORKERS = min(8, os.cpu_count() or 1)

# Reference samples scored at once inside one task (threads).  Each holds one whole
# sample in memory -- ~2 GB for Evo2 -- so this, not the core count, bounds the memory.
SAMPLE_THREADS = 4
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
