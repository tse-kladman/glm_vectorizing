"""
figure_io.py -- shared I/O and styling helpers for the article figures.

The notebooks in this folder follow a two-stage, fully reproducible workflow:

    Stage 1  (extract)   raw metric tables in  data/metrics/**   -->  aggregate
                         -->  committed numbers in  results/figure_data/*.csv

    Stage 2  (plot)      results/figure_data/*.csv   -->  render
                         -->  results/figures/{main,supplementary}/<name>.{png,pdf}

Because every figure is drawn from a small CSV that is committed to git, restyling
a plot never requires re-reading the large Excel tables or recomputing anything --
you only re-run Stage 2.  This module holds the paths, the shared colour palette
and the four helper functions that both stages use.
"""

from pathlib import Path

import matplotlib as mpl
import pandas as pd


# --------------------------------------------------------------------------- #
#  Paths -- located relative to the repository root so the notebooks work       #
#  whether they are launched from the repo root or from ``visualization/``.     #
# --------------------------------------------------------------------------- #
def _find_root(start: Path) -> Path:
    """Walk upwards until we find the folder that holds both data/ and results/."""
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() and (candidate / "results").is_dir():
            return candidate
    return start


ROOT = _find_root(Path(__file__).resolve().parent)

DATA_METRICS = ROOT / "data" / "metrics"      # inputs: raw metric tables
RESULTS = ROOT / "results"                    # everything produced by the notebooks
FIG_DATA = RESULTS / "figure_data"            # committed numbers behind each figure
FIG_MAIN = RESULTS / "figures" / "main"       # main-text figures (fig1..fig7)
FIG_SUPP = RESULTS / "figures" / "supplementary"

for _folder in (FIG_DATA, FIG_MAIN, FIG_SUPP):
    _folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Models: canonical display order, pretty names and a colour-blind-safe        #
#  palette (Okabe-Ito).                                                         #
# --------------------------------------------------------------------------- #
MODELS = ["kmer", "dnabert2", "dnabertS", "viralm", "evo2"]

DISPLAY = {
    "kmer": "4-mer",
    "dnabert2": "DNABERT-2",
    "dnabertS": "DNABERT-S",
    "viralm": "ViraLM",
    "evo2": "Evo2-7B",
}

MODEL_COLOR = {
    "kmer": "#999999",
    "dnabert2": "#0072B2",
    "dnabertS": "#E69F00",
    "viralm": "#009E73",
    "evo2": "#D55E00",
}


def set_style() -> None:
    """Apply the shared, publication-ready matplotlib defaults (call once per notebook)."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "pdf.fonttype": 42,   # embed TrueType so the PDF text stays editable
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- #
#  Stage 1 <-> Stage 2 hand-off: the committed "numbers behind each figure".     #
# --------------------------------------------------------------------------- #
def save_table(df: pd.DataFrame, name: str, *, index: bool = False) -> Path:
    """Write the exact numbers plotted in a figure to results/figure_data/<name>.csv.

    ``name`` may contain a sub-folder, e.g. ``"supplementary/evo2_ARI"``.
    """
    path = FIG_DATA / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"  numbers  ->  {path.relative_to(ROOT).as_posix()}")
    return path


def load_table(name: str, **read_csv_kwargs) -> pd.DataFrame:
    """Read a committed figure-data CSV back in (Stage 2 never touches data/metrics)."""
    return pd.read_csv(FIG_DATA / f"{name}.csv", **read_csv_kwargs)


# --------------------------------------------------------------------------- #
#  Saving the rendered figures.                                                 #
# --------------------------------------------------------------------------- #
def save_figure(fig, name: str, *, kind: str = "main", formats=("png", "pdf")) -> None:
    """Save a figure as PNG (final raster) and PDF (vector) under results/figures/<kind>/.

    kind : "main" for main-text figures, "supplementary" for the rest.
    """
    out = FIG_MAIN if kind == "main" else FIG_SUPP
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}")
    rel = (out / name).relative_to(ROOT).as_posix()
    print(f"  figure   ->  {rel}.{{{','.join(formats)}}}")
