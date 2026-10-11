"""Shared plotting style, and the hand-off between the two figure stages.

Stage 1 writes the exact numbers of a figure to ``results/figure_data/<name>.csv``
and stage 2 draws from that CSV alone.  Because the CSVs are committed, a figure
can be restyled without any metric table present, and they double as the
machine-readable source data of the article.

Every figure is saved twice: PNG at 300 dpi for the manuscript and PDF with
embedded TrueType fonts, so the text stays selectable in a vector editor.
"""

from pathlib import Path

import matplotlib as mpl
import pandas as pd

from .. import config
from ..models import MODELS


def set_style() -> None:
    """Publication defaults; call once before drawing."""
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def display(key: str) -> str:
    return MODELS[key].display if key in MODELS else key


def color(key: str) -> str:
    return MODELS[key].color if key in MODELS else "#444444"


def save_table(frame: pd.DataFrame, name: str, index: bool = False,
               float_format: str | None = None, directory: Path | None = None) -> Path:
    """Write the numbers behind a figure to results/figure_data/<name>.csv.

    Full precision by default; the significance tables pass ``%.6g`` because their
    p-values span thirty orders of magnitude and the extra digits are noise.
    """
    path = (directory or config.FIGURE_DATA) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, float_format=float_format)
    print(f"  numbers -> {config.relative(path)}")
    return path


def load_table(name: str, directory: Path | None = None, **kwargs) -> pd.DataFrame:
    return pd.read_csv((directory or config.FIGURE_DATA) / f"{name}.csv", **kwargs)


def save_figure(figure, name: str, kind: str = "main", formats=("png", "pdf")) -> None:
    """Save under figures/<kind>/; ``name`` may contain a sub-folder."""
    directory = config.FIGURE_DIRS.get(kind, config.FIG_SUPP)
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        target = Path(f"{path}.{extension}")
        target.unlink(missing_ok=True)       # re-create rather than overwrite in place
                                             # (some network/Windows mounts refuse the latter)
        figure.savefig(target)
    print(f"  figure  -> {config.relative(path)}.{{{','.join(formats)}}}")
