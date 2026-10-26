"""Reading and writing the per-sample metric tables.

Every metric is stored the same way: an Excel sheet whose rows are layers (``-``
for a layer-less model) and whose columns are the 50 reference samples.  Keeping
the per-sample values -- rather than only a mean -- is what makes the paired tests
in statistics.py possible, and what lets any figure be rebuilt without recomputing
anything.

    results/metrics/<model>/P@<k>_matrix_<virus>.xlsx          genome-level P@k
    results/metrics/<model>/cross_P@<k>_matrix_<virus>.xlsx    RC proximity ("CROSS P@k")
    results/metrics/<model>/{ARI,P5,silh}.xlsx                 reference benchmark
    results/metrics/<model>/pairwise_<A>_vs_<B>/<name>_matrix.xlsx
    results/metrics/scaling_distance/<variant>/<model>/...      the same under another
                                                                scaling / distance (Fig. 8)
    results/metrics/k_sweep/<model>_k_sweep.xlsx               word-length sweep (P@5)
    results/metrics/kmer_sweep/kmer_sweep_results.xlsx         4-mer benchmark sweep (Fig. S1a)
"""

from pathlib import Path

import pandas as pd

from .. import config
from ..models import Model

FLAT_LAYER = "-"      # the layer label of a model that has no layers


def model_dir(model: Model | str, metrics_dir: Path | None = None) -> Path:
    key = model if isinstance(model, str) else model.key
    return Path(metrics_dir or config.METRICS) / key


def scaling_distance_dir(variant: str, metrics_dir: Path | None = None) -> Path:
    """Metrics folder of one scaling/distance variant (``models.SCALING_DISTANCE``)."""
    return Path(metrics_dir or config.METRICS) / "scaling_distance" / variant


def k_sweep_path(model: Model | str, metrics_dir: Path | None = None) -> Path:
    key = model if isinstance(model, str) else model.key
    return Path(metrics_dir or config.METRICS) / "k_sweep" / f"{key}_k_sweep.xlsx"


def kmer_sweep_path(metrics_dir: Path | None = None) -> Path:
    return Path(metrics_dir or config.METRICS) / "kmer_sweep" / "kmer_sweep_results.xlsx"


def matrix_name(prefix: str, k: int, virus: str) -> str:
    return f"{prefix}@{k}_matrix_{virus}.xlsx"


def save_matrix(matrix: dict, path: Path) -> Path:
    """{layer: {sample index: value}} -> one sheet, rows = layers, columns = samples."""
    frame = pd.DataFrame(matrix).T.sort_index()
    frame.index.name = "layer"
    frame.columns = [f"sample_{column}" for column in frame.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path)
    return path


def load_matrix(path: Path) -> pd.DataFrame:
    """One sheet -> rows = layers (sorted), columns = integer sample index (sorted)."""
    frame = pd.read_excel(path).set_index("layer")
    frame.columns = [int(str(column).split("_")[1]) for column in frame.columns]
    return frame.sort_index().sort_index(axis=1)


def load_model_matrix(model: Model | str, filename: str,
                      metrics_dir: Path | None = None) -> pd.DataFrame:
    return load_matrix(model_dir(model, metrics_dir) / filename)


def load_benchmark(model: Model | str, metric: str = config.REFERENCE_METRIC,
                   metrics_dir: Path | None = None) -> pd.DataFrame:
    """Reference benchmark table: rows are layers by position, columns are samples."""
    frame = pd.read_excel(model_dir(model, metrics_dir) / f"{metric}.xlsx")
    frame.columns = [int(column) for column in frame.columns]
    return frame


def save_benchmark(frame: pd.DataFrame, model: Model | str, metric: str,
                   metrics_dir: Path | None = None) -> Path:
    """Write a benchmark table in the format ``load_benchmark`` reads."""
    path = model_dir(model, metrics_dir) / f"{metric}.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.columns = [int(column) for column in out.columns]
    out.to_excel(path, index=False)
    return path


def has_benchmark(model: Model | str, metric: str = config.REFERENCE_METRIC,
                  metrics_dir: Path | None = None) -> bool:
    return (model_dir(model, metrics_dir) / f"{metric}.xlsx").exists()
