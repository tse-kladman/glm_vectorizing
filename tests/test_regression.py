"""The refactor is only correct if the published numbers do not move.

Every CSV under ``results/figure_data`` is rebuilt from the committed metric tables
and compared against the committed copy.  The tables are in the repository, so this
runs anywhere -- no GPU, no embeddings, no cluster.

Rebuilt into a temporary directory, never over the committed files.

    python -m pytest tests/test_regression.py
"""

import pandas as pd
import pytest

from glmbench import config, models as registry
from glmbench.figures import main, supplementary

PUBLISHED = registry.resolve(registry.PUBLISHED)

MAIN_TABLES = {
    "fig1_reference_benchmark": lambda: main.build_fig1(PUBLISHED),
    "fig2_layerwise": lambda: main.build_fig2(PUBLISHED),
    "fig4_newvirus_p5": lambda: main.build_fig4(PUBLISHED),
    "fig5_cross_p5": lambda: main.build_fig5(PUBLISHED),
    "fig6_pk_vs_k": lambda: main.build_fig6(PUBLISHED),
    "fig7_pk_vs_cross": lambda: main.build_fig7(PUBLISHED),
    "supplementary/layerwise_reference": lambda: supplementary.build_reference(PUBLISHED),
    "supplementary/layerwise_newvirus_p5": lambda: supplementary.build_unseen(PUBLISHED),
}


@pytest.fixture(autouse=True)
def _write_elsewhere(tmp_path, monkeypatch):
    """Rebuild into a temporary folder so the committed numbers are never touched."""
    monkeypatch.setattr(config, "FIGURE_DATA", tmp_path)


def committed(name: str) -> pd.DataFrame:
    path = config.ROOT / "results" / "figure_data" / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{name}.csv is not committed")
    frame = pd.read_csv(path)
    # The committed CSVs now also hold the models added later (MinHash, k-mer JS); this
    # test pins the numbers of the five original models, which must not have moved.
    # The added models are checked in test_added_models_are_present.
    if "model" in frame.columns:
        frame = frame[frame["model"].isin(registry.PUBLISHED)]
    return frame.reset_index(drop=True)


@pytest.mark.parametrize("name", list(MAIN_TABLES))
def test_rebuilds_the_committed_numbers(name):
    if not (config.METRICS / "kmer").is_dir():
        pytest.skip("metric tables are not present")

    rebuilt = MAIN_TABLES[name]().reset_index(drop=True)
    expected = committed(name)

    assert list(rebuilt.columns) == list(expected.columns), f"{name}: columns changed"
    assert len(rebuilt) == len(expected), f"{name}: row count changed"

    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            pd.testing.assert_series_equal(
                rebuilt[column].astype(float), expected[column].astype(float),
                check_names=False, rtol=0, atol=1e-10,
                obj=f"{name}[{column}]")
        else:
            pd.testing.assert_series_equal(
                rebuilt[column].astype(str), expected[column].astype(str),
                check_names=False, obj=f"{name}[{column}]")


def test_added_models_are_present_in_the_figure_data():
    """The 7-model rebuild: MinHash and k-mer JS appear where their tables exist."""
    if not (config.METRICS / "minhash").is_dir():
        pytest.skip("metric tables are not present")
    fig4 = pd.read_csv(config.ROOT / "results" / "figure_data" / "fig4_newvirus_p5.csv")
    assert {"kmer_js", "minhash"} <= set(fig4["model"])
    fig1 = pd.read_csv(config.ROOT / "results" / "figure_data" / "fig1_reference_benchmark.csv")
    assert set(fig1["model"]) == set(registry.PUBLISHED)     # no benchmark table yet for the others
