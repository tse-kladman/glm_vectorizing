"""Significance marks on the figures: the helpers and the scaling/distance table."""

import numpy as np
import pandas as pd
import pytest

from glmbench import config
from glmbench.figures import significance, supplementary
from glmbench.metrics import tables


def test_adjust_and_marks_follow_benjamini_hochberg():
    adjusted = significance.adjust({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    # BH: sorted p * n / rank, then a running minimum from the largest p down
    assert adjusted["a"] == pytest.approx(0.001 * 4 / 1)
    assert adjusted["b"] == pytest.approx(0.02 * 4 / 2)
    assert adjusted["c"] == pytest.approx(0.04 * 4 / 3)
    assert adjusted["d"] == pytest.approx(0.5)
    assert significance.marks({"x": 0.0004, "y": 0.004, "z": 0.04, "w": 0.06}) == \
        {"x": "***", "y": "**", "z": "*", "w": "ns"}
    assert significance.label(float("nan")) == ""


def test_against_best_compares_every_row_with_the_best():
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.01, 50)
    table = pd.DataFrame([base, base + 0.05, base + 0.0001 * rng.normal(size=50)],
                         index=[3, 4, 5])
    best, raw = significance.against_best(table)
    assert best == 4
    assert set(raw) == {3, 5}
    assert all(p < 0.001 for p in raw.values())
    # raw p-values: the caller pools several tables into one family before adjusting
    assert raw[3] == pytest.approx(significance.paired_p(table.loc[3], table.loc[4]))


def test_paired_p_aligns_samples_by_index_not_by_position():
    rng = np.random.default_rng(3)
    a = pd.Series(rng.uniform(0.5, 0.6, 30))
    b = a + rng.normal(0.01, 0.02, 30)
    shuffled = b.sample(frac=1.0, random_state=1)        # same values, other row order
    assert significance.paired_p(a, shuffled) == pytest.approx(significance.paired_p(a, b))


def test_adjust_ignores_comparisons_without_a_p_value():
    adjusted = significance.adjust({"a": 0.01, "b": float("nan"), "c": 0.03})
    assert np.isnan(adjusted["b"])
    assert adjusted["a"] == pytest.approx(0.01 * 2 / 1)      # family of two, not three
    assert adjusted["c"] == pytest.approx(0.03)
    assert significance.label(adjusted["b"]) == ""


def test_study_p_is_symmetric_in_the_pair():
    tests = pd.DataFrame({"panel": ["P@5 X"], "model_a": ["a"], "model_b": ["b"],
                          "p_bh_all": [0.01]})
    assert significance.study_p(tests, "P@5 X", "a", "b") == 0.01
    assert significance.study_p(tests, "P@5 X", "b", "a") == 0.01
    assert np.isnan(significance.study_p(tests, "P@5 Y", "a", "b"))


def _write(directory, key, virus, rows):
    frame = pd.DataFrame(rows, columns=[f"sample_{i}" for i in range(len(rows[0]))])
    frame.insert(0, "layer", ["-"] if len(rows) == 1 else list(range(len(rows))))
    path = tables.model_dir(key, directory) / tables.matrix_name("P", 5, virus)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False)


def test_scaling_distance_tests_each_variant_against_the_main_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURE_DATA", tmp_path / "figure_data")
    rng = np.random.default_rng(1)
    base = rng.uniform(0.7, 0.8, 20)
    for virus, _ in supplementary.SCALING_GROUPS:
        _write(tmp_path, "kmer", virus, [base])
        _write(tmp_path, "kmer_js", virus, [base + 0.05])
        _write(tmp_path / "scaling_distance" / "raw_cosine", "kmer", virus, [base - 0.03])
    monkeypatch.setattr(supplementary, "SCALING_MODELS", ["kmer", "kmer_js"])
    frame = supplementary.build_scaling_distance(tmp_path)

    ego = frame[frame["group"] == "Egovirales"].set_index(["model", "variant"])
    assert np.isnan(ego.loc[("kmer", "standardized_cosine"), "p_vs_main_bh"])
    assert ego.loc[("kmer", "raw_cosine"), "p_vs_main_bh"] < 0.001
    assert ego.loc[("kmer_js", "standardized_cosine"), "p_vs_main_bh"] < 0.001
    assert ego.loc[("kmer", "raw_cosine"), "mean"] == pytest.approx(base.mean() - 0.03)
    # variants without a table are simply absent, not an error
    assert ("kmer", "raw_euclidean") not in ego.index


# ── layer numbers shown in the figures ──────────────────────────────────────

def _benchmark(directory, key, metric, rows):
    frame = pd.DataFrame(rows, columns=range(len(rows[0])))
    tables.save_benchmark(frame, key, metric, directory)


def test_fig1_and_fig2_show_viralm_layers_by_label_and_test_by_position(tmp_path, monkeypatch):
    """ViraLM's benchmark rows are positions 0..11 but its blocks are 1..12: the
    figures must say block 3 where the benchmark says position 2 (as Fig. 4 does),
    while the per-sample values are still taken from position 2."""
    from glmbench import models as registry
    from glmbench.figures import main

    monkeypatch.setattr(config, "FIGURE_DATA", tmp_path / "figure_data")
    rng = np.random.default_rng(2)
    base = rng.uniform(0.4, 0.5, 20)
    viralm_rows = [base + 0.01, base + 0.02, base + 0.2, base + 0.03]    # best: position 2
    for metric in main.FIG1_METRICS:
        _benchmark(tmp_path, "kmer", metric, [base])
        _benchmark(tmp_path, "viralm", metric, viralm_rows)
    models = registry.resolve(["kmer", "viralm"])

    fig1 = main.build_fig1(models, tmp_path).set_index(["metric", "model"])
    assert fig1.loc[("P5", "viralm"), "best_layer"] == 3
    assert fig1.loc[("P5", "viralm"), "mean"] == pytest.approx((base + 0.2).mean())
    assert fig1.loc[("P5", "viralm"), "p_vs_kmer_bh"] < 0.001

    fig2 = main.build_fig2(models, tmp_path)
    viralm = fig2[(fig2["model"] == "viralm") & (fig2["metric"] == "P5")].set_index("layer")
    assert list(viralm.index) == [1, 2, 3, 4]
    assert viralm.loc[3, "mean"] == pytest.approx((base + 0.2).mean())
    assert np.isnan(viralm.loc[3, "p_vs_best_bh"])           # the best layer itself
    assert viralm.loc[1, "p_vs_best_bh"] < 0.001


def test_supplementary_marks_the_chosen_layer_by_label(monkeypatch):
    from glmbench import models as registry
    from glmbench import selection

    monkeypatch.setattr(selection, "load", lambda *a, **k: {"viralm": {"layer": 2},
                                                            "dnabert2": {"layer": 3}})
    assert supplementary._chosen_label(registry.get("viralm")) == 3
    assert supplementary._chosen_label(registry.get("dnabert2")) == 3
    assert supplementary._chosen_label(registry.get("kmer")) is None


def test_figures_stage_builds_the_study_tests_before_the_main_figures(monkeypatch):
    """Fig. 4-6 read their marks from the study table; it must not be the stale one."""
    from glmbench import cli
    from glmbench.figures import main, statistics as statistics_figures, style

    calls = []
    monkeypatch.setattr(cli, "_models_with_tables", lambda args: [])
    monkeypatch.setattr(style, "set_style", lambda: None)
    monkeypatch.setattr(statistics_figures, "build",
                        lambda *a, **k: calls.append("stats") or (None, None))
    monkeypatch.setattr(statistics_figures, "draw_all", lambda *a, **k: calls.append("draw stats"))
    monkeypatch.setattr(main, "build_all", lambda *a, **k: calls.append("build main"))
    monkeypatch.setattr(main, "draw_all", lambda *a, **k: calls.append("draw main"))
    monkeypatch.setattr(supplementary, "build_all", lambda *a, **k: None)
    monkeypatch.setattr(supplementary, "draw_all", lambda *a, **k: None)

    args = cli.build_parser().parse_args(["figures", "--which", "main"])
    cli.stage_figures(args)
    assert calls == ["stats", "build main", "draw main"]

    calls.clear()
    cli.stage_figures(cli.build_parser().parse_args(["figures", "--plot-only"]))
    assert "stats" not in calls and "build main" not in calls


def test_a_registry_model_without_some_tables_is_left_out_not_fatal(tmp_path, monkeypatch, capsys):
    """Extra registry models (minhash1k, kmer_canon) may have been scored only in part."""
    from glmbench import models as registry
    from glmbench.figures import main

    monkeypatch.setattr(config, "FIGURE_DATA", tmp_path / "figure_data")
    monkeypatch.setattr(significance, "study_tests", lambda *a, **k: pd.DataFrame())
    rng = np.random.default_rng(4)
    for virus, _ in main.VIRUSES:
        _write(tmp_path, "kmer", virus, [rng.uniform(0.7, 0.8, 10)])
        _write(tmp_path, "minhash1k", virus, [rng.uniform(0.2, 0.3, 10)])   # P@5 only
        path = tables.model_dir("kmer", tmp_path) / tables.matrix_name("cross_P", 5, virus)
        frame = pd.DataFrame([rng.uniform(0.5, 0.6, 10)],
                             columns=[f"sample_{i}" for i in range(10)])
        frame.insert(0, "layer", ["-"])
        frame.to_excel(path, index=False)
    models = registry.resolve(["kmer", "minhash1k"])
    fig5 = main.build_fig5(models, tmp_path)
    assert set(fig5["model"]) == {"kmer"}
    assert "minhash1k: no cross_P@5" in capsys.readouterr().out
