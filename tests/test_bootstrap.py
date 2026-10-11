"""Cluster bootstrap over genomes."""

import numpy as np
import pandas as pd
import pytest

from glmbench import statistics


def test_requires_an_explicit_seed():
    with pytest.raises(ValueError, match="seed"):
        statistics.bootstrap_ci_by_genome(np.ones((3, 5)))


def test_needs_at_least_two_genomes():
    with pytest.raises(ValueError, match="two genomes"):
        statistics.bootstrap_ci_by_genome(np.ones((3, 1)), seed=0)


def test_constant_values_give_a_degenerate_interval():
    ci = statistics.bootstrap_ci_by_genome(np.full((4, 6), 0.7), n_boot=500, seed=0)
    assert ci["estimate"] == pytest.approx(0.7)
    assert ci["low"] == pytest.approx(0.7) and ci["high"] == pytest.approx(0.7)
    assert ci["n_genomes"] == 6


def test_two_genomes_resample_to_the_three_possible_means():
    """Genome values 0 and 1: a replicate mean can only be 0, 0.5 or 1."""
    ci = statistics.bootstrap_ci_by_genome(np.array([[0.0, 1.0]]), n_boot=2000, alpha=0.5, seed=1)
    assert ci["estimate"] == pytest.approx(0.5)
    assert 0.0 <= ci["low"] <= 0.5 <= ci["high"] <= 1.0
    assert ci["low"] in (0.0, 0.5) and ci["high"] in (0.5, 1.0)


def test_same_seed_same_interval_other_seed_other_interval():
    values = np.random.default_rng(0).uniform(size=(10, 12))
    a = statistics.bootstrap_ci_by_genome(values, n_boot=500, seed=3)
    b = statistics.bootstrap_ci_by_genome(values, n_boot=500, seed=3)
    c = statistics.bootstrap_ci_by_genome(values, n_boot=500, seed=4)
    assert a == b
    assert (a["low"], a["high"]) != (c["low"], c["high"])


def test_interval_narrows_with_more_genomes_and_covers_the_mean():
    rng = np.random.default_rng(2)
    few = rng.uniform(size=(5, 8))
    many = rng.uniform(size=(5, 200))
    a = statistics.bootstrap_ci_by_genome(few, n_boot=2000, seed=0)
    b = statistics.bootstrap_ci_by_genome(many, n_boot=2000, seed=0)
    assert (b["high"] - b["low"]) < (a["high"] - a["low"])
    assert b["low"] < 0.5 < b["high"]


def test_the_genome_not_the_draw_is_the_resampling_unit():
    """Many reference draws of the SAME few genomes must not shrink the interval."""
    rng = np.random.default_rng(3)
    per_genome = rng.uniform(size=8)
    one_draw = per_genome[None, :]
    fifty_draws = np.repeat(one_draw, 50, axis=0)
    a = statistics.bootstrap_ci_by_genome(one_draw, n_boot=2000, seed=0)
    b = statistics.bootstrap_ci_by_genome(fifty_draws, n_boot=2000, seed=0)
    assert a["low"] == pytest.approx(b["low"]) and a["high"] == pytest.approx(b["high"])


def test_coverage_is_roughly_nominal_for_many_genomes():
    rng = np.random.default_rng(4)
    covered = 0
    for trial in range(150):
        genomes = rng.normal(0.5, 0.1, size=(1, 40))
        ci = statistics.bootstrap_ci_by_genome(genomes, n_boot=400, seed=trial)
        covered += ci["low"] <= 0.5 <= ci["high"]
    assert 0.85 <= covered / 150 <= 1.0


def test_paired_difference_uses_the_same_genomes():
    rng = np.random.default_rng(5)
    base = rng.uniform(size=(6, 15))
    ci = statistics.bootstrap_diff_by_genome(base + 0.1, base, n_boot=500, seed=0)
    assert ci["estimate"] == pytest.approx(0.1)
    assert ci["low"] == pytest.approx(0.1) and ci["high"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        statistics.bootstrap_diff_by_genome(base, base[:, :5], seed=0)


def test_nan_cells_are_ignored():
    values = np.array([[0.2, np.nan, 0.6], [0.4, 0.8, 0.6]])
    ci = statistics.bootstrap_ci_by_genome(values, n_boot=200, seed=0)
    assert ci["estimate"] == pytest.approx(np.mean([0.3, 0.8, 0.6]))


def test_genome_matrix_pivots_the_per_genome_table():
    frame = pd.DataFrame({
        "layer": ["-"] * 6, "sample": [0, 0, 1, 1, 2, 2],
        "genome": ["a", "b"] * 3, "metric": ["P@5"] * 6, "value": [1, 2, 3, 4, 5, 6]})
    matrix = statistics.genome_matrix(frame, "P@5")
    assert matrix.shape == (3, 2) and matrix.loc[1, "b"] == 4
    layered = frame.assign(layer=["0", "0", "0", "0", "1", "1"])
    with pytest.raises(ValueError, match="which one"):
        statistics.genome_matrix(layered, "P@5")
    assert statistics.genome_matrix(layered, "P@5", layer="1").shape == (1, 2)
