"""The whole pipeline, start to finish, on a synthetic world, through the real CLI.

A tiny world is built in a scratch directory that becomes the data root
(``GLMBENCH_ROOT`` / ``config.set_root``):

    * genomes of three reference "classes" and two unseen "orders", each with its own
      composition (a random order-1 Markov chain per class; the genomes of an order are
      noisy copies of one chain),
    * a mock VMR spreadsheet that lists 12 species per class,

and then runs, via ``glmbench.cli.main([...])``:

    samples -> prep -> sweep -> select -> extract -> benchmark -> metrics
            -> metrics --scaling-distance -> stats -> figures

for the three models that need no GPU and no download (kmer, kmer_js, minhash).  The
transformer extractors are not exercised here -- they need torch, the
model weights and a GPU; their pooling arithmetic and failure handling are covered by
test_pooling.py and test_extraction.py.

The test also proves that nothing is written outside the root.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glmbench import cli, config, models

MODELS = ["kmer", "kmer_js", "minhash"]
N_REF = 6
LETTERS = np.array(list("ACGT"))


def markov_chain(rng, sharpness=3.0):
    return rng.dirichlet(np.ones(4) * sharpness, size=4)


def perturb(rng, chain, noise=0.08):
    noisy = np.clip(chain + rng.normal(0, noise, chain.shape), 0.02, None)
    return noisy / noisy.sum(axis=1, keepdims=True)


def genome(rng, chain, length):
    out = np.empty(length, dtype=np.int8)
    out[0] = rng.integers(4)
    uniform = rng.random(length)
    cumulative = chain.cumsum(axis=1)
    for i in range(1, length):
        out[i] = min(int(np.searchsorted(cumulative[out[i - 1]], uniform[i])), 3)
    return "".join(LETTERS[out])


def snapshot(root: Path) -> dict:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for sub in ("data", "results", "figures") if (root / sub).exists()
            for p in (root / sub).rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    real_root = config.ROOT
    before = snapshot(real_root)
    root = tmp_path_factory.mktemp("world")
    saved = (os.environ.get(config.ROOT_ENV), config.MEMORY_BUDGET, config.MAX_WORKERS)
    config.set_root(root)

    rng = np.random.default_rng(123)
    # three reference classes, 12 species each -> all three qualify (> 10 species)
    class_chains = {f"Class{c}viricetes": markov_chain(rng) for c in "ABC"}
    rows, fasta = [], []
    serial = 0
    for cls, chain in class_chains.items():
        for s in range(12):
            serial += 1
            accession = f"AB{200000 + serial}"
            rows.append({"Isolate ID": serial, "Class": cls, "Species": f"{cls} {s}",
                         "Exemplar or additional isolate": "E",
                         "Virus GENBANK accession": accession})
            fasta.append(f">{accession}.1\n{genome(rng, perturb(rng, chain, 0.04), 6200)}\n")
    genomes_dir = root / "genomes"
    genomes_dir.mkdir()
    (genomes_dir / "refs.fa").write_text("".join(fasta))
    vmr = root / "vmr.xlsx"
    with pd.ExcelWriter(vmr) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="VMR MSL40", index=False)

    config.UNSEEN_DIR.mkdir(parents=True)
    for order in ("Egovirales", "Mriyavirales"):
        chain = markov_chain(rng)
        (config.UNSEEN_DIR / f"{order}.fa").write_text("".join(
            f">{order[:3]}{g}\n{genome(rng, perturb(rng, chain), 11000)}\n" for g in range(6)))

    yield {"root": root, "vmr": vmr, "genomes": genomes_dir, "real_root": real_root,
           "before": before}

    config.set_root(real_root)
    config.MEMORY_BUDGET, config.MAX_WORKERS = saved[1], saved[2]
    if saved[0] is None:
        os.environ.pop(config.ROOT_ENV, None)
    else:
        os.environ[config.ROOT_ENV] = saved[0]


@pytest.fixture(scope="module")
def pipeline(world):
    """Run every stage once; the tests below inspect what it left behind."""
    models_arg = ["--models", *MODELS]
    common = ["--max-workers", "1"]
    stages = [
        ["samples", "--vmr", str(world["vmr"]), "--fasta-dir", str(world["genomes"]),
         "--n-ref", str(N_REF), "--expected-classes", "3", *common],
        ["prep", "--max-chunks", "4", *common],
        ["sweep", "--n-ref", str(N_REF), "--k-values", "2", "3", "4", *common],
        ["select", *models_arg, "--sweep-samples", "3", "--k-values", "3", "4", "5", *common],
        ["extract", *models_arg, "--n-ref", str(N_REF), *common],
        ["benchmark", *models_arg, "--n-ref", str(N_REF), *common],
        ["select", *models_arg, "--k-values", "3", "4", "5", "--sweep-samples", "3", *common],
        ["metrics", *models_arg, "--n-ref", str(N_REF), *common],
        ["metrics", *models_arg, "--n-ref", str(N_REF), "--scaling-distance",
         "--layers", "selected", *common],
        ["stats", *models_arg, *common],
        ["figures", *models_arg, *common],
    ]
    for argv in stages:
        cli.main(argv)
    return world["root"]


def table(root, model, name):
    return pd.read_excel(root / "results" / "metrics" / model / name)


# ── inputs and extraction ───────────────────────────────────────────────────

def test_reference_samples_and_fragments_exist(pipeline):
    for i in range(N_REF):
        assert (pipeline / "data" / "random_samples" / f"sample_{i}.fa").exists()
        assert (pipeline / "data" / "random_samples" / f"manifest_sample_{i}.tsv").exists()
    for name in ("Egovirales", "Mriyavirales"):
        assert (pipeline / "data" / "preproc_unk_viruses" / f"{name}_preproc.fa").exists()
        assert (pipeline / "data" / "preproc_unk_viruses"
                / f"REVERSECOMPLEMENT{name}_preproc.fa").exists()
    for model in MODELS:
        assert (pipeline / "data" / "latent_dim" / model / "sample_0.pickle").exists()
        assert (pipeline / "data" / "latent_dim" / model / "Egovirales_preproc.pickle").exists()


def test_fragments_are_complete_and_random(pipeline):
    from Bio import SeqIO

    path = pipeline / "data" / "preproc_unk_viruses" / "Egovirales_preproc.fa"
    records = list(SeqIO.parse(str(path), "fasta"))
    assert {len(r.seq) for r in records} == {2000}
    assert len(records) == 6 * 4
    sidecar = pd.read_csv(path.with_suffix(".chunks.tsv"), sep="\t", comment="#")
    assert (sidecar["start"] % 2000 == 0).all()
    assert sidecar.groupby("genome")["start"].apply(lambda s: list(s) == sorted(s)).all()


def test_selection_recorded_the_word_length_of_the_tuned_baselines(pipeline):
    chosen = json.loads((pipeline / "results" / "metrics" / "selection.json").read_text())
    assert chosen["kmer_js"]["k"] in (3, 4, 5)
    assert chosen["minhash"]["k"] in (3, 4, 5)


# ── the reference benchmark ─────────────────────────────────────────────────

def test_benchmark_numbers_are_sane(pipeline):
    for model in MODELS:
        for metric in ("ARI", "P5", "silh"):
            frame = table(pipeline, model, f"{metric}.xlsx")
            assert frame.shape == (1, N_REF)
            assert np.isfinite(frame.to_numpy()).all()
    p5 = table(pipeline, "kmer", "P5.xlsx").to_numpy()
    assert p5.min() > 0.6                      # three classes with distinct composition
    assert table(pipeline, "kmer", "ARI.xlsx").to_numpy().mean() > 0.4
    assert table(pipeline, "kmer", "silh.xlsx").to_numpy().mean() > 0.0


# ── the unseen-order metrics ────────────────────────────────────────────────

def matrix(root, model, name):
    """A metric table under ``root`` (a repository root, or a metrics folder)."""
    from glmbench.metrics import tables
    base = root / "results" / "metrics" if (root / "results").is_dir() else root
    return tables.load_matrix(base / model / name)


def test_genome_level_and_rc_proximity_tables_have_sane_values(pipeline):
    for model in MODELS:
        for virus in ("Egovirales_preproc", "Mriyavirales_preproc"):
            genome_level = matrix(pipeline, model, f"P@5_matrix_{virus}.xlsx")
            proximity = matrix(pipeline, model, f"cross_P@5_matrix_{virus}.xlsx")
            assert genome_level.shape == (1, N_REF) and proximity.shape == (1, N_REF)
            # 4 fragments per genome leave only 3 possible same-genome neighbours, so
            # P@5 is capped at 3/5; the weaker representations (sketches, JS) stay below it
            assert 0.2 < genome_level.to_numpy().min() <= 0.6 + 1e-9
            assert proximity.to_numpy().min() >= 0 and proximity.to_numpy().max() <= 1


def test_the_strand_invariant_sketch_has_its_own_rc_as_nearest_neighbour(pipeline):
    """A MinHash sketch of x equals that of RC(x): RC proximity at k = 1 is exactly 1."""
    proximity = matrix(pipeline, "minhash", "cross_P@1_matrix_Egovirales_preproc.xlsx")
    assert proximity.to_numpy().mean() == pytest.approx(1.0)


def test_the_kmer_sweep_has_every_word_length_and_sample(pipeline):
    path = pipeline / "results" / "metrics" / "kmer_sweep" / "kmer_sweep_results.xlsx"
    raw = pd.read_excel(path, sheet_name="raw")
    assert sorted(raw["k"].unique()) == [2, 3, 4]
    assert raw.groupby("k")["sample"].nunique().eq(N_REF).all()
    assert raw["P@5"].between(0, 1).all() and raw["ARI"].notna().all()
    summary = pd.read_excel(path, sheet_name="summary").set_index("k")
    assert summary.loc[4, "P@5_mean"] == pytest.approx(raw[raw["k"] == 4]["P@5"].mean())


def test_scaling_distance_variants_were_written_for_the_cosine_models_only(pipeline):
    base = pipeline / "results" / "metrics" / "scaling_distance"
    for variant in ("raw_cosine", "standardized_euclidean", "raw_euclidean"):
        for virus in ("Egovirales_preproc", "REVERSECOMPLEMENTMriyavirales_preproc"):
            frame = matrix(base / variant, "kmer", f"P@5_matrix_{virus}.xlsx")
            assert frame.shape == (1, N_REF) and np.isfinite(frame.to_numpy()).all()
        assert not (base / variant / "kmer_js").exists()
        assert not (base / variant / "kmer" / "pairwise_Egovirales_vs_Mriyavirales").exists()
    figure = pd.read_csv(pipeline / "results" / "figure_data" / "supplementary"
                         / "scaling_distance.csv")
    assert set(figure.loc[figure["model"] == "kmer", "variant"]) == {
        "standardized_cosine", "raw_cosine", "standardized_euclidean", "raw_euclidean"}


def test_pairwise_tables_exist(pipeline):
    pairwise = pipeline / "results" / "metrics" / "kmer" / "pairwise_Egovirales_vs_Mriyavirales"
    assert (pairwise / "cross_1→2_fwd_fwd_matrix.xlsx").exists()
    assert (pairwise / "pk_fwd_1_in_2_matrix.xlsx").exists()


# ── statistics and figures ──────────────────────────────────────────────────

def test_statistics_were_computed(pipeline):
    tests = pd.read_csv(pipeline / "results" / "statistics" / "pairwise_model_tests.csv")
    assert len(tests) > 0 and {"p_wilcoxon", "p_bh_all", "signif"} <= set(tests.columns)
    assert set(tests["model_a"]) | set(tests["model_b"]) == set(MODELS)


def test_figures_were_drawn(pipeline):
    main = pipeline / "figures" / "main"
    for name in ("fig1_reference_benchmark", "fig2_layerwise", "fig4_newvirus_p5",
                 "fig5_cross_p5", "fig6_pk_vs_k", "fig7_pk_vs_cross"):
        assert (main / f"{name}.png").stat().st_size > 1000, name
        assert (main / f"{name}.pdf").exists(), name
    assert (pipeline / "figures" / "main" / "fig8_scaling_distance.png").exists()
    assert (pipeline / "figures" / "supplementary" / "figS1_kmer_sweep.png").exists()
    assert (pipeline / "figures" / "statistics" / "pvalue_overview.png").exists()
    data = pd.read_csv(pipeline / "results" / "figure_data" / "fig4_newvirus_p5.csv")
    assert set(data["model"]) == set(MODELS)


# ── provenance and containment ──────────────────────────────────────────────

def test_the_manifest_records_the_run(pipeline):
    manifest = json.loads((pipeline / "results" / "run_manifest.json").read_text())
    stages = {entry["stage"] for entry in manifest["stages"]}
    assert {"samples", "prep", "sweep", "select", "extract", "benchmark", "metrics",
            "stats", "figures"} <= stages
    assert manifest["python"] and manifest["packages"]["numpy"]
    assert "commit" in manifest["git"]
    first = manifest["stages"][0]
    assert first["seeds"]["seed"] == 42 and first["command_line"][0] == "run_pipeline.py"
    assert "revision" in first["models"]["kmer"] or first["models"] == {}


def test_nothing_was_written_outside_the_root(pipeline, world):
    after = snapshot(world["real_root"])
    assert after == world["before"], "a stage wrote into (or changed) the repository"
