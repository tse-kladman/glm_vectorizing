# Viral genome embeddings: benchmarking genomic language models

Code and data accompanying the article *Assessing the Capacity of Genomic Language Models
to Generate Informative Embeddings of Viral Genome Fragments* (submitted to MDPI Viruses).
DNABERT-2, DNABERT-S, ViraLM and Evo2 are compared with four classical baselines (4-mer
profile, canonical 4-mer profile, k-mer + Jensen-Shannon, MinHash) on viral genomes: how
well each representation organises a reference set of known viruses, how well it
generalises to virus orders the models have never seen, and whether it reads the two
strands of a sequence as the same thing.

---

## Quick start (CPU only)

No GPU, no genomes, no model weights.

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # unit tests + the synthetic end-to-end run
make smoke                                # the same end-to-end run alone (~2 min)
python run_pipeline.py figures --plot-only   # redraw every figure from the committed CSVs
python run_pipeline.py stats                 # recompute the paired tests from the committed tables
python run_pipeline.py figures               # rebuild the CSVs from results/metrics, then draw
```

`tests/test_end_to_end.py` builds a tiny synthetic world (genomes with class-specific
composition, a mock ICTV table, two "orders") in a scratch directory -- the data root is
overridable with the environment variable `GLMBENCH_ROOT` -- and runs
`samples -> prep -> select -> extract -> benchmark -> metrics -> stats -> figures` through
`glmbench.cli.main` for the four CPU models (`kmer`, `kmer_canon`, `kmer_js`, `minhash`).
It asserts that every output exists, that the numbers are sane, and that nothing was
written outside the root. The transformer extractors are not part of it.

## Full reproduction (GPU)

Needs the genomes (a FASTA directory whose record ids start with the GenBank accessions of
the ICTV VMR isolates, plus the two unseen orders in `data/unk_viruses/<order>.fa`), a GPU
for the four language models, and the Evo2 / ViraLM conda environments.

```bash
python run_pipeline.py samples --fasta-dir /path/to/genomes --seed 42        # 50 reference samples
python run_pipeline.py prep --seed 42 --force                                # fragments + RC
python run_pipeline.py select --models kmer_js minhash                       # word length (needs samples)
python run_pipeline.py extract --models kmer kmer_canon kmer_js minhash
python run_pipeline.py extract --models dnabert2 dnabertS --pooling mean --force
python run_pipeline.py extract --models viralm
python run_pipeline.py extract --models evo2
python run_pipeline.py benchmark --max-workers 2 --sample-threads 2          # ARI / P@5 / silhouette
python run_pipeline.py select                                                # layers, on the benchmark
python run_pipeline.py metrics --protocol both --max-workers 2 --sample-threads 2
python run_pipeline.py stats
python run_pipeline.py figures
```

`python run_pipeline.py all --fasta-dir ...` runs the same chain in one go. Every stage
exits non-zero if anything was missing, corrupt or failed (see *Failure handling*), and
writes `results/run_manifest.json`.

---

## Repository layout

```
data/                 INPUTS ONLY -- what came from outside
    ictv_data/        ICTV Virus Metadata Resource (VMR MSL40)
    unk_viruses/      genomes of the unseen orders          not shipped, too large
    preproc_unk_viruses/  the same, cut into fragments      not shipped
    random_samples/   the 50 reference samples (+ manifests) not shipped
    latent_dim/       extracted representations             not shipped

results/              EVERY NUMBER the analysis produces
    metrics/          per-sample metric tables (xlsx), one folder per model
    metrics/selection.json   the layer and word length chosen for each model
    metrics/scaling_distance/    the same unseen-order tables under other feature scaling / distances
    metrics/legacy/   outputs of the old pipeline layout, kept for reference
    figure_data/      the exact numbers plotted in each figure (CSV, committed)
    statistics/       paired tests, genome-level bootstrap intervals (CSV)
    run_manifest.json what produced the numbers (commit, versions, seeds, command line)

figures/              EVERY PLOT  (main/, supplementary/, statistics/, exploratory/)
glmbench/             the package: all of the logic
visualization/        notebooks that demonstrate how it is used
tests/                unit tests, regression gate, end-to-end run
scripts/ Makefile     one-command entry points
```

| Module | What it holds |
|--------|---------------|
| `config.py` | where everything lives on disk (`GLMBENCH_ROOT`), shared defaults and the default seed |
| `models.py` | the registry: representation **and** distance of each model |
| `sequences.py` | genomes to fragments (complete pieces, seeded random subset) and reverse complements |
| `samples.py` | the reference samples from the ICTV VMR + a genome directory |
| `representations/` | k-mer (plain / canonical) profiles, MinHash sketches, pooling, the transformer extractors |
| `neighbours.py` | the nearest-neighbour core every metric is built on (exclusion by index) |
| `clustering.py` | K-medoids in numpy (no scikit-learn-extra) |
| `metrics/` | unseen-order metrics, pairwise metrics, the reference benchmark, the runner |
| `selection.py` | choosing the layer and the word length, always on the reference set |
| `statistics.py` | paired tests, multiple-testing corrections, cluster bootstrap over genomes |
| `provenance.py` | `results/run_manifest.json` |
| `figures/` | every plot |
| `cli.py` | the command line |

---

## Models

A model is a representation together with the distance it is compared under. The registry
in `glmbench/models.py` keeps the two together.

| Model | Representation | Distance | Free parameter |
|-------|----------------|----------|----------------|
| `kmer` | 4-mer frequency profile | standardised, cosine | -- |
| `kmer_canon` | **canonical** 4-mer profile (min of word and reverse complement; 136 entries) | standardised, cosine | -- |
| `kmer_js` | k-mer frequency profile | Jensen-Shannon | word length *k* |
| `minhash` | MinHash sketch of the canonical k-mer set | Jaccard estimate | word length *k* |
| `minhash1k` | the same with sketch size 1024 instead of 128, k = 5..12 | Jaccard estimate | word length *k* |
| `dnabert2` | DNABERT-2 embeddings | standardised, cosine | layer |
| `dnabertS` | DNABERT-S embeddings | standardised, cosine | layer |
| `viralm` | ViraLM embeddings (mean of the two 1 kb windows of a fragment) | standardised, cosine | layer |
| `evo2` | Evo2-7B embeddings | standardised, cosine | layer |

`kmer_canon` is the RC-invariant baseline against which the strand behaviour of the learned
models is read (`profile(x) == profile(RC(x))` is a unit test). MinHash is strand-invariant
by construction too (canonical k-mers). Jensen-Shannon and MinHash cannot be applied to a
transformer embedding; the code refuses rather than guesses. `Model.revision` pins a
HuggingFace revision (default `None` = unpinned; set it, it is recorded in the manifest).

DNABERT-2/S: `--pooling mean` (default) is the mean of the token states over the positions
the attention mask marks as real; `--pooling cls` takes the first token.

---

## The stages

| Stage | What it does | Needs |
|-------|--------------|-------|
| `samples` | VMR + genomes to the 50 reference samples (`--list-classes` prints the qualifying classes, needs only the VMR) | genomes |
| `prep` | genomes to 2 kb fragments (complete pieces only; at most 10, chosen at random with `--seed`), plus a reverse-complement copy | genomes |
| `extract` | fragments to vectors, one folder per model | GPU for the transformers |
| `benchmark` | ARI, P@5, silhouette per layer and reference sample -> `results/metrics/<model>/{ARI,P5,silh}.xlsx` | extracted reference samples |
| `select` | the layer and word length of each model | benchmark / reference samples |
| `metrics` | unseen-order metrics, both protocols; per-genome tables; pairwise tables | extracted vectors |
| `stats` | paired tests between models; genome-level bootstrap intervals | metric tables |
| `figures` | every plot | metric tables |
| `all` | the whole chain (`select` runs twice: word length before `extract`, layers after `benchmark`) | |

Useful across stages: `--models`, `--seed` (default 42), `--dry-run`, `--force`,
`--max-workers` (processes), `--sample-threads` (reference samples held in memory at once
per task; one Evo2 sample is ~2 GB), `--memory-budget`, `--allow-missing`.

**Failure handling.** A reference sample that is missing or unreadable, a virus file that
is absent, an RC file not aligned with its forward file: each is counted and printed, the
affected tables are *not* written, and the stage exits non-zero -- unless `--allow-missing`
is given, in which case the tables are written from the samples that did load and the loss
is stated. `extract` writes `<model dir>/failed.txt` and exits non-zero if any file failed;
`extract_evo2.py` writes `<name>.failed.txt` (record id and reason) and exits 1.

**Seeds.** Every random choice takes an explicit seed derived from `--seed`: fragment choice
(per genome, from a stable hash of seed and genome id), reference draw *i* (`seed + i`),
MinHash hash functions, K-medoids restarts, bootstrap, plot jitter. The seeds, the full
command line, the git commit, Python and package versions and the model identifiers are in
`results/run_manifest.json`, rewritten at the start of every stage.

---

## Metrics

Two different things were both called "CROSS" in earlier versions. They are unrelated and
named differently here:

* **RC proximity** (legacy "CROSS P@k", file prefix `cross_P`, Fig. 5): fragment *x* is
  searched in reference + the RC fragments of **its own genome**; success = a neighbour is
  *any* RC fragment of that genome. **Higher is better.** It does not test whether the
  neighbour is x's *own* reverse complement; a model that merely keeps the fragments of a
  genome together scores high.
* **Inter-order leakage** (Fig. 7, `pairwise_*/cross_*`): how often a fragment of order A
  retrieves a fragment of order B. **Lower is better.**

On the unseen orders, with the reference sample in the space (see `glmbench/metrics/unseen.py`):

| Name | Tables | Success means | Notes |
|------|--------|---------------|-------|
| `precision_by_genome` (legacy "P@k") | `P@<k>_matrix_*` | a neighbour is from the **same genome** | measures how tightly one genome's fragments cluster, **not** grouping by order; ceiling is (n-1)/k when a genome has n fragments (P@10 <= 0.9 with 10 fragments) |
| `precision_by_order` | `Porder@<k>_matrix_*` | a neighbour is from **another genome of the same order**; every fragment of the query's own genome is excluded (leave-genome-out) | the chance level is the share of such fragments in the space (`chance_order_precision`); needs >= 2 genomes |
| `own_rc` hit@k | `ownrc_hit@<k>_matrix_*` | the **fragment's own** reverse complement is among its k nearest neighbours (space = reference + all forward + all RC fragments of the order, x itself excluded) | RC rows are index-aligned with forward rows (asserted) |
| `own_rc` similarity | `ownrc_sim_matrix_*` | mean similarity between x and RC(x): cosine (vector models, in the standardised space), sketch agreement (MinHash), 1 - JSD (Jensen-Shannon) | high for any model that keeps a genome in one clump; read together with hit@k |
| RC proximity | `cross_P@<k>_matrix_*` | see above | legacy |

`results/metrics/<model>/per_genome_<kind>_<virus>.csv.gz` keeps the value of every genome
(`kind` = `genome`, `order`, `cross`, `ownrc`; `layer, sample, genome, metric, value`, for
k = 5 and 10), so intervals that treat the **genome** as the unit are possible:
`statistics.bootstrap_ci_by_genome` (cluster bootstrap) is run by `stats` and written to
`results/statistics/genome_bootstrap_ci.csv`.

Reference benchmark (`benchmark`; per layer and reference sample, cosine, standardised):

* **ARI** -- adjusted Rand index of a K-medoids clustering (k = number of classes, fixed
  random state, numpy implementation in `clustering.py`, `k-medoids++` start, best of 5)
  against the true labels.
* **P@5** -- of a fragment's 5 nearest neighbours (itself excluded **by index**), the share
  with its class.
* **P5_lgo** -- P@5 with every fragment of the query's own genome excluded
  (leave-genome-out; genome ids from `manifest_sample_<i>.tsv`, so only for samples drawn
  by this code). Plain P@5 counts other pieces of the same genome as hits.
* **Silhouette** -- cosine silhouette **of the true labels** (not of the K-medoids
  clustering).

Every table is per reference sample; the 50 samples are shared by all models, which makes
comparisons paired.

---

## Methods notes (read before interpreting a number)

* **What the Wilcoxon p-values mean.** The 50 reference samples are 50 overlapping draws of
  known viruses. A paired test over them asks whether A beats B when the *reference draw*
  changes; it does not test robustness to the choice of viruses (which unseen orders, which
  genomes). With n = 50 the exact two-sided p cannot go below 2/2^50 = 1.8e-15 and **hits
  that floor** whenever one model wins on every sample (36% of the distinct comparisons
  under the primary rule): "won every draw", not a probability. For uncertainty over
  genomes use the bootstrap intervals.
* **Number of comparisons.** `p_holm_all` / `p_bh_all` are over *distinct* comparisons.
  With the 7 models in the committed tables there are 21 pairs per panel; under the primary
  `reference` rule Fig. 1 is excluded (it *is* the benchmark that chooses the layer) and
  panels that read the same table are counted once: 4 (Fig. 4) + 2 (Fig. 5) + 20 (Fig. 6)
  - 2 shared (P@5 of the two orders) = 24 tables x 21 pairs = **504** distinct comparisons
  (5 not significant after FDR). With the original 5 models it was 24 x 10 = 240. The
  rules that include Fig. 1 have 534 (Fig. 1 exists only for the 5 models that have a
  benchmark).
* **Transductive scaling.** For vector models `StandardScaler` is fitted on each query
  set -- reference sample + the fragments being scored -- not once on the reference. A
  fragment's neighbours therefore depend (slightly) on which other fragments are in the
  space, and the protocols (`genome`: reference + one genome; `order`: reference + the
  whole order) scale differently. A `reference`-fitted scaler would be cleaner and has not
  been run.
* **Layer selection.** Layers are chosen on the reference benchmark (P@5) and then held
  fixed. Positions, never numbers: ViraLM's benchmark rows are 0..11 while its files and
  metric matrices are layers 1..12, so position *p* is layer *p* + 1
  (`selection.layer_label`).
* **Selection optimism.** The `oracle` and `split` rules score the same samples and differ
  only in where the layer was chosen, so their gap is the optimism of choosing a layer on
  the data it is scored on: median 0, at most +0.011. The chosen layer **differs in 5.5% of
  model x panel scores** (10 of 181; the PNG title states this number).
* **Fragments.** At most 10 complete 2 kb non-overlapping pieces per genome, chosen at
  random (seeded, per genome); the incomplete tail is dropped.
* **Reference samples.** Classes with *more than 10 species* in the VMR, 10 random genomes
  (exemplar isolates) per class, up to 10 random non-overlapping 2 kb fragments per genome,
  50 independent draws with seeds `seed + i`. On `VMR_MSL40.v2.20251013` this gives
  **39** qualifying classes (`python run_pipeline.py samples --list-classes`). The
  two unseen orders (Egovirales, Mriyavirales) do not appear in this VMR file.

---

## Evo2

`extract_evo2.py` (run in `evo2_env`) writes `{block: {label: [vectors]}}` for the 32
`blocks.<i>.mlp.l3` outputs. ViraLM is extracted with `envs_for_glms/ViraLM/viralm_hook.py`
(layers 1..12, one pickle per layer; the benchmark tables index ViraLM layers by position
0..11, so position *p* is layer *p* + 1 and `selection.json` stores the position).

---

## Figure numbering

Layers are numbered as in the extracted files everywhere (DNABERT-2/S: 0 = embeddings, 1..12 blocks; ViraLM: blocks
1..12; Evo2: blocks 0..31); `selection.json` alone stores a benchmark *position*.

| Figure | Content |
|--------|---------|
| Fig. 1 | reference benchmark, best layer per model (ARI, P@5; stars: vs the 4-mer profile) |
| Fig. 2 | layer-wise ARI and P@5 (open markers: worse than the model's best layer) |
| Fig. S1 | word-length sweep of the k-mer, Jensen-Shannon and MinHash baselines |
| Fig. 4 | unseen orders, P@5 (genome level), forward and RC |
| Fig. 5 | RC proximity P@5 (legacy "CROSS P@5") |
| Fig. 6 | P@k against k (stars: leader vs runner-up) |
| Fig. 7 | compactness vs inter-order leakage, per layer |
| Fig. 8 | feature scaling and distance (P@5, unseen orders) |
| Fig. 4b, 5b, 5c | order-level P@5; own-RC hit@5; own-RC similarity (new; appear after `metrics`) |
| `figures/statistics/` | significance plots |
| `figures/supplementary/` | per layer plots; `figS1_kmer_sweep` (k-mer length) |

## Tests

```bash
python -m pytest tests -q
```

`test_primitives` (distances, k-mer/MinHash, P@k prefixes, layer matching, corrections),
`test_sequences`, `test_pooling`, `test_unseen` (synthetic worlds with known answers),
`test_reference` (K-medoids, ARI/P@5/silhouette, agreement with the neighbours module),
`test_samples`, `test_bootstrap`, `test_extraction`, `test_runner`, `test_regression` (the
committed figure CSVs of the five original models must not move) and `test_end_to_end`.

`visualization/main_figures.ipynb` walks through the figures; `visualization/exploratory/`
holds the original exploratory notebooks (not needed to reproduce anything; they need
`requirements-dev.txt` and read legacy tables).
