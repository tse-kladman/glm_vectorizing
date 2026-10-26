# Viral genome embeddings: benchmarking genomic language models

Code and data accompanying the article *Assessing the Capacity of Genomic Language Models
to Generate Informative Embeddings of Viral Genome Fragments* (submitted to MDPI Viruses).
DNABERT-2, DNABERT-S, ViraLM and Evo2 are compared with three classical baselines (4-mer
profile, k-mer + Jensen-Shannon, MinHash) on viral genomes: how
well each representation organises a reference set of known viruses, how well it
generalises to virus orders the models have never seen, and whether it reads the two
strands of a sequence as the same thing.

> **Which settings the reported numbers use.** The committed tables are the numbers
> reported in the article. Those of DNABERT-2, DNABERT-S, ViraLM and the 4-mer profile
> were computed with mean pooling over all 512 positions for DNABERT (no attention
> mask) and with the two 1,000-nt ViraLM windows of a fragment as separate points
> (`store.VIRALM_WINDOWS = "windows"`); the article states this in its Limitations.
> The code defaults (masked pooling, one ViraLM vector per fragment) are the corrected
> settings; re-running *Full reproduction* with them gives the corrected numbers.

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
`samples -> prep -> sweep -> select -> extract -> benchmark -> metrics (main and
scaling/distance) -> stats -> figures` through `glmbench.cli.main` for the three CPU
models (`kmer`, `kmer_js`, `minhash`).
It asserts that every output exists, that the numbers are sane, and that nothing was
written outside the root. The transformer extractors are not part of it.

## Full reproduction (GPU)

Needs the genomes (a FASTA directory whose record ids start with the GenBank accessions of
the ICTV VMR isolates, plus the two unseen orders in `data/unk_viruses/<order>.fa`), a GPU
for the four language models, and the Evo2 / ViraLM conda environments (see *ViraLM* and
*Evo2* below).

```bash
python run_pipeline.py samples --fasta-dir /path/to/genomes --seed 42        # 50 reference samples
python run_pipeline.py prep --seed 42 --force                                # fragments + RC
python run_pipeline.py sweep --max-workers 4                                 # k-mer length sweep (Fig. S1a)
python run_pipeline.py select --models kmer_js minhash                       # word length (needs samples)
python run_pipeline.py extract --models kmer kmer_js minhash
python run_pipeline.py extract --models dnabert2 dnabertS --pooling mean --force
python run_pipeline.py extract --models viralm                               # needs the ViraLM model files
python run_pipeline.py extract --models evo2
python run_pipeline.py benchmark --max-workers 2 --sample-threads 2          # ARI / P@5 / silhouette
python run_pipeline.py select                                                # layers, on the benchmark
python run_pipeline.py metrics --max-workers 2 --sample-threads 2
python run_pipeline.py metrics --scaling-distance --layers selected          # Fig. 8
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
    figure_data/      the exact numbers plotted in each figure (CSV, committed)
    statistics/       paired tests between models (CSV)
    run_manifest.json what produced the numbers (commit, versions, seeds, command line)

figures/              EVERY PLOT  (main/, supplementary/, statistics/)
glmbench/             the package: all of the logic
tests/                unit tests, regression gate, end-to-end run
run_pipeline.py       the single command-line entry point
extract_evo2.py       Evo2 extractor, run inside the evo2_env conda environment
envs_for_glms/ViraLM/ ViraLM hook, run inside the viralm conda environment
                      (the ViraLM model files go to envs_for_glms/ViraLM/model/, not shipped)
requirements.txt      Python dependencies
Makefile              one-command entry points
```

| Module | What it holds |
|--------|---------------|
| `config.py` | where everything lives on disk (`GLMBENCH_ROOT`), shared defaults and the default seed |
| `models.py` | the registry: representation **and** distance of each model |
| `sequences.py` | genomes to fragments (complete pieces, seeded random subset) and reverse complements |
| `samples.py` | the reference samples from the ICTV VMR + a genome directory |
| `representations/` | k-mer profiles, MinHash sketches, pooling, the transformer extractors |
| `neighbours.py` | the nearest-neighbour core every metric is built on (exclusion by index) |
| `clustering.py` | K-medoids in numpy (no scikit-learn-extra) |
| `metrics/` | unseen-order metrics, pairwise metrics, the reference benchmark, the runner |
| `selection.py` | choosing the layer and the word length, always on the reference set |
| `statistics.py` | paired tests, multiple-testing corrections |
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
| `kmer_js` | k-mer frequency profile | Jensen-Shannon | word length *k* |
| `minhash` | MinHash sketch of the canonical k-mer set | Jaccard estimate | word length *k* |
| `dnabert2` | DNABERT-2 embeddings | standardised, cosine | layer |
| `dnabertS` | DNABERT-S embeddings | standardised, cosine | layer |
| `viralm` | ViraLM embeddings (mean of the two 1 kb windows of a fragment) | standardised, cosine | layer |
| `evo2` | Evo2-7B embeddings | standardised, cosine | layer |

MinHash is strand-invariant by construction (canonical k-mers: a fragment and its reverse
complement have the same sketch). Jensen-Shannon and MinHash cannot be applied to a
transformer embedding; the code refuses rather than guesses.

Model versions. DNABERT-2/S are loaded with `trust_remote_code=True`, so their own code comes
from the HuggingFace hub together with the weights; `Model.revision` pins both to one commit
and is recorded in `results/run_manifest.json`.

| Model | Source | Version |
|-------|--------|---------|
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` (HuggingFace) | commit `7bce263b15377fc15361f52cfab88f8b586abda0` |
| DNABERT-S | `zhihan1996/DNABERT-S` (HuggingFace) | commit `00e47f96cdea35e4b6f5df89e5419cbe47d490c6` |
| Evo2-7B | `evo2` package (https://github.com/ArcInstitute/evo2), weights `arcinstitute/evo2_7b` | `evo2` 0.3.0 |

DNABERT-2/S: `--pooling mean` (default) is the mean of the token states over the positions
the attention mask marks as real; `--pooling cls` takes the first token.

---

## The stages

| Stage | What it does | Needs |
|-------|--------------|-------|
| `samples` | VMR + genomes to the 50 reference samples (`--list-classes` prints the qualifying classes, needs only the VMR) | genomes |
| `prep` | genomes to 2 kb fragments (complete pieces only; at most 10, chosen at random with `--seed`), plus a reverse-complement copy | genomes |
| `sweep` | ARI, P@5, silhouette of the k-mer profile (standardised, cosine) for k = 1..10 on the reference samples -> `results/metrics/kmer_sweep/kmer_sweep_results.xlsx` (Fig. S1a; a profile has 4^k columns, so k = 10 needs several GB per worker) | reference samples |
| `extract` | fragments to vectors, one folder per model | GPU for the transformers |
| `benchmark` | ARI, P@5, silhouette per layer and reference sample -> `results/metrics/<model>/{ARI,P5,silh}.xlsx` | extracted reference samples |
| `select` | the layer and word length of each model (the word-length sweep -> `results/metrics/k_sweep/`, on `--sweep-samples` reference samples, default 10) | benchmark / reference samples |
| `metrics` | unseen-order metrics (P@k, RC proximity) and pairwise tables. With `--scaling-distance`: the P@k and RC proximity of the standardised-cosine models under raw features and/or the Euclidean distance -> `results/metrics/scaling_distance/<variant>/` (Fig. 8) | extracted vectors |
| `stats` | paired tests between models | metric tables |
| `figures` | every plot | metric tables |
| `all` | the whole chain (`select` runs twice: word length before `extract`, layers after `benchmark`; `metrics` runs twice: the main tables, then `--scaling-distance --layers selected`) | |

Useful across stages: `--models`, `--seed` (default 42), `--dry-run`, `--force`,
`--max-workers` (processes), `--sample-threads` (reference samples held in memory at once
per task; one Evo2 sample is ~2 GB), `--memory-budget`, `--allow-missing`.

**Failure handling.** A reference sample that is missing or unreadable, a virus file that
is absent, a representation the metric cannot take: each is counted and printed, the
affected tables are *not* written, and the stage exits non-zero -- unless `--allow-missing`
is given, in which case the tables are written from the samples that did load and the loss
is stated. `extract` writes `<model dir>/failed.txt` and exits non-zero if any file failed;
`extract_evo2.py` writes `<name>.failed.txt` (record id and reason) and exits 1.

**Seeds.** Every random choice takes an explicit seed derived from `--seed`: fragment choice
(per genome, from a stable hash of seed and genome id), reference draw *i* (`seed + i`),
MinHash hash functions, K-medoids restarts, plot jitter. The seeds, the full
command line, the git commit, Python and package versions and the model identifiers are in
`results/run_manifest.json`, rewritten at the start of every stage.

---

## Metrics

Two different set-proximity measures are stored with the prefix `cross`. They are unrelated and
named differently here:

* **RC proximity** ("CROSS P@k", file prefix `cross_P`, Fig. 5): fragment *x* is
  searched in reference + the RC fragments of **its own genome**; success = a neighbour is
  *any* RC fragment of that genome. **Higher is better.** It does not test whether the
  neighbour is x's *own* reverse complement; a model that merely keeps the fragments of a
  genome together scores high.
* **Inter-order leakage** (Fig. 7, `pairwise_*/cross_*`): how often a fragment of order A
  retrieves a fragment of order B. **Lower is better.**

On the unseen orders, with the reference sample in the space (see `glmbench/metrics/unseen.py`):

| Name | Tables | Success means | Notes |
|------|--------|---------------|-------|
| `precision_by_genome` ("P@k") | `P@<k>_matrix_*` | a neighbour is from the **same genome** | measures how tightly one genome's fragments cluster, **not** grouping by order; ceiling is (n-1)/k when a genome has n fragments (P@10 <= 0.9 with 10 fragments) |
| `precision_rc_proximity` (RC proximity) | `cross_P@<k>_matrix_*` | see above | forward fragments only (the RC file of an order is scored by `precision_by_genome` as a virus of its own) |

Reference benchmark (`benchmark`; per layer and reference sample, cosine, standardised):

* **ARI** -- adjusted Rand index of a K-medoids clustering (k = number of classes, fixed
  random state, numpy implementation in `clustering.py`, `k-medoids++` start, best of 5)
  against the true labels.
* **P@5** -- of a fragment's 5 nearest neighbours (itself excluded **by index**), the share
  with its class. Other pieces of the same genome count as hits.
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
  under the primary rule): "won every draw", not a probability.
* **Number of comparisons.** `p_holm_all` / `p_bh_all` are over *distinct* comparisons.
  With the 7 models in the committed tables there are 21 pairs per panel; under the primary
  `reference` rule Fig. 1 is excluded (it *is* the benchmark that chooses the layer) and
  panels that read the same table are counted once: 4 (Fig. 4) + 2 (Fig. 5) + 20 (Fig. 6)
  - 2 shared (P@5 of the two orders) = 24 tables x 21 pairs = **504** distinct comparisons
  (5 not significant after FDR). The rules that include Fig. 1 have 567.
* **Transductive scaling.** For vector models `StandardScaler` is fitted on each query
  set -- reference sample + the fragments being scored -- not once on the reference. A
  fragment's neighbours therefore depend (slightly) on which other fragments are in the
  space. A `reference`-fitted scaler would be cleaner and has not been run.
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
  **36** qualifying classes (`python run_pipeline.py samples --list-classes`), not the 39
  stated in the article; the cause (another VMR release, another threshold) is unknown. The
  two unseen orders (Egovirales, Mriyavirales) do not appear in this VMR file.

---

## ViraLM

ViraLM (https://github.com/ChengPENG-wolf/ViraLM) has its own licence, which you must check
before use. This repository contains only the *hook* `envs_for_glms/ViraLM/viralm_hook.py`,
adapted from ViraLM's prediction script, which dumps hidden states instead of the virus
score. To run it:

1. create the `viralm` conda environment following ViraLM's instructions;
2. download the ViraLM model files into `envs_for_glms/ViraLM/model/` (the hook's
   `--model_pth` default; this folder is git-ignored).

`extract` stops immediately, with instructions, if the model folder is missing. The
pipeline runs `conda run -n viralm python viralm_hook.py ...` with the hook's folder as the
working directory. The hook:

* accepts `--input_pth <FASTA>` `--output_pth <directory>` `--batch_size 1`
  `--layers_to_monitor L1 L2 ... L12` (encoder layers numbered **1..12**);
* cuts every record into ViraLM's 1,000-nt windows (a tail of at least 500 nt is kept;
  windows with a letter other than A/C/G/T are skipped);
* writes, for every requested layer *L*, `<output_pth>/<FASTA stem>_layer_<L>.pkl`
  (`sample_7.fa` -> `sample_7_layer_1.pkl` ... `sample_7_layer_12.pkl`), a pickled `dict`
  `{key: 1-D numpy array}` with **one entry per window**, in FASTA order, where
  `key = "<FASTA record id>_<start>_<end>"` and the vector is the mean of the layer's token
  states. The record id is `<label>_<index>`; the label is recovered by dropping the last
  three `_`-separated fields (`store._viralm_groups`). The windows of one fragment (same
  record id) are **averaged into one vector**, so a 2 kb fragment is one point of the space
  as for every other model (`store.VIRALM_WINDOWS = "windows"` keeps one point per window).
  The order of the entries is the order of the FASTA records.

**Position matching rule:** the benchmark tables (`P5.xlsx` ...) have 12 rows by position
0..11, the metric matrices have row labels 1..12, and position *p* corresponds to layer
*p* + 1; `selection.json` stores the position (ViraLM `layer: 2` is file `layer_3`).

## Evo2

`extract_evo2.py` runs in the `evo2_env` conda environment (install the `evo2` package
following https://github.com/ArcInstitute/evo2; the `evo2_7b` weights are downloaded on first
use). It writes `{block: {label: [vectors]}}` for the 32 `blocks.<i>.mlp.l3` outputs, each
the mean over the token positions. Check the licences of Evo 2, ViraLM, DNABERT-2 and
DNABERT-S yourself; none is asserted here.

---

## Figure numbering

The repository numbers its figures in the order the pipeline draws them. Layers are
numbered as in the extracted files everywhere (DNABERT-2/S: 0 = embeddings, 1..12 blocks;
ViraLM: blocks 1..12; Evo2: blocks 0..31); `selection.json` alone stores a benchmark
*position*.

| Repository | Content | Article |
|------------|---------|---------|
| `main/fig1_reference_benchmark` | reference benchmark, best layer per model (ARI, P@5) | Figure 2 |
| `main/fig2_layerwise` | layer-wise ARI and P@5 | Figure 3 |
| `main/fig4_newvirus_p5` | held-out orders, P@5 (genome level), forward and RC | Figure 4 |
| `main/fig5_cross_p5` | RC proximity P@5 | Figure 5 |
| `main/fig6_pk_vs_k` | P@k against k | Figure 6 |
| `main/fig7_pk_vs_cross` | compactness vs inter-order leakage, per layer | Figure 7 |
| `main/fig8_scaling_distance` | feature scaling and distance (P@5, held-out orders) | Figure 8 |
| `supplementary/figS1_kmer_sweep` | k-mer length sweep of the 4-mer, Jensen-Shannon and MinHash baselines | Figure S1 |
| `statistics/box_fig4` | P@5 on the held-out orders per reference sample, with paired tests | Figure S2 |
| `figure_data/supplementary/evo2_early_layer.csv` | Evo2-7B at layer 3 against layer 26 | Table S1 |
| other files in `supplementary/`, `statistics/` | per-layer plots, paired differences, heat maps of p-values | supplement |

Figure 1 (the scheme of the embedding extraction) is not drawn by the pipeline.

## Tests

```bash
python -m pytest tests -q
```

`test_primitives` (distances, k-mer/MinHash, P@k prefixes, layer matching, corrections),
`test_sequences`, `test_pooling`, `test_unseen` (synthetic worlds with known answers),
`test_reference` (K-medoids, ARI/P@5/silhouette, the k-mer sweep, agreement with the
neighbours module), `test_samples`, `test_extraction`, `test_runner`
(including the scaling/distance variants), `test_significance`, `test_store`,
`test_regression` (the committed figure CSVs must not move) and `test_end_to_end`.
