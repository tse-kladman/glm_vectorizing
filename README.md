# Viral genome embeddings: benchmarking genomic language models

Code and data accompanying the article. Several genomic language models (DNABERT-2,
DNABERT-S, ViraLM, Evo2) and three classical baselines are compared on viral genomes:
how well each representation organises a reference set of known viruses, how well it
generalises to virus orders the models have never seen, and whether it reads the two
strands of a sequence as the same thing.

```bash
pip install -r requirements.txt
python run_pipeline.py --help          # the stages
python run_pipeline.py figures         # redraw every figure from the committed numbers
```

---

## Repository layout

Three top-level folders, and nothing crosses the line between them:

```
data/                 INPUTS ONLY -- what came from outside
    ictv_data/        ICTV Virus Metadata Resource (VMR MSL40)
    unk_viruses/      genomes of the unseen orders          not shipped, too large
    preproc_unk_viruses/  the same, cut into fragments      not shipped
    random_samples/   the 50 reference samples              not shipped
    latent_dim/       extracted representations             not shipped

results/              EVERY NUMBER the analysis produces
    metrics/          per-sample metric tables (xlsx), one folder per model
    metrics/selection.json   the layer and word length chosen for each model
    figure_data/      the exact numbers plotted in each figure (CSV, committed)
    statistics/       paired tests between models (CSV, committed)

figures/              EVERY PLOT
    main/             Fig. 1-7 of the article
    supplementary/    every layer of every metric, grouped by metric
    statistics/       the significance plots
    exploratory/      output of the exploratory notebooks

glmbench/             the package: all of the logic
visualization/        notebooks that demonstrate how it is used
tests/                unit tests and the regression gate
```

The package itself is laid out in the order the analysis runs, so it can be read top
to bottom:

| Module | What it holds |
|--------|---------------|
| `config.py` | where everything lives on disk, and the shared defaults |
| `models.py` | the registry: the representation **and** distance of each model |
| `sequences.py` | genomes to fragments and reverse complements |
| `representations/` | k-mer profiles, MinHash sketches, the transformer extractors |
| `neighbours.py` | the nearest-neighbour core every metric is built on |
| `metrics/` | reference-set, unseen-virus and pairwise metrics, and their runner |
| `selection.py` | choosing the layer and the word length, always on the reference set |
| `statistics.py` | paired tests and multiple-testing corrections |
| `figures/` | every plot |
| `cli.py` | the command line |

---

## Models

A model is not just an encoder: it is a representation together with the distance that
representation is compared under, because a distance is only defined on the right kind
of vector. The registry in `glmbench/models.py` keeps the two together.

| Model | Representation | Distance | Free parameter |
|-------|----------------|----------|----------------|
| `kmer` | 4-mer frequency profile | standardised, cosine | — |
| `kmer_js` | k-mer frequency profile | Jensen-Shannon | word length *k* |
| `minhash` | MinHash sketch of the canonical k-mer set | Jaccard estimate | word length *k* |
| `dnabert2` | DNABERT-2 embeddings | standardised, cosine | layer |
| `dnabertS` | DNABERT-S embeddings | standardised, cosine | layer |
| `viralm` | ViraLM embeddings | standardised, cosine | layer |
| `evo2` | Evo2-7B embeddings | standardised, cosine | layer |

Jensen-Shannon is defined between probability distributions and MinHash compares sets,
so neither can be applied to a transformer embedding — mapping one onto the simplex
would take an arbitrary transform. The code refuses rather than guesses.

---

## How the free parameters are chosen

Every model has one free parameter — which layer to report, or how long the word
should be. Tuning either on the unseen orders would mean reporting a number that was
picked for being high, so **both are chosen on the reference viruses** and then held
fixed:

```bash
python run_pipeline.py select
```

The layer comes from the model's reference benchmark (`P5.xlsx`), the word length from
a sweep over the reference samples, and both are scored by the same quantity: P@5
inside a reference sample. The choice is written to `results/metrics/selection.json`,
and every unseen-order metric is then reported at that choice.

One subtlety the code guards against: layers are matched to the metric tables **by
position**, never by number. ViraLM numbers its layers 0..11 in the benchmark table and
1..12 in the metric matrices, so a lookup by number would silently shift it by one.

---

## The stages

| Stage | What it does | Needs |
|-------|--------------|-------|
| `prep` | genomes to 2 kb fragments, plus a reverse-complement copy | genomes |
| `extract` | fragments to vectors, one folder per model | GPU for the transformers |
| `select` | the layer and word length of each model | reference samples |
| `metrics` | P@k and CROSS P@k on the unseen orders | extracted vectors |
| `stats` | paired tests between models | metric tables |
| `figures` | every plot | metric tables |
| `all` | the whole chain | |

```bash
python run_pipeline.py prep
python run_pipeline.py extract --models kmer dnabert2
python run_pipeline.py select
python run_pipeline.py metrics --layers selected      # only the chosen layer
python run_pipeline.py figures --which main
```

Useful across all stages: `--models` to restrict the work, `--dry-run` to see it
without doing it, `--force` to redo what already exists, `--max-workers` and
`--memory-budget` to fit the machine.

**Parallelism.** Tasks (one model, one virus file) run in processes; the 50 reference
samples inside a task run in threads. The limit is memory rather than cores — every
process imports numpy/scipy and keeps one model's vectors resident — so `--max-workers`
defaults to 8 rather than to the core count, and the blockwise distances stay inside
`--memory-budget`. `--layers selected` scores only the chosen layer, which is 32 times
less work for Evo2.

Evo2 and ViraLM cannot be installed next to the rest; they run in their own conda
environments (`evo2_env`, `viralm`) through `conda run`, and write the same pickles
back.

---

## Metrics

* **P@k** — of a fragment's k nearest neighbours, the fraction that share its class.
  Higher is better.
* **CROSS P@k** — the same, but counting neighbours from the reverse-complement copy,
  which the forward fragment is searched against but is not part of. High means the
  model reads both strands alike.
* **ARI** — adjusted Rand index of a K-medoids clustering against the true labels.
* **Silhouette** — cosine silhouette of that clustering.

Every metric is a mean over the 50 reference samples, and every model is scored on the
same samples, which is what makes the comparisons paired.

---

## Reproducing the figures

The metric tables and the figure numbers are both committed, so the figures redraw
from a clean checkout with no GPU, no genomes and no extracted vectors:

```bash
python run_pipeline.py figures              # rebuild the numbers, then draw
python run_pipeline.py figures --plot-only  # draw from the committed numbers
```

`visualization/main_figures.ipynb` walks through the same thing step by step, and
`visualization/exploratory/` keeps the original working notebooks.

---

## Tests

```bash
python -m pytest tests/
```

`tests/test_primitives.py` checks the claims the analysis rests on: that Jensen-Shannon
agrees with scipy, that a MinHash sketch is identical for a fragment and its reverse
complement and estimates the Jaccard index, that P@1..P@10 read off one neighbour list
equals ten separate searches, and that layers are matched by position.

`tests/test_regression.py` rebuilds every committed figure CSV from the committed
metric tables and compares it against the committed copy, so a refactor that changes a
published number fails the suite.
