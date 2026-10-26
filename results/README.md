# `results/` -- every number the analysis produces

Nothing here is edited by hand. Plots live one level up, in `figures/`. See the main
README for what each metric measures.

```
metrics/                 per-sample metric tables, one folder per model
metrics/selection.json   the layer and word length chosen for each model
metrics/scaling_distance/     unseen-order tables under other feature scaling / distances (see below)
figure_data/             the exact numbers plotted in each figure (CSV)
statistics/              paired tests between models (CSV)
run_manifest.json        commit, versions, seeds, command line, model identifiers per stage
```

## `metrics/` -- the tables the pipeline writes

Every metric is stored the same way: rows are layers (`-` for a model without layers),
columns are the 50 reference samples (`sample_<i>`).

| File | What it holds |
|------|---------------|
| `<model>/{ARI,P5,silh}.xlsx` | reference benchmark, per layer (rows by position, columns 0..49). `silh` is the silhouette of the **true labels** |
| `<model>/P@<k>_matrix_<virus>.xlsx` | **genome-level** P@k on an unseen order: success = neighbour from the same genome (`precision_by_genome`) |
| `<model>/cross_P@<k>_matrix_<virus>.xlsx` | "CROSS P@k" = **RC proximity**: neighbour is any RC fragment of the same genome |
| `<model>/pairwise_<A>_vs_<B>/` | one unseen order against the other (**inter-order leakage**, Fig. 7) |
| `k_sweep/<model>_k_sweep.xlsx` | the word-length sweep that chose `k` (P@5, `select` stage) |
| `kmer_sweep/kmer_sweep_results.xlsx` | ARI / P@5 / silhouette of the k-mer profile (cosine) per k, sheets `summary` and `raw` (`sweep` stage, Fig. S1a) |
| `selection.json` | the chosen layer (a benchmark *position*) and word length of each model |
| `summary_unseen.xlsx` | written by `metrics`: best layer and mean per model, virus, metric and k (a quick look, not read by any figure) |

Models present on disk: `kmer`, `kmer_js`, `minhash`, `dnabert2`, `dnabertS`, `viralm`,
`evo2` (benchmark, P@k, RC proximity, pairwise).
`metrics/scaling_distance/{raw_cosine,standardized_euclidean,raw_euclidean}/<model>/` hold
the unseen-order tables of the 4-mer profile and the four language models with raw or
standardized features and the cosine or Euclidean distance (`metrics --scaling-distance`);
standardized features with the cosine distance are the main tables `metrics/<model>/`.

## `figure_data/` -- the numbers behind each figure

Committed, and read by the drawing step alone. Repository figure numbers; the article's
numbering is in the main README.

| Figure | What it shows | File | Columns |
|--------|---------------|------|---------|
| Fig. 1 | best layer of each model: ARI, P@5 | `fig1_reference_benchmark.csv` | `metric, model, best_layer, mean, std, p_vs_kmer_bh` |
| Fig. 2 | ARI and P@5 across all layers | `fig2_layerwise.csv` | `metric, model, layer, mean, std, p_vs_best_bh` |
| Fig. S1 | k-mer length sweep of the 4-mer, Jensen-Shannon and MinHash baselines | `supplementary/figS1_kmer_sweep.csv` | `method, metric, k, mean, std, n, best, p_vs_best_bh` |
| Fig. 4 | unseen orders, **genome-level** P@5, forward and RC | `fig4_newvirus_p5.csv` | `group, model, layer, mean, std, p_vs_kmer_bh` |
| Fig. 5 | **RC proximity** P@5 | `fig5_cross_p5.csv` | `group, model, layer, mean, std, p_vs_kmer_bh` |
| Fig. 6 | P@k against k | `fig6_pk_vs_k.csv` | `virus, model, k, layer, mean, leader_vs_second_bh` |
| Fig. 7 | per layer: own-genome density against **inter-order leakage** (`cross_p_at_10`) | `fig7_pk_vs_cross.csv` | `model, layer, p_at_10, cross_p_at_10` |
| Fig. 8 | P@5 on the unseen orders under four scaling/distance variants | `supplementary/scaling_distance.csv` | `group, model, variant, label, layer, mean, std, p_vs_main_bh` |
| supplementary | per-layer tables; scaling and distance; early Evo2 layer | `supplementary/layerwise_reference.csv`, `layerwise_newvirus_p5.csv`, `scaling_distance.csv`, `evo2_early_layer.csv` | |

`p_*_bh` columns are Benjamini-Hochberg adjusted p-values of the two-sided paired Wilcoxon
test over the reference samples (`glmbench/figures/significance.py`): Fig. 4-6 take them
from `statistics/pairwise_model_tests.csv` (selection `reference`, study-wide family), the
other figures form one family per figure. `layer` / `best_layer` are layer *labels* as in
the extracted files -- ViraLM's blocks are 1..12, so its benchmark position 2 is shown as
layer 3 in every figure.

Fig. 4, 5 and 6 report each model at the layer chosen by its **reference benchmark**, not at
the layer that scores best on the unseen order; the `layer` column says which one. Fig. 1 and
2 are the benchmark itself, so they show its own best layer. Fig. 7 shows every layer.

## `statistics/` -- the tests

| File | What it holds |
|------|---------------|
| `pairwise_model_tests.csv` | one row per pair of models, per panel, per selection rule |
| `panel_samples.csv` | the per-sample values those tests are computed from |

The tests are paired over the 50 reference samples: Wilcoxon signed-rank (primary,
two-sided), a paired t-test, `rank_biserial_r` as effect size. **They test robustness to the
reference draw only**, not to the choice of viruses; the exact Wilcoxon p-value cannot go
below 2/2^50 = 1.8e-15 and sits exactly on that floor whenever a model wins every sample.

Corrections: `p_holm_panel` within a panel; `p_holm_all` and `p_bh_all` over the distinct
comparisons. Distinct = (source table, model pair): panels reading the same table count once.
With 7 models (21 pairs): **504** distinct comparisons under `reference` / `reference_split`
(Fig. 1 excluded; 24 distinct tables x 21), **567** under the rules that include Fig. 1
(3 more tables x 21).
`signif` follows `p_bh_all`: `***` < 0.001, `**` < 0.01, `*` < 0.05, `ns`. Not significant after
FDR: 5 (`reference`), 9, 7, 4, 5 (`reference_split`, `split`, `in_sample`, `oracle`).

The `selection` column records how each model's layer was chosen:

| `selection` | Layer chosen on | Reported on | Role |
|-------------|-----------------|-------------|------|
| `reference` | the model's reference benchmark, all 50 | all 50 | **primary** for the unseen-order panels |
| `reference_split` | the benchmark, even samples | odd samples | choice and report share no sample |
| `split` | even samples of the same table | odd samples | **primary** for Fig. 1 (which *is* the benchmark) |
| `in_sample` | all 50 of the same table | all 50 | reproduces the published figures |
| `oracle` | odd samples of the same table | odd samples | diagnostic: the gap to `split` is the selection optimism |

Measured on these tables, the optimism is negligible (median 0, at most +0.011) and the layer
chosen by `oracle` differs from the one chosen by `split` in **5.5%** of model x panel scores
(10 of 181).

## Conventions

* `mean` / `std` are taken across the reference samples.
* Model order, display names and colours come from `glmbench/models.py` (Okabe-Ito).
