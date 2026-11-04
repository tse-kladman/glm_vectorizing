# `results/` -- every number the analysis produces

Nothing here is edited by hand. Plots live one level up, in `figures/`. See the main
README for what each metric measures.

```
metrics/                 per-sample metric tables, one folder per model
metrics/selection.json   the layer and word length chosen for each model
metrics/scaling_distance/     unseen-order tables under other feature scaling / distances (see below)
metrics/legacy/          legacy outputs kept for reference (see below)
figure_data/             the exact numbers plotted in each figure (CSV)
statistics/              paired tests, genome-level bootstrap intervals (CSV)
run_manifest.json        commit, versions, seeds, command line, model identifiers per stage
```

## `metrics/` -- the tables the pipeline writes

Every metric is stored the same way: rows are layers (`-` for a model without layers),
columns are the 50 reference samples (`sample_<i>`).

| File | What it holds |
|------|---------------|
| `<model>/{ARI,P5,silh}.xlsx` | reference benchmark, per layer (rows by position, columns 0..49). `silh` is the silhouette of the **true labels** |
| `<model>/P@<k>_matrix_<virus>.xlsx` | **genome-level** P@k on an unseen order: success = neighbour from the same genome (`precision_by_genome`) |
| `<model>/Porder@<k>_matrix_<virus>.xlsx` | **order-level** P@k, leave-genome-out (`precision_by_order`) |
| `<model>/cross_P@<k>_matrix_<virus>.xlsx` | legacy "CROSS P@k" = **RC proximity**: neighbour is any RC fragment of the same genome |
| `<model>/ownrc_hit@<k>_matrix_<virus>.xlsx` | own-RC hit@k: the fragment's own reverse complement is among its k nearest -- new |
| `<model>/ownrc_sim_matrix_<virus>.xlsx` | mean similarity between a fragment and its RC (cosine / sketch agreement / 1 - JSD) -- new |
| `<model>/per_genome_<kind>_<virus>.csv.gz` | the same per genome (k = 5, 10): `layer, sample, genome, metric, value`; `kind` = `genome`, `order`, `cross`, `ownrc` -- new |
| `<model>/pairwise_<A>_vs_<B>/` | one unseen order against the other (**inter-order leakage**, Fig. 7) |
| `k_sweep/<model>_k_sweep.xlsx` | the word-length sweep that chose `k` |
| `kmer_sweep/` | the k-mer (cosine) word-length sweep behind Fig. S1 |
| `selection.json` | the chosen layer (a benchmark *position*) and word length of each model |

Models present on disk today: `kmer`, `dnabert2`, `dnabertS`, `viralm`, `evo2` (full: benchmark,
P@k, RC proximity, pairwise) and `kmer_js`, `minhash` (P@k and RC proximity only; no benchmark,
no pairwise). Absent: `kmer_canon`.

### Legacy material (moved out of `data/metrics/`, nothing numeric deleted)

`data/` is inputs only, so the old `data/metrics/` was relocated:

| Old location | New location | Note |
|--------------|--------------|------|
| `data/metrics/P1..P3/<model>/` | `metrics/scaling_distance/{raw_cosine,standardized_euclidean,raw_euclidean}/<model>/` | the 4-mer profile and the four models with raw or standardized features and the cosine or Euclidean distance. Standardized features with the cosine distance are the main tables `metrics/<model>/` (the former copy was identical up to column order and was removed). Used by `figures/main/fig8_scaling_distance.png` |
| `data/metrics/P4/kmer`, `P5/kmer` | **deleted** | k-mer profile under old code paths, never used; MinHash and Jensen-Shannon are the models `minhash` and `kmer_js` |
| `data/metrics/P4/minhash_k6`, `P5/kmer_k4` | **deleted** | identical copies of `metrics/minhash` and `metrics/kmer_js` (verified with `diff -r` before the move) |
| `data/metrics/k_sweep/` | `metrics/legacy/k_sweep/` | older sweep on a different random subset of samples; `best_k.json` agrees with `selection.json` (js 4, minhash 6) |
| `data/metrics/metric_axis_summary.xlsx`, `p@k_for_k/*.parquet` | `metrics/legacy/` | |
| `data/metrics/p@k_for_k/P@K_K.png` | **deleted** | byte-identical to `figures/exploratory/P@K_K.png` |
| `data/metrics/compare_metrics/for_article/*.png` | `figures/exploratory/legacy_compare_metrics/` | 8 files that were byte-identical to files already in `figures/exploratory/` were deleted, 44 moved |
| `data/metrics/kmer_sweep/kmer_sweep_plot.png` | `figures/exploratory/legacy_kmer_sweep_plot.png` | |

`metrics/P@K_new_virus/`, `metrics/cross_virus/`, `metrics/pairwise_cross_viruses.xlsx` and
`metrics/summary_new_viruses.xlsx` are outputs of the pre-package pipeline; nothing in the
current code writes or reads them (the exploratory notebooks do). Treat them as legacy.

## `figure_data/` -- the numbers behind each figure

Committed, and read by the drawing step alone. Repository figure numbers; the article's
numbering is to be confirmed (main README).

| Figure | What it shows | File | Columns |
|--------|---------------|------|---------|
| Fig. 1 | best layer of each model: ARI, P@5 | `fig1_reference_benchmark.csv` | `metric, model, best_layer, mean, std, p_vs_kmer_bh` |
| Fig. 2 | ARI and P@5 across all layers | `fig2_layerwise.csv` | `metric, model, layer, mean, std, p_vs_best_bh` |
| Fig. S1 | k-mer length sweep of the 4-mer, Jensen-Shannon and MinHash baselines | `supplementary/figS1_kmer_sweep.csv` | `method, metric, k, mean, std, n, best, p_vs_best_bh` |
| Fig. 4 | unseen orders, **genome-level** P@5, forward and RC | `fig4_newvirus_p5.csv` | `group, model, layer, mean, std, p_vs_kmer_bh` |
| Fig. 5 | **RC proximity** P@5 (file keeps the legacy name `cross`) | `fig5_cross_p5.csv` | `group, model, layer, mean, std, p_vs_kmer_bh` |
| Fig. 6 | P@k against k | `fig6_pk_vs_k.csv` | `virus, model, k, layer, mean, leader_vs_second_bh` |
| Fig. 7 | per layer: own-genome density against **inter-order leakage** (columns keep the old `cross_p_at_10`) | `fig7_pk_vs_cross.csv` | `model, layer, p_at_10, cross_p_at_10` |
| Fig. 4b / 5b / 5c | order-level P@5 / own-RC hit@5 / own-RC similarity -- written only once `metrics` has produced the new tables | `fig4b_order_p5.csv`, `fig5b_own_rc_hit5.csv`, `fig5c_own_rc_similarity.csv` | as Fig. 4 |
| Fig. 8 | P@5 on the unseen orders under four scaling/distance variants | `supplementary/scaling_distance.csv` | `group, model, variant, label, layer, mean, std, p_vs_main_bh` |
| supplementary | per-layer tables; scaling and distance; early Evo2 layer | `supplementary/layerwise_reference.csv`, `layerwise_newvirus_p5.csv`, `scaling_distance.csv`, `evo2_early_layer.csv` | |

**What changed in the committed CSVs with this release** (rebuilt from the same committed
metric tables, no new computation): Fig. 4, 5, 6 now also contain `kmer_js` and `minhash`
(seven models, 28/14/140 rows); Fig. 1, 2, 7 and the layer-wise supplementary files keep the
five original models, because MinHash and k-mer JS have no reference benchmark and no
pairwise tables yet (the figures state this). The numbers of the original five models are
unchanged -- `tests/test_regression.py` pins them. Fig. 5 and the statistics panels were
renamed "RC proximity" (labels only). `supplementary/scaling_distance.csv` replaces the former distance-protocol table.

`p_*_bh` columns are Benjamini-Hochberg adjusted p-values of the two-sided paired Wilcoxon
test over the reference samples (`glmbench/figures/significance.py`): Fig. 4-6 take them
from `statistics/pairwise_model_tests.csv` (selection `reference`, study-wide family), the
other figures form one family per figure. `layer` / `best_layer` are layer *labels* as in
the extracted files -- ViraLM's blocks are 1..12, so its benchmark position 2 is shown as
layer 3 in every figure (Fig. 1/2 showed the position before; the scores are unchanged).

Fig. 4, 5 and 6 report each model at the layer chosen by its **reference benchmark**, not at
the layer that scores best on the unseen order; the `layer` column says which one. Fig. 1 and
2 are the benchmark itself, so they show its own best layer. Fig. 7 shows every layer.

## `statistics/` -- the tests

| File | What it holds |
|------|---------------|
| `pairwise_model_tests.csv` | one row per pair of models, per panel, per selection rule |
| `panel_samples.csv` | the per-sample values those tests are computed from |
| `genome_bootstrap_ci.csv` | cluster-bootstrap CIs over genomes (written once per-genome tables exist; **not present yet**) |

The tests are paired over the 50 reference samples: Wilcoxon signed-rank (primary,
two-sided), a paired t-test, `rank_biserial_r` as effect size. **They test robustness to the
reference draw only**, not to the choice of viruses; the exact Wilcoxon p-value cannot go
below 2/2^50 = 1.8e-15 and sits exactly on that floor whenever a model wins every sample.

Corrections: `p_holm_panel` within a panel; `p_holm_all` and `p_bh_all` over the distinct
comparisons. Distinct = (source table, model pair): panels reading the same table count once.
With 7 models (21 pairs): **504** distinct comparisons under `reference` / `reference_split`
(Fig. 1 excluded; 24 distinct tables x 21), **534** under the rules that include Fig. 1
(30 more, from the 5 models with a benchmark). With the original 5 models: 240 / 270.
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
(10 of 181), not "about half" as an earlier version of this file said.

## Conventions

* `mean` / `std` are taken across the reference samples.
* Model order, display names and colours come from `glmbench/models.py` (Okabe-Ito).
* Per-genome tables exist so that the **genome** can be the resampling unit; fragments of one
  genome and the overlapping reference draws are not independent observations.
