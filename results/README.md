# `results/` — every number the analysis produces

Nothing here is edited by hand. Plots live one level up, in `figures/`.

```
metrics/            per-sample metric tables, one folder per model
metrics/selection.json   the layer and word length chosen for each model
figure_data/        the exact numbers plotted in each figure (CSV)
statistics/         paired tests between models (CSV)
```

## `metrics/` — the tables the pipeline writes

Every metric is stored the same way: rows are layers (`-` for a model without layers),
columns are the 50 reference samples. Keeping the per-sample values rather than only a
mean is what makes the paired tests possible and lets any figure be rebuilt without
recomputing anything.

| File | What it holds |
|------|---------------|
| `<model>/{ARI,P5,silh}.xlsx` | the reference benchmark, per layer |
| `<model>/P@<k>_matrix_<virus>.xlsx` | P@k on an unseen order |
| `<model>/cross_P@<k>_matrix_<virus>.xlsx` | CROSS P@k, forward against reverse complement |
| `<model>/pairwise_<A>_vs_<B>/` | one unseen order against the other (Fig. 7) |
| `cross_virus/` | per-strand cross-virus tables |
| `kmer_sweep/` | the k-mer word-length sweep behind Fig. 3 |
| `k_sweep/<model>_k_sweep.xlsx` | the word-length sweep that chose `k` |
| `selection.json` | the chosen layer and word length of each model |

## `figure_data/` — the numbers behind each figure

Committed, and read by the drawing step alone, so restyling a figure never touches a
metric table. They double as the machine-readable source data of the article.

| Figure | What it shows | File | Columns |
|--------|---------------|------|---------|
| `fig1_reference_benchmark` | best layer of each model: ARI, P@5, silhouette | `fig1_reference_benchmark.csv` | `metric, model, best_layer, mean, std` |
| `fig2_layerwise` | ARI and P@5 across all layers | `fig2_layerwise.csv` | `metric, model, layer, mean, std` |
| `fig3_kmer_sweep` | k-mer baseline against word length k | `fig3_kmer_sweep.csv` | `k, P@5_mean, P@5_std, ARI_mean, ARI_std, Silhouette_mean, Silhouette_std` |
| `fig4_newvirus_p5` | unseen orders: P@5, forward and reverse complement | `fig4_newvirus_p5.csv` | `group, model, layer, mean, std` |
| `fig5_cross_p5` | forward vs reverse-complement proximity: CROSS P@5 | `fig5_cross_p5.csv` | `group, model, layer, mean, std` |
| `fig6_pk_vs_k` | P@k against the neighbourhood size k | `fig6_pk_vs_k.csv` | `virus, model, k, layer, mean` |
| `fig7_pk_vs_cross` | per layer: own-order density against cross-order proximity | `fig7_pk_vs_cross.csv` | `model, layer, p_at_10, cross_p_at_10` |
| per-layer supplementary | every layer of every metric | `supplementary/layerwise_reference.csv`, `supplementary/layerwise_newvirus_p5.csv` | |

Fig. 4, 5 and 6 report each model at the layer chosen by its **reference benchmark**,
not at the layer that scores best on the unseen order itself; the `layer` column says
which one that was. Fig. 1 and 2 are the benchmark itself, so they show its own best
layer. Fig. 7 shows every layer, so no choice arises.

## `statistics/` — the tests

| File | What it holds |
|------|---------------|
| `pairwise_model_tests.csv` | one row per pair of models, per panel, per selection rule |
| `panel_samples.csv` | the per-sample values those tests are computed from |

Every model is scored on the same reference samples, so the tests are paired: Wilcoxon
signed-rank (primary, two-sided) and a paired t-test, with `rank_biserial_r` as the
effect size.

Corrections: `p_holm_panel` within one panel, `p_holm_all` and `p_bh_all` over the
distinct comparisons of the whole study — distinct because some panels read the same
table and must not be counted twice. `signif` follows `p_bh_all`:
`***` < 0.001, `**` < 0.01, `*` < 0.05, `ns`.

The `selection` column records how each model's layer was chosen:

| `selection` | Layer chosen on | Reported on | Role |
|-------------|-----------------|-------------|------|
| `reference` | the model's reference benchmark, all 50 | all 50 | **primary** for the unseen-order panels |
| `reference_split` | the benchmark, even samples | odd samples | choice and report share no sample |
| `split` | even samples of the same table | odd samples | **primary** for Fig. 1, which *is* the benchmark |
| `in_sample` | all 50 of the same table | all 50 | reproduces the published figures |
| `oracle` | odd samples of the same table | odd samples | diagnostic: same samples as `split`, so the gap between them is the selection optimism |

Measured on these tables, that optimism is negligible — median 0, at most +0.011 — but
the chosen layer differs in roughly half the panels, which is why the rule matters.

## Conventions

* `mean` / `std` are taken across the 50 reference samples.
* Model order, display names and colours come from `glmbench/models.py`
  (Okabe-Ito palette, colour-blind safe); change them there and every figure follows.
* The Wilcoxon test has a floor: with n samples the smallest two-sided p it can return
  is 2/2^n, so values at that floor mean "better on every sample", not an exact
  probability.
