# `results/` — everything the figure notebooks produce

Nothing in this folder is edited by hand and nothing outside it is written to by the
notebooks. Two kinds of content live here:

* **`figure_data/`** — the exact numbers plotted in each figure, as CSV. These are
  committed to git, and Stage 2 of each notebook reads *only* these files. Restyling a
  figure therefore never requires re-reading `data/metrics/**.xlsx` or recomputing
  anything.
* **`figures/`** — the rendered figures, each saved twice: `.png` (300 dpi, for the
  manuscript) and `.pdf` (vector, with editable text).

```
figure_data/           numbers behind the main-text figures
figure_data/supplementary/
figure_data/statistics/  per-sample values and the pairwise tests between models
figures/main/          Fig. 1-7 of the article
figures/supplementary/
figures/statistics/    box plots and heat maps of the model comparisons
figures/exploratory/   output of visualization/exploratory/ (not used in the article)
```

---

## Main-text figures

Produced by [`visualization/main_figures.ipynb`](../visualization/main_figures.ipynb).

| Figure | What it shows | Numbers | Columns |
|--------|---------------|---------|---------|
| `fig1_reference_benchmark` | Best layer of each model on the reference set: ARI, P@5, silhouette | `figure_data/fig1_reference_benchmark.csv` | `metric, model, best_layer, mean, std` |
| `fig2_layerwise` | ARI and P@5 across all transformer layers | `figure_data/fig2_layerwise.csv` | `metric, model, layer, mean, std` |
| `fig3_kmer_sweep` | k-mer baseline: ARI / P@5 / silhouette vs word length k | `figure_data/fig3_kmer_sweep.csv` | `k, P@5_mean, P@5_std, ARI_mean, ARI_std, Silhouette_mean, Silhouette_std` |
| `fig4_newvirus_p5` | Generalisation to unseen orders: P@5, forward and reverse complement, at the reference-chosen layer | `figure_data/fig4_newvirus_p5.csv` | `group, model, layer, mean, std` |
| `fig5_cross_p5` | Forward vs reverse-complement proximity: CROSS P@5, at the reference-chosen layer | `figure_data/fig5_cross_p5.csv` | `group, model, layer, mean, std` |
| `fig6_pk_vs_k` | P@k as a function of the neighbourhood size k (k = 1..10), at the reference-chosen layer | `figure_data/fig6_pk_vs_k.csv` | `virus, model, k, layer, mean` |
| `fig7_pk_vs_cross` | Egovirales vs Mriyavirales, per layer: own-order density (P@10) vs cross-order proximity (CROSS P@10) | `figure_data/fig7_pk_vs_cross.csv` | `model, layer, p_at_10, cross_p_at_10` |

Each figure exists as `figures/main/<name>.png` and `figures/main/<name>.pdf`.

## Supplementary figures

Produced by
[`visualization/supplementary_figures.ipynb`](../visualization/supplementary_figures.ipynb).
Fig. 1, 2, 4 and 5 summarise each model by its best layer; these show the full per-layer
picture behind that summary.

| Figures | What they show | Numbers | Columns |
|---------|----------------|---------|---------|
| `layerwise_<model>_{ARI,P5,silh}` | Reference-set metric across all layers, one panel per model | `figure_data/supplementary/layerwise_reference.csv` | `metric, model, layer, mean, std` |
| `newvirus_p5_<model>_<virus>` | P@5 across all layers for each unseen order, forward and RC (`_rc`) | `figure_data/supplementary/layerwise_newvirus_p5.csv` | `virus, model, layer, mean, std` |

`<model>` is one of `kmer`, `dnabert2`, `dnabertS`, `viralm`, `evo2`; `<virus>` is
`Egovirales` or `Mriyavirales`, with the `_rc` suffix for reverse-complement input. The
4-mer baseline has no layers and so appears only in the `layerwise_kmer_*` figures as a
single bar.

## Statistical tests between models

Produced by [`visualization/model_significance.py`](../visualization/model_significance.py)
(`python visualization/model_significance.py`; reads only `data/metrics/**.xlsx`).

Every metric is computed on the same 50 reference samples for every model, so the tests
are paired. Every pair of models is compared with the Wilcoxon signed-rank test
(primary, two-sided) and the paired t-test.

The layer each model is taken at is itself chosen from the data, so the tables carry
three selection rules in the `selection` column:

| `selection` | Layer chosen on | Reported on | Role |
|-------------|-----------------|-------------|------|
| `reference` | the model's reference benchmark (`P5.xlsx`), all 50 | all 50 (n = 50) | **primary for the unseen-order panels** (Fig. 4-6): the layer is picked on a different task and never on the numbers reported |
| `reference_split` | reference benchmark, even samples | odd samples (n = 25) | as above, but choice and report do not even share a sample draw |
| `split` | even samples of the same table | odd samples (n = 25) | **primary for Fig. 1**, which *is* the reference benchmark |
| `in_sample` | all 50 of the same table | all 50 (n = 50) | reproduces the published figures |
| `oracle` | odd samples of the same table | odd samples (n = 25) | diagnostic: same scored samples as `split`, so the gap between them *is* the selection optimism |

Reference-chosen layers are matched to the metric tables **by position**, not by value:
ViraLM numbers its layers 0..11 in the benchmark table and 1..12 in the P@K matrices, so
looking the layer up by its number would silently shift it by one.

Two things this rule shows. Choosing the layer on the panel itself inflates the score only
slightly (`oracle` vs `split`: median 0, at most +0.011), but it changes *which layer wins*
in roughly half the panels -- and for Evo2 that costs 0.19 of P@5 on Egovirales, enough to
move it from first place to last. Fig. 1 keeps `split`, since choosing its layer from the
benchmark it is itself measuring would be circular.

| File | What it holds |
|------|---------------|
| `figure_data/statistics/pairwise_model_tests.csv` | one row per pair of models, per panel of Fig. 1, 4, 5 and 6 |
| `figure_data/statistics/panel_samples.csv` | the per-sample values behind those tests (`figure, panel, model, layer, sample, value`) |

Columns of the test table: `mean_a`, `mean_b`, `mean_diff` (A - B), `rank_biserial_r`
(matched-pairs effect size, positive = A higher), `p_wilcoxon`, `p_ttest`, and three
multiple-testing corrections of the Wilcoxon p-value:

| Column | Family | Controls |
|--------|--------|----------|
| `p_holm_panel` | the 10 pairs of one panel | FWER within a panel read on its own |
| `p_holm_all` | all 270 distinct comparisons | FWER of the whole study (most conservative) |
| `p_bh_all` | all 270 distinct comparisons | FDR; `signif` and every figure annotation use this one |

The three selection rules are alternative analyses of one dataset rather than extra
hypotheses, so each is corrected within itself.

Fig. 4 and the `k = 5` panels of Fig. 6 are built from the same metric tables; those 20
duplicated comparisons are counted once by the study-wide corrections (`source` and
`comparison` identify them).

`signif`: `***` < 0.001, `**` < 0.01, `*` < 0.05, `ns`.

| Figure | What it shows |
|--------|---------------|
| `figures/statistics/box_fig{1,4,5}` | distribution of each model over the reference samples, one box per model, sorted by mean and annotated with significance brackets (neighbouring pairs, plus every pair that comes out *not* significant) |
| `figures/statistics/paired_fig{1,4,5}` | the 50 *paired* differences against the best model of each panel -- the quantity the test ranks |
| `figures/statistics/heatmap_fig{1,4,5}` | every pair: mean difference (colour) and BH-adjusted significance |
| `figures/statistics/pvalue_overview` | all comparisons, raw vs BH-adjusted p, with the exact-Wilcoxon floor 2/2^50 |
| `figures/statistics/selection_effect` | how much choosing the layer on the scored samples inflates the result |

Because the only source of randomness is the choice of reference sample, these p-values
say that a ranking is stable across reference sets -- not that it would hold for other
viruses or other training runs.

## Distance protocols: alternative-metric baselines vs the models

Produced by [`visualization/metric_axis_figures.py`](../visualization/metric_axis_figures.py)
(`python visualization/metric_axis_figures.py`). Two baselines are drawn next to every
model under the published protocol P0: the 4-mer profile under Jensen-Shannon (P5, from
`data/metrics/P5/kmer/`) and MinHash sketches under the Jaccard index (P4, from
`data/metrics/P4/minhash/` — build the sketches with `utils/minhash_sketch.py` first).
`--extra_protocols` adds P1-P3 on the 4-mer profile.

| Figure | What it shows | Numbers |
|--------|---------------|---------|
| `figures/metric_axis/fig4_newvirus_p5` | Fig. 4 with the extra baseline series | `figure_data/metric_axis/fig4_newvirus_p5.csv` |
| `figures/metric_axis/fig5_cross_p5` | Fig. 5 with the extra baseline series | `figure_data/metric_axis/fig5_cross_p5.csv` |
| `figures/metric_axis/fig6_pk_vs_k` | Fig. 6 with the extra baseline series (dashed) | `figure_data/metric_axis/fig6_pk_vs_k.csv` |
| `figures/metric_axis/box_fig{4,5}` | per-sample boxes with BH-adjusted brackets (split-sample) | `figure_data/metric_axis/panel_samples.csv`, `pairwise_tests.csv` |

## Conventions

* Model order, display names and colours come from `visualization/figure_io.py`
  (Okabe-Ito palette, colour-blind safe). Change them there and every figure follows.
* `mean` / `std` are always taken across the 50 reference samples.
* Fig. 4, 5 and 6 report each model at the layer chosen by its **reference benchmark**
  (`data/metrics/<model>/P5.xlsx`), not at the layer that scores best on the unseen
  order itself, so no selection happens on the reported numbers. One layer per model is
  used across all of those panels; `figure_io.reference_layer` matches the benchmark to
  the metric tables by position, because ViraLM numbers its layers 0..11 in one and
  1..12 in the other.
* `layer` is an integer, except for the layer-less 4-mer baseline, which is written as `-`
  (it appears in `fig7_pk_vs_cross.csv` and is drawn as a single star in Fig. 7).
* All PDFs embed TrueType fonts (`pdf.fonttype = 42`), so the text stays selectable and
  editable in a vector editor.
