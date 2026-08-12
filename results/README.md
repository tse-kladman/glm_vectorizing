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
figure_data/         numbers behind the main-text figures
figure_data/supplementary/
figures/main/        Fig. 1-7 of the article
figures/supplementary/
figures/exploratory/ output of visualization/exploratory/ (not used in the article)
```

---

## Main-text figures

Produced by [`visualization/main_figures.ipynb`](../visualization/main_figures.ipynb).

| Figure | What it shows | Numbers | Columns |
|--------|---------------|---------|---------|
| `fig1_reference_benchmark` | Best layer of each model on the reference set: ARI, P@5, silhouette | `figure_data/fig1_reference_benchmark.csv` | `metric, model, best_layer, mean, std` |
| `fig2_layerwise` | ARI and P@5 across all transformer layers | `figure_data/fig2_layerwise.csv` | `metric, model, layer, mean, std` |
| `fig3_kmer_sweep` | k-mer baseline: ARI / P@5 / silhouette vs word length k | `figure_data/fig3_kmer_sweep.csv` | `k, P@5_mean, P@5_std, ARI_mean, ARI_std, Silhouette_mean, Silhouette_std` |
| `fig4_newvirus_p5` | Generalisation to unseen orders: P@5, forward and reverse complement | `figure_data/fig4_newvirus_p5.csv` | `group, model, mean, std` |
| `fig5_cross_p5` | Forward vs reverse-complement proximity: CROSS P@5 | `figure_data/fig5_cross_p5.csv` | `group, model, mean, std` |
| `fig6_pk_vs_k` | P@k as a function of the neighbourhood size k (k = 1..10) | `figure_data/fig6_pk_vs_k.csv` | `virus, model, k, mean` |
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

## Conventions

* Model order, display names and colours come from `visualization/figure_io.py`
  (Okabe-Ito palette, colour-blind safe). Change them there and every figure follows.
* `mean` / `std` are always taken across the 50 reference samples.
* `layer` is an integer, except for the layer-less 4-mer baseline, which is written as `-`
  (it appears in `fig7_pk_vs_cross.csv` and is drawn as a single star in Fig. 7).
* All PDFs embed TrueType fonts (`pdf.fonttype = 42`), so the text stays selectable and
  editable in a vector editor.
