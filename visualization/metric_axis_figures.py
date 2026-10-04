"""
metric_axis_figures.py -- Fig. 4-6 redrawn with the distribution-only protocols.

The 4-mer profile is a probability distribution, so besides the historical protocol
(P0: StandardScaler + cosine) it can be compared with metrics defined on the simplex:

    P5  sqrt(Jensen-Shannon)        on the 4-mer profile itself

MinHash goes further: it drops the frequency vector and describes a fragment by the
*set* of its k-mers, estimating the Jaccard index between two such sets.  That is a
representation of its own, built by utils/minhash_sketch.py, and it enters as

    P4  MinHash (Jaccard)           on sketches, model "minhash"

This script puts those baselines next to every model under P0 --
the protocol behind all published tables -- and redraws the figures we already have
for that comparison:

    fig4_newvirus_p5   P@5 on the unseen orders, forward and RC           (bars)
    fig5_cross_p5      CROSS P@5, forward vs reverse complement           (bars)
    fig6_pk_vs_k       P@k as a function of k                             (lines)
    box_fig4, box_fig5 per-sample distributions with significance brackets

Inputs
    data/metrics/<model>/P@k_matrix_*.xlsx       P0, every model (published tables)
    data/metrics/<P4|P5>/kmer/P@k_matrix_*.xlsx  written by utils/metric_axis.py:

        python utils/metric_axis.py --latent_dir data/latent_dim --metrics_dir data/metrics \\
            --names Egovirales_preproc Mriyavirales_preproc --models kmer --protocols P4 P5

Like the figure notebooks, the script has two stages:

  Stage 1  data/metrics/**.xlsx  ->  results/figure_data/metric_axis/*.csv
  Stage 2  results/figure_data/metric_axis/*.csv  ->  results/figures/metric_axis/*.{png,pdf}

    python visualization/metric_axis_figures.py
    python visualization/metric_axis_figures.py --plot_only            # restyle only
    python visualization/metric_axis_figures.py --extra_protocols P1 P2 P3 P4 P5

Layer choice follows the rest of the repository: the bar and line figures take each
model at the layer with the highest mean over all 50 samples (as the published
figures do); the box plots and tests use the split-sample rule of
model_significance.py (layer chosen on even samples, scored on odd ones).  The 4-mer
has no layers, so for it both rules coincide.
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import figure_io as fio
from figure_io import MODELS, DISPLAY, MODEL_COLOR, save_figure
from model_significance import (_best_row, _draw_brackets, _grid, _significance_pairs,
                                benjamini_hochberg, holm, rank_biserial, stars)

FIG_DATA_NAME = "metric_axis"
FIG_DATA = fio.FIG_DATA / FIG_DATA_NAME

VIRUSES = [("Egovirales_preproc", "Egovirales"), ("Mriyavirales_preproc", "Mriyavirales")]
FIG4_GROUPS = [("Egovirales_preproc", "Egovirales"),
               ("Mriyavirales_preproc", "Mriyavirales"),
               ("REVERSECOMPLEMENTEgovirales_preproc", "Egovirales (RC)"),
               ("REVERSECOMPLEMENTMriyavirales_preproc", "Mriyavirales (RC)")]
KS = range(1, 11)

# Protocol labels match utils/compute_metrics.PROTOCOLS; kept here so that Stage 2
# does not need the computation code.
PROTOCOL_LABEL = {
    "P0": "std + cosine", "P1": "raw + cosine", "P2": "std + euclidean",
    "P3": "raw + euclidean", "P4": "MinHash (Jaccard)", "P5": "Jensen-Shannon",
}
# Which representation each protocol needs.  P1-P3 and P5 read the 4-mer profile;
# P4 reads MinHash sketches, which are a representation of their own and live under
# their own model key (see utils/minhash_sketch.py).
PROTOCOL_MODEL = {"P1": "kmer", "P2": "kmer", "P3": "kmer",
                  "P4": "minhash", "P5": "kmer"}


def resolve_models(metrics_dir: Path) -> dict:
    """Point P4/P5 at the word length that utils/k_sweep.py chose on the reference set.

    The sweep writes ``data/metrics/k_sweep/best_k.json``; with it, P5 reads
    ``kmer_k<K>`` and P4 reads ``minhash_k<K>`` instead of the k = 4 defaults.
    """
    models = dict(PROTOCOL_MODEL)
    path = Path(metrics_dir) / "k_sweep" / "best_k.json"
    if not path.exists():
        return models
    chosen = json.loads(path.read_text())
    if "js" in chosen:
        models["P5"] = f"kmer_k{chosen['js']}"
    if "minhash" in chosen:
        models["P4"] = f"minhash_k{chosen['minhash']}"
    return models


SERIES_DISPLAY = {**DISPLAY, "minhash": "MinHash"}


def series_name(model: str) -> str:
    """4-mer-like model key -> display name carrying its word length, e.g. '6-mer'."""
    if model.startswith("kmer"):
        return f"{model.split('_k')[1]}-mer" if "_k" in model else DISPLAY["kmer"]
    if model.startswith("minhash"):
        return "MinHash"
    return SERIES_DISPLAY.get(model, model)
# Okabe-Ito colours not used by the models; the hatch marks "4-mer, other metric".
PROTOCOL_COLOR = {
    "P1": "#F0E442", "P2": "#882255", "P3": "#000000",
    "P4": "#CC79A7", "P5": "#56B4E9",
}
KMER_HATCH = "///"


# --------------------------------------------------------------------------- #
#  Series: one (model, protocol) pair = one bar / line / box.                   #
# --------------------------------------------------------------------------- #
def series_table(extra_protocols, protocol_model=None) -> pd.DataFrame:
    """Plot order: the alternative-metric baselines first, then the language models."""
    protocol_model = protocol_model or PROTOCOL_MODEL
    rows = [{"series": "kmer", "model": "kmer", "protocol": "P0"}]
    rows += [{"series": f"{protocol_model[p]}_{p}", "model": protocol_model[p],
              "protocol": p} for p in extra_protocols]
    rows += [{"series": m, "model": m, "protocol": "P0"} for m in MODELS if m != "kmer"]
    return pd.DataFrame(rows)


def series_style(row) -> dict:
    """Label, colour and hatch of one series, derived from its model and protocol."""
    if row["protocol"] == "P0":
        # P0 is the protocol of every published series, so it is left unnamed: a bare
        # "4-mer" means the standard protocol (StandardScaler + cosine), exactly like
        # the language-model series next to it.  Only the alternatives name a metric.
        return {"label": DISPLAY[row["model"]], "color": MODEL_COLOR[row["model"]],
                "hatch": None, "ls": "-"}
    # MinHash is its own representation, so the protocol label already names it in full
    label = (PROTOCOL_LABEL[row["protocol"]] if row["model"].startswith("minhash") else
             f"{series_name(row['model'])} ({PROTOCOL_LABEL[row['protocol']]})")
    return {"label": label, "color": PROTOCOL_COLOR[row["protocol"]],
            "hatch": KMER_HATCH, "ls": "--"}


def styles(series: pd.DataFrame) -> dict:
    return {row["series"]: series_style(row) for _, row in series.iterrows()}


# --------------------------------------------------------------------------- #
#  Stage 1 -- read the metric tables.                                           #
# --------------------------------------------------------------------------- #
def matrix_path(metrics_dir: Path, model: str, protocol: str, filename: str) -> Path:
    """P0 comes from the published tables; other protocols from metric_axis.py."""
    base = metrics_dir / model if protocol == "P0" else metrics_dir / protocol / model
    return base / filename


def load_matrix(path: Path) -> pd.DataFrame:
    """Rows = layers ('-' for the 4-mer), columns = reference sample index 0..49."""
    if not path.exists():
        model = path.parent.name
        sketches = ("  python utils/minhash_sketch.py --ref_dir data/random_samples "
                    "--virus_dir data/preproc_unk_viruses --names Egovirales_preproc "
                    "Mriyavirales_preproc --rc --outdir data/latent_dim/minhash\n"
                    if model == "minhash" else "")
        # P0 tables live directly under <metrics_dir>/<model>/, the others under
        # <metrics_dir>/<protocol>/<model>/, so only the latter name a protocol.
        parent = path.parent.parent.name
        protocol = parent if re.fullmatch(r"P\d+", parent) else "P0"
        if protocol == "P0":
            raise SystemExit(
                f"ERROR: {path} not found.\n"
                f"This is a published table of the main pipeline; run run_pipeline.py "
                f"(or point --metrics_dir at the directory that holds it).")
        raise SystemExit(
            f"ERROR: {path} not found.\n"
            f"Compute the missing protocol first:\n" + sketches +
            f"  python utils/metric_axis.py --latent_dir data/latent_dim "
            f"--metrics_dir data/metrics --names Egovirales_preproc Mriyavirales_preproc "
            f"--models {model} --protocols {protocol}")
    mat = pd.read_excel(path).set_index("layer")
    mat.columns = [int(str(c).split("_")[1]) for c in mat.columns]
    return mat.sort_index(axis=1)


def panels():
    """(figure, panel label, file name) of every panel the script draws or tests."""
    for name, label in FIG4_GROUPS:
        yield "fig4", label, f"P@5_matrix_{name}.xlsx"
    for name, label in VIRUSES:
        yield "fig5", label, f"cross_P@5_matrix_{name}.xlsx"
    for name, label in VIRUSES:
        for k in KS:
            yield "fig6", f"P@{k} {label}", f"P@{k}_matrix_{name}.xlsx"


def extract(series: pd.DataFrame, metrics_dir: Path):
    """Per-sample values under both selection rules, summary rows and tests."""
    samples, summary = [], []
    for figure, panel, filename in panels():
        for _, s in series.iterrows():
            mat = load_matrix(matrix_path(metrics_dir, s["model"], s["protocol"], filename))
            for selection in ("in_sample", "split"):
                layer, values = _best_row(mat, selection)
                if s["model"].startswith(("kmer", "minhash")):
                    layer = "-"
                for sample, value in values.items():
                    samples.append({"selection": selection, "figure": figure, "panel": panel,
                                    "source": filename, "series": s["series"],
                                    "layer": layer, "sample": int(sample),
                                    "value": float(value)})
                if selection == "in_sample":
                    summary.append({"figure": figure, "panel": panel, "series": s["series"],
                                    "model": s["model"], "protocol": s["protocol"],
                                    "layer": layer, "mean": float(values.mean()),
                                    "std": float(values.std(ddof=0))})
    return pd.DataFrame(samples), pd.DataFrame(summary)


def pairwise_tests(samples: pd.DataFrame, order: list) -> pd.DataFrame:
    """Wilcoxon + paired t for every pair of series, per panel; Holm and BH corrections.

    Same conventions as model_significance.py: Holm within a panel, Holm and BH over
    the distinct comparisons of the whole set (Fig. 4 and the k = 5 panels of Fig. 6
    share their tables, so those comparisons are counted once).
    """
    rows = []
    for (figure, panel, source), sub in samples.groupby(["figure", "panel", "source"],
                                                        sort=False):
        wide = sub.pivot(index="sample", columns="series", values="value")
        layer = sub.groupby("series")["layer"].first()
        panel_rows = []
        for a, b in combinations([s for s in order if s in wide.columns], 2):
            pair = wide[[a, b]].dropna()
            d = (pair[a] - pair[b]).to_numpy()
            panel_rows.append({
                "figure": figure, "panel": panel, "source": source,
                "series_a": a, "layer_a": layer[a], "series_b": b, "layer_b": layer[b],
                "n": len(d), "mean_a": pair[a].mean(), "mean_b": pair[b].mean(),
                "mean_diff": d.mean(), "rank_biserial_r": rank_biserial(d),
                "p_wilcoxon": stats.wilcoxon(pair[a], pair[b]).pvalue if d.any() else 1.0,
                "p_ttest": stats.ttest_rel(pair[a], pair[b]).pvalue if d.any() else 1.0,
            })
        panel_df = pd.DataFrame(panel_rows)
        panel_df["p_holm_panel"] = holm(panel_df["p_wilcoxon"].to_numpy())
        rows.append(panel_df)

    tests = pd.concat(rows, ignore_index=True)
    tests["comparison"] = tests["source"] + "|" + tests["series_a"] + "|" + tests["series_b"]
    unique = tests.drop_duplicates("comparison").set_index("comparison")
    p = unique["p_wilcoxon"].to_numpy()
    adjusted = pd.DataFrame({"p_holm_all": holm(p), "p_bh_all": benjamini_hochberg(p)},
                            index=unique.index)
    tests = tests.join(adjusted, on="comparison")
    tests["signif"] = tests["p_bh_all"].map(stars)
    return tests


def stage1(series: pd.DataFrame, metrics_dir: Path):
    samples, summary = extract(series, metrics_dir)
    order = series["series"].tolist()
    split = samples[samples["selection"] == "split"]
    tests = pairwise_tests(split, order)

    fig4 = summary[summary["figure"] == "fig4"].rename(columns={"panel": "group"})
    fig5 = summary[summary["figure"] == "fig5"].rename(columns={"panel": "group"})
    fig6 = summary[summary["figure"] == "fig6"].copy()
    fig6[["k", "virus"]] = fig6["panel"].str.extract(r"P@(\d+) (\w+)")
    fig6["k"] = fig6["k"].astype(int)

    cols = ["series", "model", "protocol", "layer", "mean", "std"]
    fio.save_table(series, f"{FIG_DATA_NAME}/series")
    fio.save_table(fig4[["group"] + cols], f"{FIG_DATA_NAME}/fig4_newvirus_p5")
    fio.save_table(fig5[["group"] + cols], f"{FIG_DATA_NAME}/fig5_cross_p5")
    fio.save_table(fig6[["virus", "k"] + cols], f"{FIG_DATA_NAME}/fig6_pk_vs_k")
    fio.save_table(split.drop(columns="selection"), f"{FIG_DATA_NAME}/panel_samples")
    fio.save_table(tests, f"{FIG_DATA_NAME}/pairwise_tests")


# --------------------------------------------------------------------------- #
#  Stage 2 -- figures, from the CSVs only.                                      #
# --------------------------------------------------------------------------- #
def _legend_below(ax, ncol):
    ax.legend(ncol=ncol, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))


def plot_bars(df: pd.DataFrame, order_groups, style, name, ylabel, title, width):
    order = df["series"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(width, 5.0))
    w = 0.8 / len(order)
    x = np.arange(len(order_groups))
    for i, s in enumerate(order):
        d = df[df["series"] == s].set_index("group").reindex(order_groups)
        st = style[s]
        ax.bar(x + (i - (len(order) - 1) / 2) * w, d["mean"], w, yerr=d["std"], capsize=2,
               color=st["color"], hatch=st["hatch"], label=st["label"],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(order_groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_below(ax, ncol=4)
    fig.tight_layout()
    save_figure(fig, name, kind="metric_axis")
    plt.close(fig)


def plot_fig6(df: pd.DataFrame, style):
    order = df["series"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (_, virus) in zip(axes, VIRUSES):
        for s in order:
            d = df[(df["virus"] == virus) & (df["series"] == s)].sort_values("k")
            st = style[s]
            ax.plot(d["k"], d["mean"], st["ls"], marker="o", ms=4, color=st["color"],
                    label=st["label"])
        ax.set_xlabel("k (number of neighbours)")
        ax.set_title(virus)
        ax.set_xticks(list(KS))
    axes[0].set_ylabel("Precision@k")
    axes[0].legend(fontsize=9, ncol=2)
    fig.suptitle("Precision@k as a function of k: 4-mer under distribution metrics "
                 "vs the models", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig6_pk_vs_k", kind="metric_axis")
    plt.close(fig)


def plot_boxes(samples: pd.DataFrame, tests: pd.DataFrame, style, figure, title, ncols):
    """Same layout as model_significance.plot_boxes, one box per series."""
    rng = np.random.default_rng(0)
    sub_fig = samples[samples["figure"] == figure]
    panel_names = sub_fig["panel"].unique()
    fig, axes = _grid(len(panel_names), ncols, panel_size=(5.0, 4.4))

    for ax, panel in zip(axes, panel_names):
        sub = sub_fig[sub_fig["panel"] == panel]
        order = (sub.groupby("series")["value"].mean()
                 .sort_values(ascending=False).index.tolist())
        values = [sub[sub["series"] == s]["value"].to_numpy() for s in order]

        bp = ax.boxplot(values, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.6))
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(style[s]["color"])
            patch.set_hatch(style[s]["hatch"])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        for pos, (vals, s) in enumerate(zip(values, order), start=1):
            jitter = rng.uniform(-0.13, 0.13, len(vals))
            ax.plot(pos + jitter, vals, "o", ms=2.4, mew=0, color=style[s]["color"],
                    alpha=0.65, zorder=3)

        layers = sub.groupby("series")["layer"].first()
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([style[s]["label"] if str(layers[s]) == "-" else
                            f"{style[s]['label']}\n(L{layers[s]})" for s in order],
                           fontsize=8.5, rotation=45, ha="right")
        ax.set_title(panel, fontsize=12)
        ax.set_ylabel("Score per reference sample")

        panel_tests = (tests[(tests["figure"] == figure) & (tests["panel"] == panel)]
                       .rename(columns={"series_a": "model_a", "series_b": "model_b"}))
        allv = np.concatenate(values)
        spread = allv.max() - allv.min()
        _draw_brackets(ax, _significance_pairs(panel_tests, order),
                       top=allv.max() + spread * 0.04, step=spread * 0.11)

    fig.suptitle(f"{title}\ndistribution over the held-out reference samples "
                 f"(brackets: BH-adjusted significance)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"box_{figure}", kind="metric_axis")
    plt.close(fig)


def print_focus(tests: pd.DataFrame, series: pd.DataFrame):
    """Console summary: every non-P0 4-mer series against every other series."""
    focus = series.loc[series["protocol"] != "P0", "series"].tolist()
    look = tests[tests["figure"].isin(["fig4", "fig5"])
                 & (tests["series_a"].isin(focus) | tests["series_b"].isin(focus))]
    if look.empty:
        return
    table = look[["panel", "series_a", "series_b", "mean_a", "mean_b", "mean_diff",
                  "p_bh_all", "signif"]]
    print("\nNew protocols vs the rest (split-sample, BH over the whole set):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def stage2():
    series = fio.load_table(f"{FIG_DATA_NAME}/series")
    style = styles(series)
    fig4 = fio.load_table(f"{FIG_DATA_NAME}/fig4_newvirus_p5")
    fig5 = fio.load_table(f"{FIG_DATA_NAME}/fig5_cross_p5")
    fig6 = fio.load_table(f"{FIG_DATA_NAME}/fig6_pk_vs_k")
    samples = fio.load_table(f"{FIG_DATA_NAME}/panel_samples")
    tests = fio.load_table(f"{FIG_DATA_NAME}/pairwise_tests")

    plot_bars(fig4, [g for _, g in FIG4_GROUPS], style, "fig4_newvirus_p5",
              "Mean Precision@5",
              "Generalisation to unseen viruses (Precision@5): distance protocols", 11)
    plot_bars(fig5, [g for _, g in VIRUSES], style, "fig5_cross_p5",
              "Mean CROSS Precision@5",
              "Forward vs reverse-complement proximity (CROSS Precision@5)", 9)
    plot_fig6(fig6, style)
    plot_boxes(samples, tests, style, "fig4", "Generalisation to unseen orders (Fig. 4)", 2)
    plot_boxes(samples, tests, style, "fig5", "Forward vs reverse complement (Fig. 5)", 2)
    print_focus(tests, series)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--metrics_dir", default=str(fio.DATA_METRICS))
    parser.add_argument("--extra_protocols", nargs="+", default=["P4", "P5"],
                        choices=[p for p in PROTOCOL_LABEL if p != "P0"],
                        help="alternative-metric baselines to draw next to P0; each "
                             "reads the representation it needs (see PROTOCOL_MODEL)")
    parser.add_argument("--plot_only", action="store_true",
                        help="skip Stage 1, redraw from results/figure_data/metric_axis/")
    args = parser.parse_args()

    fio.set_style()
    if not args.plot_only:
        protocol_model = resolve_models(Path(args.metrics_dir))
        chosen = {p: protocol_model[p] for p in args.extra_protocols}
        print(f"series: {chosen}")
        stage1(series_table(args.extra_protocols, protocol_model), Path(args.metrics_dir))
    stage2()


if __name__ == "__main__":
    main()
