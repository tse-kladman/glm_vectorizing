"""Fig. 1, 2, 4-7 of the article and the k-mer sweep (Fig. S1).

Each figure has a ``build_*`` that turns metric tables into the CSV behind it and a
``draw_*`` that renders that CSV.  The scaling/distance comparison (Fig. 8) lives in
``supplementary.py`` next to the other sensitivity checks.

Numbering.  The repository numbers its figures in the order the pipeline draws them;
the mapping to the article's numbering is the table in ``results/README.md``.  The
former Fig. 3 (k-mer length sweep) is now Fig. S1 (``build_fig3`` / ``draw_fig3``
keep their names).
Models without the tables a figure needs (MinHash and Jensen-Shannon have no
reference benchmark yet, so no Fig. 1/2 bars) are left out of that figure only, and
the figure says so.

Terminology.  Two unrelated things were both called "CROSS" in earlier versions:
*RC proximity* (Fig. 5: how close a forward fragment lies to the reverse-complement
fragments of its own genome; higher is better) and *inter-order leakage* (Fig. 7: how
often a fragment of one order retrieves a fragment of the other; lower is better).
Column names of the committed CSVs keep the old names; titles and axes use these.

Where the layer comes from.  Fig. 1 and 2 *are* the reference benchmark, so they
show the best layer of that benchmark itself.  Fig. 4, 5 and 6 report every model
at the layer the benchmark picked (selection.py) -- never at the layer that happens
to score best on the unseen order -- so one layer per model is used across all of
them, and nothing is tuned on the numbers being reported.  Fig. 7 shows every layer
at once, so no choice arises.

Layer numbers.  Every ``layer`` / ``best_layer`` written here is the layer *label*
(``reference.layer_label``), the number the extracted files use: DNABERT-2/S 0 =
embedding output and 1..12 the encoder blocks, ViraLM its blocks 1..12, Evo2 blocks
0..31.  The benchmark tables are indexed by position, which for ViraLM is one less;
per-sample values are always looked up by position, labels are only shown.
"""

import numpy as np
import pandas as pd

from .. import config, selection
from ..metrics import reference, tables
from ..models import Model
from . import significance
from .style import color, display, load_table, save_figure, save_table

VIRUSES = [("Egovirales_preproc", "Egovirales"),
           ("Mriyavirales_preproc", "Mriyavirales")]
FIG4_GROUPS = VIRUSES + [("REVERSECOMPLEMENTEgovirales_preproc", "Egovirales (RC)"),
                         ("REVERSECOMPLEMENTMriyavirales_preproc", "Mriyavirales (RC)")]
PAIRWISE_DIR = "pairwise_Egovirales_vs_Mriyavirales"


def _has_table(model: Model, filename: str, metrics_dir=None, quiet=False) -> bool:
    if (tables.model_dir(model, metrics_dir) / filename).exists():
        return True
    if not quiet:
        print(f"  {model.key}: no {filename}, left out of this figure")
    return False


def _reference_row(model: Model, filename: str, metrics_dir=None):
    """Per-sample values of one metric table at the reference-chosen layer."""
    matrix = tables.load_model_matrix(model, filename, metrics_dir)
    return selection.row_of(model, matrix, metrics_dir=metrics_dir)


# ── Fig. 1: reference benchmark, best layer of each model ───────────────────

FIG1_METRICS = ("ARI", "P5")        # the silhouette is supplementary only
BASELINE = "kmer"                    # every star on a bar is "this model against 4-mer"


def build_fig1(models, metrics_dir=None) -> pd.DataFrame:
    """Best layer of each model per metric, and its test against the 4-mer profile.

    The per-sample values at the best layer are compared with the 4-mer values on the
    same reference samples (aligned by sample number); one BH family over the whole
    figure.  The best layer is chosen on these same samples (as the bar shows it), so
    the mark is the in-sample comparison; the split-sample version is in
    ``figures/statistics`` (box_fig1).
    """
    models = reference.available(models, metrics_dir=metrics_dir)
    frame = pd.concat([reference.summary(models, metric, metrics_dir)
                       for metric in FIG1_METRICS], ignore_index=True)

    def values(key, metric):
        """Per-sample values at the best layer, looked up by POSITION (the
        ``best_layer`` column holds the label, which differs for ViraLM)."""
        table = reference.load(key, metric, metrics_dir)
        return table.iloc[reference.best_position(key, metric, metrics_dir)]

    raw = {}
    for metric in FIG1_METRICS:
        keys = list(frame.loc[frame["metric"] == metric, "model"])
        if BASELINE not in keys:
            continue
        base = values(BASELINE, metric)
        for key in keys:
            if key != BASELINE:
                raw[(metric, key)] = significance.paired_p(values(key, metric), base)
    adjusted = significance.adjust(raw)
    frame["p_vs_kmer_bh"] = [adjusted.get((m, k), np.nan)
                             for m, k in zip(frame["metric"], frame["model"])]
    save_table(frame, "fig1_reference_benchmark")
    return frame


def draw_fig1(models):
    import matplotlib.pyplot as plt

    frame = load_table("fig1_reference_benchmark")
    frame = frame[frame["metric"].isin(FIG1_METRICS)]
    requested = list(models)
    models = [model for model in models if model.key in set(frame["model"])]
    keys = [model.key for model in models]
    panels = [(metric, reference.METRIC_TITLES[metric]) for metric in FIG1_METRICS]

    figure, axes = plt.subplots(1, len(panels), figsize=(9.5, 4.4))
    for axis, (metric, title) in zip(axes, panels):
        data = frame[frame["metric"] == metric].set_index("model").reindex(keys)
        axis.bar(range(len(keys)), data["mean"], yerr=data["std"], capsize=4,
                 color=[color(key) for key in keys], edgecolor="black", linewidth=0.5)
        pad = 0.02 * float(np.nanmax(data["mean"] + data["std"]))
        for index, key in enumerate(keys):
            if "p_vs_kmer_bh" in data:
                significance.put(axis, index, data.loc[key, "mean"] + data.loc[key, "std"] + pad,
                                 significance.label(data.loc[key, "p_vs_kmer_bh"]))
        labels = [display(key) if np.isnan(data.loc[key, "best_layer"]) or
                  not models[index].layered else
                  f"{display(key)}\n(L{int(data.loc[key, 'best_layer'])})"
                  for index, key in enumerate(keys)]
        axis.set_xticks(range(len(keys)))
        axis.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.margins(y=0.18)
    figure.suptitle("Reference benchmark: best layer of each model", fontsize=15, y=1.04)
    absent = [display(model.key) for model in requested if model.key not in keys]
    if absent:
        figure.text(0.5, -0.04, "no reference benchmark computed for: " + ", ".join(absent),
                    ha="center", fontsize=8, color="0.4")
    figure.tight_layout()
    save_figure(figure, "fig1_reference_benchmark", kind="main")
    plt.close(figure)


# ── Fig. 2: layer-wise ARI and P@5 ──────────────────────────────────────────

def build_fig2(models, metrics_dir=None) -> pd.DataFrame:
    """Every layer of every model, with the test of each layer against the model's best
    layer (same reference samples; one BH family over the whole figure)."""
    models = reference.available(models, metrics_dir=metrics_dir)
    frame = pd.concat([reference.layerwise(models, metric, metrics_dir)
                       for metric in ("ARI", "P5")], ignore_index=True)
    raw = {}
    for model in models:
        if not model.layered:
            continue
        for metric in ("ARI", "P5"):
            table = reference.load(model, metric, metrics_dir)
            table.index = [reference.layer_label(model, position)
                           for position in range(len(table))]
            _, against = significance.against_best(table)
            raw.update({(model.key, metric, layer): p for layer, p in against.items()})
    adjusted = significance.adjust(raw)
    frame["p_vs_best_bh"] = [adjusted.get((m, t, int(l)), np.nan)
                             for m, t, l in zip(frame["model"], frame["metric"], frame["layer"])]
    save_table(frame, "fig2_layerwise")
    return frame


def draw_fig2(models):
    """Filled marker: the layer is not significantly worse than the best layer of the
    model (or is the best); open marker: significantly worse."""
    import matplotlib.pyplot as plt

    frame = load_table("fig2_layerwise")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, (metric, title) in zip(axes, [("ARI", "Adjusted Rand Index"),
                                            ("P5", "Precision@5")]):
        for model in models:
            data = frame[(frame["metric"] == metric)
                         & (frame["model"] == model.key)].sort_values("layer")
            if data.empty:
                continue
            if not model.layered:
                axis.axhline(data["mean"].iloc[0], ls=":", lw=2,
                             color=model.color, label=model.display)
                continue
            axis.plot(data["layer"], data["mean"], "-", color=model.color, label=model.display)
            axis.fill_between(data["layer"], data["mean"] - data["std"],
                              data["mean"] + data["std"], color=model.color, alpha=0.12)
            worse = (data["p_vs_best_bh"] < 0.05 if "p_vs_best_bh" in data
                     else pd.Series(False, index=data.index))
            axis.plot(data["layer"][~worse], data["mean"][~worse], "o", ms=5,
                      color=model.color)
            axis.plot(data["layer"][worse], data["mean"][worse], "o", ms=5,
                      markerfacecolor="white", markeredgecolor=model.color)
        axis.set_xlabel("Layer")
        axis.set_ylabel(title)
        axis.set_title(title)
    axes[0].legend(loc="lower center", ncol=4, fontsize=8)
    figure.suptitle("Clustering quality across transformer layers", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "fig2_layerwise", kind="main")
    plt.close(figure)


# ── Fig. 3 (supplementary): k-mer length sweep of the classical baselines ────

def _sweep_tables(metrics_dir=None) -> dict:
    """{(method, metric): table rows = k, columns = reference samples}."""
    base = metrics_dir or config.METRICS
    out = {}
    path = base / "kmer_sweep" / "kmer_sweep_results.xlsx"
    if path.exists():
        raw = pd.read_excel(path, sheet_name="raw")
        for metric, column in (("ARI", "ARI"), ("P5", "P@5")):
            out[("kmer", metric)] = raw.pivot(index="k", columns="sample", values=column)
    for key in ("kmer_js", "minhash"):
        path = base / "k_sweep" / f"{key}_k_sweep.xlsx"
        if path.exists():
            table = pd.read_excel(path).set_index("k")
            table.columns = [int(str(column).split("_")[1]) for column in table.columns]
            out[(key, "P5")] = table.sort_index(axis=1)
    return out


def build_fig3(metrics_dir=None) -> pd.DataFrame | None:
    """Mean and spread per k, and the test of every k against the best k of that
    method and metric (paired over the reference samples; one BH family)."""
    sweeps = _sweep_tables(metrics_dir)
    if not sweeps:
        print("  no k-mer sweep tables found -> keeping the committed CSV")
        return None
    raw, best = {}, {}
    for (method, metric), table in sweeps.items():
        best[(method, metric)], against = significance.against_best(table)
        raw.update({(method, metric, k): p for k, p in against.items()})
    adjusted = significance.adjust(raw)
    rows = [{"method": method, "metric": metric, "k": k,
             "mean": float(table.loc[k].mean()), "std": float(table.loc[k].std(ddof=0)),
             "n": int(table.loc[k].notna().sum()), "best": k == best[(method, metric)],
             "p_vs_best_bh": adjusted.get((method, metric, k), np.nan)}
            for (method, metric), table in sweeps.items() for k in table.index]
    frame = pd.DataFrame(rows)
    save_table(frame, "supplementary/figS1_kmer_sweep")
    return frame


def draw_fig3():
    """(a) ARI and P@5 of the k-mer profile (cosine distance; the baseline uses k = 4)
    against k; (b) P@5 of the k-mer profile with cosine and with Jensen-Shannon, and of
    MinHash, against k.  Open markers: k significantly worse than the best k of that
    curve."""
    import matplotlib.pyplot as plt

    path = config.FIGURE_DATA / "supplementary" / "figS1_kmer_sweep.csv"
    if not path.exists():
        print("  figS1: no k-mer sweep summary, not drawn")
        return
    frame = pd.read_csv(path)
    curves = [
        (0, "kmer", "ARI", "Adjusted Rand Index", "#0072B2", "-"),
        (0, "kmer", "P5", "Precision@5", "#D55E00", "-"),
        (1, "kmer", "P5", "k-mer profile, cosine", color("kmer"), "--"),
        (1, "kmer_js", "P5", "k-mer profile, Jensen-Shannon", color("kmer_js"), "-"),
        (1, "minhash", "P5", "MinHash", color("minhash"), "-"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for panel, method, metric, label, line_color, style in curves:
        data = frame[(frame["method"] == method) & (frame["metric"] == metric)].sort_values("k")
        if data.empty:
            continue
        axis = axes[panel]
        axis.errorbar(data["k"], data["mean"], yerr=data["std"], ls=style, capsize=3,
                      color=line_color, label=label)
        worse = data["p_vs_best_bh"] < 0.05
        axis.plot(data["k"][~worse], data["mean"][~worse], "o", color=line_color)
        axis.plot(data["k"][worse], data["mean"][worse], "o",
                  markerfacecolor="white", markeredgecolor=line_color)
    axes[0].set_title("(a) k-mer profile, cosine distance")
    axes[0].set_ylabel("Score")
    axes[1].set_title("(b) Precision@5 of the classical baselines")
    axes[1].set_ylabel("Precision@5")
    for axis in axes:
        axis.set_xlabel("k (k-mer length)")
        axis.legend(fontsize=8)
    figure.tight_layout()
    save_figure(figure, "figS1_kmer_sweep", kind="supplementary")
    plt.close(figure)


# ── Fig. 4 and 5: the unseen orders ─────────────────────────────────────────

def _unseen_table(models, groups, prefix, name, metrics_dir=None, filename=None,
                  optional=False, panel=None) -> pd.DataFrame:
    """Mean and spread over the reference samples of one table, at the reference layer.

    ``optional`` tables (the order-level and own-RC metrics, which only exist after a
    re-run) skip models that lack the file and are not written at all if none has it.
    A model that lacks a required table (e.g. a registry model scored with only part
    of the protocols) is left out of the figure and named, as in the statistics.
    ``panel`` maps a group label to its panel name in the study-wide test table; when
    given, the BH p of every model against the 4-mer profile is added.
    """
    rows = []
    for virus, label in groups:
        for model in models:
            table = filename(virus) if filename else tables.matrix_name(
                prefix, config.PK_FOR_FIGURES, virus)
            if not _has_table(model, table, metrics_dir, quiet=optional):
                continue
            layer, values = _reference_row(model, table, metrics_dir)
            rows.append({"group": label, "model": model.key, "layer": layer,
                         "mean": float(values.mean()), "std": float(values.std(ddof=0))})
    frame = pd.DataFrame(rows)
    if optional and frame.empty:
        return frame
    if panel:
        tests = significance.study_tests()
        frame["p_vs_kmer_bh"] = [
            np.nan if key == BASELINE else
            significance.study_p(tests, panel(group), key, BASELINE)
            for group, key in zip(frame["group"], frame["model"])]
    save_table(frame, name)
    return frame


def build_fig4(models, metrics_dir=None) -> pd.DataFrame:
    return _unseen_table(models, FIG4_GROUPS, "P", "fig4_newvirus_p5", metrics_dir,
                         panel=lambda group: f"P@5 {group}")


def build_fig5(models, metrics_dir=None) -> pd.DataFrame:
    return _unseen_table(models, VIRUSES, "cross_P", "fig5_cross_p5", metrics_dir,
                         panel=lambda group: f"RC-proximity P@5 {group}")


def build_fig4b(models, metrics_dir=None) -> pd.DataFrame:
    """Order-level P@5 (leave-genome-out): success = another genome of the same order."""
    return _unseen_table(models, FIG4_GROUPS, "Porder", "fig4b_order_p5", metrics_dir,
                         optional=True)


def build_fig5b(models, metrics_dir=None) -> pd.DataFrame:
    """Own-RC hit@5: the fragment's own reverse complement is among its 5 neighbours."""
    return _unseen_table(models, VIRUSES, None, "fig5b_own_rc_hit5", metrics_dir,
                         filename=lambda virus: tables.own_rc_names(config.PK_FOR_FIGURES, virus),
                         optional=True)


def build_fig5c(models, metrics_dir=None) -> pd.DataFrame:
    """Mean similarity between a fragment and its own reverse complement."""
    return _unseen_table(models, VIRUSES, None, "fig5c_own_rc_similarity", metrics_dir,
                         filename=tables.own_rc_similarity_name, optional=True)


def _grouped_bars(frame, models, order, ylabel, title, name):
    import matplotlib.pyplot as plt

    models = [model for model in models if model.key in set(frame["model"])]

    figure, axis = plt.subplots(figsize=(10 if len(order) > 2 else 8, 4.8))
    width, positions = 0.8 / len(models), np.arange(len(order))
    for index, model in enumerate(models):
        data = frame[frame["model"] == model.key].set_index("group").reindex(order)
        x = positions + (index - (len(models) - 1) / 2) * width
        axis.bar(x, data["mean"], width,
                 yerr=data["std"], capsize=2, color=model.color, label=model.display,
                 edgecolor="black", linewidth=0.3)
        if "p_vs_kmer_bh" in data:
            for position, (_, row) in zip(x, data.iterrows()):
                significance.put(axis, position, row["mean"] + row["std"] + 0.01,
                                 significance.label(row["p_vs_kmer_bh"]), fontsize=7.5)
    axis.set_xticks(positions)
    axis.set_xticklabels(order)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.margins(y=0.12)
    axis.legend(ncol=min(len(models), 4), fontsize=9, loc="upper center",
                bbox_to_anchor=(0.5, -0.12))
    figure.tight_layout()
    save_figure(figure, name, kind="main")
    plt.close(figure)


def draw_fig4(models):
    frame = load_table("fig4_newvirus_p5")
    _grouped_bars(frame, models, [label for _, label in FIG4_GROUPS],
                  "Mean Precision@5", "Held-out viruses (Precision@5)",
                  "fig4_newvirus_p5")


def draw_fig5(models):
    frame = load_table("fig5_cross_p5")
    _grouped_bars(frame, models, [label for _, label in VIRUSES],
                  "RC proximity (Precision@5)",
                  "RC proximity: share of the 5 nearest neighbors that are reverse-complement\n"
                  "fragments of the SAME genome (not necessarily the fragment's own RC)",
                  "fig5_cross_p5")


def _draw_optional(name, groups, models, ylabel, title):
    path = config.FIGURE_DATA / f"{name}.csv"
    if path.exists():
        _grouped_bars(load_table(name), models, [label for _, label in groups],
                      ylabel, title, name)


def draw_fig4b(models):
    _draw_optional("fig4b_order_p5", FIG4_GROUPS, models, "Mean order-level Precision@5",
                   "Order-level P@5, leave-genome-out: neighbors from ANOTHER genome of\n"
                   "the same unseen order (own genome excluded)")


def draw_fig5b(models):
    _draw_optional("fig5b_own_rc_hit5", VIRUSES, models, "Mean own-RC hit@5",
                   "Own reverse complement among the 5 nearest neighbors")


def draw_fig5c(models):
    _draw_optional("fig5c_own_rc_similarity", VIRUSES, models,
                   "Mean similarity of x and RC(x)",
                   "Similarity of a fragment and its reverse complement\n"
                   "(cosine / sketch agreement / 1 - JSD, by model)")


# ── Fig. 6: P@k against k ───────────────────────────────────────────────────

def build_fig6(models, metrics_dir=None) -> pd.DataFrame:
    rows = []
    for virus, label in VIRUSES:
        for model in models:
            for k in config.KS:
                name = tables.matrix_name("P", k, virus)
                if not _has_table(model, name, metrics_dir):
                    continue
                layer, values = _reference_row(model, name, metrics_dir)
                rows.append({"virus": label, "model": model.key, "k": k,
                             "layer": layer, "mean": float(values.mean())})
    frame = pd.DataFrame(rows)
    # at every k: is the leading model significantly ahead of the second one?
    tests = significance.study_tests()
    frame["leader_vs_second_bh"] = np.nan
    for (virus, k), group in frame.groupby(["virus", "k"]):
        ranked = group.sort_values("mean", ascending=False)
        if len(ranked) < 2:
            continue
        first, second = ranked["model"].iloc[0], ranked["model"].iloc[1]
        frame.loc[ranked.index[0], "leader_vs_second_bh"] = significance.study_p(
            tests, f"P@{k} {virus}", first, second)
    save_table(frame, "fig6_pk_vs_k")
    return frame


def draw_fig6(models):
    import matplotlib.pyplot as plt

    frame = load_table("fig6_pk_vs_k")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, (_, label) in zip(axes, VIRUSES):
        for model in models:
            data = frame[(frame["virus"] == label)
                         & (frame["model"] == model.key)].sort_values("k")
            if data.empty:
                continue
            axis.plot(data["k"], data["mean"], "-o", ms=4,
                      color=model.color, label=model.display)
        if "leader_vs_second_bh" in frame:
            leaders = frame[(frame["virus"] == label) & frame["leader_vs_second_bh"].notna()]
            for _, row in leaders.iterrows():
                significance.put(axis, row["k"], row["mean"] + 0.015,
                                 significance.label(row["leader_vs_second_bh"]), fontsize=8)
        axis.set_xlabel("k (number of neighbors)")
        axis.set_title(label)
        axis.set_xticks(list(config.KS))
    axes[0].set_ylabel("Precision@k")
    axes[0].legend(fontsize=8, ncol=2)
    figure.suptitle("Precision@k as a function of k (held-out viruses)", fontsize=15, y=1.02)
    figure.tight_layout()
    save_figure(figure, "fig6_pk_vs_k", kind="main")
    plt.close(figure)


# ── Fig. 7: compactness against separation, per layer ───────────────────────

def build_fig7(models, metrics_dir=None) -> pd.DataFrame:
    rows = []
    for model in models:
        directory = tables.model_dir(model, metrics_dir) / PAIRWISE_DIR
        if not directory.is_dir():
            continue

        def layer_mean(filename):
            return pd.read_excel(directory / filename).set_index("layer").mean(axis=1)

        own = (layer_mean("pk_fwd_1_in_2_matrix.xlsx")
               + layer_mean("pk_fwd_2_in_1_matrix.xlsx")) / 2
        cross = layer_mean("cross_1→2_fwd_fwd_matrix.xlsx")
        for layer in own.index:
            rows.append({"model": model.key, "layer": layer,
                         "p_at_10": float(own[layer]),
                         "cross_p_at_10": float(cross[layer])})
    frame = pd.DataFrame(rows)
    save_table(frame, "fig7_pk_vs_cross")
    return frame


def draw_fig7(models):
    import matplotlib.pyplot as plt

    frame = load_table("fig7_pk_vs_cross")
    figure, axis = plt.subplots(figsize=(9, 6))
    flat_markers = iter(["*", "P", "X", "D", "s", "^"])
    for model in models:
        data = frame[frame["model"] == model.key]
        if data.empty:
            continue
        flat = not model.layered
        axis.scatter(data["cross_p_at_10"], data["p_at_10"],
                     marker=next(flat_markers) if flat else "o", s=200 if flat else 55,
                     color=model.color, label=model.display,
                     edgecolor="white", linewidth=0.6, zorder=3)
        for _, row in data.iterrows():
            layer = "".join(char for char in str(row["layer"]) if char.isdigit())
            if layer:
                axis.annotate(layer, (row["cross_p_at_10"], row["p_at_10"]),
                              fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Inter-order leakage: P@10 of Egovirales fragments retrieving "
                    "Mriyavirales (lower = better separation)")
    axis.set_ylabel("Precision@10 within the own order (higher = denser cluster)")
    axis.set_title("Egovirales vs Mriyavirales: cluster compactness vs separation, per layer")
    axis.legend(title="Model")
    figure.tight_layout()
    save_figure(figure, "fig7_pk_vs_cross", kind="main")
    plt.close(figure)


# ── orchestration ───────────────────────────────────────────────────────────

BUILDERS = {
    "fig1": build_fig1, "fig2": build_fig2, "fig4": build_fig4,
    "fig5": build_fig5, "fig6": build_fig6, "fig7": build_fig7,
    "fig4b": build_fig4b, "fig5b": build_fig5b, "fig5c": build_fig5c,
}


def build_all(models, metrics_dir=None, which=None) -> None:
    wanted = set(which or BUILDERS)
    for name, builder in BUILDERS.items():
        if name in wanted:
            builder(models, metrics_dir)
    if "fig3" in wanted or which is None:
        build_fig3(metrics_dir)


def draw_all(models, which=None) -> None:
    wanted = set(which or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7",
                           "fig4b", "fig5b", "fig5c"])
    if "fig1" in wanted:
        draw_fig1(models)
    if "fig2" in wanted:
        draw_fig2(models)
    if "fig3" in wanted:
        draw_fig3()
    if "fig4" in wanted:
        draw_fig4(models)
    if "fig5" in wanted:
        draw_fig5(models)
    if "fig6" in wanted:
        draw_fig6(models)
    if "fig7" in wanted:
        draw_fig7(models)
    if "fig4b" in wanted:
        draw_fig4b(models)
    if "fig5b" in wanted:
        draw_fig5b(models)
    if "fig5c" in wanted:
        draw_fig5c(models)
