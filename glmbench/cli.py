"""The command line: one entry point, one stage per subcommand.

    prep      genomes            -> fragments and their reverse complements
    extract   fragments          -> vectors, one folder per model
    select    reference samples  -> the layer and word length of each model
    metrics   vectors            -> the metric tables
    stats     metric tables      -> paired tests between models
    figures   metric tables      -> every plot
    all       the whole chain

Stages are separate because they cost wildly different amounts: extraction needs a
GPU and hours, everything after it is minutes on a laptop.  Run ``--help`` on any
stage for its own options.
"""

import argparse
import textwrap
from pathlib import Path

import pandas as pd

from . import config, models as registry, selection, sequences
from .figures import main as main_figures
from .figures import statistics as statistics_figures
from .figures import style
from .figures import supplementary as supplementary_figures
from .metrics import runner, tables
from .representations import extract as extraction

EPILOG = textwrap.dedent("""
    examples:
      run_pipeline.py all                          every stage, every model
      run_pipeline.py extract --models kmer evo2   just those two
      run_pipeline.py select                       pick layers and word lengths
      run_pipeline.py metrics --layers selected    only the chosen layer (much faster)
      run_pipeline.py figures --which main         redraw the article figures
      run_pipeline.py metrics --dry-run            show the work without doing it
""")


# ── shared options ──────────────────────────────────────────────────────────

def _common(parser: argparse.ArgumentParser, with_models=True) -> None:
    if with_models:
        parser.add_argument("--models", nargs="+", metavar="MODEL",
                            choices=registry.ORDER, default=None,
                            help=f"models to act on (default: all of "
                                 f"{', '.join(registry.ORDER)})")
    parser.add_argument("--max-workers", type=int, default=config.MAX_WORKERS,
                        help=f"worker processes (default {config.MAX_WORKERS}); each "
                             f"holds its own copy of the data, so lower it if memory "
                             f"is tight")
    parser.add_argument("--memory-budget", type=int, default=config.MEMORY_BUDGET,
                        metavar="BYTES",
                        help="cap on one intermediate distance block "
                             f"(default {config.MEMORY_BUDGET >> 20} MB)")
    parser.add_argument("--metrics-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--latent-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true",
                        help="redo work whose output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run, compute nothing")


def _viruses(preproc_dir=None) -> dict:
    prepared = sequences.existing(preproc_dir)
    if not prepared:
        raise SystemExit(f"no prepared fragments in {config.relative(config.PREPROC_DIR)}; "
                         f"run the 'prep' stage first")
    return prepared


def _models(args) -> list:
    return registry.resolve(args.models)


def _models_with_tables(args) -> list:
    """Models that actually have metric tables; the rest are reported and skipped.

    Lets the default ``--models`` (all of them) work before every model has been
    through the pipeline, instead of failing on the first missing file.
    """
    chosen = _models(args)
    present = [model for model in chosen
               if tables.model_dir(model, args.metrics_dir).is_dir()]
    missing = [model.key for model in chosen if model not in present]
    if missing:
        print(f"  no metric tables yet, skipped: {', '.join(missing)}")
    if not present:
        raise SystemExit("no metric tables found; run the 'metrics' stage first")
    return present


# ── stages ──────────────────────────────────────────────────────────────────

def stage_prep(args) -> None:
    """Cut genomes into fragments and write their reverse complements."""
    prepared = sequences.prepare(size=args.chunk_size, max_chunks=args.max_chunks,
                                 force=args.force)
    if not prepared:
        raise SystemExit(f"no input genomes in {config.relative(config.UNSEEN_DIR)}")
    print(f"prepared {len(prepared)} orders: {', '.join(sorted(prepared))}")


def stage_extract(args) -> None:
    """Turn fragments into vectors, one folder per model."""
    extraction.run(_models(args), _viruses(), latent_dir=args.latent_dir,
                   n_ref=args.n_ref, k=args.k, force=args.force, dry_run=args.dry_run)


def stage_select(args) -> None:
    """Choose the layer and the word length of every model, on the reference set."""
    chosen = {}
    for model in _models(args):
        if model.tunes_k:
            if args.dry_run:
                print(f"  {model.key}: would sweep k over {list(model.k_values)}")
                continue
            table = selection.sweep_k(model, args.k_values, n_ref=args.n_ref,
                                      max_workers=args.max_workers)
            means = table.mean(axis=1, skipna=True)
            best = int(means.idxmax())
            path = (config.METRICS / "k_sweep" / f"{model.key}_k_sweep.xlsx")
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_excel(path)
            for k, score in means.items():
                print(f"    k={k:<3} {score:.4f}" + ("   <- chosen" if k == best else ""))
            selection.record(model, k=best, score=means[best],
                             chosen_by=f"reference P@{args.k_neighbours} sweep")
            chosen[model.key] = f"k={best}"

        elif model.layered:
            if not tables.has_benchmark(model, args.select_by, args.metrics_dir):
                print(f"  {model.key}: no {args.select_by}.xlsx benchmark, skipped")
                continue
            if args.dry_run:
                print(f"  {model.key}: would choose a layer by {args.select_by}")
                continue
            layer = selection.choose_layer(model, args.select_by, args.metrics_dir)
            score = selection.reference.layer_means(
                model, args.select_by, args.metrics_dir).iloc[layer]
            selection.record(model, layer=layer, score=score,
                             chosen_by=f"reference {args.select_by}")
            chosen[model.key] = f"layer={layer}"
        else:
            chosen[model.key] = "nothing to choose"

    if not args.dry_run:
        print(f"\nselection -> {config.relative(config.SELECTION_FILE)}")
    for key, value in chosen.items():
        print(f"  {key:<10} {value}")


def stage_metrics(args) -> None:
    """Score every model on the unseen orders."""
    prepared = _viruses()
    viruses = [(forward, reverse) for forward, reverse in prepared.values()]
    rows = runner.run(_models(args), viruses, n_ref=args.n_ref, latent_dir=args.latent_dir,
                      metrics_dir=args.metrics_dir, layers_mode=args.layers,
                      max_workers=args.max_workers, dry_run=args.dry_run)
    if rows:
        directory = Path(args.metrics_dir) if args.metrics_dir else config.METRICS
        path = directory / "summary_unseen.xlsx"
        pd.DataFrame(rows).to_excel(path, index=False)
        print(f"summary -> {config.relative(path)}  ({len(rows)} rows)")


def stage_stats(args) -> None:
    """Paired tests between models, with multiple-testing corrections."""
    if args.dry_run:
        print(f"  would test {len(_models(args))} models under "
              f"{len(statistics_figures.SELECTIONS)} selection rules")
        return
    style.set_style()
    statistics_figures.build(_models_with_tables(args), args.metrics_dir)


def stage_figures(args) -> None:
    """Draw everything, from the committed numbers when they are present."""
    chosen_models = _models(args) if args.plot_only else _models_with_tables(args)
    if args.dry_run:
        print(f"  would draw {args.which} for {len(chosen_models)} models")
        return

    style.set_style()
    wanted = {"main", "supplementary", "statistics"} if args.which == "all" else {args.which}

    if "main" in wanted:
        if not args.plot_only:
            main_figures.build_all(chosen_models, args.metrics_dir)
        main_figures.draw_all(chosen_models)
    if "supplementary" in wanted:
        if not args.plot_only:
            supplementary_figures.build_all(chosen_models, args.metrics_dir)
        supplementary_figures.draw_all(chosen_models, args.metrics_dir)
    if "statistics" in wanted:
        samples = tests = None
        if not args.plot_only:
            samples, tests = statistics_figures.build(chosen_models, args.metrics_dir)
        statistics_figures.draw_all(chosen_models, samples, tests)


def stage_all(args) -> None:
    """Every stage in order."""
    for name, stage in (("prep", stage_prep), ("extract", stage_extract),
                        ("select", stage_select), ("metrics", stage_metrics),
                        ("stats", stage_stats), ("figures", stage_figures)):
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        stage(args)


# ── parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Benchmarking genomic language models on viral genomes.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")

    prep = subparsers.add_parser("prep", help="genomes -> fragments + reverse complements",
                                 description=stage_prep.__doc__)
    prep.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE,
                      help=f"fragment length in bp (default {config.CHUNK_SIZE})")
    prep.add_argument("--max-chunks", type=int, default=config.MAX_CHUNKS,
                      help=f"fragments kept per sequence (default {config.MAX_CHUNKS})")
    _common(prep, with_models=False)
    prep.set_defaults(func=stage_prep)

    extract = subparsers.add_parser("extract", help="fragments -> vectors",
                                    description=stage_extract.__doc__)
    extract.add_argument("--n-ref", type=int, default=config.N_REFERENCE,
                         help=f"reference samples to encode (default {config.N_REFERENCE})")
    extract.add_argument("--k", type=int, default=None,
                         help="word length for the k-mer baselines; default is the one "
                              "chosen by the 'select' stage")
    _common(extract)
    extract.set_defaults(func=stage_extract)

    select = subparsers.add_parser("select", help="choose layers and word lengths",
                                   description=stage_select.__doc__)
    select.add_argument("--select-by", default=config.REFERENCE_METRIC,
                        choices=["P5", "ARI", "silh"],
                        help=f"reference metric that picks the layer "
                             f"(default {config.REFERENCE_METRIC})")
    select.add_argument("--n-ref", type=int, default=10,
                        help="reference samples used for the word-length sweep "
                             "(default 10; a sub-sample is enough to pick a parameter)")
    select.add_argument("--k-values", nargs="*", type=int, default=None,
                        help="word lengths to try (default: the model's own range)")
    select.add_argument("--k-neighbours", type=int, default=config.PK_FOR_FIGURES,
                        help=f"the K of the P@K used to score a word length "
                             f"(default {config.PK_FOR_FIGURES})")
    _common(select)
    select.set_defaults(func=stage_select)

    metrics = subparsers.add_parser("metrics", help="vectors -> metric tables",
                                    description=stage_metrics.__doc__)
    metrics.add_argument("--n-ref", type=int, default=config.N_REFERENCE,
                         help=f"reference samples to score against "
                              f"(default {config.N_REFERENCE})")
    metrics.add_argument("--layers", choices=["all", "selected"], default="all",
                         help="score every layer, or only the one chosen on the "
                              "reference set (default all; 'selected' is up to 32x "
                              "less work but leaves the per-layer figures empty)")
    _common(metrics)
    metrics.set_defaults(func=stage_metrics)

    stats = subparsers.add_parser("stats", help="metric tables -> paired tests",
                                  description=stage_stats.__doc__)
    _common(stats)
    stats.set_defaults(func=stage_stats)

    figures = subparsers.add_parser("figures", help="metric tables -> plots",
                                    description=stage_figures.__doc__)
    figures.add_argument("--which", default="all",
                         choices=["all", "main", "supplementary", "statistics"],
                         help="which family of figures to draw (default all)")
    figures.add_argument("--plot-only", action="store_true",
                         help="skip rebuilding the CSVs and redraw from the committed ones")
    _common(figures)
    figures.set_defaults(func=stage_figures)

    everything = subparsers.add_parser("all", help="run every stage in order",
                                       description=stage_all.__doc__)
    everything.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    everything.add_argument("--max-chunks", type=int, default=config.MAX_CHUNKS)
    everything.add_argument("--n-ref", type=int, default=config.N_REFERENCE)
    everything.add_argument("--k", type=int, default=None)
    everything.add_argument("--k-values", nargs="*", type=int, default=None)
    everything.add_argument("--k-neighbours", type=int, default=config.PK_FOR_FIGURES)
    everything.add_argument("--select-by", default=config.REFERENCE_METRIC,
                            choices=["P5", "ARI", "silh"])
    everything.add_argument("--layers", choices=["all", "selected"], default="all")
    everything.add_argument("--which", default="all",
                            choices=["all", "main", "supplementary", "statistics"])
    everything.add_argument("--plot-only", action="store_true")
    _common(everything)
    everything.set_defaults(func=stage_all)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config.MEMORY_BUDGET = args.memory_budget
    config.MAX_WORKERS = args.max_workers
    config.ensure_dirs(config.METRICS, config.FIGURE_DATA, config.STATISTICS)
    args.func(args)
