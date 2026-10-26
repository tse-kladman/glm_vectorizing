"""The command line: one entry point, one stage per subcommand.

    prep       genomes            -> fragments and their reverse complements
    samples    VMR + genomes      -> the 50 reference samples
    sweep      reference samples  -> ARI / P@5 / silhouette of the k-mer profile per k
    extract    fragments          -> vectors, one folder per model
    benchmark  reference samples  -> ARI / P@5 / silhouette per layer
    select     reference samples  -> the layer and word length of each model
    metrics    vectors            -> the metric tables (and, with --scaling-distance,
                                     the same under other scaling / distances)
    stats      metric tables      -> paired tests between models
    figures    metric tables      -> every plot
    all        the whole chain

Stages are separate because they cost wildly different amounts: extraction needs a
GPU and hours, everything after it is minutes on a laptop.  Run ``--help`` on any
stage for its own options.  Every stage first records what it is about to do in
``results/run_manifest.json`` (provenance.py).
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

from . import config, models as registry, provenance, selection, sequences
from .figures import main as main_figures
from .figures import statistics as statistics_figures
from .figures import style
from .figures import supplementary as supplementary_figures
from .metrics import reference, runner, tables
from .representations import extract as extraction
from .representations import pooling as pooling_module
from .representations import store

EPILOG = textwrap.dedent("""
    examples:
      run_pipeline.py all --fasta-dir genomes/     every stage, every model
      run_pipeline.py samples --list-classes       which VMR classes qualify (no genomes needed)
      run_pipeline.py extract --models kmer evo2   just those two
      run_pipeline.py benchmark --models kmer      ARI / P@5 / silhouette on the reference samples
      run_pipeline.py select                       pick layers and word lengths
      run_pipeline.py metrics --scaling-distance --layers selected
                                                   the tables of Fig. 8
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
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"the seed every random choice is derived from "
                             f"(default {config.SEED}); recorded in the run manifest")
    parser.add_argument("--max-workers", type=int, default=config.MAX_WORKERS,
                        help=f"worker processes (default {config.MAX_WORKERS}); each "
                             f"holds its own copy of the data, so lower it if memory "
                             f"is tight")
    parser.add_argument("--sample-threads", type=int, default=config.SAMPLE_THREADS,
                        help=f"reference samples held in memory at once per task "
                             f"(default {config.SAMPLE_THREADS}; one Evo2 sample is ~2 GB)")
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


def _prep_options(parser) -> None:
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE,
                        help=f"fragment length in bp (default {config.CHUNK_SIZE})")
    parser.add_argument("--max-chunks", type=int, default=config.MAX_CHUNKS,
                        help=f"fragments kept per sequence, chosen at random with --seed "
                             f"(default {config.MAX_CHUNKS})")
    parser.add_argument("--min-length", type=int, default=None,
                        help="shortest fragment kept (default: the chunk size, i.e. the "
                             "incomplete tail of a genome is dropped)")


def _samples_options(parser) -> None:
    parser.add_argument("--vmr", default=None, help="ICTV VMR spreadsheet "
                        "(default: the one in data/ictv_data)")
    parser.add_argument("--fasta-dir", default=None,
                        help="directory of FASTA files whose record ids start with the "
                             "GenBank accession of the VMR isolates")
    parser.add_argument("--accession-map", default=None,
                        help="TSV 'accession<TAB>fasta file' for genomes kept elsewhere")
    parser.add_argument("--n-genomes", type=int, default=config.REFERENCE_GENOMES_PER_CLASS,
                        help="genomes drawn per class (default %(default)s)")
    parser.add_argument("--n-chunks", type=int, default=config.REFERENCE_CHUNKS_PER_GENOME,
                        help="fragments drawn per genome (default %(default)s)")
    parser.add_argument("--include-additional", action="store_true",
                        help="also draw from non-exemplar isolates (default: exemplars only)")
    parser.add_argument("--expected-classes", type=int, default=39,
                        help="number of qualifying classes the article reports; a "
                             "different count prints a warning (default %(default)s)")
    parser.add_argument("--list-classes", action="store_true",
                        help="only print which classes qualify, build nothing")


def _viruses(preproc_dir=None) -> dict:
    prepared = sequences.existing(preproc_dir)
    if not prepared:
        raise SystemExit(f"no prepared fragments in {config.relative(config.PREPROC_DIR)}; "
                         f"run the 'prep' stage first")
    return prepared


def _models(args) -> list:
    return registry.resolve(args.models)


def _models_with_vectors(args) -> list:
    """The requested models; when none were named, only those that have been extracted."""
    chosen = _models(args)
    if args.models:
        return chosen
    present = [model for model in chosen if store.available(model, args.latent_dir)]
    skipped = [model.key for model in chosen if model not in present]
    if skipped:
        print(f"  no extracted vectors yet, skipped: {', '.join(skipped)}")
    if not present:
        raise SystemExit("no extracted vectors found; run the 'extract' stage first")
    return present


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


def _models_in_figure_data(args) -> list:
    """For ``--plot-only``: the requested models that the committed CSVs actually hold."""
    chosen = _models(args)
    path = config.FIGURE_DATA / "fig4_newvirus_p5.csv"
    if not path.exists():
        return chosen
    present = set(pd.read_csv(path)["model"])
    kept = [model for model in chosen if model.key in present]
    dropped = [model.key for model in chosen if model not in kept]
    if dropped:
        print(f"  not in the committed figure data, skipped: {', '.join(dropped)}")
    return kept


# ── stages ──────────────────────────────────────────────────────────────────

def stage_prep(args) -> None:
    """Cut genomes into fragments and write their reverse complements."""
    prepared = sequences.prepare(size=args.chunk_size, max_chunks=args.max_chunks,
                                 force=args.force, seed=args.seed,
                                 min_length=args.min_length)
    if not prepared:
        raise SystemExit(f"no input genomes in {config.relative(config.UNSEEN_DIR)}")
    print(f"prepared {len(prepared)} orders: {', '.join(sorted(prepared))}")


def stage_samples(args) -> None:
    """Build the reference samples from the ICTV VMR and a directory of genomes."""
    from . import samples

    vmr = samples.load_vmr(args.vmr)
    if args.list_classes:
        samples.report_classes(vmr, args.expected_classes)
        return

    wanted = [config.REFERENCE_DIR / f"sample_{i}.fa" for i in range(args.n_ref)]
    if not args.force and all(path.exists() for path in wanted):
        print(f"all {args.n_ref} reference samples already exist in "
              f"{config.relative(config.REFERENCE_DIR)}; use --force to redraw them")
        return
    if not (args.fasta_dir or args.accession_map):
        raise SystemExit("building the reference samples needs the genomes: pass "
                         "--fasta-dir (and/or --accession-map). Run with --list-classes "
                         "to see which classes qualify without them.")
    if args.dry_run:
        samples.report_classes(vmr, args.expected_classes)
        print(f"  would draw {args.n_ref} samples with seeds {args.seed}..{args.seed + args.n_ref - 1}")
        return
    index = samples.FastaIndex.from_dir(args.fasta_dir, args.accession_map)
    print(f"  {len(index)} accessions found in the FASTA files")
    samples.build_samples(vmr, index, n_ref=args.n_ref, seed=args.seed,
                          n_genomes=args.n_genomes, n_chunks=args.n_chunks,
                          include_additional=args.include_additional, force=args.force,
                          expected_classes=args.expected_classes)


def stage_extract(args) -> None:
    """Turn fragments into vectors, one folder per model."""
    extraction.run(_models(args), _viruses(), latent_dir=args.latent_dir,
                   n_ref=args.n_ref, k=args.k, force=args.force, dry_run=args.dry_run,
                   pooling=args.pooling, seed=args.seed)


def stage_benchmark(args) -> None:
    """ARI, P@5 and silhouette of every layer on every reference sample."""
    for model in _models_with_vectors(args):
        if args.dry_run:
            print(f"  {model.key}: would benchmark {args.n_ref} reference samples")
            continue
        existing = all(tables.has_benchmark(model, metric, args.metrics_dir)
                       for metric in reference.METRICS)
        if existing and not args.force:
            print(f"  {model.key}: benchmark tables exist (use --force to recompute)")
            continue
        print(f"  {model.key}: benchmarking {args.n_ref} reference samples")
        reference.compute(model, n_ref=args.n_ref, latent_dir=args.latent_dir,
                          metrics_dir=args.metrics_dir, seed=args.seed,
                          allow_missing=args.allow_missing, max_workers=args.sample_threads)


def stage_select(args) -> None:
    """Choose the layer and the word length of every model, on the reference set."""
    chosen = {}
    for model in _models(args):
        if model.tunes_k:
            if args.dry_run:
                print(f"  {model.key}: would sweep k over {list(model.k_values)}")
                continue
            if not (config.REFERENCE_DIR / "sample_0.fa").exists():
                print(f"  {model.key}: no reference samples in "
                      f"{config.relative(config.REFERENCE_DIR)}, word length not swept")
                continue
            table = selection.sweep_k(model, args.k_values, n_ref=args.sweep_samples,
                                      k_neighbours=args.k_neighbours,
                                      max_workers=args.max_workers, seed=args.seed)
            means = table.mean(axis=1, skipna=True)
            best = int(means.idxmax())
            path = tables.k_sweep_path(model, args.metrics_dir)
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


def stage_sweep(args) -> None:
    """ARI, P@5 and silhouette of the k-mer profile at every word length (Fig. S1a)."""
    model = registry.get("kmer")
    k_values = getattr(args, "sweep_k_values", None) or list(range(1, 11))
    if args.dry_run:
        print(f"  {model.key}: would benchmark k = {k_values} on {args.n_ref} "
              f"reference samples")
        return
    path = tables.kmer_sweep_path(args.metrics_dir)
    if path.exists() and not args.force:
        print(f"  {config.relative(path)} exists (use --force to recompute)")
        return
    summary, raw = selection.sweep_benchmark(model, k_values, n_ref=args.n_ref,
                                             max_workers=args.max_workers, seed=args.seed)
    selection.save_sweep_benchmark(summary, raw, path)
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"sweep -> {config.relative(path)}")


def stage_metrics(args) -> None:
    """Score every model on the unseen orders."""
    prepared = _viruses()
    viruses = [(forward, reverse) for forward, reverse in prepared.values()]
    models = _models_with_vectors(args)
    variants = [None]
    if args.scaling_distance:
        # Fig. 8 reads the P@k of the models compared by standardised cosine; only
        # those have the variants, and no pairwise tables are needed.
        skipped = [model.key for model in models if model.metric != "cosine"]
        if skipped:
            print(f"  no scaling/distance variants for: {', '.join(skipped)}")
        models = [model for model in models if model.metric == "cosine"]
        variants = list(registry.SCALING_DISTANCE)

    for variant in variants:
        metrics_dir = (tables.scaling_distance_dir(variant, args.metrics_dir) if variant
                       else args.metrics_dir)
        rows = runner.run(models, viruses, n_ref=args.n_ref,
                          latent_dir=args.latent_dir, metrics_dir=metrics_dir,
                          layers_mode=args.layers, max_workers=args.max_workers,
                          dry_run=args.dry_run, allow_missing=args.allow_missing,
                          with_pairwise=not (args.no_pairwise or variant),
                          variant=variant)
        if rows:
            path = Path(metrics_dir or config.METRICS) / "summary_unseen.xlsx"
            pd.DataFrame(rows).to_excel(path, index=False)
            print(f"summary -> {config.relative(path)}  ({len(rows)} rows)")


def stage_stats(args) -> None:
    """Paired tests between models, with multiple-testing corrections."""
    if args.dry_run:
        print(f"  would test {len(_models(args))} models under "
              f"{len(statistics_figures.SELECTIONS)} selection rules")
        return
    style.set_style()
    models = _models_with_tables(args)
    statistics_figures.build(models, args.metrics_dir)


def stage_figures(args) -> None:
    """Draw everything, from the committed numbers when they are present."""
    chosen_models = (_models_in_figure_data(args) if args.plot_only
                     else _models_with_tables(args))
    if args.dry_run:
        print(f"  would draw {args.which} for {len(chosen_models)} models")
        return

    style.set_style()
    wanted = {"main", "supplementary", "statistics"} if args.which == "all" else {args.which}

    # The study-wide test table is built FIRST: the marks of Fig. 4-6 are read from
    # it, so building the main figures before it would draw them from the previous
    # run's table (stale, or for other models).
    samples = tests = None
    if not args.plot_only and wanted & {"main", "statistics"}:
        samples, tests = statistics_figures.build(chosen_models, args.metrics_dir)
    if "main" in wanted:
        if not args.plot_only:
            main_figures.build_all(chosen_models, args.metrics_dir)
        main_figures.draw_all(chosen_models)
    if "supplementary" in wanted:
        if not args.plot_only:
            supplementary_figures.build_all(chosen_models, args.metrics_dir)
        supplementary_figures.draw_all(chosen_models, args.metrics_dir)
    if "statistics" in wanted:
        statistics_figures.draw_all(chosen_models, samples, tests)


def stage_scaling_distance(args) -> None:
    """The tables of Fig. 8, at the layer chosen on the reference set."""
    stage_metrics(argparse.Namespace(**{**vars(args), "scaling_distance": True,
                                        "layers": "selected"}))


ALL_STAGES = (("prep", stage_prep), ("samples", stage_samples), ("sweep", stage_sweep),
              ("select (word length)", stage_select), ("extract", stage_extract),
              ("benchmark", stage_benchmark), ("select (layer)", stage_select),
              ("metrics", stage_metrics),
              ("metrics (scaling and distance)", stage_scaling_distance),
              ("stats", stage_stats), ("figures", stage_figures))


def stage_all(args) -> None:
    """Every stage in order.

    ``select`` runs twice: the word length of the k-mer baselines is needed to
    extract them, the layer of the transformers needs their benchmark, which needs
    their extraction.  ``metrics`` runs twice too: the main tables, then the
    scaling/distance variants of Fig. 8 at the chosen layer.
    """
    for name, stage in ALL_STAGES:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        _record(name.split(" ")[0], args)
        stage(args)


# ── parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Benchmarking genomic language models on viral genomes.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")

    def n_ref(sub):
        sub.add_argument("--n-ref", type=int, default=config.N_REFERENCE,
                         help=f"number of reference samples (default {config.N_REFERENCE})")

    def allow_missing(sub):
        sub.add_argument("--allow-missing", action="store_true",
                         help="accept results computed from the reference samples that "
                              "did load (default: fail if any is missing or corrupt)")

    prep = subparsers.add_parser("prep", help="genomes -> fragments + reverse complements",
                                 description=stage_prep.__doc__)
    _prep_options(prep)
    _common(prep, with_models=False)
    prep.set_defaults(func=stage_prep)

    samples = subparsers.add_parser("samples", help="VMR + genomes -> reference samples",
                                    description=stage_samples.__doc__)
    n_ref(samples)
    _samples_options(samples)
    _common(samples, with_models=False)
    samples.set_defaults(func=stage_samples)

    sweep = subparsers.add_parser("sweep", help="reference samples -> k-mer length "
                                  "sweep (Fig. S1a)", description=stage_sweep.__doc__)
    n_ref(sweep)
    sweep.add_argument("--k-values", nargs="*", type=int, default=None,
                       dest="sweep_k_values",
                       help="word lengths to benchmark (default 1..10; a profile has 4^k "
                            "columns, so k = 10 needs several GB per worker)")
    _common(sweep, with_models=False)
    sweep.set_defaults(func=stage_sweep)

    extract = subparsers.add_parser("extract", help="fragments -> vectors",
                                    description=stage_extract.__doc__)
    n_ref(extract)
    extract.add_argument("--k", type=int, default=None,
                         help="word length for the k-mer baselines; default is the one "
                              "chosen by the 'select' stage")
    extract.add_argument("--pooling", choices=pooling_module.MODES,
                         default=pooling_module.DEFAULT_MODE,
                         help="how DNABERT-2/S token states become one vector: masked "
                              "mean over real tokens (default) or the [CLS] state")
    _common(extract)
    extract.set_defaults(func=stage_extract)

    benchmark = subparsers.add_parser("benchmark", help="reference samples -> ARI/P@5/silhouette",
                                      description=stage_benchmark.__doc__)
    n_ref(benchmark)
    allow_missing(benchmark)
    _common(benchmark)
    benchmark.set_defaults(func=stage_benchmark)

    select = subparsers.add_parser("select", help="choose layers and word lengths",
                                   description=stage_select.__doc__)
    select.add_argument("--select-by", default=config.REFERENCE_METRIC,
                        choices=["P5", "ARI", "silh"],
                        help=f"reference metric that picks the layer "
                             f"(default {config.REFERENCE_METRIC})")
    select.add_argument("--sweep-samples", type=int, default=10,
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
    n_ref(metrics)
    allow_missing(metrics)
    metrics.add_argument("--layers", choices=["all", "selected"], default="all",
                         help="score every layer, or only the one chosen on the "
                              "reference set (default all; 'selected' is up to 32x "
                              "less work but leaves the per-layer figures empty)")
    metrics.add_argument("--no-pairwise", action="store_true",
                         help="skip the order-against-order tables of Fig. 7")
    metrics.add_argument("--scaling-distance", action="store_true",
                         help="instead of the main tables, score the models compared by "
                              "standardised cosine under raw features and/or the "
                              "Euclidean distance, into metrics/scaling_distance/ "
                              "(Fig. 8; '--layers selected' is enough for the figure)")
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
    _prep_options(everything)
    _samples_options(everything)
    n_ref(everything)
    allow_missing(everything)
    everything.add_argument("--k", type=int, default=None)
    everything.add_argument("--pooling", choices=pooling_module.MODES,
                            default=pooling_module.DEFAULT_MODE)
    everything.add_argument("--k-values", nargs="*", type=int, default=None)
    everything.add_argument("--sweep-samples", type=int, default=10)
    everything.add_argument("--k-neighbours", type=int, default=config.PK_FOR_FIGURES)
    everything.add_argument("--select-by", default=config.REFERENCE_METRIC,
                            choices=["P5", "ARI", "silh"])
    everything.add_argument("--layers", choices=["all", "selected"], default="all")
    everything.add_argument("--no-pairwise", action="store_true")
    everything.add_argument("--scaling-distance", action="store_true", help=argparse.SUPPRESS)
    everything.add_argument("--which", default="all",
                            choices=["all", "main", "supplementary", "statistics"])
    everything.add_argument("--plot-only", action="store_true")
    _common(everything)
    everything.set_defaults(func=stage_all)

    return parser


_ARGV = None


def _record(stage: str, args) -> None:
    """Write the run manifest for a stage, before it does anything."""
    if getattr(args, "dry_run", False):
        return
    seeds = {"seed": args.seed}
    extra = {key: getattr(args, key) for key in
             ("n_ref", "sweep_samples", "pooling", "layers", "chunk_size",
              "max_chunks", "min_length", "n_genomes", "n_chunks", "k", "allow_missing",
              "scaling_distance")
             if hasattr(args, key)}
    provenance.write(stage, argv=_ARGV, seeds=seeds,
                     models=[m.key for m in _models(args)] if hasattr(args, "models") else None,
                     extra=extra)


def main(argv=None) -> None:
    global _ARGV
    args = build_parser().parse_args(argv)
    _ARGV = (["run_pipeline.py"] + list(argv)) if argv is not None else list(sys.argv)
    config.MEMORY_BUDGET = args.memory_budget
    config.MAX_WORKERS = args.max_workers
    config.SAMPLE_THREADS = args.sample_threads
    config.ensure_dirs(config.METRICS, config.FIGURE_DATA, config.STATISTICS)
    if args.stage != "all":
        _record(args.stage, args)
    args.func(args)
