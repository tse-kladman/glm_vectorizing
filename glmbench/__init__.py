"""glmbench -- benchmarking genomic language models on viral genomes.

The package is laid out in the order the analysis runs, so it can be read top to
bottom:

    config.py          where everything lives on disk, and the shared defaults
    models.py          the model registry: representation + distance of each model
    sequences.py       FASTA -> 2 kb chunks -> reverse complement
    representations/   turning sequences into vectors (k-mer, MinHash, transformers)
    neighbours.py      the nearest-neighbour core shared by every metric
    metrics/           reference-set, unseen-virus and pairwise metrics
    selection.py       choosing the layer and the word length on the reference set
    statistics.py      paired tests between models and multiple-testing corrections
    figures/           every plot

``run_pipeline.py`` at the repository root is the single command-line entry point
to all of it.
"""

__all__ = ["config", "models"]
