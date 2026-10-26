"""Turning sequences into the vectors each model compares.

    kmer.py          k-mer frequency profiles of any word length
    minhash.py       MinHash sketches of the canonical k-mer set
    pooling.py       token states -> one vector per fragment (masked mean or [CLS])
    transformers.py  DNABERT-2/S in process, Evo2 and ViraLM through their own envs
    store.py         reading and writing data/latent_dim/
    extract.py       running every extractor over the reference samples and viruses

Every extractor writes the same structure -- ``{label: array of rows}``, one pickle
per FASTA file -- so everything downstream treats the models identically.
"""

from . import kmer, minhash, store  # noqa: F401
