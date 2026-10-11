"""The model registry -- the single place that defines what a model *is*.

A model is not just an encoder: it is a representation together with the distance
that representation is compared under.  Those two cannot be separated, because a
distance is only defined on the right kind of vector:

    vector        arbitrary real embedding   -> standardised, compared by cosine
    distribution  k-mer frequency profile    -> a point on the simplex, so metrics
                                                defined there (Jensen-Shannon) apply
    sketch        MinHash of a k-mer set     -> compared by the Jaccard estimate

Keeping that pairing in one table is what lets the rest of the code treat every
model identically: ``metrics`` only ever asks the registry how to transform a
matrix and how to measure distance in it.

Two models tune their word length ``k`` on the reference set (see selection.py);
the transformers tune their layer there instead.  Nothing is ever tuned on the
unseen orders.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str                 # command-line name and folder name on disk
    display: str             # label in figures
    color: str               # Okabe-Ito, colour-blind safe
    kind: str                # 'vector' | 'distribution' | 'sketch'
    transform: str           # 'standardize' | 'none'
    metric: str              # 'cosine' | 'jensenshannon' | 'minhash'
    extractor: str           # 'kmer' | 'minhash' | 'bert' | 'evo2' | 'viralm'
    layered: bool            # does it have layers to choose between?
    hf_name: str = ""        # HuggingFace id, for the in-process transformers
    conda_env: str = ""      # separate environment, for the subprocess extractors
    fixed_k: int | None = None       # word length, when it is not tuned
    k_values: tuple = ()             # word lengths tried on the reference set
    sketch_size: int = 0             # MinHash only
    canonical: bool = False          # k-mer profile over canonical k-mers (RC-invariant)
    revision: str | None = None      # HuggingFace commit / release pinned for the run;
                                     # None = whatever the hub served (recorded in the
                                     # run manifest only if you set it, so set it)

    @property
    def tunes_k(self) -> bool:
        return bool(self.k_values)

    @property
    def tunes_layer(self) -> bool:
        return self.layered


MODELS: dict[str, Model] = {}


def _register(model: Model) -> Model:
    MODELS[model.key] = model
    return model


# ── classical baselines ─────────────────────────────────────────────────────
#
# The word length of the two tuned baselines is swept over a range chosen from how
# the metrics saturate: short words put every k-mer in every fragment (the Jaccard
# index goes to one, profiles become indistinguishable), long words make fragments
# share nothing (Jaccard goes to zero, Jensen-Shannon saturates at its 1-bit
# ceiling).  Only in between does the metric carry signal.

KMER = _register(Model(
    key="kmer", display="k-mer", color="#999999",
    kind="distribution", transform="standardize", metric="cosine",
    extractor="kmer", layered=False, fixed_k=4,
))

# The same profile, but over canonical k-mers: every word is replaced by
# min(word, reverse complement) before counting, so the profile of a fragment and of
# its reverse complement are identical by construction.  It is the RC-invariant
# baseline against which the reverse-complement behaviour of the learned models is read.
KMER_CANON = _register(Model(
    key="kmer_canon", display="k-mer (canonical)", color="#000000",
    kind="distribution", transform="standardize", metric="cosine",
    extractor="kmer", layered=False, fixed_k=4, canonical=True,
))

KMER_JS = _register(Model(
    key="kmer_js", display="k-mer (JS)", color="#56B4E9",
    kind="distribution", transform="none", metric="jensenshannon",
    extractor="kmer", layered=False, k_values=(3, 4, 5, 6, 7),
))

MINHASH = _register(Model(
    key="minhash", display="MinHash", color="#CC79A7",
    kind="sketch", transform="none", metric="minhash",
    extractor="minhash", layered=False, k_values=(6, 7, 8, 9, 10), sketch_size=128,
))

# ── genomic language models ─────────────────────────────────────────────────

DNABERT2 = _register(Model(
    key="dnabert2", display="DNABERT-2", color="#0072B2",
    kind="vector", transform="standardize", metric="cosine",
    extractor="bert", layered=True, hf_name="zhihan1996/DNABERT-2-117M",
))

DNABERTS = _register(Model(
    key="dnabertS", display="DNABERT-S", color="#E69F00",
    kind="vector", transform="standardize", metric="cosine",
    extractor="bert", layered=True, hf_name="zhihan1996/DNABERT-S",
))

VIRALM = _register(Model(
    key="viralm", display="ViraLM", color="#009E73",
    kind="vector", transform="standardize", metric="cosine",
    extractor="viralm", layered=True, conda_env="viralm",
))

EVO2 = _register(Model(
    key="evo2", display="Evo2-7B", color="#D55E00",
    kind="vector", transform="standardize", metric="cosine",
    extractor="evo2", layered=True, conda_env="evo2_env",
))


# Canonical order for figures and tables: baselines first, then the models.
ORDER = ["kmer", "kmer_canon", "kmer_js", "minhash", "dnabert2", "dnabertS", "viralm", "evo2"]

# The five models of the first release of the tables; kept so that the regression
# tests can still reproduce those numbers exactly (the later additions -- MinHash,
# Jensen-Shannon, canonical k-mers -- are tested separately).
PUBLISHED = ["kmer", "dnabert2", "dnabertS", "viralm", "evo2"]


def get(key: str) -> Model:
    try:
        return MODELS[key]
    except KeyError:
        raise SystemExit(f"unknown model {key!r}; known models: {', '.join(ORDER)}")


def resolve(keys=None) -> list[Model]:
    """Model objects for a list of keys, in canonical order; None means all."""
    if not keys:
        return [MODELS[k] for k in ORDER]
    chosen = {get(k).key for k in keys}
    return [MODELS[k] for k in ORDER if k in chosen]


def display(key: str) -> str:
    return MODELS[key].display if key in MODELS else key


def color(key: str) -> str:
    return MODELS[key].color if key in MODELS else "#444444"
