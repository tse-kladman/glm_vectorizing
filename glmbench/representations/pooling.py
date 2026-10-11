"""How the token states of a transformer become one vector per fragment.

Kept free of torch on purpose: the pooling arithmetic is plain numpy, so it can be
unit-tested on a CPU-only machine, and the transformer code only has to hand it the
hidden states of one layer and the attention mask.

    mean   average of the token states over the positions the attention mask marks
           as real -- **padding is excluded**.  (The first release of the DNABERT
           embeddings averaged over all 512 positions, padding included, and without
           a mask; see the README for what that changes.)
    cls    the state of the first token ([CLS]).

A fragment shorter than the padded length has padding positions whose states are
either zeros (DNABERT-2's unpadded attention re-pads with zeros) or the embedding of
the pad token (the embedding layer).  Either way they are not part of the fragment.
"""

import numpy as np

MODES = ("mean", "cls")
DEFAULT_MODE = "mean"


def pool(hidden: np.ndarray, attention_mask: np.ndarray, mode: str = DEFAULT_MODE) -> np.ndarray:
    """Pool the states of one layer: ``hidden`` (length, width), mask (length,) -> (width,).

    A batch is also accepted: ``hidden`` (batch, length, width), mask (batch, length)
    -> (batch, width).
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    mask = np.asarray(attention_mask)
    if mode not in MODES:
        raise ValueError(f"unknown pooling {mode!r}; choose from {MODES}")
    if hidden.ndim == 2:
        return pool(hidden[None], mask[None], mode)[0]
    if hidden.shape[:2] != mask.shape:
        raise ValueError(f"mask {mask.shape} does not match hidden states {hidden.shape}")

    if mode == "cls":
        return hidden[:, 0, :]
    weights = (mask != 0).astype(np.float64)
    counts = weights.sum(axis=1, keepdims=True)
    if (counts == 0).any():
        raise ValueError("a fragment has no real token: its attention mask is all zeros")
    return (hidden * weights[:, :, None]).sum(axis=1) / counts
