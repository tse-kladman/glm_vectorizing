"""Pooling of token states: padding must never enter a fragment's vector."""

import numpy as np
import pytest

from glmbench.representations import pooling


def test_masked_mean_ignores_padding():
    hidden = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0], [100.0, 100.0]])
    mask = np.array([1, 1, 0, 0])
    np.testing.assert_allclose(pooling.pool(hidden, mask), [2.0, 3.0])


def test_unmasked_mean_would_differ():
    """The defect being fixed: averaging all positions dilutes the vector with padding."""
    hidden = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    mask = np.array([1, 1, 0, 0])
    unmasked = hidden.mean(axis=0)
    assert not np.allclose(unmasked, pooling.pool(hidden, mask))
    np.testing.assert_allclose(unmasked * 2, pooling.pool(hidden, mask))    # scaled by n/L


def test_masked_mean_is_independent_of_the_amount_of_padding():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(7, 5))
    short = np.vstack([real, np.full((3, 5), 9.0)])
    long = np.vstack([real, np.full((50, 5), -4.0)])
    a = pooling.pool(short, np.r_[np.ones(7), np.zeros(3)])
    b = pooling.pool(long, np.r_[np.ones(7), np.zeros(50)])
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, real.mean(axis=0))


def test_cls_takes_the_first_token():
    hidden = np.arange(12.0).reshape(4, 3)
    np.testing.assert_allclose(pooling.pool(hidden, np.array([1, 1, 1, 0]), "cls"), [0, 1, 2])


def test_batch_matches_single_fragments():
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(3, 6, 4))
    mask = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    batch = pooling.pool(hidden, mask)
    for i in range(3):
        np.testing.assert_allclose(batch[i], pooling.pool(hidden[i], mask[i]))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="unknown pooling"):
        pooling.pool(np.zeros((2, 2)), np.ones(2), "max")
    with pytest.raises(ValueError, match="all zeros"):
        pooling.pool(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="does not match"):
        pooling.pool(np.zeros((3, 2)), np.ones(2))


def test_default_is_masked_mean():
    assert pooling.DEFAULT_MODE == "mean"


class FakeTensor:
    """Just enough of a torch tensor for the extraction code path, so no torch is needed."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def test_extraction_pools_padded_and_unpadded_layers_identically():
    """DNABERT-2's encoder returns only the real tokens; the embedding layer is padded.

    Both must give the mean over the real tokens, whichever shape arrives.
    """
    from glmbench.representations import transformers

    rng = np.random.default_rng(2)
    real = rng.normal(size=(5, 4)).astype("float32")
    mask = np.array([1] * 5 + [0] * 3)
    padded = np.vstack([real, np.zeros((3, 4), dtype="float32")])

    from_padded = transformers._pooled(FakeTensor(padded), mask, "mean")
    from_unpadded = transformers._pooled(FakeTensor(real), mask, "mean")
    np.testing.assert_allclose(from_padded, real.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(from_unpadded, real.mean(axis=0), rtol=1e-5)

    with pytest.raises(ValueError, match="neither"):
        transformers._pooled(FakeTensor(np.zeros((6, 4))), mask, "mean")
