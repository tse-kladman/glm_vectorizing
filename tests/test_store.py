"""How the stored representations are read back: ViraLM's windows."""

import numpy as np
import pytest

from glmbench.representations import store


def _windows():
    # two 2,000-nt fragments of class A and one of class B, each as two 1,000-nt windows
    return {
        "A_0_0_1000": np.array([1.0, 0.0]), "A_0_1000_2000": np.array([3.0, 2.0]),
        "A_1_0_1000": np.array([5.0, 5.0]), "A_1_1000_2000": np.array([7.0, 5.0]),
        "B_0_0_1000": np.array([0.0, 1.0]), "B_0_1000_2000": np.array([0.0, 3.0]),
    }


def test_viralm_windows_of_one_fragment_become_one_point():
    """A sister window must not be a separate point: it is always the nearest
    'same class' neighbour and inflates every P@k of ViraLM."""
    groups = store._viralm_groups(_windows())
    assert sorted(groups) == ["A", "B"]
    assert len(groups["A"]) == 2 and len(groups["B"]) == 1
    np.testing.assert_allclose(groups["A"][0], [2.0, 1.0])     # mean of the two windows
    np.testing.assert_allclose(groups["A"][1], [6.0, 5.0])     # FASTA order kept
    np.testing.assert_allclose(groups["B"][0], [0.0, 2.0])


def test_viralm_labels_with_underscores_and_a_skipped_window():
    raw = {"REVERSECOMPLEMENTMZ_123.1_4_0_1000": np.array([1.0]),
           "REVERSECOMPLEMENTMZ_123.1_4_1000_2000": np.array([3.0]),
           "REVERSECOMPLEMENTMZ_123.1_5_1000_2000": np.array([9.0])}   # window 0 skipped
    groups = store._viralm_groups(raw)
    assert list(groups) == ["REVERSECOMPLEMENTMZ_123.1"]
    np.testing.assert_allclose(np.vstack(groups["REVERSECOMPLEMENTMZ_123.1"]), [[2.0], [9.0]])


def test_viralm_window_mode_of_the_first_release_is_still_available():
    groups = store._viralm_groups(_windows(), windows="windows")
    assert len(groups["A"]) == 4 and len(groups["B"]) == 2
    with pytest.raises(ValueError):
        store._viralm_groups(_windows(), windows="max")
