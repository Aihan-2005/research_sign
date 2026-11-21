import numpy as np
import pytest
from persian_sea.semantic.candidates import select_candidate_indices


def test_overlapping_signs_can_have_unsorted_midpoints():
    starts = np.array([0., 1., 2.])
    ends = np.array([9., 1.1, 2.1])
    # mids are [4.5, 1.05, 2.05], not sorted despite sorted starts.
    indices = select_candidate_indices(starts, ends, 1.0, 1.1,
                                       radius=0.05, min_candidates=1,
                                       max_candidates=3)
    assert indices.tolist() == [1]


def test_invalid_timestamps_rejected():
    with pytest.raises(ValueError, match='finite'):
        select_candidate_indices(np.array([0., np.nan]), np.array([1., 2.]),
                                 0.1, 0.5, 0., 1, 2)
