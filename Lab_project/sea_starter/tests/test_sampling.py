import numpy as np

from persian_sea.features.sampling import sample_times, times_to_indices


def test_sample_times_are_inside_interval():
    t = sample_times(1.0, 3.0, 4)
    assert len(t) == 4
    assert np.all(t > 1.0)
    assert np.all(t < 3.0)


def test_times_to_indices_clips():
    idx = times_to_indices(np.array([-1.0, 0.0, 10.0]), fps=25.0, length=100)
    assert idx.tolist() == [0, 0, 99]
