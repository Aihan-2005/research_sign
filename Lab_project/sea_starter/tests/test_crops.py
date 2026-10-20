import numpy as np

from persian_sea.features.crops import _square_box_from_points


def test_square_box():
    pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
    box = _square_box_from_points(pts, width=100, height=100, padding=1.0, min_side=1)
    assert box is not None
    assert box.width == box.height
