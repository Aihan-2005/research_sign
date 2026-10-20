from persian_sea.segmentation.metrics import (
    BoundaryCounts,
    SegmentationMetrics,
    interval_iou,
)
from persian_sea.types import SignSegment


def test_interval_iou():
    a = SignSegment(0.0, 2.0)
    b = SignSegment(1.0, 3.0)
    assert abs(interval_iou(a, b) - 1 / 3) < 1e-8


def test_perfect_segmentation_f1():
    pred = [SignSegment(1.0, 2.0), SignSegment(3.0, 4.0)]
    gold = [SignSegment(1.0, 2.0), SignSegment(3.0, 4.0)]
    metric = SegmentationMetrics(0.5)
    metric.update(pred, gold)
    assert metric.f1 == 1.0


def test_boundary_counts_are_video_local():
    counts = BoundaryCounts()
    counts.update([1.0], [1.0], 0.01)
    counts.update([1.0], [2.0], 0.01)
    assert counts.tp == 1
    assert counts.fp == 1
    assert counts.fn == 1
