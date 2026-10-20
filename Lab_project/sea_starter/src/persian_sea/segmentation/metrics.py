from dataclasses import dataclass
import math

from persian_sea.types import SignSegment


def interval_iou(a: SignSegment, b: SignSegment) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0.0:
        return 0.0
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def _greedy_iou_match(
    predicted: list[SignSegment],
    gold: list[SignSegment],
    min_iou: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predicted):
        for gi, target in enumerate(gold):
            score = interval_iou(pred, target)
            if score >= min_iou:
                candidates.append((score, pi, gi))

    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []


    for score, pi, gi in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        matches.append((pi, gi, score))

    return matches


@dataclass
class SegmentationMetrics:
    iou_threshold: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_iou_sum: float = 0.0
    start_abs_error_sum: float = 0.0
    end_abs_error_sum: float = 0.0
    matched_count: int = 0

    def update(
        self,
        predicted: list[SignSegment],
        gold: list[SignSegment],
    ) -> None:
        matches = _greedy_iou_match(
            predicted,
            gold,
            self.iou_threshold,
        )

        tp = len(matches)
        self.tp += tp
        self.fp += len(predicted) - tp
        self.fn += len(gold) - tp

        for pi, gi, iou in matches:
            pred = predicted[pi]
            target = gold[gi]
            self.matched_iou_sum += iou
            self.start_abs_error_sum += abs(pred.start - target.start)
            self.end_abs_error_sum += abs(pred.end - target.end)
            self.matched_count += 1

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_matched_iou(self) -> float:
        return (
            self.matched_iou_sum / self.matched_count
            if self.matched_count
            else 0.0
        )

    @property
    def start_mae(self) -> float:
        return (
            self.start_abs_error_sum / self.matched_count
            if self.matched_count
            else math.inf
        )

    @property
    def end_mae(self) -> float:
        return (
            self.end_abs_error_sum / self.matched_count
            if self.matched_count
            else math.inf
        )

    def as_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "start_mae_seconds": self.start_mae,
            "end_mae_seconds": self.end_mae,
        }


@dataclass
class BoundaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(
        self,
        predicted_times: list[float],
        gold_times: list[float],
        tolerance_seconds: float,
    ) -> None:
        candidates: list[tuple[float, int, int]] = []
        for pi, p in enumerate(predicted_times):
            for gi, g in enumerate(gold_times):
                distance = abs(p - g)
                if distance <= tolerance_seconds:
                    candidates.append((distance, pi, gi))

        candidates.sort()
        used_pred: set[int] = set()
        used_gold: set[int] = set()
        local_tp = 0

        for _, pi, gi in candidates:
            if pi in used_pred or gi in used_gold:
                continue
            used_pred.add(pi)
            used_gold.add(gi)
            local_tp += 1

        self.tp += local_tp
        self.fp += len(predicted_times) - local_tp
        self.fn += len(gold_times) - local_tp

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
