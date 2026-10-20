
from __future__ import annotations
from persian_sea.core.registry import Registry
from persian_sea.semantic.global_decode import monotonic_global_decode


def greedy_monotonic(candidates_per_cue, rough_times,
                     overlap_penalty, gap_change_penalty):

    answer = []
    for i, group in enumerate(candidates_per_cue):
        valid = [x for x in group if not answer or x.start >= answer[-1].start]
        if not valid:
            raise ValueError(f"No monotonic path at subtitle {i}")
        answer.append(max(valid, key=lambda x: x.score))
    return answer


decoders = Registry('semantic global decoder')
decoders.register('monotonic_dp', lambda **_: monotonic_global_decode)
decoders.register('greedy_monotonic', lambda **_: greedy_monotonic)
