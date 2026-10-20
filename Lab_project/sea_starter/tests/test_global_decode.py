from persian_sea.semantic.global_decode import (
    AlignmentCandidate,
    monotonic_global_decode,
)


def test_global_decode_prefers_monotonic_path():
    candidates = [
        [
            AlignmentCandidate(1.0, 2.0, 5.0, 0),
            AlignmentCandidate(4.0, 5.0, 4.0, 1),
        ],
        [
            AlignmentCandidate(0.5, 1.5, 10.0, 0),  # backward; disallowed
            AlignmentCandidate(2.2, 3.2, 4.0, 1),
        ],
    ]

    result = monotonic_global_decode(
        candidates,
        rough_times=[(1.0, 2.0), (2.2, 3.2)],
        overlap_penalty=2.0,
        gap_change_penalty=0.1,
    )

    assert result[0].start == 1.0
    assert result[1].start == 2.2
