from __future__ import annotations

import numpy as np
import pytest

from persian_sea.alignment.semantic_dp import SemanticDPSettings, align_semantic_dp
from persian_sea.io.subtitles import SubtitleCue
from persian_sea.types import SignSegment


def signs(start, count):
    return [SignSegment(float(i), float(i) + .85) for i in range(start, start + count)]


def config(**kwargs):
    return SemanticDPSettings(**kwargs)


def test_similarity_selects_later_group_despite_earlier_subtitle():
    chunks = signs(0, 8)
    subtitle = [SubtitleCue(0., 2., "علامت مربوط به ثانیه 4")]
    sims = np.array([[-.5, -.5, -.5, -.5, 1., 1., -.5, -.5]])
    result = align_semantic_dp(subtitle, chunks, sims, config(
        search_radius_seconds=10., max_group_segments=2,
        semantic_weight=30., duration_weight=0., gap_weight=0.,
        skipped_sign_cost=.1, null_cue_cost=100.,
    ))
    assert len(result.proposals) == 1
    assert result.proposals[0].first_sign_index == 4
    assert result.proposals[0].last_sign_index in (4, 5)


def test_monotonic_nonoverlapping_assignments():
    chunks = signs(0, 7)
    cues = [SubtitleCue(0, 2, 'a'), SubtitleCue(3, 5, 'b')]
    sim = np.array([[1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0]], dtype=float)
    result = align_semantic_dp(cues, chunks, sim, config(
        search_radius_seconds=4, max_group_segments=2,
        semantic_weight=15, null_cue_cost=100))
    assert len(result.proposals) == 2
    assert result.proposals[0].last_sign_index < result.proposals[1].first_sign_index
    for p in result.proposals:
        assert p.start == chunks[p.first_sign_index].start
        assert p.end == chunks[p.last_sign_index].end


def test_explicit_null_when_there_is_no_local_candidate():
    result = align_semantic_dp([SubtitleCue(40, 42, 'unmatched')], signs(0, 3),
                               np.zeros((1,3)), config(search_radius_seconds=1))
    assert not result.proposals
    assert result.unmatched_cue_indices == (0,)
    assert result.candidate_counts == (0,)


def test_no_null_path_fails_closed():
    with pytest.raises(ValueError, match='No feasible candidate'):
        align_semantic_dp([SubtitleCue(40, 42, 'missing')], signs(0, 2),
                          np.zeros((1,2)), config(search_radius_seconds=0,
                                                  allow_null_alignment=False))


def test_invalid_matrix_and_unsorted_inputs_rejected():
    with pytest.raises(ValueError, match='shape'):
        align_semantic_dp([SubtitleCue(0,2,'a')], signs(0,2),
                          np.ones((2,2)), config())
    with pytest.raises(ValueError, match='NaN/Inf'):
        align_semantic_dp([SubtitleCue(0,2,'a')], signs(0,2),
                          np.array([[np.nan,0]]), config())
    with pytest.raises(ValueError, match='sorted'):
        align_semantic_dp([SubtitleCue(3,4,'a'), SubtitleCue(0,1,'b')], signs(0,2),
                          np.zeros((2,2)), config())


def test_search_restricts_both_group_boundaries():
    result = align_semantic_dp([SubtitleCue(3,4,'x')], signs(0,7),
                               np.ones((1,7)), config(search_radius_seconds=.1,
                                                      null_cue_cost=999))
    assert len(result.proposals) == 1
    assert 2 <= result.proposals[0].first_sign_index <= 4
    assert result.proposals[0].last_sign_index <= 4


def test_skipping_unmatched_signs_and_reporting_coverage():
    result = align_semantic_dp([SubtitleCue(2,3,'x')], signs(0,5),
                               np.array([[0,0,1,0,0]], dtype=float),
                               config(search_radius_seconds=1, max_group_segments=1,
                                      semantic_weight=20, null_cue_cost=100))
    assert result.proposals[0].first_sign_index == 2
    assert result.skipped_signs == 4


def test_empty_cues_is_noop():
    result = align_semantic_dp([], signs(0,3), np.zeros((0,3)), config())
    assert result.proposals == ()
    assert result.unmatched_cue_indices == ()
    assert result.skipped_signs == 3
