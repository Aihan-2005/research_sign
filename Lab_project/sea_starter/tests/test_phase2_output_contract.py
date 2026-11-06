from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from persian_sea.alignment.semantic_dp import SemanticDPSettings, SemanticDPResult
from persian_sea.alignment_sources import read_review_actions, verify_review_matches_proposal
from persian_sea.io.subtitles import SubtitleCue, read_subtitles
from persian_sea.types import Alignment


def _script_module():
    path = Path(__file__).parents[1] / 'scripts/align_sea_semantic_vlp.py'
    spec = importlib.util.spec_from_file_location('phase2_sea_script_for_test', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _dummy_paths(tmp_path):
    result = []
    for name in ('vlp.pt','visual.npz','text.npz'):
        path = tmp_path / name
        path.write_bytes(b'reproducible-sha256-test-only')
        result.append(path)
    return result


def test_output_null_does_not_create_trainable_proposed_vtt(tmp_path):
    script = _script_module()
    model, visual, text = _dummy_paths(tmp_path)
    raw = [SubtitleCue(1,2,'الف'), SubtitleCue(4,5,'ب')]
    result = SemanticDPResult((Alignment(0,1,2,0,0),), (1,), (1,0), 8., 2)
    proposed, report = script.create_outputs(
        'test', raw, raw, result, tmp_path/'out',
        settings=SemanticDPSettings(), ckpt_path=model, visual_path=visual,
        text_path=text, review_threshold=1.5, overwrite=False,
        similarity_matrix=np.ones((2,3)),
    )
    assert proposed is None and not (tmp_path/'out/test.proposed.vtt').exists()
    info = json.loads(report.read_text(encoding='utf8'))
    assert info['status'] == 'incomplete_abstained'
    assert info['num_unmatched_cues'] == 1
    assert info['decisions'][1]['action'] == 'no_proposal'


def test_complete_proposed_vtt_compatible_with_phase1_review_gate(tmp_path):
    script = _script_module()
    model, visual, text = _dummy_paths(tmp_path)
    raw = [SubtitleCue(1,2,'سلام')]
    result = SemanticDPResult((Alignment(0,1.1,2.1,0,1),), (), (2,), 1., 1)
    proposed, report = script.create_outputs(
        'test', raw, raw, result, tmp_path/'out',
        settings=SemanticDPSettings(), ckpt_path=model, visual_path=visual,
        text_path=text, review_threshold=1.5, overwrite=False,
        similarity_matrix=np.ones((1,3)),
    )
    assert proposed is not None
    assert read_subtitles(proposed)[0].start == 1.1
    assert read_review_actions(proposed, 1, required=True) == ['suggested_only']
    verify_review_matches_proposal(proposed, read_subtitles(proposed))
    assert json.loads(report.read_text(encoding='utf8'))['proposals'][0]['mean_cosine_similarity'] == 1.0
