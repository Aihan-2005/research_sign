
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

from persian_sea.io.subtitles import SubtitleCue, write_vtt
from persian_sea.diagnostics.oracle_coverage import CandidatePolicy, evaluate_oracle
from persian_sea.diagnostics.alignment_benchmark import benchmark_alignments
from persian_sea.config import load_config
from persian_sea.video_text.config import VideoTextConfig


def _fixture(tmp_path: Path):
    raw, gt, cache, alignment = [tmp_path / name for name in
                                 ('raw', 'gt', 'features', 'aligned')]
    for path in (raw, gt, cache, alignment):
        path.mkdir()
    write_vtt([SubtitleCue(1., 3., 'فردا باران'),
               SubtitleCue(3., 5., 'هوا خوب است')], raw / 'episode1.vtt')
    write_vtt([SubtitleCue(1., 3., 'فردا باران'),
               SubtitleCue(3., 5., 'هوا خوب است')], gt / 'episode1.vtt')
    np.savez_compressed(cache / 'episode1.npz', starts=np.arange(7, dtype=np.float32),
                        ends=np.arange(1, 8, dtype=np.float32))
    ids = tmp_path / 'val_ids.txt'
    ids.write_text('episode1\n', encoding='utf-8')
    return raw, gt, cache, alignment, ids


def _policy(name='semantic', width=1):
    return (CandidatePolicy('semantic', 0.0, 2, 7,
                            max_span_segments=width, max_span_seconds=12)
            if name == 'semantic' else CandidatePolicy('video_text', 0, 2, 7,
                                                      span_sizes=(width,)))


def _run(paths, policy):
    raw, gt, features, _, ids = paths
    return evaluate_oracle(ids, subtitle_dir=raw, verified_gt_dir=gt,
                           feature_dir=features, policy=policy)


def test_oracle_is_strictly_bounded_by_available_spans(tmp_path):
    parts = _fixture(tmp_path)
    report = _run(parts, _policy(width=1))
    assert report['summary']['verified_cues'] == 2
    assert report['summary']['mean_oracle_iou'] < 1
    assert report['summary']['mean_window_unrestricted_iou'] == 1.0
    assert report['per_cue'][0]['loss_from_span_constraints'] > 0
    assert report['per_cue'][0]['num_candidate_spans'] >= 1
    assert report['summary']['oracle_recall_at_iou_0.75'] == 0.0


def test_longer_spans_reach_verified_window(tmp_path):
    parts = _fixture(tmp_path)
    short = _run(parts, _policy(width=1))
    long = _run(parts, _policy(width=2))
    assert long['summary']['mean_oracle_iou'] == 1.0
    assert long['summary']['mean_oracle_iou'] > short['summary']['mean_oracle_iou']


def test_video_text_span_sizes_are_not_semantic_contiguous_ranges(tmp_path):
    parts = _fixture(tmp_path)
    narrow = _run(parts, _policy('video_text', 1))
    wide = _run(parts, _policy('video_text', 2))
    assert narrow['summary']['mean_oracle_iou'] == 0.5
    assert wide['summary']['mean_oracle_iou'] == 1.0
    cfg = VideoTextConfig(span_sizes=(1,2),batch_size=2)
    assert CandidatePolicy.video_text(cfg).span_sizes == (1,2)


def test_missing_verified_gt_is_never_treated_as_raw_gt(tmp_path):
    parts = _fixture(tmp_path)
    (parts[1] / 'episode1.vtt').unlink()
    with pytest.raises(ValueError, match='No matching HUMAN-VERIFIED'):
        _run(parts, _policy(width=2))


def test_partial_gt_is_reported_explicitly(tmp_path):
    parts = _fixture(tmp_path)
    write_vtt([SubtitleCue(1.,3., 'فردا باران')], parts[1] / 'episode1.vtt')
    report = _run(parts, _policy(width=2))
    assert report['summary']['verified_cues'] == 1
    assert report['summary']['unmatched_cues_in_gt_videos'] == 1
    with pytest.raises(ValueError, match='Incomplete VERIFIED GT'):
        evaluate_oracle(parts[4], subtitle_dir=parts[0], verified_gt_dir=parts[1],
                        feature_dir=parts[2], policy=_policy(width=2),
                        require_complete_gt=True)


def test_invalid_segment_timing_fails_closed(tmp_path):
    parts = _fixture(tmp_path)
    np.savez_compressed(parts[2] / 'episode1.npz', starts=np.array([0.,float('nan')]),
                        ends=np.array([1.,3.]))
    with pytest.raises(ValueError, match='Invalid visual feature timestamps'):
        _run(parts, _policy(width=2))


def test_missing_visual_cache_fails_loudly(tmp_path):
    parts = _fixture(tmp_path)
    (parts[2] / 'episode1.npz').unlink()
    with pytest.raises(FileNotFoundError, match='Visual feature cache missing'):
        _run(parts, _policy(width=2))


def test_invalid_policy(tmp_path):
    with pytest.raises(ValueError, match='segment'):
        CandidatePolicy('semantic', 0, 5, 2, max_span_segments=3,max_span_seconds=4)
    with pytest.raises(ValueError, match='span_sizes'):
        CandidatePolicy('video_text', 4, 2, 8)


def _cfg(parts):
    raw, gt, _, out, _ = parts
    return SimpleNamespace(paths=SimpleNamespace(subtitles_raw=raw, subtitles_gt=gt,
                                                 alignment_output=out))


def test_alignment_suite_uses_identical_verified_cues(tmp_path):
    parts = _fixture(tmp_path)
    raw, gt, _, out, ids = parts

    write_vtt([SubtitleCue(1, 3, 'فردا باران'),
               SubtitleCue(2, 4, 'هوا خوب است')], out / 'episode1.proposed.vtt')
    (out / 'semantic').mkdir()
    write_vtt([SubtitleCue(1, 3, 'فردا باران'),
               SubtitleCue(3, 5, 'هوا خوب است')], out / 'semantic/episode1.proposed.vtt')
    report = benchmark_alignments(_cfg(parts), ids, ('raw','sea_proposed','semantic_proposed'))
    assert report['common_verified_cues'] == 2
    m = report['matched_common_method_metrics']
    assert m['raw']['mean_iou'] == pytest.approx(1.0)
    assert m['sea_proposed']['mean_iou'] < m['semantic_proposed']['mean_iou']
    assert m['sea_proposed']['cues'] == 2


def test_alignment_suite_missing_preds_reported_and_fair_comparison_empty(tmp_path):
    parts = _fixture(tmp_path)
    report = benchmark_alignments(_cfg(parts), parts[4], ('raw','semantic_proposed'))
    assert report['common_verified_cues'] == 0
    assert report['missing_prediction_videos']['semantic_proposed'] == ['episode1']
    assert report['each_method_on_available_cues_NOT_directly_comparable']['raw']['cues'] == 2
    with pytest.raises(ValueError, match='Incomplete benchmark'):
        benchmark_alignments(_cfg(parts), parts[4], ('raw','semantic_proposed'), require_all=True)


def test_alignment_suite_rejects_mismatched_text_and_order(tmp_path):
    parts = _fixture(tmp_path)
    write_vtt([SubtitleCue(1,3,'wrong text'), SubtitleCue(3,5,'هوا خوب است')],
              parts[3] / 'episode1.proposed.vtt')
    with pytest.raises(ValueError, match='text/order'):
        benchmark_alignments(_cfg(parts), parts[4], ('raw','sea_proposed'))


def test_alignment_suite_not_allows_raw_gt_substitution(tmp_path):
    parts = _fixture(tmp_path)
    (parts[1] / 'episode1.vtt').unlink()
    with pytest.raises(ValueError, match='No independently VERIFIED GT'):
        benchmark_alignments(_cfg(parts), parts[4], ('raw',))


def test_entrypoints_include_all_research_commands():
    path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    content = path.read_text(encoding='utf-8')
    script_section = content.split('[project.scripts]', 1)[1].split('[tool.', 1)[0]
    for script in ('persian-sea-oracle-coverage','persian-sea-eval-alignment-suite',
                   'persian-sea-preflight','persian-sea-preprocess',
                   'persian-sea-train-video-text','persian-sea-audit-signers'):
        assert f'{script} = ' in script_section


def test_backend_specific_preflight_calls_text_cache_identity(tmp_path,monkeypatch,capsys):
    from persian_sea.cli import preflight
    path = tmp_path / 'text' / 'bge_m3'
    path.mkdir(parents=True)
    (path/'episode1.npz').touch()
    ids = tmp_path/'ids.txt';ids.write_text('episode1\n')
    fake = SimpleNamespace(paths=SimpleNamespace(video_ids=ids, videos=tmp_path,
        subtitles_raw=tmp_path, poses=tmp_path, feature_cache=tmp_path,
        text_feature_cache=tmp_path/'text'),
        scanner=SimpleNamespace(rgb_model='facebook/dinov2-base',samples_per_segment=8,
                                resolved_views=lambda:['signer']),
        semantic=SimpleNamespace(text_model='BAAI/bge-m3'))
    captured = {}
    monkeypatch.setattr(preflight,'load_config',lambda _:fake)
    monkeypatch.setattr(preflight,'text_cache_directory',lambda _:path)
    monkeypatch.setattr(preflight,'inspect_input',lambda *a:{'video_id':'episode1','findings':[]})
    def cache(*args,**kwargs):
        captured['text_dir']=args[2]
        return {'findings':[]}
    monkeypatch.setattr(preflight,'inspect_cache_pair',cache)
    def verify(cache,cfg,video_id):
        captured['verified']=str(cache)
    monkeypatch.setattr(preflight,'verify_text_cache',verify)
    monkeypatch.setattr(sys,'argv',['preflight','--config','dummy.yaml', '--check-caches'])
    preflight.main()
    assert captured['text_dir']==path
    assert captured['verified']==str(path/'episode1.npz')
    assert json.loads(capsys.readouterr().out)['errors']==0


def test_preflight_cache_identity_mismatch_marks_error(tmp_path,monkeypatch,capsys):
    from persian_sea.cli import preflight
    path = tmp_path/'text';path.mkdir()
    (path/'episode1.npz').touch()
    ids=tmp_path/'ids';ids.write_text('episode1\n')
    fake = SimpleNamespace(paths=SimpleNamespace(video_ids=ids,videos=tmp_path,
        subtitles_raw=tmp_path,poses=tmp_path,feature_cache=tmp_path,
        text_feature_cache=path),
        scanner=SimpleNamespace(rgb_model='rgb',samples_per_segment=8,
                                resolved_views=lambda:[]),
        semantic=SimpleNamespace(text_model='txt'))
    monkeypatch.setattr(preflight,'load_config',lambda _:fake)
    monkeypatch.setattr(preflight,'text_cache_directory',lambda _:path)
    monkeypatch.setattr(preflight,'inspect_input',lambda *a:{'findings':[]})
    monkeypatch.setattr(preflight,'inspect_cache_pair',lambda *a,**k:{'findings':[]})
    def fail(*_): raise ValueError('Wrong text model revision')
    monkeypatch.setattr(preflight,'verify_text_cache',fail)
    monkeypatch.setattr(sys,'argv',['preflight','--config','dummy','--check-caches'])
    with pytest.raises(SystemExit) as exc: preflight.main()
    assert exc.value.code==2
    report=json.loads(capsys.readouterr().out)
    assert report['videos'][0]['cache']['findings'][0]['code']=='text_cache_provenance'


def test_oracle_rejects_approved_pair_source_to_prevent_easy_oracle():
    with pytest.raises(ValueError, match='pair_source=raw'):
        CandidatePolicy.video_text(VideoTextConfig(pair_source='approved'))
