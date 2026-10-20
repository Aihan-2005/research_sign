from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from persian_sea.data.manifest import read_manifest, audit_manifest, write_id_files
from persian_sea.features.preflight import inspect_input
from persian_sea.features.cache_audit import inspect_cache_pair
from persian_sea.data.processing import run_steps
from persian_sea.config import load_config


def manifest(tmp_path, rows):
    f = tmp_path / 'recordings.csv'
    f.write_text('video_id,source_recording_id,signer_id,split\n' + rows, encoding='utf-8')
    return f


def test_manifest_writes_ids_only_if_allowed(tmp_path):
    rows = 'a,s1,person1,train\nb,s2,person2,val\nc,s3,person3,test\n'
    items = read_manifest(manifest(tmp_path, rows))
    report = audit_manifest(items, strict_signer=True)
    assert report['total_videos'] == 3
    files = write_id_files(items, tmp_path/'data')
    assert Path(files['train_ids.txt']).read_text().strip() == 'a'
    with pytest.raises(FileExistsError):
        write_id_files(items, tmp_path/'data')
    assert write_id_files(items, tmp_path/'data', overwrite=True)


def test_manifest_rejects_recording_leak(tmp_path):
    f=manifest(tmp_path,'a,original,one,train\nb,original,one,test\n')
    with pytest.raises(ValueError, match='leakage'):
        audit_manifest(read_manifest(f))


def test_manifest_signer_overlap_can_be_strict(tmp_path):
    rows='a,s1,one,train\nb,s2,one,test\n'
    items=read_manifest(manifest(tmp_path,rows))
    assert 'one' in audit_manifest(items)['signers_across_splits']
    with pytest.raises(ValueError,match='Signer-disjoint'):
        audit_manifest(items, strict_signer=True)


def test_manifest_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError,match='Invalid video_id'):
        read_manifest(manifest(tmp_path,'../secret,s1,x,train\n'))


def _data(tmp_path, text='سلام'):
    vs=tmp_path/'videos';ss=tmp_path/'subtitles';fs=tmp_path/'features';ts=tmp_path/'text'
    for p in (vs,ss,fs,ts):p.mkdir()
    (vs/'a.mp4').write_bytes(b'example')
    (ss/'a.vtt').write_text('WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n'+text+'\n', encoding='utf-8')
    starts=np.array([0,1,2],np.float32);ends=starts+.8
    np.savez_compressed(fs/'a.npz',pose=np.ones((3,2,75,4),np.float16),
                        rgb=np.ones((3,2,4,8),np.float16), starts=starts,ends=ends)
    np.savez_compressed(ts/'a.npz',texts=np.array([text]), embeddings=np.ones((1,8),np.float16))
    return vs,ss,fs,ts


def test_preflight_missing_subtitle(tmp_path):
    vs,ss,fs,ts=_data(tmp_path)
    (ss/'a.vtt').unlink()
    a=inspect_input('a',vs,ss)
    assert any(f['code']=='subtitle_missing' for f in a['findings'])


def test_cache_validates_feature_shapes_and_text(tmp_path):
    vs,ss,fs,ts=_data(tmp_path)
    ok=inspect_cache_pair('a',fs,ts,ss)
    assert not [f for f in ok['findings'] if f['severity']=='error']
    assert ok['num_signs']==3
    np.savez_compressed(ts/'a.npz',texts=np.array(['اشتباه']), embeddings=np.ones((1,8),np.float16))
    bad=inspect_cache_pair('a',fs,ts,ss)
    assert any(f['code']=='stale_text_cache' for f in bad['findings'])


def test_cache_warns_on_missing_pose_confidence(tmp_path):
    vs,ss,fs,ts=_data(tmp_path)
    with np.load(fs/'a.npz') as file:
        p=file['pose'].copy();r=file['rgb'];s=file['starts'];e=file['ends']
    p[...,3]=0
    np.savez_compressed(fs/'a.npz',pose=p,rgb=r,starts=s,ends=e)
    out=inspect_cache_pair('a',fs,ts,ss)
    assert any(f['code']=='pose_low_visibility' for f in out['findings'])


def test_preprocessing_stage_registry_dry_run(tmp_path):
    # Import the example YAML copied from the checked-out repository.
    config_path=Path(__file__).resolve().parents[1] / 'configs' / 'baseline.yaml'
    cfg=load_config(config_path)
    ids=tmp_path/'ids.txt';ids.write_text('a\n',encoding='utf-8')
    assert run_steps(cfg,['pose','segment','visual','text'],ids_file=ids,dry_run=True)==['pose','segment','visual','text']
    with pytest.raises(ValueError,match='Unknown step'):
        run_steps(cfg,['pose','bad'],ids_file=ids,dry_run=True)
    with pytest.raises(ValueError,match='No video IDs'):
        ids.write_text('',encoding='utf-8');run_steps(cfg,['pose'],ids_file=ids,dry_run=True)


def test_pose_video_length_check():
    from persian_sea.features.pose_quality import compare_pose_duration
    good=compare_pose_duration(pose_frames=300,pose_fps=30.,video_seconds=10.0)
    assert good['within_tolerance']
    bad=compare_pose_duration(pose_frames=300,pose_fps=30.,video_seconds=12.0)
    assert not bad['within_tolerance']
    with pytest.raises(ValueError):
        compare_pose_duration(pose_frames=0,pose_fps=30.,video_seconds=1)

def test_input_end_to_end_real_mp4(tmp_path):
    import shutil,subprocess
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe not installed')
    vs=tmp_path/'videos';ss=tmp_path/'subtitles';vs.mkdir();ss.mkdir()
    cmd=['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=320x240:r=10',
         '-t','2','-c:v','mpeg4','-y',str(vs/'real.mp4')]
    subprocess.run(cmd,check=True,capture_output=True)
    (ss/'real.vtt').write_text('WEBVTT\n\n00:00:00.200 --> 00:00:01.000\nنمونه\n',encoding='utf-8')
    result=inspect_input('real',vs,ss)
    assert result['duration_seconds'] >= 1.9
    assert result['video_dimensions'] == [320,240]
    assert not [f for f in result['findings'] if f['severity']=='error']


def test_motion_encoder_detects_motion_and_rejects_bad_shapes():
    import torch
    from persian_sea.model.motion_encoder import PoseDeltaEncoder
    model=PoseDeltaEncoder(output_dim=16,hidden_dim=16).eval()
    p=torch.ones(2,4,75,4)
    p[..., :3]=0.0
    with torch.no_grad():
        no_motion=model(p)
    assert no_motion.shape == (2,16)
    assert torch.equal(no_motion,torch.zeros_like(no_motion))
    p[:,1:,33:54,0]=torch.tensor([.2,.4,.6])[None,:,None]
    with torch.no_grad():
        moving=model(p)
    assert torch.isfinite(moving).all()
    assert moving.norm(dim=-1).min()>0
    with pytest.raises(ValueError,match='Expected pose'):
        model(torch.ones(2,4,74,4))


def test_detects_outdated_cache_model(tmp_path):
    import json
    vs,ss,fs,ts=_data(tmp_path)
    (fs/'a.json').write_text(json.dumps({'rgb_model':'old_model'}),encoding='utf-8')
    (ts/'a.json').write_text(json.dumps({'text_model':'BAAI/bge-m3'}),encoding='utf-8')
    out=inspect_cache_pair('a',fs,ts,ss,expected_rgb_model='facebook/dinov2-base',
                           expected_text_model='BAAI/bge-m3')
    assert any(x['code']=='stale_model_cache' for x in out['findings'])


def test_motion_backend_uses_same_visual_encoder_interface(monkeypatch):
    import sys, types, torch
    from torch import nn
    import torch.nn.functional as F
    from persian_sea.model.builders import build_visual_encoder, visual_backends

    class StubBase(nn.Module):
        def __init__(self, output_dim, **kwargs):
            super().__init__()
            self.projection = nn.Linear(4, output_dim)
        def forward(self, pose, rgb):
            pooled = pose.mean(dim=(1,2))
            return {'fused': F.normalize(self.projection(pooled),dim=-1)}

    fake = types.ModuleType('persian_sea.model.visual_encoder')
    fake.PoseRGBVisualEncoder = StubBase
    monkeypatch.setitem(sys.modules, 'persian_sea.model.visual_encoder', fake)
    assert 'pose_rgb_motion' in visual_backends.options
    model=build_visual_encoder('pose_rgb_motion', output_dim=8,
                               model_dim=12,pose_input_dim=4,
                               rgb_input_dim=6,num_views=4)
    p=torch.ones(2,3,75,4)
    p[:,:,33:54,0]=torch.arange(3)[None,:,None] * .2
    rgb=torch.randn(2,3,4,6)
    obj=model(p,rgb)
    assert obj['fused'].shape==(2,8) and obj['motion'].shape==(2,8)
    assert torch.isfinite(obj['fused']).all()
    obj['fused'].sum().backward()
    assert model.motion.projection.weight.grad is not None
